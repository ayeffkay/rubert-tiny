import math
import os
import time
import shutil

import psutil

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset
from transformers import get_constant_schedule_with_warmup
from utils import logger
import sys
from pathlib import Path


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AutoConfig

import custom_step
import hyptorch.nn as hypnn
import hyptorch.pmath as hypmath
import delta

class Distiller:
    def __init__(
        self, params: dict, 
        student: nn.Module, teacher: nn.Module,
        train_dataset: LmSeqsDataset, valid_dataset: LmSeqsDataset,
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu

        self.student = student
        self.teacher = teacher

        self.student_config = AutoConfig.from_pretrained(params.student_name)
        self.student_vocab_size = student.config.vocab_size
        self.teacher_vocab_size = teacher.config.vocab_size

        logger.info("Train size: {}, valid_size: {}".format(len(train_dataset), len(valid_dataset)))
        if params.gpus <= 1:
            train_sampler = RandomSampler(train_dataset)
            valid_sampler = RandomSampler(valid_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
            valid_sampler = DistributedSampler(valid_dataset)

        if params.group_by_size:
            train_groups = create_lengths_groups(lengths=train_dataset.st_len,
                                                 k=params.max_model_input_size)
            train_sampler = GroupedBatchSampler(sampler=train_sampler,
                                                group_ids=train_groups,
                                                batch_size=params.batch_size)
            valid_groups = create_lengths_groups(lengths=valid_dataset.st_len,
                                                 k=params.max_model_input_size)
            valid_sampler = GroupedBatchSampler(sampler=valid_sampler,
                                                group_ids=valid_groups,
                                                batch_size=params.batch_size)
        else:
            train_sampler = BatchSampler(sampler=train_sampler, batch_size=params.batch_size, drop_last=False)
            valid_sampler = BatchSampler(sampler=valid_sampler, batch_size=params.batch_size, drop_last=False)

        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_sampler=train_sampler,
                                           collate_fn=train_dataset.batch_sequences)
        self.valid_dataloader = DataLoader(dataset=valid_dataset,
                                           batch_sampler=valid_sampler,
                                           collate_fn=valid_dataset.batch_sequences)
        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos
        self.alpha_contrastive = params.alpha_contrastive 

        if self.alpha_mlm:
            logger.info("Using MLM loss for LM step.")
            self.mlm_mask_prop = params.mlm_mask_prop
            assert 0.0 <= self.mlm_mask_prop <= 1.0
            assert params.word_mask + params.word_keep + params.word_rand == 1.0
            self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
            self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.gpus > 0 else self.pred_probs
            if params.student_token_probs is not None:
                self.student_token_probs = params.student_token_probs.to(f"cuda:{params.local_rank}") if params.gpus > 0 else params.student_token_probs
            else:
                self.student_token_probs = None

        self.epoch = 0

        self.n_train_iter_epoch = 0
        self.n_train_iter_total = 0

        self.n_gradient_updates_epoch = 0
        self.n_gradient_updates_total = 0

        self.n_sequences_epoch = 0
        self.n_sequences_total = 0

        self.last_train_loss = 0
        self.total_train_loss_epoch = 0
        self.total_valid_loss_epoch = 0
        self.best_total_valid_loss_epoch = sys.maxsize
        self.n_valid_iter_epoch = 0
        self.last_log = 0


        if self.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
            self.student = DistributedDataParallel(
                self.student,
                device_ids=[params.local_rank],
                output_device=params.local_rank,
                find_unused_parameters=True,
            )

        self.is_master = params.is_master
        if self.is_master:
            logger.info("--- Initializing Tensorboard")
            Path(self.params.tensorboard_logs_path).mkdir(parents=True, exist_ok=True)
            log_dir = os.path.join(self.params.tensorboard_logs_path, self.params.tensorboard_log_name)
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            self.tensorboard = SummaryWriter(log_dir=log_dir, flush_secs=1)
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            }
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        self.num_steps_epoch = len(self.train_dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )
        self.warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.const_scheduler_with_warmup = get_constant_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=self.warmup_steps)
        self.reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', threshold_mode='abs', 
                                                                            factor=params.reduce_factor, patience=params.valid_epochs_patience, 
                                                                            threshold=1e-2, cooldown=params.valid_epochs_patience, 
                                                                            eps=1e-10, min_lr=1e-6, verbose=True)
                                                        
        if self.alpha_ce > 0.0:   
            self.last_loss_ce = 0
            self.last_valid_loss_ce_epoch = 0
            self.ce_loss_fct = nn.KLDivLoss(reduction='sum')
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = 0
            self.last_valid_loss_mlm_epoch = 0
            self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        layers_ct = 4 if self.params.projection_strategy in ['last', 'skip', 'average'] else 1
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
            self.last_valid_loss_mse_epoch = 0
            self.mse_loss_fct = nn.MSELoss(reduction='sum')

            if self.params.project_to == 'teacher':
                self.hid_projectors_mse_student = nn.ModuleList([nn.Linear(student.config.hidden_size, 
                                                                    teacher.config.hidden_size).to(f'cuda:{params.local_rank}') for _ in range(layers_ct)])
                self.hid_projectors_mse_teacher = None

            elif self.params.project_to == 'student':
                self.hid_projectors_mse_teacher = nn.ModuleList([nn.Linear(teacher.config.hidden_size, 
                                                                student.config.hidden_size).to(f'cuda:{params.local_rank}') for _ in range(layers_ct)])
                self.hid_projectors_mse_student = None
        
            else:
                self.hid_projectors_mse_teacher = nn.ModuleList([nn.Linear(teacher.config.hidden_size, 
                                                                self.params.intermediate_dim).to(f'cuda:{params.local_rank}') for _ in range(layers_ct)])
                self.hid_projectors_mse_student = nn.ModuleList([nn.Linear(student.config.hidden_size, 
                                                                self.params.intermediate_dim).to(f'cuda:{params.local_rank}') for _ in range(layers_ct)])

        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
            self.last_valid_loss_cos_epoch = 0
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
            ## TODO: self.attn_projector

        self.s_hid_dim = student.config.hidden_size
        self.t_hid_dim = teacher.config.hidden_size

        self.do_hyperbolic_mapping_in_step = True 
        if self.alpha_contrastive > 0.0:
            self.last_loss_contrastive = 0
            self.last_valid_loss_contrastive_epoch = 0


            if params.hidden_distil_type == 'hyperbolic':
                self.similarity_metric = hypmath.dist
                self.c = params.c
                if params.init_c == 'precompute_from_teacher':
                    delta_, diam = delta.get_delta(self.teacher, self.valid_dataloader, 
                                                    't_ids', 't_lengths', 
                                                    self.params.local_rank, self.multi_gpu, 
                                                    self.params.n_samples_to_precompute_c, 
                                                    self.params.n_components, 
                                                    self.params.n_tries)
                    self.c = delta.calculate_c(delta_, diam)
                elif self.params.init_c == 'precompute_from_student':
                    if self.params.align_hiddens == 'reduce':
                        s_ids, s_lengths = 't2s_ids', 't2s_lengths'
                    else:
                        s_ids, s_lengths = 's_ids', 's_lengths'
                    delta_, diam = delta.get_delta(self.student, self.valid_dataloader, 
                                                    s_ids, s_lengths, 
                                                    self.params.local_rank, self.multi_gpu, 
                                                    self.params.n_samples_to_precompute_c, 
                                                    self.params.n_components, 
                                                    self.params.n_tries)
                    self.c = delta.calculate_c(delta_, diam)
                else:
                    self.c = self.params.c

                if params.use_hyperbolic_projections:

                    s_ball_dim = self.s_hid_dim
                    t_ball_dim = self.t_hid_dim
                    if self.params.project_to == 'teacher':
                        self.hid_projectors_contrastive_teacher = None
                        self.hid_projectors_contrastive_student = nn.ModuleList([hypnn.HypLinear(self.s_hid_dim, self.t_hid_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{params.local_rank}') for _ in range(layers_ct)])
                    elif params.project_to =='student':
                        self.hid_projectors_contrastive_student = None
                        self.hid_projectors_contrastive_teacher = nn.ModuleList([hypnn.HypLinear(self.t_hid_dim, self.s_hid_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{params.local_rank}') for _ in range(layers_ct)])
                    else:
                        self.hid_projectors_contrastive_student = nn.ModuleList([hypnn.HypLinear(self.s_hid_dim, params.intermediate_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{params.local_rank}') for _ in range(layers_ct)])
                        self.hid_projectors_contrastive_teacher = nn.ModuleList([hypnn.HypLinear(self.t_hid_dim, params.intermediate_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{params.local_rank}') for _ in range(layers_ct)])
                else:
                    if params.project_to == 'teacher':
                        s_ball_dim = self.t_hid_dim
                        t_ball_dim = self.t_hid_dim
                    elif params.project_to == 'student':
                        s_ball_dim = self.s_hid_dim
                        t_ball_dim = self.s_hid_dim
                    else:
                        s_ball_dim = params.intermediate_dim
                        t_ball_dim = params.intermediate_dim
                
                self.student_to_poincare = hypnn.ToPoincare(self.c, train_c=params.adjust_c == 'train_exp_map_student', 
                                                            train_x=params.train_x, ball_dim=s_ball_dim, riemannian=params.riemannian)
                self.teacher_to_poincare = hypnn.ToPoincare(self.c, train_c=params.adjust_c == 'train_exp_map_teacher', 
                                                            train_x=params.train_x, ball_dim=t_ball_dim, riemannian=params.riemannian)
                if params.adjust_c == 'train_exp_map_student':
                    self.optimizer.add_param_group({'params': self.student_to_poincare.parameters()})
                elif params.adjust_c == 'train_exp_map_teacher':
                    self.optimizer.add_param_group({'params': self.teacher_to_poincare.parameters()})
            else:
                self.similarity_metric = custom_step.cosine_similarity


            if params.hidden_distil_type is None or (params.hidden_distil_type == 'hyperbolic' and not params.use_hyperbolic_projections):
                self.do_hyperbolic_mapping_in_step = True
                if params.project_to == 'teacher':
                    self.hid_projectors_contrastive_teacher = None
                    self.hid_projectors_contrastive_student = nn.ModuleList([nn.Linear(self.s_hid_dim, self.t_hid_dim).to(f'cuda:{self.params.local_rank}') for _ in range(layers_ct)])
                elif params.project_to =='student':
                    self.hid_projectors_contrastive_student = None
                    self.hid_projectors_contrastive_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, self.s_hid_dim).to(f'cuda:{self.params.local_rank}') for _ in range(layers_ct)])
                else:
                    self.hid_projectors_contrastive_student = nn.ModuleList([nn.Linear(self.s_hid_dim, params.intermediate_dim).to(f'cuda:{self.params.local_rank}') for _ in range(layers_ct)])
                    self.hid_projectors_contrastive_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, params.intermediate_dim).to(f'cuda:{self.params.local_rank}') for _ in range(layers_ct)])
 

    @staticmethod
    def generate_padding_mask(seq_len, lengths):
        padding_mask = torch.arange(seq_len, dtype=torch.long, device=lengths.device) < lengths[:,None]
        return padding_mask


    def prepare_batch_mlm(self, tok_ids, lengths, pad_token_id, mask_token_id, token_probs):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        attn_mask = torch.arange(tok_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]

        bs, max_seq_len = tok_ids.size()
        mlm_labels = tok_ids.new(tok_ids.size()).copy_(tok_ids)

        x_prob = token_probs[tok_ids.flatten()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)

        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=tok_ids.device
        )
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)
        pred_mask[tok_ids == pad_token_id] = 0

        _token_ids_real = tok_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.student_vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(mask_token_id)
        probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True)
        _tok_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )
        tok_ids = tok_ids.masked_scatter(pred_mask, _tok_ids)
        mlm_labels[~pred_mask] = -100

        return tok_ids, attn_mask, mlm_labels

    def prepare_batch(self, batch):
        batch_cuda = {arg_name: arg_value.to(f"cuda:{self.params.local_rank}") for arg_name, arg_value in batch.items()}
        batch_cuda['t_attn_mask'] = self.generate_padding_mask(batch_cuda['t_ids'].size(1), batch_cuda['t_lengths'])

        if self.alpha_mlm > 0.0 and self.student_token_probs is not None:
            s_tok_ids, s_attn_mask, s_lm_labels = self.prepare_batch_mlm(batch_cuda['s_ids'],
                                                                         batch_cuda['s_lengths'],
                                                                         self.params.student_tok_ids['pad_token'],
                                                                         self.params.student_tok_ids['mask_token'],
                                                                         self.student_token_probs
                                                                         )
            batch_cuda['s_ids'] = s_tok_ids
            batch_cuda['s_lm_labels'] = s_lm_labels
        else:
            s_attn_mask = self.generate_padding_mask(batch_cuda['s_ids'].size(1), batch_cuda['s_lengths'])
        batch_cuda['s_attn_mask'] = s_attn_mask

        if self.params.t2s_mapping is not None:
            batch_cuda['t2s_attn_mask'] = self.generate_padding_mask(batch_cuda['t2s_ids'].size(1),
                                                                     batch_cuda['t2s_lengths'])
        return batch_cuda

    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()

        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            self.student.train()
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            iter_bar = tqdm(self.train_dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                batch_cuda = self.prepare_batch(batch)
                batch_cuda['grad_on'] = True
                self.step(**batch_cuda)

                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_train_loss:.5f}", "Avg_cum_loss": f"{self.total_train_loss_epoch/self.n_train_iter_epoch:.2f}"}
                )

            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")

            self.validate()
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")

    def validate(self):
        self.teacher.eval()
        self.student.eval()

        if self.is_master:
            logger.info(f"--- Validating epoch {self.epoch}/{self.params.n_epoch-1}")
        if self.multi_gpu:
            torch.distributed.barrier()
        iter_bar = tqdm(self.valid_dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
        for batch in iter_bar:
            batch_cuda = self.prepare_batch(batch)
            batch_cuda['grad_on'] = False
            self.step(**batch_cuda)
            iter_bar.update()
            iter_bar.set_postfix(
                {"Avg_cum_valid_loss": f"{self.total_valid_loss_epoch/self.n_valid_iter_epoch:.2f}"}
            )
        iter_bar.close()

        if self.n_gradient_updates_total > self.warmup_steps:
            self.reduce_on_plateau.step(self.total_valid_loss_epoch/self.n_valid_iter_epoch)

        if self.is_master:
            logger.info(f"--- Ending validation epoch {self.epoch}/{self.params.n_epoch-1}")

    def step(self, t_ids=None, t_attn_mask=None,
             s_ids=None, s_attn_mask=None, s_lm_labels=None,
             t2s_ids=None, t2s_attn_mask=None,
             student_split_ids=None, teacher_mask=None, student_mask=None,
             grad_on=True, **kwargs):
        with torch.no_grad():
            t_out = self.teacher(input_ids=t_ids, attention_mask=t_attn_mask)
            t_logits = t_out.logits

        with torch.set_grad_enabled(grad_on):
            s_out = self.student(input_ids=s_ids, attention_mask=s_attn_mask)
            s_logits = s_out.logits

        loss = 0.0
        if self.params.t2s_mapping is not None:
            with torch.set_grad_enabled(grad_on):
                t2s_out = self.student(input_ids=t2s_ids,
                                       attention_mask=t2s_attn_mask)
            t2s_logits = t2s_out.logits
            if self.params.t2s_vocab_padded is not None:
                student_mapped_logits = custom_step.map_step(t2s_logits,
                                                             student_split_ids, self.params.student_tok_ids['pad_token'],
                                                             t2s_vocab_padded=self.params.t2s_vocab_padded,
                                                             s2t_vocab_padded=self.params.s2t_vocab_padded,
                                                             s2t_idxs_padded=self.params.s2t_idxs_padded,
                                                             sum_probs=self.params.sum_probs)
                student_mapped_logits = custom_step.masked_select_reshape_2d(student_mapped_logits, t_attn_mask, self.teacher_vocab_size)
                teacher_mapped_logits = custom_step.masked_select_reshape_2d(t_logits, t_attn_mask, self.teacher_vocab_size)
                if self.params.t2s_mapped_ids is not None:
                    student_mapped_logits = student_mapped_logits[:, self.params.t2s_mapped_ids]
                    teacher_mapped_logits = teacher_mapped_logits[:, self.params.t2s_mapped_ids]

                if self.alpha_ce > 0.0:
                    loss_ce = custom_step.ce_step(student_mapped_logits, teacher_mapped_logits, self.ce_loss_fct, self.temperature)
        # reduce-map was skipped
        if self.params.matching_ids is not None and self.params.t2s_vocab_padded is None:
            # use only logits for teacher tokens which could be represented by student tokens
            # PAD -> PAD, UNK -> UNK, CLS -> CLS, unused1 -> skip, unused2 -> skip, ...
            student_mapped_logits = custom_step.match_step(s_logits, student_mask, 1, self.params.student_matched)
            teacher_mapped_logits = custom_step.match_step(t_logits, teacher_mask, 1, self.params.teacher_matched)

            if self.alpha_ce > 0.0:
                loss_ce = custom_step.ce_step(student_mapped_logits, teacher_mapped_logits, 
                                              self.ce_loss_fct, self.temperature)

        if (self.alpha_contrastive > 0.0 and self.params.hidden_distil_type == 'hyperbolic' and 
            self.params.use_hyperbolic_projections):

                t_hiddens = [self.teacher_to_poincare(t, c=self.c) for t in t_out.hidden_states]
                if self.params.align_hiddens == 'reduce':
                    s_hiddens = [self.student_to_poincare(s, c=self.c) for s in t2s_out.hidden_states]
                else:
                    s_hiddens = [self.student_to_poincare(s, c=self.c) for s in s_out.hidden_states]
            
        else:
            t_hiddens = t_out.hidden_states
            if self.params.align_hiddens == 'reduce':
                s_hiddens = t2s_out.hidden_states
            else:
                s_hiddens = s_out.hidden_states

        # match strategy, use s outputs
        if self.params.align_hiddens == 'match' and self.params.matching_ids is not None:
            if self.alpha_mse > 0.0:

                loss_mse = custom_step.mse_step(self.hid_projectors_mse_student, 
                                                self.hid_projectors_mse_teacher, 
                                                self.mse_loss_fct, 
                                                s_hiddens, t_hiddens,
                                                student_mask, teacher_mask, 
                                                1, self.params.projection_strategy, 
                                                t_s_layers_ids=self.params.t_s_layers_ids)
            if self.alpha_contrastive > 0.0:
                loss_contrastive = custom_step.contrastive_step(self.params.train_cardinality, 
                                                                self.hid_projectors_contrastive_student,
                                                                self.hid_projectors_contrastive_teacher, 
                                                                s_hiddens, t_hiddens, student_mask, teacher_mask, 
                                                                0, 1, self.params.projection_strategy,
                                                                negative_sampling_strategy=self.params.negative_sampling_strategy,
                                                                use_mismatched_ids=self.params.use_mismatched_ids,
                                                                n_negative_samples=self.params.n_negative_samples,
                                                                teacher_student_prop=self.params.teacher_student_prop,
                                                                temperature=self.temperature,
                                                                from_one_sample=self.params.from_one_sample,
                                                                add_neg_size_constant=self.params.add_neg_size_constant, 
                                                                t_s_layers_ids=self.params.t_s_layers_ids, 
                                                                similarity_metric=self.similarity_metric, 
                                                                use_hyp_mapping_in_step=self.do_hyperbolic_mapping_in_step, 
                                                                c=self.c if hasattr(self, 'c') else None, 
                                                                teacher_to_poincare=self.teacher_to_poincare if hasattr(self, 'teacher_to_poincare') else None, 
                                                                student_to_poincare=self.student_to_poincare if hasattr(self, 'student_to_poincare') else None)

        # reduce strategy, use t2s hiddens and t2s mapping
        elif self.params.align_hiddens == 'reduce' and self.params.t2s_mapping is not None:
            if self.alpha_mse > 0.0:
                loss_mse = custom_step.mse_step(self.hid_projectors_mse_student, 
                                                self.hid_projectors_mse_teacher, 
                                                self.mse_loss_fct,
                                                s_hiddens, t_hiddens,
                                                t_attn_mask, t_attn_mask, 1,
                                                self.params.projection_strategy,
                                                student_split_ids, self.params.student_tok_ids['pad_token'], 
                                                t_s_layers_ids=self.params.t_s_layers_ids)
            if self.alpha_contrastive > 0.0:
                loss_contrastive = custom_step.contrastive_step(self.params.train_cardinality,
                                                                self.hid_projectors_contrastive_student, 
                                                                self.hid_projectors_contrastive_teacher, 
                                                                s_hiddens, t_hiddens,
                                                                t_attn_mask, t_attn_mask, 0, 1,
                                                                self.params.projection_strategy,
                                                                student_split_ids,
                                                                self.params.student_tok_ids['pad_token'],
                                                                self.params.negative_sampling_strategy,
                                                                self.params.use_mismatched_ids,
                                                                self.params.n_negative_samples,
                                                                self.params.teacher_student_prop,
                                                                self.temperature, self.params.from_one_sample,
                                                                self.params.add_neg_size_constant, 
                                                                t_s_layers_ids=self.params.t_s_layers_ids, 
                                                                similarity_metric=self.similarity_metric, 
                                                                use_hyp_mapping_in_step=self.do_hyperbolic_mapping_in_step, 
                                                                c=self.c if hasattr(self, 'c') else None, 
                                                                teacher_to_poincare=self.teacher_to_poincare if hasattr(self, 'teacher_to_poincare') else None, 
                                                                student_to_poincare=self.student_to_poincare if hasattr(self, 'student_to_poincare') else None)
        # no alignment strategy for hiddens specified
        elif self.params.align_hiddens is None and self.params.matching_ids is not None:
            if self.alpha_contrastive > 0.0:
                # TODO: correct for hyperbolic layers
                loss_contrastive = custom_step.contrastive_step_v0(self.params.train_cardinality,
                                                                   self.hid_projectors_contrastive_student, 
                                                                   s_hiddens, t_hiddens,
                                                                   student_mask, teacher_mask, 0, 1,
                                                                   self.params.projection_strategy,
                                                                   self.params.student_tok_ids['pad_token'],
                                                                   self.params.add_neg_size_constant)

        loss = 0.
        with torch.set_grad_enabled(grad_on):
            """
                KLDiv loss
            """
            if self.alpha_ce > 0.0:
                loss += self.alpha_ce * loss_ce
            """
                MLM loss
            """
            if self.alpha_mlm > 0.0:
                loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), s_lm_labels.view(-1))
                loss += self.alpha_mlm * loss_mlm
            """
                L2 loss
            """
            if self.alpha_mse > 0.0:
                loss += self.alpha_mse * loss_mse

            """
                Contrastive loss
            """
            if self.alpha_contrastive > 0.0:
                loss += self.alpha_contrastive * loss_contrastive

        if grad_on:
            self.total_train_loss_epoch += loss.item()
            self.last_train_loss = loss.item()
            if self.alpha_ce > 0.0:
                self.last_loss_ce = loss_ce.item()
            if self.alpha_mlm > 0.0:
                self.last_loss_mlm = loss_mlm.item()
            if self.alpha_mse > 0.0:
                self.last_loss_mse = loss_mse.item()
            if self.alpha_contrastive > 0.0:
                self.last_loss_contrastive = loss_contrastive.item()
            self.n_sequences_epoch += t_ids.size(0) * self.params.gpus
            self.n_sequences_total += t_ids.size(0) * self.params.gpus
            self.optimize(loss)
        else:
            self.total_valid_loss_epoch += loss.item()
            if self.alpha_ce > 0.0:
                self.last_valid_loss_ce_epoch += loss_ce.item()
            if self.alpha_mlm > 0.0:
                self.last_valid_loss_mlm_epoch += loss_mlm.item()
            if self.alpha_mse > 0.0:
                self.last_valid_loss_mse_epoch += loss_mse.item()
            if self.alpha_contrastive > 0.0:
                self.last_valid_loss_contrastive_epoch += loss_contrastive.item()

            self.n_valid_iter_epoch += 1

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        loss.backward()  # sync point
        self.n_train_iter_epoch += 1
        self.n_train_iter_total += 1

        if self.n_train_iter_epoch % self.params.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if hasattr(self.params, 'c'):
                if self.params.adjust_c == 'train_exp_map_teacher':
                    self.c = self.teacher_to_poincare.c
                elif self.params.adjust_c == 'train_exp_map_student':
                    self.c = self.student_to_poincare.c

            self.n_gradient_updates_epoch += 1
            self.n_gradient_updates_total += 1

            if self.n_gradient_updates_total < self.warmup_steps:
                self.const_scheduler_with_warmup.step()

            # self.scheduler.step()

            if self.n_gradient_updates_epoch % self.params.log_interval == 0:
                self.log_tensorboard()
                self.last_log = time.time()
            if self.n_gradient_updates_epoch % self.params.checkpoint_interval == 0:
                self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_gradient_updates_total
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_gradient_updates_total
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_gradient_updates_total
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_gradient_updates_total
            )

        self.tensorboard.add_scalar(
            tag="train_losses/cum_avg_train_loss_epoch",
            scalar_value=self.total_train_loss_epoch / self.n_train_iter_epoch,
            global_step=self.n_gradient_updates_total
        )
        self.tensorboard.add_scalar(
            tag="train_losses/cum_avg_train_loss_epoch_samples",
            scalar_value=self.total_train_loss_epoch / self.n_train_iter_epoch,
            global_step=self.n_sequences_total
        )
        self.tensorboard.add_scalar(tag="train_losses/train_loss",
                                    scalar_value=self.last_train_loss,
                                    global_step=self.n_gradient_updates_total
                                    )
        self.tensorboard.add_scalar(tag="train_losses/train_loss_samples",
                                    scalar_value=self.last_train_loss,
                                    global_step=self.n_sequences_total
                                    )
        self.tensorboard.add_scalar(
            tag="learning_rate_steps/lr",
            scalar_value=self.optimizer.param_groups[0]['lr'],
            global_step=self.n_gradient_updates_total
        )

        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_gradient_updates_total,
        )
        self.tensorboard.add_scalar(
            tag="global/speed", scalar_value=time.time() - self.last_log, global_step=self.n_gradient_updates_total
        )

        if self.alpha_ce > 0.0:
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_ce",
                scalar_value=self.last_loss_ce,
                global_step=self.n_gradient_updates_total
            )
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_ce_samples",
                scalar_value=self.last_loss_ce,
                global_step=self.n_sequences_total
            )

        if self.alpha_mlm > 0.0:
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_mlm",
                scalar_value=self.last_loss_mlm,
                global_step=self.n_gradient_updates_total
            )
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_mlm_samples",
                scalar_value=self.last_loss_mlm,
                global_step=self.n_sequences_total
            )
        if self.alpha_mse > 0.0:
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_mse",
                scalar_value=self.last_loss_mse,
                global_step=self.n_gradient_updates_total
            )
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_mse_samples",
                scalar_value=self.last_loss_mse,
                global_step=self.n_sequences_total
            )
        if self.alpha_cos > 0.0:
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_cos",
                scalar_value=self.last_loss_cos,
                global_step=self.n_gradient_updates_total
            )
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_cos_samples",
                scalar_value=self.last_loss_cos,
                global_step=self.n_sequences_total
            )
        if self.alpha_contrastive > 0.0:
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_contrastive",
                scalar_value=self.last_loss_contrastive,
                global_step=self.n_gradient_updates_total
            )
            self.tensorboard.add_scalar(
                tag="train_losses/train_loss_contrastive_samples",
                scalar_value=self.last_loss_contrastive,
                global_step=self.n_sequences_total
            )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
            self.tensorboard.add_scalar(
                tag="epoch/learning_rate",
                scalar_value=self.optimizer.param_groups[0]['lr'],
                global_step=self.epoch + 1
            )
            self.tensorboard.add_scalar(
                tag="epoch/train_loss_epoch", scalar_value=self.total_train_loss_epoch / self.n_train_iter_epoch, global_step=self.epoch
            )
            mean_valid_loss = self.total_valid_loss_epoch / self.n_valid_iter_epoch
            self.tensorboard.add_scalar(
                tag="epoch/valid_loss", scalar_value=mean_valid_loss,
                global_step=self.epoch + 1
            )
            
            if self.alpha_ce:
                self.last_valid_loss_ce_epoch /= self.n_valid_iter_epoch
                self.tensorboard.add_scalar(
                    tag="epoch/valid_ce_loss", scalar_value=self.last_valid_loss_ce_epoch,
                    global_step=self.epoch + 1
                )
                self.last_valid_loss_ce_epoch = 0
                
            if self.alpha_mlm > 0.0:
                self.last_valid_loss_mlm_epoch /= self.n_valid_iter_epoch
                self.tensorboard.add_scalar(
                    tag="epoch/valid_mlm_loss", scalar_value=self.last_valid_loss_mlm_epoch,
                    global_step=self.epoch + 1
                )
                self.last_valid_loss_mlm_epoch = 0
                
            if self.alpha_mse > 0.0:
                self.last_valid_loss_mse_epoch /= self.n_valid_iter_epoch
                self.tensorboard.add_scalar(
                    tag="epoch/valid_mse_loss", scalar_value=self.last_valid_loss_mse_epoch,
                    global_step=self.epoch + 1
                )
                self.last_valid_loss_mse_epoch = 0
                
            if self.alpha_cos > 0.0:
                self.last_valid_loss_cos_epoch /= self.n_valid_iter_epoch
                self.tensorboard.add_scalar(
                    tag="epoch/valid_cos_loss", scalar_value=self.last_valid_loss_cos_epoch,
                    global_step=self.epoch + 1
                )
                self.last_valid_loss_cos_epoch = 0
                
            if self.alpha_contrastive > 0.0:
                self.last_valid_loss_contrastive_epoch /= self.n_valid_iter_epoch
                self.tensorboard.add_scalar(
                    tag="epoch/valid_contrastive_loss", scalar_value=self.last_valid_loss_contrastive_epoch,
                    global_step=self.epoch + 1
                )
                self.last_valid_loss_contrastive_epoch = 0
                
            if mean_valid_loss < self.best_total_valid_loss_epoch:
                self.best_total_valid_loss_epoch = mean_valid_loss
                self.save_checkpoint(checkpoint_name="best_valid.pth")
                self.save_checkpoint(checkpoint_name="best_model.bin")
                logger.info("Best validation loss was improved to {:.4f}".format(mean_valid_loss))
            else:
                logger.info("Best validation loss was not improved! Best {:4f} < current {:4f}".format(self.best_total_valid_loss_epoch, mean_valid_loss))

        self.epoch += 1
        self.n_train_iter_epoch = 0
        self.n_gradient_updates_epoch = 0
        self.n_sequences_epoch = 0
        self.n_valid_iter = 0
        self.total_train_loss_epoch = 0
        self.total_valid_loss_epoch = 0
        self.n_valid_iter_epoch = 0

        # recompute from student anyway, as teacher is frozen
        if hasattr(self, 'recompute_c') and self.recompute_c:
            if self.params.align_hiddens == 'reduce':
                s_ids, s_lengths = 't2s_ids', 't2s_lengths'
            else:
                s_ids, s_lengths = 's_ids', 's_lengths'
            delta_, diam = delta.get_delta(self.student, self.valid_dataloader, 
                                           s_ids, s_lengths, 
                                           self.params.local_rank, self.multi_gpu, 
                                           self.params.n_samples_to_precompute_c, 
                                           self.params.n_components, self.params.n_tries
                                          )
            self.c = delta.calculate_c(delta_, diam)


    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
