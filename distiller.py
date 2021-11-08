import math
import os
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset
from transformers import get_cosine_schedule_with_warmup
from utils import logger
import sys


from setup_logger import setup_logger

import pickle

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AutoConfig



class MyIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, indices, inverted_indices):
        # tensor: 2-d tensor (bs * seq_len) x (student_vocab_size + 1)
        ctx.tensor_shape = tensor.shape
        ctx.inverted_indices = inverted_indices
        return tensor[:, indices]
        
    @staticmethod
    def backward(ctx, grad_output):
        # 3-d grad_output, gradients in last dimension are equal (as grad_output is result of torch.sum)
        # (bs * seq_len) x teacher_vocab_size x X
        x = grad_output[:,:, 0]
        #device = x.device
        #x = torch.cat([x, torch.zeros((grad_output.shape[0], 1), device=device)], dim=-1)
        return torch.sum(x[:, ctx.inverted_indices], dim=-1), None, None
        

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook
    

class Distiller:
    def __init__(
        self, params: dict, teacher_token_probs: torch.tensor, student_token_probs: torch.tensor, 
        student: nn.Module, teacher: nn.Module, 
        train_dataset: LmSeqsDataset, valid_dataset: LmSeqsDataset, 
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

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
        # nn.Parameter(params.temperature)
        #self.temperature = nn.Parameter(torch.tensor([params.temperature], dtype=torch.float))
        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos

        self.mlm = params.mlm
        if self.mlm:
            logger.info("Using MLM loss for LM step.")
            self.mlm_mask_prop = params.mlm_mask_prop
            assert 0.0 <= self.mlm_mask_prop <= 1.0
            assert params.word_mask + params.word_keep + params.word_rand == 1.0
            self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
            self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.gpus > 0 else self.pred_probs
            self.teacher_token_probs = teacher_token_probs.to(f"cuda:{params.local_rank}") if params.gpus > 0 else teacher_token_probs
            self.student_token_probs = student_token_probs.to(f"cuda:{params.local_rank}") if params.gpus > 0 else student_token_probs
            if self.fp16:
                self.pred_probs = self.pred_probs.half()
                self.token_probs = self.token_probs.half()

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.total_valid_loss_epoch = 0
        self.best_total_valid_loss_epoch = sys.maxsize
        self.last_valid_loss_ce_epoch = 0
        self.last_valid_loss_mlm_epoch = 0
        if self.alpha_mse > 0.0:
            self.last_valid_loss_mse_epoch = 0
        if self.alpha_cos > 0.0:
            self.last_valid_loss_cos_epoch = 0
        
        self.n_valid_iter = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        #self.ce_loss_fct = CrossEntropy()
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.train_dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

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
        """
            {
                "params": self.temperature
            }
        """
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=warmup_steps, 
                                                         num_training_steps=num_train_optimization_steps)

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                
                from torch.nn.parallel import DistributedDataParallel

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
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)

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

        mlm_labels[~pred_mask] = -100  # previously `mlm_labels[1-pred_mask] = -1`, cf pytorch 1.2.0 compatibility



        return tok_ids, attn_mask, mlm_labels

 

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
                #if self.params.gpus > 0:
                batch_cuda = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)

                t_tok_ids, t_attn_mask, t_lm_labels = self.prepare_batch_mlm(batch_cuda[0], 
                                                                             batch_cuda[1], 
                                                                             self.params.teacher_tok_ids['pad_token'], 
                                                                             self.params.teacher_tok_ids['mask_token'], self.teacher_token_probs
                                                                            )
                st_tok_ids, st_attn_mask, st_lm_labels = self.prepare_batch_mlm(batch_cuda[2], 
                                                                                batch_cuda[3], self.params.student_tok_ids['pad_token'], 
                                                                                self.params.student_tok_ids['mask_token'], self.student_token_probs
                                                                               )

                t2s_tok_ids, t2s_attn_mask, t2s_lm_labels = self.prepare_batch_mlm(batch_cuda[4], 
                                                                                   batch_cuda[5], self.params.student_tok_ids['pad_token'],
                                                                                   self.params.student_tok_ids['mask_token'], 
                                                                                   self.student_token_probs)
               
                self.step(t_tok_ids, t_attn_mask, t_lm_labels, 
                          st_tok_ids, st_attn_mask, st_lm_labels, 
                          t2s_tok_ids, t2s_attn_mask, t2s_lm_labels, batch_cuda[-1],
                          grad_on=True)
                
                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_loss:.5f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
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
        self.student.eval()
        if self.is_master:
            logger.info(f"--- Validating epoch {self.epoch}/{self.params.n_epoch-1}")
        if self.multi_gpu:
            torch.distributed.barrier()
        iter_bar = tqdm(self.valid_dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])

        for batch in iter_bar:
            #if self.params.gpus > 0:
            batch_cuda = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)

            t_tok_ids, t_attn_mask, t_lm_labels = self.prepare_batch_mlm(batch_cuda[0], 
                                                                         batch_cuda[1], 
                                                                         self.params.teacher_tok_ids['pad_token'], 
                                                                         self.params.teacher_tok_ids['mask_token'], 
                                                                         self.teacher_token_probs
                                                                        )
            st_tok_ids, st_attn_mask, st_lm_labels = self.prepare_batch_mlm(batch_cuda[2], 
                                                                            batch_cuda[3], self.params.student_tok_ids['pad_token'], 
                                                                            self.params.student_tok_ids['mask_token'], self.student_token_probs
                                                                           )

            t2s_tok_ids, t2s_attn_mask, t2s_lm_labels = self.prepare_batch_mlm(batch_cuda[4], 
                                                                            batch_cuda[5], self.params.student_tok_ids['pad_token'], 
                                                                            self.params.student_tok_ids['mask_token'], self.student_token_probs
                                                                           )

            loss = self.step(t_tok_ids, t_attn_mask, t_lm_labels, 
                      st_tok_ids, st_attn_mask, st_lm_labels, 
                      t2s_tok_ids, t2s_attn_mask, t2s_lm_labels, batch[6],
                      grad_on=False)

            iter_bar.update()
            iter_bar.set_postfix(
                {"Avg_cum_valid_loss": f"{self.total_valid_loss_epoch/self.n_valid_iter:.2f}"}
            )
        iter_bar.close()
        

        if self.is_master:
            logger.info(f"--- Ending validation epoch {self.epoch}/{self.params.n_epoch-1}")
        
        
    def step(self, t_tok_ids, t_attn_mask, t_lm_labels, 
             st_tok_ids, st_attn_mask, st_lm_labels, 
             t2s_tok_ids, t2s_attn_mask, t2s_lm_labels,
             offset, grad_on: bool=True):


        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=t_tok_ids, attention_mask=t_attn_mask
            )
            
            t_logits = t_out.logits.detach()
            t_hidden_states = t_out.hidden_states
            t_attentions = t_out.attentions


        with torch.set_grad_enabled(grad_on):
            s_out = self.student(
                input_ids=st_tok_ids, attention_mask=st_attn_mask
            )
            s_logits = s_out.logits
            s_hidden_states = s_out.hidden_states
            s_attentions = s_out.attentions

            t2s_out = self.student(
                input_ids=t2s_tok_ids, attention_mask=t2s_attn_mask
            ) 

            t2s_logits = t2s_out.logits
            t2s_hidden_states = t2s_out.hidden_states
            t2s_attentions = t2s_out.attentions
                                    
        if self.params.restrict_ce_to_mask:
            mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        else:
            t_attn_mask = t_attn_mask.unsqueeze(-1).expand_as(t_logits)  # (bs, seq_length, voc_size)
            st_attn_mask = st_attn_mask.unsqueeze(-1).expand_as(s_logits)
            t2s_attn_mask = t2s_attn_mask.unsqueeze(-1).expand_as(t2s_logits)
        
        """
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        """
        
        def reduce_student_logits_to_teacher(t2s_logits, idxs_padded):
            bs, stu_seq_len, stu_voc_size = t2s_logits.shape
            fake_seq = self.params.student_tok_ids['pad_token'] * torch.ones(bs, 1, stu_voc_size).to(t2s_logits.device)
            #fake_seq.requires_grad_()
            padded_seq = torch.cat((t2s_logits, fake_seq), dim=1)
            batch_idx = torch.tensor([[[i]] for i in range(bs)], dtype=torch.long).to(t2s_logits.device)
            reduced_seq = torch.sum(padded_seq[batch_idx, idxs_padded], dim=2)
            return reduced_seq
        
        def map_student_logits_to_teacher(reduced_t2s_logits):
            bs, seq_len, stu_voc_size = reduced_t2s_logits.shape
            reshaped = reduced_t2s_logits.reshape(-1, stu_voc_size)
            mapped_logits = torch.sum(reshaped[:,self.params.t2s_vocab_padded], dim=-1).reshape(bs, seq_len, self.teacher_vocab_size)
            #reshaped.register_hook(set_grad(reshaped))
            return mapped_logits
        
        loss = 0.        
        #self.temperature.to(device)
        with torch.set_grad_enabled(grad_on):
            reduced_t2s_logits = reduce_student_logits_to_teacher(t2s_logits, offset)
            
            # v1 without MyIndex
            """
            mapped_t2s_logits = map_student_logits_to_teacher(reduced_t2s_logits)
            """

            myindex= MyIndex.apply
            bs, teacher_seq_len, stu_voc_size = reduced_t2s_logits.shape
            reshaped = reduced_t2s_logits.reshape(-1, stu_voc_size)
            teacher_vocab_size = t_logits.size(2)

            mapped_t2s_logits = torch.sum(myindex(reshaped, 
                                              self.params.t2s_vocab_padded, 
                                              self.params.s2t_vocab_padded), 
                                      dim=-1).reshape(bs, teacher_seq_len, teacher_vocab_size)
          

            loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(mapped_t2s_logits / self.temperature, dim=-1),
                F.softmax(t_logits / self.temperature, dim=-1),
            )
            * self.temperature ** 2
            )

            loss = self.alpha_ce * loss_ce


            if self.alpha_mlm > 0.0:
                loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), st_lm_labels.view(-1))
                loss = loss + self.alpha_mlm * loss_mlm




                                    
            if grad_on:
                self.total_loss_epoch += loss.item()
                self.last_loss = loss.item()
                self.last_loss_ce = loss_ce.item()
                if self.alpha_mlm > 0.0:
                    self.last_loss_mlm = loss_mlm.item()
                if self.alpha_mse > 0.0:
                    self.last_loss_mse = loss_mse.item()
                if self.alpha_cos > 0.0:
                    self.last_loss_cos = loss_cos.item()
                self.optimize(loss)
                self.n_sequences_epoch += t_tok_ids.size(0)
            else:
                self.total_valid_loss_epoch += loss.item()
                self.last_valid_loss_ce_epoch += loss_ce.item()
                if self.alpha_mlm > 0.0:
                    self.last_valid_loss_mlm_epoch = self.last_valid_loss_mlm_epoch + loss_mlm.item()
                if self.alpha_mse > 0.0:
                    self.last_valid_loss_mse_epoch += loss_mse.item()
                if self.alpha_cos > 0.0:
                    self.last_valid_loss_cos_epoch += loss_cos.item()
                self.n_valid_iter += 1

        return loss


    def optimize(self, loss, grad_on: bool=True):
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

        loss.backward()
        self.iter()
        
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()
            

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        
        self.tensorboard.add_scalar(
            tag="losses/loss_ce", scalar_value=self.last_loss_ce, global_step=self.n_total_iter
        )
        
        
        if self.alpha_mlm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mlm", scalar_value=self.last_loss_mlm, global_step=self.n_total_iter
            )
        if self.alpha_mse > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_mse", scalar_value=self.last_loss_mse, global_step=self.n_total_iter
            )
        if self.alpha_cos > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_cos", scalar_value=self.last_loss_cos, global_step=self.n_total_iter
            )
        self.tensorboard.add_scalar(
            tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter
        )

        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="global/speed", scalar_value=time.time() - self.last_log, global_step=self.n_total_iter
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
                tag="epoch/loss", scalar_value=self.total_loss_epoch / self.n_iter, global_step=self.epoch
            )
            mean_valid_loss = self.total_valid_loss_epoch / self.n_valid_iter
            self.tensorboard.add_scalar(
                tag="epoch/valid_loss", scalar_value=mean_valid_loss, 
                global_step=self.epoch
            )
            self.last_valid_loss_ce_epoch /= self.n_valid_iter
            self.tensorboard.add_scalar(
                tag="epoch/valid_ce_loss", scalar_value=self.last_valid_loss_ce_epoch, 
                global_step=self.epoch
            )
            self.last_valid_loss_ce_epoch = 0
            if self.alpha_mlm > 0.0:
                self.last_valid_loss_mlm_epoch /= self.n_valid_iter
                self.tensorboard.add_scalar(
                    tag="epoch/valid_mlm_loss", scalar_value=self.last_valid_loss_mlm_epoch, 
                    global_step=self.epoch
                )
                self.last_valid_loss_mlm_epoch = 0
            if self.alpha_mse > 0.0:
                self.last_valid_loss_mse_epoch /= self.n_valid_iter
                self.tensorboard.add_scalar(
                    tag="epoch/valid_mse_loss", scalar_value=self.last_valid_loss_mse_epoch, 
                    global_step=self.epoch
                )
                self.last_valid_loss_mse_epoch = 0
            if self.alpha_cos > 0.0:
                self.last_valid_loss_cos_epoch /= self.n_valid_iter
                self.tensorboard.add_scalar(
                    tag="epoch/valid_cos_loss", scalar_value=self.last_valid_loss_cos_epoch, 
                    global_step=self.epoch
                )
                self.last_valid_loss_cos_epoch = 0
            if mean_valid_loss < self.best_total_valid_loss_epoch:
                self.best_total_valid_loss_epoch = mean_valid_loss
                self.save_checkpoint(checkpoint_name=f"best_valid.pth")
                self.save_checkpoint(checkpoint_name="best_model.bin")
                logger.info("Best validation loss was improved to {:.4f}".format(mean_valid_loss))
            else:
                logger.info("Best validation loss was not improved! Best {:4f} < current {:4f}".format(self.best_total_valid_loss_epoch, mean_valid_loss))

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.n_valid_iter = 0
        self.total_loss_epoch = 0
        self.total_valid_loss_epoch = 0

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

