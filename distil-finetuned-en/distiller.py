import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.optimization import AdamW
from tqdm import tqdm

import load_data
import delta_
import os

import sys
sys.path.insert(1, '../')
import hyptorch.nn as hypnn
import hyptorch.pmath as hypmath

import custom_step
from utils import logger

import datasets
import wandb


class Distiller(object):
    def __init__(self, params):
        self.params = params
        self.run = params.run
        self.run.define_metric('epoch')
        self.run.define_metric('train/*', step_metric='epoch')
        self.run.define_metric('valid/*', step_metric='epoch')
        self.run.define_metric('valid/loss_epoch', step_metric='epoch', mode='min')

        self.main_metric = datasets.load_metric('glue', params.glue_dataset)
        self.metric_name = ''
        self.train_metric_value = 0
        self.valid_metric_value = 0
        self.best_valid_metric_value = sys.maxsize
        self.summary_table = wandb.Table(columns=['Task', 'Metric', 'Validation score', 'Test score'])

        # TODO: load data using different tokenizers (now we assume that vocabularies match)
        train_data = load_data.load_glue_dataset(params.glue_dataset, 
                                                            params.tokenizer_name, 
                                                            params.padding, 
                                                            params.truncation, 
                                                            'train', 0, 
                                                            params.seed)
        valid_data = load_data.load_glue_dataset(params.glue_dataset, 
                                                params.tokenizer_name, 
                                                params.padding, 
                                                params.truncation, 
                                                'validation')
        self.train_loader = DataLoader(train_data, params.batch_size, shuffle=True, collate_fn=load_data.collate_fn)
        self.valid_loader = DataLoader(valid_data, params.batch_size, shuffle=False, collate_fn=load_data.collate_fn)
        self.test_loader = DataLoader(valid_data, params.batch_size, shuffle=False, collate_fn=load_data.collate_fn)

        self.gpu_id = params.gpu_id
        self.n_classes = train_data.features['labels'].num_classes
        config = AutoConfig.from_pretrained(params.student_name, num_labels=self.n_classes, output_hidden_states=True, return_dict=True)
        self.student = AutoModelForSequenceClassification.from_config(config).to(f'cuda:{self.gpu_id}')
        self.teacher = AutoModelForSequenceClassification.from_pretrained(params.teacher_name, num_labels=self.n_classes, output_hidden_states=True, return_dict=True).to(f'cuda:{self.gpu_id}')
        if params.teacher_weights is not None:
            state_dict = torch.load(params.teacher_weights)['model_state_dict']
            self.teacher.load_state_dict(state_dict)

        self.optimizer = AdamW(self.student.parameters(), lr=params.lr)
        self.scheduler = getattr(torch.optim.lr_scheduler, params.scheduler)(self.optimizer, **params.scheduler_params)

        if self.params.train_temperature:
            self.temperature = nn.Parameter(torch.tensor([self.params.temperature]))
            self.optimizer.add_param_group({"params": [self.temperature]})
        else:
            self.temperature = self.params.temperature

        self.alpha_ce = params.alpha_ce
        self.alpha_kl = params.alpha_kl
        self.alpha_mse = params.alpha_mse
        self.alpha_contrastive = params.alpha_contrastive

        self.s_hid_dim = self.student.config.hidden_size
        self.t_hid_dim = self.teacher.config.hidden_size

        if self.alpha_ce > 0.0:
            self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
            self.train_loss_ce_epoch = 0.
            self.valid_loss_ce_epoch = 0.

        if self.alpha_kl > 0.0:
            self.kl_loss = nn.KLDivLoss(reduction='sum')
            self.train_loss_kl_epoch = 0.
            self.valid_loss_kl_epoch = 0.
        
        layers_ct = 4 if self.params.projection_strategy in ['last', 'skip', 'average'] else 1
        if self.alpha_mse > 0.0:
            self.mse_loss = nn.MSELoss(reduction='sum')
            self.train_loss_mse_epoch = 0.
            self.valid_loss_mse_epoch = 0.
            if params.project_to == 'teacher':
                self.hid_projectors_mse_teacher = None
                self.hid_projectors_mse_student = nn.ModuleList([nn.Linear(self.s_hid_dim, self.t_hid_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
            elif params.project_to =='student':
                self.hid_projectors_mse_student = None
                self.hid_projectors_mse_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, self.s_hid_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
            else:
                self.hid_projectors_mse_student = nn.ModuleList([nn.Linear(self.s_hid_dim, params.intermediate_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                self.hid_projectors_mse_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, params.intermediate_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])

        if self.alpha_contrastive > 0.0:
            self.train_loss_contrastive_epoch = 0.
            self.valid_loss_contrastive_epoch = 0.
            if params.hidden_distil_type == 'hyperbolic':
                self.similarity_metric = hypmath.dist
                self.c = params.c
                if params.init_c == 'precompute_from_teacher':
                    d, diam = delta_.get_delta(self.teacher, self.valid_loader, 
                                                   cuda_no=self.gpu_id, n_samples_slct=params.n_samples_to_precompute_c, 
                                                   n_components=params.reduce_to_n_components, n_tries=params.n_tries)
                    self.c = delta_.calculate_c(d, diam)
                elif params.init_c == 'precompute_from_student': 
                    d, diam = delta_.get_delta(self.teacher, self.valid_loader, 
                                                   cuda_no=self.gpu_id, n_samples_slct=params.n_samples_to_precompute_c, 
                                                   n_components=params.reduce_to_n_components, n_tries=params.n_tries)
                    self.c = delta_.calculate_c(d, diam)

                if params.use_hyperbolic_projections:
                    s_ball_dim = self.s_hid_dim
                    t_ball_dim = self.t_hid_dim
                    if self.params.project_to == 'teacher':
                        self.hid_projectors_contrastive_teacher = None
                        self.hid_projectors_contrastive_student = nn.ModuleList([hypnn.HypLinear(self.s_hid_dim, self.t_hid_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                    elif params.project_to =='student':
                        self.hid_projectors_contrastive_student = None
                        self.hid_projectors_contrastive_teacher = nn.ModuleList([hypnn.HypLinear(self.t_hid_dim, self.s_hid_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                    else:
                        self.hid_projectors_contrastive_student = nn.ModuleList([hypnn.HypLinear(self.s_hid_dim, params.intermediate_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                        self.hid_projectors_contrastive_teacher = nn.ModuleList([hypnn.HypLinear(self.t_hid_dim, params.intermediate_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
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
                
                self.student_to_poincare = hypnn.ToPoincare(self.c, train_c=params.train_c == 'train_exp_map_student', 
                                                            train_x=params.train_x, ball_dim=s_ball_dim, riemannian=params.riemannian)
                self.teacher_to_poincare = hypnn.ToPoincare(self.c, train_c=params.train_c == 'train_exp_map_teacher', 
                                                            train_x=params.train_x, ball_dim=t_ball_dim, riemannian=params.riemannian)
                if params.train_c == 'train_exp_map_student':
                    self.optimizer.add_param_group({'params': self.student_to_poincare.parameters()})
                elif params.train_c == 'train_exp_map_teacher':
                    self.optimizer.add_param_group({'params': self.teacher_to_poincare.parameters()})
            else:
                self.similarity_metric = custom_step.cosine_similarity

            if params.hidden_distil_type is None or (params.hidden_distil_type == 'hyperbolic' and not params.use_hyperbolic_projections):
                if params.project_to == 'teacher':
                    self.hid_projectors_contrastive_teacher = None
                    self.hid_projectors_contrastive_student = nn.ModuleList([nn.Linear(self.s_hid_dim, self.t_hid_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                elif params.project_to =='student':
                    self.hid_projectors_contrastive_student = None
                    self.hid_projectors_contrastive_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, self.s_hid_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                else:
                    self.hid_projectors_contrastive_student = nn.ModuleList([nn.Linear(self.s_hid_dim, params.intermediate_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                    self.hid_projectors_contrastive_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, params.intermediate_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])

        self.epoch = 0
        self.n_epochs = params.n_epochs
        self.train_loss_epoch = 0.
        self.valid_loss_epoch = 0.
        self.best_valid_loss_epoch = sys.maxsize
        self.n_train_batches_seen = 0
        self.n_valid_batches_seen = 0


    def step(self, input_ids, attention_mask, labels, token_type_ids=None, grad_on=True, is_predict_step=False, **kwargs):
        with torch.set_grad_enabled(grad_on):
            t_out = self.teacher(input_ids, attention_mask, token_type_ids)
            s_out = self.student(input_ids, attention_mask)

            predictions = torch.max(s_out.logits, dim=1)[1]
            self.main_metric.add_batch(predictions=predictions, references=labels)
            if is_predict_step:
                return 

            loss = 0.
            if self.alpha_ce > 0.0:
                b_size = s_out.logits.size(0)
                loss_ce = self.ce_loss(s_out.logits, labels) / b_size
                loss += self.alpha_ce * loss_ce

            if self.alpha_contrastive > 0.0:
                hyp_kwargs = dict(use_hyp_mapping_in_step=False)
                if self.params.hidden_distil_type == 'hyperbolic':
                    if self.params.use_hyperbolic_projections:
                        t_hiddens = [self.teacher_to_poincare(t, c=self.c) for t in t_out.hidden_states]
                        s_hiddens = [self.student_to_poincare(s, c=self.c) for s in s_out.hidden_states]
                    else:
                        hyp_kwargs = dict(use_hyp_mappings_in_step=True, 
                                        c=self.c, 
                                        student_to_poincare=self.student_to_poincare, 
                                        teacher_to_poincare=self.teacher_to_poincare)
                        t_hiddens = t_out.hidden_states
                        s_hiddens = s_out.hidden_states

                else:
                    t_hiddens = t_out.hidden_states
                    s_hiddens = s_out.hidden_states
                loss_contrastive = custom_step.contrastive_step(None, 
                                                                self.hid_projectors_contrastive_student, 
                                                                self.hid_projectors_contrastive_teacher, 
                                                                s_hiddens, t_hiddens, attention_mask, attention_mask, 
                                                                proj_strategy=self.params.projection_strategy, 
                                                                negative_sampling_strategy=self.params.negative_sampling_strategy, 
                                                                n_negative_samples=self.params.n_negative_samples, 
                                                                teacher_student_prop=self.params.teacher_student_prop, 
                                                                temperature=self.temperature, 
                                                                from_one_sample=self.params.from_one_sample, 
                                                                add_neg_size_constant=self.params.add_neg_size_constant, 
                                                                t_s_layers_ids=self.params.t_s_layers_ids, 
                                                                similarity_metric=self.similarity_metric, **hyp_kwargs)
                loss += self.alpha_contrastive * loss_contrastive
            if self.alpha_mse > 0.0:
                loss_mse = custom_step.mse_step(self.hid_projectors_mse_student, self.hid_projectors_mse_teacher, self.mse_loss, 
                                                s_out.hidden_states, t_out.hidden_states, attention_mask, attention_mask, 
                                                proj_strategy=self.params.projection_strategy, 
                                                t_s_layers_ids=self.params.t_s_layers_ids)
                loss += self.alpha_mse * loss_mse
            if self.alpha_kl > 0.0: 
                loss_kl = custom_step.ce_step(s_out.logits, t_out.logits, self.kl_loss, self.temperature)
                loss += self.alpha_kl * loss_kl

        if grad_on:
            self.train_loss_epoch += loss.item()
            if self.alpha_ce > 0.0:
                self.train_loss_ce_epoch += loss_ce.item()
            if self.alpha_kl > 0.0:
                self.train_loss_kl_epoch += loss_kl.item()
            if self.alpha_mse > 0.0:
                self.train_loss_mse_epoch += loss_mse.item()
            if self.alpha_contrastive > 0.0:
                self.train_loss_contrastive_epoch += loss_contrastive.item()
            self.optimize(loss)
        else:
            self.valid_loss_epoch += loss.item()
            if self.alpha_ce > 0.0:
                self.valid_loss_ce_epoch += loss_ce.item()
            if self.alpha_kl > 0.0:
                self.valid_loss_kl_epoch += loss_kl.item()
            if self.alpha_mse > 0.0:
                self.valid_loss_mse_epoch += loss_mse.item()
            if self.alpha_contrastive > 0.0:
                self.valid_loss_contrastive_epoch += loss_contrastive.item()


    def train(self):
        logger.info("Starting training")
        self.teacher.eval()

        for _ in range(self.n_epochs):
            self.student.train()
            logger.info(f"--- Starting epoch {self.epoch}/{self.n_epochs}")

            iter_bar = tqdm(self.train_loader, desc="-Iter")
            for batch in iter_bar:
                input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.teacher.forward.__code__.co_varnames}
                input_batch['grad_on'] = True
                self.step(**input_batch)
                self.n_train_batches_seen += 1
                iter_bar.update()
                iter_bar.set_postfix(
                    {"Avg_cum_loss": f"{self.train_loss_epoch/self.n_train_batches_seen:.2f}"}
                )

            iter_bar.close()
            logger.info(f"--- Ending epoch {self.epoch}/{self.n_epochs}")
            
            train_score = list(self.main_metric.compute().items())[0]
            self.metric_name = train_score[0]
            self.train_metric_value = train_score[1]

            self.validate()
            self.end_epoch()
        
        logger.info("Training is finished.")
        logger.info("Starting testing...")
        self.test()
        logger.info("Testing is finished.")


    def validate(self):
        self.teacher.eval()
        self.student.eval()
        
        logger.info(f"--- Validating epoch {self.epoch}/{self.n_epochs}")
        iter_bar = tqdm(self.valid_loader, desc="-Iter")
        for batch in iter_bar:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.teacher.forward.__code__.co_varnames}
            input_batch['grad_on'] = False
            self.step(**input_batch)
            self.n_valid_batches_seen += 1
            iter_bar.update()
            iter_bar.set_postfix(
                {"Avg_cum_valid_loss": f"{self.valid_loss_epoch /self.n_valid_batches_seen:.2f}"}
            )
        iter_bar.close()
        valid_score = list(self.main_metric.compute().items())[0]
        self.metric_name = valid_score[0]
        self.valid_metric_value = valid_score[1]
        self.optimizer.step()
        logger.info(f"--- Ending validation epoch {self.epoch}/{self.n_epochs}")


    def test(self):
        self.teacher.eval()
        self.student.eval()

        best_model = torch.load(os.path.join(self.params.dumps_dir, 'best_model.pth'))
        self.student.load_state_dict(best_model)

        logger.info(f"--- Testing...")
        iter_bar = tqdm(self.test_loader, desc="-Iter")
        for batch in iter_bar:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.teacher.__code__.co_varnames}
            input_batch['grad_on'] = False
            input_batch['is_predict_step'] = True
            self.step(**input_batch)

        test_scores = list(self.main_metric.compute())
        self.run.summary[f'test/{self.metric_name}'] = test_scores[1]
        self.run.summary[f'valid/{self.metric_name}'] = self.best_valid_metric_value
        
        self.summary_table.add_row([self.params.glue_dataset, self.metric_name, 
                                    self.best_valid_metric_value, test_scores[1]])
        self.run.log({'summary': self.summary_table})

    
    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.params.hidden_distil_type == 'hyperbolic' and hasattr(self.params, 'c'):
            if self.params.train_c == 'train_exp_map_teacher':
                self.c = self.teacher_to_poincare.c
            elif self.params.train_c == 'train_exp_map_student':
                self.c = self.student_to_poincare.c


    def end_epoch(self):
        self.epoch += 1
        self.log()

        cur_valid_loss = self.valid_loss_epoch / self.n_valid_batches_seen
        if cur_valid_loss < self.best_valid_loss_epoch:
            self.best_valid_loss_epoch = cur_valid_loss
        if self.valid_metric_value > self.best_valid_metric_value:
            self.best_valid_metric_value = self.valid_metric_value
            self.save_checkpoint('best_model.pth')

        self.save_checkpoint('last_checkpoint.pth', mode='full')
        self.save_checkpoint(f'epoch_{self.epoch}.pth')
        

        self.train_loss_epoch = 0
        self.valid_loss_epoch = 0

        if self.alpha_ce > 0.0:
            self.train_loss_ce_epoch = 0.0
            self.valid_loss_ce_epoch = 0.0

        if self.alpha_kl > 0.0:
            self.train_loss_kl_epoch = 0.0
            self.valid_loss_kl_epoch = 0.0

        if self.alpha_mse > 0.0:
            self.train_loss_mse_epoch = 0.0
            self.valid_loss_mse_epoch = 0.0

        if self.alpha_contrastive > 0.0:
            self.train_loss_contrastive_epoch = 0.0
            self.valid_loss_contrastive_epoch = 0.0

    
    def log(self):
        log_dict = {'epoch': self.epoch, 
                    'temperature': self.temperature.item() if self.params.train_temperature else self.temperature, 
                    'train/loss_epoch': self.train_loss_epoch / self.n_train_batches_seen, 
                    'valid/loss_epoch': self.valid_loss_epoch / self.n_valid_batches_seen, 
                    'lr': self.optimizer.param_groups[0]['lr'], 
                    f'train/{self.metric_name}': self.train_metric_value, 
                    f'valid/{self.metric_name}': self.valid_metric_value
        }

        if self.params.hidden_distil_type == 'hyperbolic':
            log_dict['curvature'] = self.c.item() if self.params.train_c is not None and 'train' in self.params.train_c else self.c

        if self.alpha_ce > 0.0:
            log_dict['train/loss_ce_epoch'] = self.train_loss_ce_epoch / self.n_train_batches_seen
            log_dict['valid/loss_ce_epoch'] = self.valid_loss_ce_epoch / self.n_valid_batches_seen
        
        if self.alpha_kl > 0.0:
            log_dict['train/loss_kl_epoch'] = self.train_loss_kl_epoch / self.n_train_batches_seen
            log_dict['valid/loss_kl_epoch'] = self.valid_loss_kl_epoch / self.n_valid_batches_seen

        if self.alpha_mse > 0.0:
            log_dict['train/loss_mse_epoch'] = self.train_loss_mse_epoch / self.n_train_batches_seen
            log_dict['valid/loss_mse_epoch'] = self.valid_loss_mse_epoch / self.n_valid_batches_seen

        if self.alpha_contrastive > 0.0:
            log_dict['train/loss_contrastive_epoch'] = self.train_loss_contrastive_epoch / self.n_train_batches_seen
            log_dict['valid/loss_contrastive_epoch'] = self.valid_loss_contrastive_epoch / self.n_valid_batches_seen

        self.run.log(log_dict)

    
    def save_checkpoint(self, checkpoint_name, mode=None):
        if mode == 'full':
            torch.save(dict(epoch=self.epoch, 
                            model_state_dict=self.student.state_dict(), 
                            optimizer_state_dict=self.optimizer.state_dict(), 
                            best_valid_loss=self.best_valid_loss_epoch, 
                            cur_valid_loss=self.valid_loss_epoch / self.n_valid_batches_seen, 
                            cur_valid_metric=self.valid_metric_value, 
                            best_valid_metric_value=self.best_valid_metric_value), 
                        os.path.join(self.params.dumps_dir, checkpoint_name))
        else:
            mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
            mdl_to_save.config.save_pretrained(self.params.dumps_dir)
            state_dict = mdl_to_save.state_dict()
            torch.save(state_dict, os.path.join(self.params.dumps_dir, checkpoint_name))

