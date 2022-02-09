import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from progressbar import ProgressBar

import load_data
import delta_
import os
import json

import sys
sys.path.insert(1, '../')
import hyptorch.nn as hypnn
import hyptorch.pmath as hypmath

import custom_step
from utils import logger

import datasets
import wandb

from transformers import get_linear_schedule_with_warmup
import math

import random
import numpy as np


class Distiller(object):
    def __init__(self, params):

        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)

        self.params = params
        self.run = params.run
        
        self.run.define_metric('step')
        self.run.define_metric('step/*', step_metric='step')
        self.run.define_metric('batch/*', step_metric='step')

        self.metric_name = ''
        self.train_metric_value = 0
        self.valid_metric_value = 0
        self.best_valid_metric_value = 0

        self.gradient_accumulation_steps = params.gradient_accumulation_steps
        self.n_gradient_updates = 0

        self.summary_table = wandb.Table(columns=['Task', 'Config_name', 'Metric', 'Train score', 'Validation score', 'Test score'])

        tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_name)
        train_data, _ = load_data.load_glue_dataset(params.glue_dataset, 
                                                 tokenizer, 
                                                 params.tokenizer_params, 
                                                 'train')
        valid_data, _ = load_data.load_glue_dataset(params.glue_dataset, 
                                                 tokenizer, 
                                                 params.tokenizer_params, 
                                                'validation')
        self.train_loader = DataLoader(train_data, params.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_data, params.batch_size, shuffle=False)

        self.log_after_n_steps = params.log_after_n_steps
        if params.log_after_epoch:
            self.log_after_n_steps = len(self.train_loader)

        self.gpu_id = params.gpu_id
        self.n_classes = train_data.features['labels'].num_classes
        self.alpha_task = params.alpha_task
        if params.alpha_task > 0.0:
            if self.n_classes == 1:
                self.task_loss = nn.MSELoss(reduction='sum')
            else:
                self.task_loss = nn.CrossEntropyLoss(reduction='sum')

            self.train_loss_task_step = 0.
            self.train_loss_task_batch = 0.
            self.valid_loss_task_step = 0.

        config = AutoConfig.from_pretrained(params.student_name, num_labels=self.n_classes, output_hidden_states=True, return_dict=True)
        self.student = AutoModelForSequenceClassification.from_config(config).to(f'cuda:{self.gpu_id}')
        self.teacher = AutoModelForSequenceClassification.from_pretrained(params.teacher_name, num_labels=self.n_classes, output_hidden_states=True, return_dict=True).to(f'cuda:{self.gpu_id}')
        if params.teacher_weights is not None:
            state_dict = torch.load(params.teacher_weights, map_location=torch.device(f'cuda:{self.gpu_id}'))
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.teacher.load_state_dict(state_dict)
        if params.student_weights is not None:
            state_dict = torch.load(params.student_weights, map_location=torch.device(f'cuda:{self.gpu_id}'))
            self.student.load_state_dict(state_dict)

        optimizer_grouped_parameters = []
        custom_step.add_param_group(optimizer_grouped_parameters, self.student, params.weight_decay)
        
        if params.train_temperature:
            self.temperature = nn.Parameter(torch.tensor([params.temperature], 
                                            requires_grad=True, 
                                            device=f'cuda:{params.gpu_id}'))
            custom_step.add_param_group(optimizer_grouped_parameters, self.temperature, params.weight_decay)
        else:
            self.temperature = params.temperature

        self.alpha_kl = params.alpha_kl
        self.alpha_mse = params.alpha_mse
        self.alpha_contrastive = params.alpha_contrastive

        self.s_hid_dim = self.student.config.hidden_size
        self.t_hid_dim = self.teacher.config.hidden_size

        if self.alpha_kl > 0.0:
            self.kl_loss = nn.KLDivLoss(reduction='sum')
            self.train_loss_kl_step = 0.
            self.valid_loss_kl_step = 0.
        
        layers_ct = 4 if self.params.projection_strategy in ['last', 'skip', 'average'] else 1
        if self.params.projection_strategy == 'weighted_average_by_layers':
            a = torch.rand(self.teacher.config.num_hidden_layers + 1, requires_grad=True, device=f'cuda:{self.gpu_id}')
            b = torch.rand(self.student.config.num_hidden_layers + 1, requires_grad=True, device=f'cuda:{self.gpu_id}')
            self.teacher_weights = nn.Parameter(a)
            self.student_weights = nn.Parameter(b)
            custom_step.add_param_group(optimizer_grouped_parameters, self.teacher_weights, params.weight_decay)
            custom_step.add_param_group(optimizer_grouped_parameters, self.student_weights, params.weight_decay)
        else:
            self.teacher_weights = None
            self.student_weights = None

        if self.alpha_mse > 0.0:
            self.mse_loss = nn.MSELoss(reduction='sum')
            self.train_loss_mse_step = 0.
            self.train_loss_mse_batch = 0.
            self.valid_loss_mse_step = 0.

            if params.project_to == 'teacher':
                self.hid_projectors_mse_teacher = None
                self.hid_projectors_mse_student = nn.ModuleList([nn.Linear(self.s_hid_dim, self.t_hid_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                
                custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_mse_student, params.weight_decay)

            elif params.project_to =='student':
                self.hid_projectors_mse_student = None
                self.hid_projectors_mse_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, self.s_hid_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])

                custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_mse_teacher, params.weight_decay)
            else:
                self.hid_projectors_mse_student = nn.ModuleList([nn.Linear(self.s_hid_dim, params.intermediate_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                self.hid_projectors_mse_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, params.intermediate_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_mse_student, params.weight_decay)
                custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_mse_teacher, params.weight_decay)

        if self.alpha_contrastive > 0.0:
            self.train_loss_contrastive_step = 0.
            self.train_loss_contrastive_batch = 0.
            self.valid_loss_contrastive_step = 0.

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
                logger.info('Curvature was set to {:.4f}'.format(self.c))

                if params.use_hyperbolic_projections:
                    s_ball_dim = self.s_hid_dim
                    t_ball_dim = self.t_hid_dim
                    if self.params.project_to == 'teacher':
                        self.hid_projectors_contrastive_teacher = None
                        self.hid_projectors_contrastive_student = nn.ModuleList([hypnn.HypLinear(self.s_hid_dim, self.t_hid_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                        custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_contrastive_student, params.weight_decay)
                    elif params.project_to =='student':
                        self.hid_projectors_contrastive_student = None
                        self.hid_projectors_contrastive_teacher = nn.ModuleList([hypnn.HypLinear(self.t_hid_dim, self.s_hid_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                        custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_contrastive_teacher, params.weight_decay)
                    else:
                        self.hid_projectors_contrastive_student = nn.ModuleList([hypnn.HypLinear(self.s_hid_dim, params.intermediate_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                        self.hid_projectors_contrastive_teacher = nn.ModuleList([hypnn.HypLinear(self.t_hid_dim, params.intermediate_dim, 
                                                                                                 self.c, params.use_bias).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                        custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_contrastive_student, params.weight_decay)
                        custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_contrastive_teacher, params.weight_decay)
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
                    custom_step.add_param_group(optimizer_grouped_parameters, self.student_to_poincare, params.weight_decay)
                elif params.train_c == 'train_exp_map_teacher':
                    custom_step.add_param_group(optimizer_grouped_parameters, self.teacher_to_poincare, params.weight_decay)
            else:
                self.similarity_metric = custom_step.cosine_similarity

            if params.hidden_distil_type is None or (params.hidden_distil_type == 'hyperbolic' and not params.use_hyperbolic_projections):
                if params.project_to == 'teacher':
                    self.hid_projectors_contrastive_teacher = None
                    self.hid_projectors_contrastive_student = nn.ModuleList([nn.Linear(self.s_hid_dim, self.t_hid_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                    custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_contrastive_student, params.weight_decay)
                elif params.project_to =='student':
                    self.hid_projectors_contrastive_student = None
                    self.hid_projectors_contrastive_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, self.s_hid_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                    custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_contrastive_teacher, params.weight_decay)
                else:
                    self.hid_projectors_contrastive_student = nn.ModuleList([nn.Linear(self.s_hid_dim, params.intermediate_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                    self.hid_projectors_contrastive_teacher = nn.ModuleList([nn.Linear(self.t_hid_dim, params.intermediate_dim).to(f'cuda:{self.gpu_id}') for _ in range(layers_ct)])
                    custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_contrastive_student, params.weight_decay)
                    custom_step.add_param_group(optimizer_grouped_parameters, self.hid_projectors_contrastive_teacher, params.weight_decay)

        self.train_loss_step = 0.
        self.train_loss_batch = 0.
        self.valid_loss_step = 0.
        self.best_valid_loss_step = sys.maxsize
        self.n_train_batches_seen = 0
        self.n_total_train_batches_seen = 0
        self.n_valid_batches_seen = 0
        self.cur_epoch = 0

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=params.lr)
        self.num_steps_epoch = len(self.train_loader)
        self.num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epochs) + 1
        )
        self.warmup_steps = math.ceil(self.num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=self.warmup_steps, 
                                                         num_training_steps=self.num_train_optimization_steps
                                                        )


    def train(self):
        logger.info("Starting training")
        self.save_checkpoint('best_model.pth')

        train_metric = datasets.load_metric('glue', self.params.glue_dataset)
        bar = ProgressBar(max_value=self.gradient_accumulation_steps * self.log_after_n_steps)
        ct = 0
        for _ in range(self.params.n_epochs):
            self.cur_epoch += 1
            for batch in self.train_loader:
                bar.update(ct)
                ct += 1

                input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.student.forward.__code__.co_varnames}
                input_batch['grad_on'] = True
                self.n_train_batches_seen += 1
                self.n_total_train_batches_seen += 1

                predictions = self.step(**input_batch)
                train_metric.add_batch(predictions=predictions.view(-1), references=batch['labels'].view(-1))

                if self.n_total_train_batches_seen / self.gradient_accumulation_steps == self.n_gradient_updates and self.n_gradient_updates % self.log_after_n_steps == 0:
                    train_res = list(train_metric.compute().items())[0]
                    self.metric_name = train_res[0]
                    self.train_metric_value = train_res[1]
                    self.validate()
                    self.log_step()
                    self.end_step()
                    ct = 0
        
        logger.info("Training is finished.")
        logger.info("Starting testing...")
        self.test()
        logger.info("Testing is finished.")

    def validate(self):
        logger.info(f"--- Validating step...")
        valid_metric = datasets.load_metric('glue', self.params.glue_dataset)
        for batch in self.valid_loader:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.teacher.forward.__code__.co_varnames}
            input_batch['grad_on'] = False
            predictions = self.step(**input_batch)
            valid_metric.add_batch(predictions=predictions.view(-1), references=batch['labels'].view(-1))
            self.n_valid_batches_seen += 1
        
        valid_res = list(valid_metric.compute().items())[0]
        self.metric_name = valid_res[0]
        self.valid_metric_value = valid_res[1]
        logger.info(f"--- Ending validating step.")

    def test(self):
        best_model = torch.load(os.path.join(self.params.dumps_dir, 'best_model.pth'))
        self.student.load_state_dict(best_model)

        logger.info(f"--- Testing...")

        test_metric = datasets.load_metric('glue', self.params.glue_dataset)
        for batch in self.valid_loader:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.teacher.forward.__code__.co_varnames}
            input_batch['grad_on'] = False
            input_batch['is_predict_step'] = True
            predictions = self.step(**input_batch)
            test_metric.add_batch(predictions=predictions.view(-1), references=batch['labels'].view(-1))

        test_scores = list(test_metric.compute().items())[0]
        self.run.summary[f'test/{self.metric_name}'] = test_scores[1]
        self.run.summary[f'valid/{self.metric_name}'] = self.best_valid_metric_value
        self.run.summary[f'train/{self.metric_name}'] = self.train_metric_value
        
        self.summary_table.add_data(self.params.glue_dataset, 
                                    self.params.run_id, 
                                    self.metric_name, 
                                    self.train_metric_value, 
                                    self.best_valid_metric_value, test_scores[1])
        self.run.log({'summary': self.summary_table})

    
    def step(self, input_ids, attention_mask, labels, token_type_ids=None, grad_on=True, is_predict_step=False, **kwargs):
        self.student.train() if grad_on else self.student.eval()

        with torch.set_grad_enabled(grad_on):
            t_out = self.teacher(input_ids, attention_mask, token_type_ids)
            s_out = self.student(input_ids, attention_mask)

            with torch.no_grad():
                predictions = torch.max(s_out.logits.detach(), dim=1)[1]

            if is_predict_step:
                return predictions

            loss = 0.
            if self.alpha_task > 0.0:
                b_size = s_out.logits.size(0)
                loss_task = self.task_loss(s_out.logits, labels) / b_size
                loss += self.alpha_task * loss_task

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
                                                                similarity_metric=self.similarity_metric, 
                                                                student_weights=self.student_weights, 
                                                                teacher_weights=self.teacher_weights, **hyp_kwargs)
                loss += self.alpha_contrastive * loss_contrastive
            if self.alpha_mse > 0.0:
                loss_mse = custom_step.mse_step(self.hid_projectors_mse_student, self.hid_projectors_mse_teacher, self.mse_loss, 
                                                s_out.hidden_states, t_out.hidden_states, attention_mask, attention_mask, 
                                                proj_strategy=self.params.projection_strategy, 
                                                t_s_layers_ids=self.params.t_s_layers_ids, 
                                                student_weights=self.student_weights, 
                                                teacher_weights=self.teacher_weights)
                loss += self.alpha_mse * loss_mse
            if self.alpha_kl > 0.0: 
                loss_kl = custom_step.ce_step(s_out.logits, t_out.logits, self.kl_loss, self.temperature)
                loss += self.alpha_kl * loss_kl

        if grad_on:
            self.train_loss_step += loss.item()
            self.train_loss_batch = loss.item()
            if self.alpha_task > 0.0:
                self.train_loss_task_step += loss_task.item()
                self.train_loss_task_batch = loss_task.item()
            if self.alpha_kl > 0.0:
                self.train_loss_kl_step += loss_kl.item()
                self.train_loss_kl_batch = loss_kl.item()
            if self.alpha_mse > 0.0:
                self.train_loss_mse_step += loss_mse.item()
                self.train_loss_mse_batch = loss_mse.item()
            if self.alpha_contrastive > 0.0:
                self.train_loss_contrastive_step += loss_contrastive.item()
                self.train_loss_contrastive_batch = loss_contrastive.item()
            self.optimize(loss)
        else:
            self.valid_loss_step += loss.item()
            if self.alpha_task > 0.0:
                self.valid_loss_task_step += loss_task.item()
            if self.alpha_kl > 0.0:
                self.valid_loss_kl_step += loss_kl.item()
            if self.alpha_mse > 0.0:
                self.valid_loss_mse_step += loss_mse.item()
            if self.alpha_contrastive > 0.0:
                self.valid_loss_contrastive_step += loss_contrastive.item()
        return predictions

    
    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        loss.backward()

        if (self.n_total_train_batches_seen // self.gradient_accumulation_steps > 0 and 
            self.n_total_train_batches_seen % self.gradient_accumulation_steps == 0):

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.n_gradient_updates += 1
            self.scheduler.step()

            if self.params.hidden_distil_type == 'hyperbolic' and hasattr(self.params, 'c'):
                if self.params.train_c == 'train_exp_map_teacher':
                    self.c = self.teacher_to_poincare.c
                elif self.params.train_c == 'train_exp_map_student':
                    self.c = self.student_to_poincare.c


    def end_step(self):
        cur_valid_loss = self.valid_loss_step / self.n_valid_batches_seen
        if cur_valid_loss < self.best_valid_loss_step:
            self.best_valid_loss_step = cur_valid_loss
        if self.valid_metric_value > self.best_valid_metric_value:
            self.best_valid_metric_value = self.valid_metric_value
            self.save_checkpoint('best_model.pth')

        self.save_checkpoint('last_checkpoint.pth', mode='full')
        self.save_checkpoint(f'step_{self.cur_epoch}.pth')

        self.train_loss_step = 0
        self.valid_loss_step = 0

        if self.alpha_task > 0.0:
            self.train_loss_task_step = 0.0
            self.valid_loss_task_step = 0.0

        if self.alpha_kl > 0.0:
            self.train_loss_kl_step = 0.0
            self.valid_loss_kl_step = 0.0

        if self.alpha_mse > 0.0:
            self.train_loss_mse_step = 0.0
            self.valid_loss_mse_step = 0.0

        if self.alpha_contrastive > 0.0:
            self.train_loss_contrastive_step = 0.0
            self.valid_loss_contrastive_step = 0.0

        self.n_train_batches_seen = 0
        self.n_valid_batches_seen = 0


    def log_step(self):
        log_dict = {'step': self.n_gradient_updates // self.log_after_n_steps, 
                    'step/temperature': self.temperature.item() if self.params.train_temperature else self.temperature, 
                    'step/train_loss_cum_avg': self.train_loss_step / self.n_train_batches_seen, 
                    'batch/train_loss': self.train_loss_batch, 
                    'step/valid_loss_cum_avg': self.valid_loss_step / self.n_valid_batches_seen,
                    'step/lr': self.optimizer.param_groups[0]['lr'],  
                    f'step/train_{self.metric_name}': self.train_metric_value, 
                    f'step/valid_{self.metric_name}': self.valid_metric_value
                    }
        if self.params.hidden_distil_type == 'hyperbolic':
            log_dict['step/curvature'] = self.c.item() if self.params.train_c is not None and 'train' in self.params.train_c else self.c

        if self.alpha_task > 0.0:
            log_dict['batch/train_loss_task'] = self.train_loss_task_batch
            log_dict['step/train_loss_task_cum_avg'] = self.train_loss_task_step / self.n_train_batches_seen
            log_dict['step/valid_loss_task_cum_avg'] = self.valid_loss_task_step / self.n_valid_batches_seen
        
        if self.alpha_kl > 0.0:
            log_dict['batch/train_loss_kl'] = self.train_loss_kl_batch
            log_dict['step/train_loss_kl_cum_avg'] = self.train_loss_kl_step / self.n_train_batches_seen
            log_dict['step/valid_loss_kl_cum_avg'] = self.valid_loss_kl_step / self.n_valid_batches_seen

        if self.alpha_mse > 0.0:
            log_dict['batch/train_loss_mse'] = self.train_loss_mse_batch
            log_dict['step/train_loss_mse_cum_avg'] = self.train_loss_mse_step / self.n_train_batches_seen
            log_dict['step/valid_loss_mse_cum_avg'] = self.valid_loss_mse_step / self.n_valid_batches_seen

        if self.alpha_contrastive > 0.0:
            log_dict['batch/train_loss_contrastive'] = self.train_loss_contrastive_batch
            log_dict['step/train_loss_contrastive_cum_avg'] = self.train_loss_contrastive_step / self.n_train_batches_seen
            log_dict['step/valid_loss_contrastive_cum_avg'] = self.valid_loss_contrastive_step / self.n_valid_batches_seen

        self.run.log(log_dict)
        logger.info(json.dumps(log_dict))


    def save_checkpoint(self, checkpoint_name, mode=None):
        if mode == 'full':
            torch.save(dict(step=self.n_gradient_updates / self.log_after_n_steps, 
                            model_state_dict=self.student.state_dict(), 
                            optimizer_state_dict=self.optimizer.state_dict(), 
                            best_valid_loss=self.best_valid_loss_step, 
                            cur_valid_loss=self.valid_loss_step / self.n_valid_batches_seen, 
                            cur_valid_metric=self.valid_metric_value, 
                            best_valid_metric_value=self.best_valid_metric_value), 
                        os.path.join(self.params.dumps_dir, checkpoint_name))
        else:
            mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
            mdl_to_save.config.save_pretrained(self.params.dumps_dir)
            state_dict = mdl_to_save.state_dict()
            torch.save(state_dict, os.path.join(self.params.dumps_dir, checkpoint_name))
