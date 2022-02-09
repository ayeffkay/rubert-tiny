import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW
from progressbar import ProgressBar

import load_data
import os

import sys
sys.path.insert(1, '../')
from utils import logger

import datasets
import wandb
import json

import random
import numpy as np


class StudentTrainer(object):
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

        self.lr_drop_patience = params.lr_drop_patience
        self.lr_drop_div = params.lr_drop_div
        self.lr_drop_ct = 0

        self.valid_patience = params.valid_patience
        self.validation_patience_ct = 0
        self.log_after_n_steps = params.log_after_n_steps

        self.gradient_accumulation_steps = params.gradient_accumulation_steps
        self.n_gradient_updates = 0

        self.end_train_flag = False

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

        self.train_loader = DataLoader(train_data, params.batch_size, shuffle=True, collate_fn=load_data.collate_fn)
        self.valid_loader = DataLoader(valid_data, params.batch_size, shuffle=False, collate_fn=load_data.collate_fn)
        self.test_loader = DataLoader(valid_data, params.batch_size, shuffle=False, collate_fn=load_data.collate_fn)

        self.log_after_n_steps = params.log_after_n_steps
        if params.log_after_epoch:
            self.log_after_n_steps = len(self.train_loader)
            
        self.gpu_id = params.gpu_id
        self.n_classes = train_data.features['labels'].num_classes
        config = AutoConfig.from_pretrained(params.model_name, num_labels=self.n_classes, output_hidden_states=True, return_dict=True)
        if params.from_pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(params.model_name, config=config).to(f'cuda:{self.gpu_id}')
        else:
            self.model = AutoModelForSequenceClassification.from_config(config).to(f'cuda:{self.gpu_id}')

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': params.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=params.lr)

        self.alpha_task = params.alpha_task
        assert self.alpha_task > 0.0

        if self.n_classes > 1:
            self.task_loss = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.task_loss = nn.MSELoss()

        self.train_task_loss_step = 0.
        self.valid_task_loss_step = 0.
        self.train_task_loss_batch = 0.
        self.best_valid_task_loss_step = sys.maxsize

        self.n_train_batches_seen = 0
        self.n_total_train_batches_seen = 0
        self.n_valid_batches_seen = 0


    def step(self, input_ids, attention_mask, labels, 
             grad_on=True, is_predict_step=False, token_type_ids=None, **kwargs):
        self.model.train() if grad_on else self.model.eval()

        with torch.set_grad_enabled(grad_on):
            s_out = self.model(input_ids, attention_mask, token_type_ids)

            with torch.no_grad():
                predictions = torch.max(s_out.logits.detach(), dim=1)[1]

            if is_predict_step:
                return predictions

            task_loss = self.alpha_task * self.task_loss(s_out.logits, labels)

        if grad_on:
            self.train_task_loss_batch = task_loss.item()
            self.train_task_loss_step += task_loss.item()
            self.optimize(task_loss)
        else:
            self.valid_task_loss_step += task_loss.item()

        return predictions


    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()
        if self.gradient_accumulation_steps > 1:
            loss /= self.gradient_accumulation_steps
        loss.backward()
        
        if (self.n_total_train_batches_seen // self.gradient_accumulation_steps > 0 and 
            self.n_total_train_batches_seen % self.gradient_accumulation_steps == 0):

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.n_gradient_updates += 1


    def train(self):
        logger.info("Starting training")
        self.save_checkpoint('best_model.pth')

        train_metric = datasets.load_metric('glue', self.params.glue_dataset)
        bar = ProgressBar(max_value=self.log_after_n_steps * self.gradient_accumulation_steps)
        i = 0
        while True:
            if self.end_train_flag:
                break
            
            for batch in self.train_loader:
                i += 1
                bar.update(i)

                input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.model.forward.__code__.co_varnames}
                input_batch['grad_on'] = True
                self.n_train_batches_seen += 1
                self.n_total_train_batches_seen += 1

                predictions = self.step(**input_batch)
                train_metric.add_batch(predictions=predictions.view(-1), references=batch['labels'].view(-1))

                if self.n_gradient_updates > 0:
                    if self.n_total_train_batches_seen / self.gradient_accumulation_steps == self.n_gradient_updates and self.n_gradient_updates % self.log_after_n_steps == 0:
                        train_res = list(train_metric.compute().items())[0]
                        self.metric_name = train_res[0]
                        self.train_metric_value = train_res[1]
                        self.validate()
                        self.log()
                        self.end_step()
                        i = 0

                if self.end_train_flag:
                    break
        
        logger.info("Training is finished.")
        logger.info("Starting testing...")
        self.test()
        logger.info("Testing is finished.")


    def validate(self):
        logger.info(f"--- Validating step...")
        valid_metric = datasets.load_metric('glue', self.params.glue_dataset)
        for batch in self.valid_loader:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.model.forward.__code__.co_varnames}
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
        self.model.load_state_dict(best_model)

        logger.info(f"--- Testing...")

        test_metric = datasets.load_metric('glue', self.params.glue_dataset)
        for batch in self.test_loader:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.model.forward.__code__.co_varnames}
            input_batch['grad_on'] = False
            input_batch['is_predict_step'] = True
            predictions = self.step(**input_batch)
            test_metric.add_batch(predictions=predictions.view(-1), references=batch['labels'].view(-1))

        test_scores = list(test_metric.compute().items())[0]
        self.run.summary[f'train/{self.metric_name}'] = self.train_metric_value
        self.run.summary[f'test/{self.metric_name}'] = test_scores[1]
        self.run.summary[f'valid/{self.metric_name}'] = self.best_valid_metric_value
        
        self.summary_table.add_data(self.params.glue_dataset, 
                                    self.params.run_id, 
                                    self.metric_name, 
                                    self.train_metric_value, 
                                    self.best_valid_metric_value, test_scores[1])
        self.run.log({'summary': self.summary_table})



    def end_step(self):
        cur_valid_loss = self.valid_task_loss_step / self.n_valid_batches_seen
        if cur_valid_loss < self.best_valid_task_loss_step:
            self.best_valid_task_loss_step = cur_valid_loss
        if self.valid_metric_value > self.best_valid_metric_value:
            self.best_valid_metric_value = self.valid_metric_value
            self.save_checkpoint('best_model.pth')
            self.lr_drop_ct = 0
            self.validation_patience_ct = 0
        else:
            self.lr_drop_ct += 1
            self.validation_patience_ct += 1
            logger.info(f'LR drop patience increased: {self.lr_drop_ct}/{self.lr_drop_patience}')
            logger.info(f'Validation patience increased: {self.validation_patience_ct}/{self.valid_patience}')

            if (self.lr_drop_ct == self.lr_drop_patience and 
                self.optimizer.param_groups[0]['lr'] > self.params.min_lr): 

                self.optimizer.param_groups[0]['lr'] /= self.lr_drop_div
                self.lr_drop_ct = 0
            
            if self.validation_patience_ct == self.valid_patience:
                self.end_train_flag = True

        self.save_checkpoint('last_checkpoint.pth', mode='full')
        self.save_checkpoint(f'step_{self.n_gradient_updates // self.log_after_n_steps}.pth') 

        self.train_task_loss_step = 0
        self.valid_task_loss_step = 0
        self.n_train_batches_seen = 0
        self.n_valid_batches_seen = 0
    
    def log(self):
        log_dict = {'step': self.n_gradient_updates / self.log_after_n_steps, 
                    'step/train_loss_cum_avg': self.train_task_loss_step / self.n_train_batches_seen, 
                    'step/valid_loss_cum_avg': self.valid_task_loss_step / self.n_valid_batches_seen, 
                    'step/lr': self.optimizer.param_groups[0]['lr'], 
                    f'step/train_{self.metric_name}': self.train_metric_value, 
                    f'step/valid_{self.metric_name}': self.valid_metric_value, 
                    'batch/train_loss': self.train_task_loss_batch
        }
        self.run.log(log_dict)
        logger.info(json.dumps(log_dict))

    
    def save_checkpoint(self, checkpoint_name, mode=None):
        if mode == 'full':
            torch.save(dict(step=self.n_gradient_updates // self.log_after_n_steps, 
                            model_state_dict=self.model.state_dict(), 
                            optimizer_state_dict=self.optimizer.state_dict(), 
                            best_valid_loss=self.best_valid_task_loss_step, 
                            cur_valid_loss=self.valid_task_loss_step / self.n_valid_batches_seen, 
                            cur_valid_metric=self.valid_metric_value, 
                            best_valid_metric_value=self.best_valid_metric_value), 
                            os.path.join(self.params.dumps_dir, checkpoint_name))
        else:
            mdl_to_save = self.model.module if hasattr(self.model, "module") else self.model
            mdl_to_save.config.save_pretrained(self.params.dumps_dir)
            state_dict = mdl_to_save.state_dict()
            torch.save(state_dict, os.path.join(self.params.dumps_dir, checkpoint_name))

    
