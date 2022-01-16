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

class StudentTrainer(object):
    def __init__(self, params):
        self.params = params
        self.run = params.run
        self.run.define_metric('step')
        self.run.define_metric('train/*', step_metric='step')
        self.run.define_metric('valid/*', step_metric='step')

        self.metric_name = ''
        self.train_metric_value = 0
        self.valid_metric_value = 0
        self.best_valid_metric_value = 0

        self.lr_drop_patience = params.lr_drop_patience
        self.lr_drop_div = params.lr_drop_div
        self.valid_patience = params.valid_patience
        self.val_every_n_batches = params.val_every_n_batches
        self.end_train_flag = False

        self.summary_table = wandb.Table(columns=['Task', 'Metric', 'Validation score', 'Test score'])

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
        
        if params.val_after_epoch:
            self.val_every_n_batches = len(self.train_loader)

        self.gpu_id = params.gpu_id
        self.n_classes = train_data.features['labels'].num_classes
        config = AutoConfig.from_pretrained(params.student_name, num_labels=self.n_classes, output_hidden_states=True, return_dict=True)
        if params.from_pretrained:
            self.student = AutoModelForSequenceClassification.from_pretrained(params.student_name, config=config).to(f'cuda:{self.gpu_id}')
        else:
            self.student = AutoModelForSequenceClassification.from_config(config).to(f'cuda:{self.gpu_id}')
        self.optimizer = AdamW(self.student.parameters(), lr=params.lr)
        #self.scheduler = getattr(torch.optim.lr_scheduler, params.scheduler)(self.optimizer, **params.scheduler_params)

        self.alpha_ce = params.alpha_ce
        assert self.alpha_ce > 0.0

        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.train_loss_ce_step = 0.
        self.valid_loss_ce_step = 0.
        self.best_valid_loss_ce_step = sys.maxsize

        self.n_train_batches_seen = 0
        self.n_total_train_batches_seen = 0
        self.n_valid_batches_seen = 0
        self.lr_drop_ct = 0
        self.validation_patience_ct = 0
        self.n_log_step = 0

    def step(self, input_ids, attention_mask, labels, 
             grad_on=True, is_predict_step=False, token_type_ids=None, **kwargs):
        self.student.train() if grad_on else self.student.eval()

        with torch.set_grad_enabled(grad_on):
            s_out = self.student(input_ids, attention_mask, token_type_ids)

            with torch.no_grad():
                predictions = torch.max(s_out.logits.detach(), dim=1)[1]

            if is_predict_step:
                return predictions
            loss_ce = self.ce_loss(s_out.logits, labels)

        if grad_on:
            self.train_loss_ce_step += loss_ce.item()
            self.optimize(loss_ce)
        else:
            self.valid_loss_ce_step += loss_ce.item()

        return predictions


    def train(self):
        logger.info("Starting training")
        torch.save(self.student.state_dict(), 'best_model.pth')

        while True:
            if self.end_train_flag:
                break
            train_metric = datasets.load_metric('glue', self.params.glue_dataset)
            bar = ProgressBar(max_value=self.val_every_n_batches)
            i = 0
            for batch in self.train_loader:
                input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.student.forward.__code__.co_varnames}
                input_batch['grad_on'] = True
                predictions = self.step(**input_batch)
                train_metric.add_batch(predictions=predictions, references=batch['labels'])
                self.n_train_batches_seen += 1
                self.n_total_train_batches_seen += 1

                if self.n_total_train_batches_seen % self.val_every_n_batches == 0:
                    train_res = list(train_metric.compute().items())[0]
                    self.metric_name = train_res[0]
                    self.train_metric_value = train_res[1]

                    self.validate()
                    self.end_step()
                    i = 0

                if self.end_train_flag:
                    break
                i += 1
                bar.update(i)
        
        logger.info("Training is finished.")
        logger.info("Starting testing...")
        self.test()
        logger.info("Testing is finished.")


    def validate(self):
        logger.info(f"--- Validating step...")
        valid_metric = datasets.load_metric('glue', self.params.glue_dataset)
        for batch in self.valid_loader:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.student.forward.__code__.co_varnames}
            input_batch['grad_on'] = False
            predictions = self.step(**input_batch)
            valid_metric.add_batch(predictions=predictions, references=batch['labels'])
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
        for batch in self.test_loader:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.student.forward.__code__.co_varnames}
            input_batch['grad_on'] = False
            input_batch['is_predict_step'] = True
            predictions = self.step(**input_batch)
            test_metric.add_batch(predictions=predictions, references=batch['labels'])

        test_scores = list(test_metric.compute().items())[0]
        self.run.summary[f'test/{self.metric_name}'] = test_scores[1]
        self.run.summary[f'valid/{self.metric_name}'] = self.best_valid_metric_value
        
        self.summary_table.add_data(self.params.glue_dataset, self.metric_name, 
                                    self.best_valid_metric_value, test_scores[1])
        self.run.log({'summary': self.summary_table})

    
    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


    def end_step(self):
        self.n_log_step += 1
        self.log()

        cur_valid_loss = self.valid_loss_ce_step / self.n_valid_batches_seen
        if cur_valid_loss < self.best_valid_loss_ce_step:
            self.best_valid_loss_ce_step = cur_valid_loss
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
        self.save_checkpoint(f'step_{self.n_log_step}.pth') 

        self.train_loss_ce_step = 0
        self.valid_loss_ce_step = 0
        self.n_train_batches_seen = 0
        self.n_valid_batches_seen = 0


    
    def log(self):
        log_dict = {'step': self.n_log_step, 
                    'train/loss_ce_step': self.train_loss_ce_step / self.n_train_batches_seen, 
                    'valid/loss_ce_step': self.valid_loss_ce_step / self.n_valid_batches_seen, 
                    'lr': self.optimizer.param_groups[0]['lr'], 
                    f'train/{self.metric_name}': self.train_metric_value, 
                    f'valid/{self.metric_name}': self.valid_metric_value
        }
        self.run.log(log_dict)
        logger.info(json.dumps(log_dict))

    
    def save_checkpoint(self, checkpoint_name, mode=None):
        if mode == 'full':
            torch.save(dict(step=self.n_log_step, 
                            model_state_dict=self.student.state_dict(), 
                            optimizer_state_dict=self.optimizer.state_dict(), 
                            best_valid_loss=self.best_valid_loss_ce_step, 
                            cur_valid_loss=self.valid_loss_ce_step / self.n_valid_batches_seen, 
                            cur_valid_metric=self.valid_metric_value, 
                            best_valid_metric_value=self.best_valid_metric_value), 
                        os.path.join(self.params.dumps_dir, checkpoint_name))
        else:
            mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
            mdl_to_save.config.save_pretrained(self.params.dumps_dir)
            state_dict = mdl_to_save.state_dict()
            torch.save(state_dict, os.path.join(self.params.dumps_dir, checkpoint_name))

    
