import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.optimization import AdamW
from tqdm import tqdm

import load_data
import os

import sys
sys.path.insert(1, '../')
from utils import logger

import datasets
import wandb

class StudentTrainer(object):
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

        train_data, valid_data = load_data.load_glue_dataset(params.glue_dataset, 
                                                            params.tokenizer_name, 
                                                            params.padding, 
                                                            params.truncation, 
                                                            'train', params.valid_prop, 
                                                            params.seed)
        test_data = load_data.load_glue_dataset(params.glue_dataset, 
                                                params.tokenizer_name, 
                                                params.padding, 
                                                params.truncation, 
                                                'validation')
        self.train_loader = DataLoader(train_data, params.batch_size, shuffle=True, collate_fn=load_data.collate_fn)
        self.valid_loader = DataLoader(valid_data, params.batch_size, shuffle=False, collate_fn=load_data.collate_fn)
        self.test_loader = DataLoader(test_data, params.batch_size, shuffle=False, collate_fn=load_data.collate_fn)


        self.gpu_id = params.gpu_id
        self.n_classes = self.train_data.features['labels'].num_classes
        config = AutoConfig.from_pretrained(params.student_name, num_labels=self.n_classes, output_hidden_states=True, return_dict=True)
        self.student = AutoModelForSequenceClassification.from_config(config).to(f'cuda:{self.gpu_id}')
        self.optimizer = AdamW(self.student.parameters(), lr=params.lr)
        self.scheduler = getattr(torch.optim.lr_scheduler, params.scheduler)(self.optimizer, **params.scheduler_params)

        self.alpha_ce = params.alpha_ce
        
        if self.alpha_ce > 0.0:
            self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
            self.train_loss_ce_epoch = 0.
            self.valid_loss_ce_epoch = 0.

        self.epoch = 0
        self.n_epochs = params.n_epochs
        self.train_loss_epoch = 0.
        self.valid_loss_epoch = 0.
        self.best_valid_loss_epoch = sys.maxsize
        self.n_train_batches_seen = 0
        self.n_valid_batches_seen = 0

    def step(self, input_ids, attention_mask, labels, grad_on=True, is_predict_step=False, **kwargs):
        with torch.set_grad_enabled(grad_on):
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

        if grad_on:
            self.train_loss_epoch += loss.item()
            if self.alpha_ce > 0.0:
                self.train_loss_ce_epoch += loss_ce.item()
                self.optimize(loss)
        else:
            self.valid_loss_epoch += loss.item()
            if self.alpha_ce > 0.0:
                self.valid_loss_ce_epoch += loss_ce.item()


    def train(self):
        logger.info("Starting training")

        for _ in range(self.n_epochs):
            self.student.train()
            logger.info(f"--- Starting epoch {self.epoch}/{self.n_epochs}")

            iter_bar = tqdm(self.train_loader, desc="-Iter")
            for batch in iter_bar:
                input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.student.forward.__code__.co_varnames}
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

        self.student.eval()
        logger.info(f"--- Validating epoch {self.epoch}/{self.n_epochs}")
        iter_bar = tqdm(self.valid_loader, desc="-Iter")
        for batch in iter_bar:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.student.forward.__code__.co_varnames}
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
        self.student.eval()
        best_model = torch.load(self.params.dumps_dir, 'best_model.pth')
        self.student.load_state_dict(best_model)

        logger.info(f"--- Testing...")
        iter_bar = tqdm(self.test_loader, desc="-Iter")
        for batch in iter_bar:
            input_batch = {name: value.to(f'cuda:{self.gpu_id}') for name, value in batch.items() if name in self.student.__code__.co_varnames}
            input_batch['grad_on'] = False
            input_batch['is_predict_step'] = True
            self.step(**input_batch)

        test_scores = list(self.main_metric.compute())[0]
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

    
    def log(self):
        log_dict = {'epoch': self.epoch, 
                    'train/loss_epoch': self.train_loss_epoch / self.n_train_batches_seen, 
                    'valid/loss_epoch': self.valid_loss_epoch / self.n_valid_batches_seen, 
                    'lr': self.optimizer.param_groups[0]['lr'], 
                    f'train/{self.metric_name}': self.train_metric_value, 
                    f'valid/{self.metric_name}': self.valid_metric_value
        }


        if self.alpha_ce > 0.0:
            log_dict['train/loss_ce_epoch'] = self.train_loss_ce_epoch / self.n_train_batches_seen
            log_dict['valid/loss_ce_epoch'] = self.valid_loss_ce_epoch / self.n_valid_batches_seen

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

    
