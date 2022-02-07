#!/bin/bash

pkill -f train.py

python train.py --glue_dataset sst2 \
                --tokenizer_name bert-base-uncased \
                --tokenizer_params scripts/tokenizer.yaml \
                --batch_size 128 \
                --gradient_accumulation_steps 4 \
                --log_after_n_steps 20 \
                --lr 5e-05 \
                --gpu_id 0 \
                --alpha_task 1 \
                --seed 42 \
                --dumps_dir distilbert_tiny_uncased_raw_sst2 \
                --wandb_config scripts/wandb_config.yaml \
                --run_id distilbert_tiny_uncased_raw_sst2 \
                train_single --model_name distilbert-tiny-uncased \
                --lr_drop_patience 2 \
                --lr_drop_div 1.5 \
                --valid_patience 20 \
                --min_lr 1e-7
