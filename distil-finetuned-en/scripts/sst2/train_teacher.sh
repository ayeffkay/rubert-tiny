#!/bin/bash

python train.py --glue_dataset sst2 \
                --tokenizer_name bert-base-uncased \
                --tokenizer_params scripts/sst2/tokenizer.yaml \
                --batch_size 128 \
                --gradient_accumulation_steps 4 \
                --log_after_n_steps 20 \
                --lr 5e-05 \
                --gpu_id $1 \
                --alpha_task $2 \
                --seed 42 \
                --dumps_dir "sst2/$3" \
                --wandb_config scripts/sst2/wandb_config.yaml \
                --run_id $3 \
                train_single --model_name $4 \
                --lr_drop_patience 2 \
                --lr_drop_div 1.5 \
                --valid_patience 20 \
                --min_lr 1e-7
