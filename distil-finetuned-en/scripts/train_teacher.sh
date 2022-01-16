#!/bin/bash

pkill -f train.py

python train.py --glue_dataset sst2 \
                --tokenizer_name distilbert-base-uncased \
                --tokenizer_params scripts/tokenizer.yaml \
                --batch_size 128 \
                --lr 1e-05 \
                --gpu_id 0 \
                --alpha_ce 1 \
                --seed 42 \
                --dumps_dir student_cola \
                --wandb_config scripts/wandb_config.yaml \
                --run_id student_sst2 \
                --val_every_n_batches 250 \
                --lr_drop_patience 3 \
                --lr_drop_div 2.0 \
                --valid_patience 10 \
                --min_lr 1e-7 \
                train_student --student_name distilbert-base-uncased --from_pretrained