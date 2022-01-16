#!/bin/bash

pkill -f train.py

python train.py --glue_dataset sst2 \
                --tokenizer_name bert-base-uncased \
                --tokenizer_params scripts/tokenizer.yaml \
                --batch_size 128 \
                --lr 4e-05 \
                --val_every_n_batches 250 \
                --valid_patience 10 \
                --lr_drop_patience 3 \
                --lr_drop_div 2.0 \
                --gpu_id 0 \
                --seed 42 \
                --dumps_dir sst2_baseline \
                --valid_prop 0.1 \
                --wandb_config scripts/wandb_config.yaml \
                --run_id sst2_baseline_distil \
                --alpha_ce 0.5 \
                distil_teacher --teacher_name bert-base-uncased \
                --teacher_weights teacher_cola/best_model.pth \
                --student_name distilbert-tiny-uncased \
                --alpha_kl 2.0 --temperature 2


