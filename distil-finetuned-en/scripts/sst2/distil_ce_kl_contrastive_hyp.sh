#!/bin/bash

python train.py --glue_dataset sst2 \
                --tokenizer_name bert-base-uncased \
                --tokenizer_params scripts/sst2/tokenizer.yaml \
                --batch_size 32 \
                --gradient_accumulation_steps 4 \
                --lr 1e-05 \
                --log_after_n_steps 10 \
                --gpu_id $1 \
                --seed 42 \
                --dumps_dir "sst2/$2" \
                --wandb_config scripts/sst2/wandb_config.yaml \
                --run_id $2 \
                --alpha_task $3 \
                distil_teacher --n_epochs 4 --warmup_prop 0.0001 \
                --teacher_name bert-base-uncased \
                --teacher_weights sst2/teacher_weights.pth \
                --student_name distilbert-tiny-uncased \
                --student_weights sst2/student_init_weights.pth \
                --alpha_kl $4 --temperature 5 --alpha_contrastive $5 \
                --project_to teacher --projection_strategy average_by_layers \
                --n_negative_samples $6 --negative_sampling_strategy student \
                hyperbolic 
