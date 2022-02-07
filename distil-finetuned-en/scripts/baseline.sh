#!/bin/bash

: '
python train.py --glue_dataset sst2 \
                --tokenizer_name bert-base-uncased \
                --tokenizer_params scripts/tokenizer.yaml \
                --batch_size 64 \
                --gradient_accumulation_steps 4 \
                --lr 5e-05 \
                --log_after_n_steps 25 \
                --warmup_prop 0.001 \
                --gpu_id 0 \
                --seed 42 \
                --dumps_dir distilbert_tiny_uncased_ce_kl \
                --wandb_config scripts/wandb_config.yaml \
                --run_id distilbert_tiny_uncased_ce_kl \
                --alpha_task 0.5 \
                distil_teacher --n_epochs 4 \
                --teacher_name bert-base-uncased \
                --teacher_weights bert_base_uncased_pretrained_sst2/best_model.pth \
                --student_name distilbert-tiny-uncased \
                --alpha_kl 2.0 --temperature 2 --alpha_contrastive 0.1 \
                --project_to teacher --projection_strategy average_by_layers \
                --n_negative_samples -1 --negative_sampling_strategy student
'
python train.py --glue_dataset sst2 \
                --tokenizer_name bert-base-uncased \
                --tokenizer_params scripts/tokenizer.yaml \
                --gradient_accumulation_steps 4 \
                --batch_size 32 \
                --lr 5e-05 \
                --log_after_n_steps 5 \
                --gpu_id 0 \
                --seed 42 \
                --dumps_dir distilbert_tiny_uncased_sst2_ce_kl_init_teacher-1 \
                --wandb_config scripts/wandb_config.yaml \
                --run_id distilbert_tiny_uncased_sst2_ce_kl_init_teacher-1 \
                --alpha_task 0.5 \
                distil_teacher --warmup_prop 0.01 \
                --n_epochs 5 --teacher_name bert-base-uncased \
                --teacher_weights bert_base_uncased_pretrained_sst2/best_model.pth \
                --student_weights student_init_weights.pth \
                --student_name distilbert-tiny-uncased \
                --alpha_kl 2.0 --temperature 5


