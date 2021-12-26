#!/bin/bash

pkill -f train.py

python train.py --glue_dataset cola \
                --padding false \
                --truncation longest_first \
                --tokenizer_name bert-base-uncased \
                --batch_size 128 \
                --n_epochs 256 \
                --lr 1e-03 \
                --scheduler ReduceLROnPlateau \
                --scheduler_params scripts/scheduler.yaml \
                --gpu_id 9 \
                --alpha_ce 0.5 \
                --seed 42 \
                --dumps_dir cola_hyperbolic_baseline \
                --valid_prop 0.1 \
                --wandb_config scripts/wandb_config.yaml \
                --run_id cola_hyperbolic_baseline_distil \
                distil_teacher --teacher_name bert-base-uncased \
                --teacher_weights /home/ayeffkay/.deeppavlov/models/classifiers/glue_cola_torch_uncased_bert/model.pth.tar \
                --student_name distilbert-tiny-uncased \
                --project_to student \
                --projection_strategy average_by_layers \
                --alpha_kl 1.0 --temperature 2 \
                --alpha_contrastive 0.1 \
                --n_negative_samples 32 \
                --negative_sampling_strategy student \
                hyperbolic --use_hyperbolic_projections