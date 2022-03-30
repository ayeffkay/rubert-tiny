#!/bin/bash

pkill -f 'python -u train.py'
export NODE_RANK=0
export N_NODES=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
export N_GPU_NODE=7
export WORLD_SIZE=7


python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    train.py \
    --force \
    --dump_path ../rubert_tiny/distilrubert-tiny-reduce-contrastive-v1-kl-mlm-negt2s64-avg-all-dump \
    --tensorboard_logs_path ../rubert_tiny/tensorboard_logs_rubert_tiny_updated \
    --tensorboard_log_name distilrubert-tiny-reduce-contrastive-v1-kl-mlm-negt2s64-avg-all-dump \
    --binarized_data_folder ../rubert_tiny/processed_binarized \
    --student_name ../rubert_tiny/distilrubert_tiny_cased_convers \
    --student_pretrained_weights ../rubert_tiny/distilrubert_tiny_weights.pth \
    --teacher_name ../rubert_tiny/ru_convers \
    --temperature 2 \
    --alpha_ce 2.0 --alpha_mlm 0.5 --alpha_contrastive 0.1 --projection_strategy average_by_layers \
    --align_logits match \
    --align_hiddens reduce \
    --train_projections \
    --n_negative_samples 64 \
    --negative_sampling_strategy student \
    --student_token_counts ../rubert_tiny/student_counts.pickle \
    --n_epoch 64 --batch_size 4 --group_by_size \
    --gradient_accumulation_steps 128 \
    --learning_rate 5e-4 --valid_epochs_patience 3 --reduce_factor 5e-1 \
    --gpus $WORLD_SIZE \
    --seed 42 --log_interval 4 \
    --t2s_mapping ../rubert_tiny/teacher2student.pickle \
    --matching_ids ../rubert_tiny/matched_tokens.pickle