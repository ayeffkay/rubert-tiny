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
    --dump_path distilrubert-tiny-baseline-contrastive-dump \
    --tensorboard_logs_path tensorboard_logs \
    --tensorboard_log_name distilrubert_tiny_baseline_contrastive \
    --binarized_data_folder processed_binarized \
    --student_name distilrubert_tiny_cased_convers \
    --student_pretrained_weights distilrubert_tiny_weights.pth \
    --teacher_name ru_convers \
    --temperature 5 \
    --alpha_ce 2.0 --alpha_mlm 0.5 --alpha_contrastive 0.1 --projection_strategy average_by_layers \
    --n_negative_samples -1 --negative_sampling_strategy student \
    --student_token_counts student_counts.pickle \
    --n_epoch 64 --batch_size 3 --group_by_size \
    --gradient_accumulation_steps 256 \
    --learning_rate 5e-4 --gpus $WORLD_SIZE \
    --seed 42 --log_interval 500 \
    --t2s_mapping teacher2student.pickle \
    matched_tokens --matching_ids matched_tokens.pickle