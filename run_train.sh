#!/bin/bash

pkill -f 'python -u train.py'
export NODE_RANK=0
export N_NODES=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPU_NODE=8
export WORLD_SIZE=8


python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    train.py \
    --force \
    --dump_path distilrubert-tiny-dumps \
    --binarized_data_folder processed_binarized \
    --student_name distilrubert_tiny_cased_convers \
    --student_pretrained_weights rubert_tiny_weights.pth \
    --teacher_name ru_convers \
    --temperature 2 \
    --alpha_ce 2.0 --alpha_mlm 0.5 --alpha_mse 0.0 --alpha_cos 0.0 --mlm \
    --n_epoch 64 --batch_size 2 --gradient_accumulation_steps 64 \
    --learning_rate 1e-5 --gpus $WORLD_SIZE \
    --teacher_mapping teacher2student.pickle --t2s_vocab_padded t2s_padded.pickle \
    --s2t_vocab_padded s2t_padded.pickle \
    --seed 42 --log_interval 1000 \
    --teacher_token_counts teacher_counts.pickle \
    --student_token_counts student_counts.pickle \
    --group_by_size

