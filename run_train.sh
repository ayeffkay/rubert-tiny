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
    --dump_path distilrubert-tiny-dump \
    --binarized_data_folder processed_binarized \
    --student_name distilrubert_tiny_cased_convers \
    --student_pretrained_weights distilrubert_tiny_weights.pth \
    --teacher_name ru_convers \
    --temperature 2 \
    --alpha_ce 2.0 --alpha_mlm 0.5 --alpha_mse 0.0 --projection_strategy average_by_layers --alpha_cos 0.0 --alpha_contrastive 0.0 \
    --teacher_token_counts teacher_counts.pickle \
    --student_token_counts student_counts.pickle \
    --n_epoch 64 --batch_size 2 --group_by_size \
    --gradient_accumulation_steps 128 \
    --learning_rate 1e-5 --gpus $WORLD_SIZE \
    --seed 42 --log_interval 1000 \
    --t2s_mapping teacher2student.pickle \
    tokens_mapping --t2s_vocab_padded t2s_padded.pickle 
    
# to apply backward optimization with my index, add 
#--s2t_vocab_padded s2t_padded.pickle
