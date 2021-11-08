import argparse
import json
import os
import pickle
import shutil

import numpy as np
import torch
import glob

from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer
)
from utils import init_gpu_params, logger, set_seed
import torch

from distiller import Distiller
from lm_seqs_dataset import LmSeqsDataset

MODEL_CLASSES = {
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer)
}

def get_special_tokens_map(tokenizer):
    special_tok_ids = {}
    for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
        idx = tokenizer.all_special_tokens.index(tok_symbol)
        special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
    return special_tok_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite dump_path if it already exists.")
    parser.add_argument(
         "--dump_path", type=str, required=True, help="The output directory (log, checkpoints, parameters, etc.)"
    )
    parser.add_argument(
        "--binarized_data_folder"
    )
    parser.add_argument("--student_name", type=str, required=True, help="Path to the student configuration.")
    parser.add_argument(
        "--student_pretrained_weights", default=None, type=str, help="Load student initialization checkpoint."
    )
    parser.add_argument("--teacher_name", type=str, required=True, help="The teacher model.")
    parser.add_argument("--temperature", default=2.0, type=float, help="Temperature for the softmax temperature.")
    parser.add_argument(
        "--alpha_ce", default=0.5, type=float, help="Linear weight for the distillation loss. Must be >=0."
    )
    parser.add_argument(
        "--alpha_mlm", 
        default=0.0, 
        type=float, 
        help="Linear weight for the MLM loss. Must be >=0. Should be used in coonjunction with `mlm` flag.",
    )
    parser.add_argument(
        "--mlm", action="store_true", help="The LM step: MLM or CLM. If `mlm` is True, the MLM is used over CLM."
    )
    parser.add_argument(
        "--mlm_mask_prop", 
        default=0.15, 
        type=float,
        help="Proportion of tokens for which we need to make a prediction.",
    )
    parser.add_argument("--student_token_counts")
    parser.add_argument("--word_mask", default=0.8, type=float, help="Proportion of tokens to mask out.")
    parser.add_argument("--word_keep", default=0.1, type=float, help="Proportion of tokens to keep.")
    parser.add_argument("--word_rand", default=0.1, type=float, help="Proportion of tokens to randomly replace.")
    parser.add_argument(
        "--mlm_smoothing",
        default=0.7,
        type=float,
        help="Smoothing parameter to emphasize more rare tokens (see XLM, similar to word2vec).",
    )
    parser.add_argument(
        "--restrict_ce_to_mask", 
        action="store_true", 
        help="If true, compute the distilation loss only the [MLM] prediction distribution.",
        )
    parser.add_argument("--n_epoch", type=int, default=3, help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size (for each process).")
    parser.add_argument(
        "--group_by_size", 
        action="store_true", 
        help="If true, group sequences that have similar length into the same batch. Default is false.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=500, 
        help="Gradient accumulation for larger training batches.",
    )
    parser.add_argument("--warmup_prop", default=0.05, type=float, help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float, help="Random initialization range.")

    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56, help="Random seed")

    parser.add_argument("--log_interval", type=int, default=500, help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=4000, help="Checkpoint interval.")
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    
    parser.add_argument('--teacher_mapping')
    parser.add_argument('--t2s_vocab_padded')
    parser.add_argument('--s2t_vocab_padded')

    args = parser.parse_args()
    init_gpu_params(args)
    set_seed(args)

    if args.is_master:
        if os.path.exists(args.dump_path):
            if not args.force:
                raise ValueError(
                        f"Serialization dir {args.dump_path} already exists, but you have not precised wheter to overwrite it"
                        "Use `--force` if you want to overwrite it"
                    )
            else:
                shutil.rmtree(args.dump_path)

        if not os.path.exists(args.dump_path):
            os.makedirs(args.dump_path)
        logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

        # SAVE PARAMS #
        logger.info(f"Param: {args}")
        with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    student_config_class, student_model_class, student_tokenizer_class = MODEL_CLASSES["distilbert"]
    teacher_config_class, teacher_model_class, teacher_tokenizer_class = MODEL_CLASSES["bert"]

    # TOKENIZER #
    teacher_tokenizer = teacher_tokenizer_class.from_pretrained(args.teacher_name, do_lower_case=False)
    student_tokenizer = student_tokenizer_class.from_pretrained(args.student_name, do_lower_case=False)
    
    teacher_special_tok_ids = get_special_tokens_map(teacher_tokenizer)
    student_special_tok_ids = get_special_tokens_map(student_tokenizer)
    
    logger.info(f"Teacher special tokens {teacher_special_tok_ids}")
    logger.info(f"Student special tokens {student_special_tok_ids}")
    args.teacher_tok_ids = teacher_special_tok_ids
    args.student_tok_ids = student_special_tok_ids
    args.max_model_input_size = teacher_tokenizer.max_model_input_sizes.get(args.teacher_name, 512)

    # DATA LOADER #
    #logger.info(f"Loading data from {args.data_file}")
    logger.info(f"Loading data from {args.binarized_data_folder}")
    all_shards = list(sorted(glob.glob(args.binarized_data_folder + '/*')))
    shards_per_worker = len(all_shards) // int(os.environ['WORLD_SIZE'])
    shards_slct = all_shards[args.local_rank * shards_per_worker:(args.local_rank + 1) * shards_per_worker]
    train_data = []
    for shard_file in shards_slct:
        with open(shard_file, 'rb') as fp:
            train_data.extend(pickle.load(fp))
    
    def get_token_probs(tok_counts_file, special_tok_ids):
        logger.info(f"Loading token counts from {tok_counts_file} (already pre-computed)")
        with open(tok_counts_file, "rb") as fp:
            counts = pickle.load(fp)
        token_probs = np.maximum(counts, 1) ** -args.mlm_smoothing
        for idx in special_tok_ids.values():
            token_probs[idx] = 0.0  # do not predict special tokens
        token_probs = torch.from_numpy(token_probs)
        return token_probs
    
    if args.mlm:
        student_token_probs = get_token_probs(args.student_token_counts, student_special_tok_ids)
    else:
        teacher_token_probs = None
        student_token_probs = None
        
    n = len(train_data)
    m = int(0.1 * n)
    valid_data = train_data[n - m:]
    train_data = train_data[:n - m]
    train_lm_seq_dataset = LmSeqsDataset(params=args, all_tokens=train_data)
    valid_lm_seq_dataset = LmSeqsDataset(params=args, all_tokens=valid_data)
    
    logger.info("Data loader created.")

    # STUDENT #
    logger.info(f"Loading student config from {args.student_name}")
    stu_architecture_config = student_config_class.from_pretrained(args.student_name)
    stu_architecture_config.output_hidden_states = True
    stu_architecture_config.output_attentions = True

    if args.student_pretrained_weights is not None:
        logger.info(f"Loading pretrained weights from {args.student_pretrained_weights}")
        student = student_model_class.from_pretrained(args.student_pretrained_weights, 
                                                      config=stu_architecture_config)
    else:
        student = student_model_class(stu_architecture_config)

    if args.gpus > 0:
        student.to(f"cuda:{args.local_rank}")
    logger.info("Student loaded.")

    # TEACHER #
    teacher = teacher_model_class.from_pretrained(args.teacher_name, 
                                                  output_hidden_states=True, 
                                                  output_attentions=True)
    if args.gpus > 0:
        teacher.to(f"cuda:{args.local_rank}")
    logger.info(f"Teacher loaded from {args.teacher_name}")
    
    
    with open(args.teacher_mapping, 'rb') as f:
        args.teacher_mapping = pickle.load(f)
    logger.info("Loaded teacher2student mapping file.")
    
    
    with open(args.t2s_vocab_padded, 'rb') as f:
        args.t2s_vocab_padded = torch.tensor(pickle.load(f)).to(f'cuda:{args.local_rank}')
    logger.info("Loaded padded teacher2student mapping file")
    

    with open(args.s2t_vocab_padded, 'rb') as f:
        args.s2t_vocab_padded = torch.tensor(pickle.load(f)).to(f'cuda:{args.local_rank}')
    
    
    # DISTILLER #
    torch.cuda.empty_cache()
    distiller = Distiller(params=args, train_dataset=train_lm_seq_dataset, 
                          valid_dataset=valid_lm_seq_dataset,
                          student_token_probs=student_token_probs, 
                          student=student, teacher=teacher)
    distiller.train()
    logger.info("Let's go get some drinks.")
    
if __name__ == "__main__":
    main()




