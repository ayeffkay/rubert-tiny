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


def select_shards(data_folder, gpus, local_rank, n_shards=-1):
    all_shards = list(sorted(glob.glob(data_folder + '/*')))
    shards_per_worker = len(all_shards) // gpus if n_shards == -1 else n_shards
    shards_slct = all_shards[local_rank * shards_per_worker:(local_rank + 1) * shards_per_worker]
    return shards_slct


def load_data_from_shards(shards_slct):
    train_data = []
    for shard_file in shards_slct:
        with open(shard_file, 'rb') as fp:
            train_data.extend(pickle.load(fp))
    return train_data


def get_token_probs(tok_counts_file, special_tok_ids, mlm_smoothing):
    logger.info(f"Loading token counts from {tok_counts_file} (already pre-computed)")
    with open(tok_counts_file, "rb") as fp:
        counts = pickle.load(fp)
    token_probs = np.maximum(counts, 1) ** mlm_smoothing
    for idx in special_tok_ids.values():
        token_probs[idx] = 0.0  # do not predict special tokens
    token_probs = torch.from_numpy(token_probs)
    return token_probs


def get_matched_ts_ids(ids_file):
    with open(ids_file, 'rb') as f:
        ids_vocab = pickle.load(f)  # {'token': [teacher_id, student_id]}
        teacher_ids = [ids[0] for ids in ids_vocab.values()]
        student_ids = [ids[1] for ids in ids_vocab.values()]
    return teacher_ids, student_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite dump_path if it already exists.")
    parser.add_argument(
         "--dump_path", type=str, required=True, help="The output directory (log, checkpoints, parameters, etc.)"
    )

    tb_group = parser.add_argument_group('tensorboard_args')
    tb_group.add_argument('--tensorboard_logs_path')
    tb_group.add_argument('--tensorboard_log_name')

    parser.add_argument('--binarized_data_folder')
    parser.add_argument('--student_name', type=str, required=True, help="Path to the student configuration.")
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
        "--mlm_smoothing",
        default=0.7,
        type=float,
        help="Smoothing parameter to emphasize more rare tokens.",
    )

    parser.add_argument(
        "--mlm_mask_prop",
        default=0.15,
        type=float,
        help="Proportion of tokens for which we need to make a prediction.",
    )
    parser.add_argument("--word_mask", default=0.8, type=float, help="Proportion of tokens to mask out.")
    parser.add_argument("--word_keep", default=0.1, type=float, help="Proportion of tokens to keep.")
    parser.add_argument("--word_rand", default=0.1, type=float, help="Proportion of tokens to randomly replace.")

    parser.add_argument("--alpha_mse", default=0.0, type=float, help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument('--projection_strategy', choices=['last', 'skip', 'average', 'average_by_layers', None], default=None,
                        help="""How to use student and teacher hidden representations for MSE loss.
                        last -- use last states of teacher and student (1-1 mapping), 
                        skip -- use intermediate states from teacher and student (1-1 mapping), 
                        average -- average teacher layers output sequentially to fit number of student hidden layers (1-n mapping)
                        average_by_layers -- average student and teacher hidden representations by layers (m-n mapping to 1-1 mapping)
                        """)

    parser.add_argument(
        "--alpha_cos", default=0.0, type=float, help="Linear weight of the cosine embedding loss. Must be >=0."
    )

    contrastive = parser.add_argument_group('contrastive_loss')
    contrastive.add_argument('--alpha_contrastive', default=0.0, type=float)
    contrastive.add_argument('--use_mismatched_ids', action='store_true')
    contrastive.add_argument('--from_one_sample', action='store_true')
    contrastive.add_argument('--n_negative_samples', type=int, default=-1)
    contrastive.add_argument('--teacher_student_prop', nargs='?', type=float, default=0.5)
    contrastive.add_argument('--negative_sampling_strategy', choices=['teacher', 'student', 'teacher_and_student', None], default=None)
    contrastive.add_argument('--add_neg_size_constant', action='store_true')

    parser.add_argument("--teacher_token_counts", nargs='?', type=str, help="The token counts in the data_file for MLM.")
    parser.add_argument("--student_token_counts", nargs='?')

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
    parser.add_argument("--valid_epochs_patience", type=int, default=3)
    parser.add_argument("--reduce_factor", type=float, default=1e-1)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float, help="Random initialization range.")

    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56, help="Random seed")

    parser.add_argument("--log_interval", type=int, default=500, help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=4000, help="Checkpoint interval.")
    parser.add_argument("--valid_prop", type=float, default=0.1)

    parser.add_argument('--t2s_mapping', nargs='?')
    parser.add_argument('--t2s_mapped_ids', nargs='?')
    parser.add_argument("--sum_probs", action="store_true", help="sum probabilities instead of logits")

    kl_reduce_map_group = parser.add_argument_group(title='reduce-map')
    kl_reduce_map_group.add_argument('--t2s_vocab_padded', nargs='?')
    kl_reduce_map_group.add_argument('--s2t_vocab_padded', nargs='?')
    kl_reduce_map_group.add_argument('--s2t_idxs_padded', nargs='?')

    kl_match_group = parser.add_argument_group(title='match')
    kl_match_group.add_argument('--matching_ids', nargs='?')

    parser.add_argument('--align_hiddens', choices=['match', 'reduce', None], default=None)

    args = parser.parse_args()

    # as current implementation can't track mismatches which belong to the specified sample
    assert not(args.use_mismatched_ids and args.from_one_sample)

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

        logger.info(f"Param: {args}")
        with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    student_config_class, student_model_class, student_tokenizer_class = MODEL_CLASSES["distilbert"]
    _, teacher_model_class, teacher_tokenizer_class = MODEL_CLASSES["bert"]

    """
    Tokenizers and special tokens mapping
    """
    teacher_tokenizer = teacher_tokenizer_class.from_pretrained(args.teacher_name, do_lower_case=False)
    student_tokenizer = student_tokenizer_class.from_pretrained(args.student_name, do_lower_case=False)

    teacher_special_tok_ids = get_special_tokens_map(teacher_tokenizer)
    student_special_tok_ids = get_special_tokens_map(student_tokenizer)

    logger.info(f"Teacher special tokens {teacher_special_tok_ids}")
    logger.info(f"Student special tokens {student_special_tok_ids}")


    args.teacher_tok_ids = teacher_special_tok_ids
    args.student_tok_ids = student_special_tok_ids
    args.max_model_input_size = teacher_tokenizer.max_model_input_sizes.get(args.teacher_name, 512)

    """
    Data subset selection for worker
    """
    logger.info(f"Loading data from {args.binarized_data_folder}")
    shards_slct = select_shards(args.binarized_data_folder, args.gpus, args.local_rank)
    train_data = load_data_from_shards(shards_slct)

    """
    Counting probs for MLM
    """
    if args.alpha_mlm:
        assert args.alpha_mlm and (args.teacher_token_counts is not None or args.student_token_counts is not None)
        args.teacher_token_probs = get_token_probs(args.teacher_token_counts,
                                                   teacher_special_tok_ids, args.mlm_smoothing) if args.teacher_token_counts else None
        args.student_token_probs = get_token_probs(args.student_token_counts,
                                                   student_special_tok_ids, args.mlm_smoothing) if args.student_token_counts else None
    else:
        args.teacher_token_probs = None
        args.student_token_probs = None

    """
    Train/validation split
    """
    n = len(train_data)
    m = int(args.valid_prop * n)
    valid_data = train_data[n - m:]
    train_data = train_data[:n - m]
    args.train_size = len(train_data)
    args.valid_size = len(valid_data)

    negative_samples_col = 0 if args.negative_sampling_strategy in 'teacher' else 1
    args.train_cardinality = sum(len(seq) for seq in train_data[negative_samples_col])

    train_lm_seq_dataset = LmSeqsDataset(params=args, all_tokens=train_data)
    valid_lm_seq_dataset = LmSeqsDataset(params=args, all_tokens=valid_data)

    logger.info("Train and validation data sets created.")

    """
    Student initialization
    """
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

    """
    Teacher initialization
    """
    teacher = teacher_model_class.from_pretrained(args.teacher_name,
                                                  output_hidden_states=True,
                                                  output_attentions=True)
    if args.gpus > 0:
        teacher.to(f"cuda:{args.local_rank}")
    logger.info(f"Teacher loaded from {args.teacher_name}")

    """
    Special t2s/s2t mappings
    """
    if args.t2s_mapping is not None:
        with open(args.t2s_mapping, 'rb') as f:
            args.t2s_mapping = pickle.load(f)
        logger.info("Loaded teacher2student mapping file.")

    if args.t2s_mapped_ids is not None:
        with open(args.t2s_mapped_ids, 'rb') as f:
            args.t2s_mapped_ids = torch.tensor(pickle.load(f)).to(f'cuda:{args.local_rank}')
        logger.info("Loaded mapped teacher tokens.")

    if args.t2s_vocab_padded is not None:
        with open(args.t2s_vocab_padded, 'rb') as f:
            args.t2s_vocab_padded = torch.tensor(pickle.load(f)).to(f'cuda:{args.local_rank}')
        logger.info("Loaded padded teacher2student mapping file")

    if args.s2t_vocab_padded is not None:
        with open(args.s2t_vocab_padded, 'rb') as f:
            args.s2t_vocab_padded = torch.tensor(pickle.load(f)).to(f'cuda:{args.local_rank}')
        logger.info("Loaded padded student2teacher mapping file")

    if args.s2t_idxs_padded is not None:
        with open(args.s2t_idxs_padded, 'rb') as f:
            args.s2t_idxs_padded = torch.tensor(pickle.load(f)).to(f'cuda:{args.local_rank}')
        logger.info("Loaded padded student2teacher idxs mapping file")

    """
    Matching tokens loading
    """
    if args.matching_ids is not None:
        teacher_matched, student_matched = get_matched_ts_ids(args.matching_ids)
        args.teacher_matched = torch.tensor(teacher_matched).to(f'cuda:{args.local_rank}')
        args.student_matched = torch.tensor(student_matched).to(f'cuda:{args.local_rank}')
        logger.info("Loaded teacher and student matched ids.")

    """
    Initializing distillation wrapper
    """
    torch.cuda.empty_cache()
    distiller = Distiller(params=args,
                          train_dataset=train_lm_seq_dataset,
                          valid_dataset=valid_lm_seq_dataset,
                          student=student, teacher=teacher)
    distiller.train()


if __name__ == "__main__":
    main()
