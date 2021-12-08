import numpy as np
import pickle
import glob
import multiprocessing as mp
from collections import defaultdict
import logging
from transformers import BertTokenizer
import progressbar
from argparse import ArgumentParser
import json
from pathlib import Path
import os
import itertools

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def divide_chunks(l, n, cls_id=None, sep_id=None):
    chunks = []
    for i in range(0, len(l), n):
        cur = l[i : i + n]
        if cls_id and cur[0] != cls_id:
            cur = np.insert(cur, 0, cls_id)
        if sep_id and cur[-1] != sep_id:
            cur = np.insert(cur, len(cur), sep_id)
        chunks.append(cur)
    return chunks

def count_unk_seqs(seqs, unk_id, threshold=0.5):
    return any(np.count_nonzero(seq == unk_id) > threshold * len(seq) for seq in seqs)

def count_unk_seq(seq, unk_id, threshold=0.5):
    return np.count_nonzero(seq == unk_id) > threshold * len(seq)

def generate_t2s_mapping(seq, mapping_dict):
    return list(itertools.chain.from_iterable([mapping_dict[tok_id] for tok_id in seq]))

def check_sequences(file, teacher_special_tokens, student_special_tokens, min_len, max_len, output_folder):
    ct = defaultdict(int)
    res = []
    logger.info(f'Processing shard {file} is started...')
    with open(file, 'rb+') as f:
        shard = pickle.load(f)
        bar = progressbar.ProgressBar(max_value=len(shard))
        for i, seq in enumerate(shard):
            if all(1 if s in teacher_special_tokens.values() else 0 for s in set(seq[0])) or all(1 if s in student_special_tokens.values() else 0 for s in set(seq[1])):
                ct['special_removed'] += 1
                continue
            if any(len(s) < min_len for s in seq):
                ct['empty_removed'] += 1
            elif count_unk_seq(seq[0], teacher_special_tokens['cls_token']) or count_unk_seq(seq[1], student_special_tokens['cls_token']):
                ct['unk_removed'] += 1
            elif any(len(s) > max_len for s in seq):
                divided_t = divide_chunks(seq[0], max_len - 2, 
                              teacher_special_tokens['cls_token'], 
                              teacher_special_tokens['sep_token'])
                divided_s = divide_chunks(seq[1], max_len - 2, 
                                          student_special_tokens['cls_token'], 
                                          student_special_tokens['sep_token']
                                         )
                
                ct['splitted'] += 1
                for d_t, d_s in zip(divided_t, divided_s):
                    if (count_unk_seq(d_t, teacher_special_tokens['unk_token'])) or (count_unk_seq(d_s, student_special_tokens['unk_token'])):
                        ct['unk_removed'] += 1
                    else:
                        ct['saved'] += 1
                        res.append((d_t, d_s))
            else:
                ct['saved'] += 1
                res.append(seq)
                
            if i % 10**5:
                bar.update(i)
    logger.info(f'Process done. Shard stats: {json.dumps(ct)}')
    output_file = os.path.join(output_folder, os.path.split(file)[-1])
    with open(output_file, 'wb') as g:
        pickle.dump(res, g)
        
def get_special_tokens(tokenizer_name):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    special_tok_ids = {}
    for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
        idx = tokenizer.all_special_tokens.index(tok_symbol)
        special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
    return special_tok_ids
    
    
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--binarized_shards')
    parser.add_argument('--teacher_name')
    parser.add_argument('--student_name')
    parser.add_argument('--min_len', type=int)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--output_folder')
    
    args, _ = parser.parse_known_args()
    args.teacher_special_tokens = get_special_tokens(args.teacher_name)
    args.student_special_tokens = get_special_tokens(args.student_name)
    
    return args

def main():
    args = get_args()
    procs = []
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    for file in glob.glob(f'{args.binarized_shards}/*'):
        p = mp.Process(target=check_sequences, args=(file, 
                                                     args.teacher_special_tokens, 
                                                     args.student_special_tokens, 
                                                     args.min_len, 
                                                     args.max_len, 
                                                     args.output_folder
                                                    ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    
if __name__ == '__main__':
    main()