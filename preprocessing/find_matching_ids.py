from argparse import ArgumentParser
import multiprocessing as mp
import pickle
from pathlib import Path
from transformers import BertTokenizer, DistilBertTokenizer
import os
import logging
import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



def get_args():
    parser = ArgumentParser()
    parser.add_argument('--binarized_data_folder')
    parser.add_argument('--teacher_tokenizer')
    parser.add_argument('--student_tokenizer')
    
    args, _ = parser.parse_known_args()
    args.teacher_tokenizer = BertTokenizer.from_pretrained(args.teacher_tokenizer)
    args.student_tokenizer = DistilBertTokenizer.from_pretrained(args.student_tokenizer)
    return args

def find_matches(data_folder, data_file, teacher_tokenizer, student_tokenizer):
    shard_path = Path(data_folder)/data_file
    with open(shard_path, 'rb') as f:
        shard = pickle.load(f)
    res = []
    
    logger.info(f'Start processing {shard_path}.')
    for seq_t, seq_s in shard:
        t_tokens = teacher_tokenizer.convert_ids_to_tokens(seq_t)
        s_tokens = student_tokenizer.convert_ids_to_tokens(seq_s)
        
        t_ids = []
        s_ids = []
        last_s = 0
        for i, t_token in enumerate(t_tokens):
            try:
                j = s_tokens.index(t_token, last_s)
                t_ids.append(i)
                s_ids.append(j)
                last_s = j + 1
                if last_s >= len(s_tokens):
                    break
            except:
                continue
        t_ids_map_mask = np.zeros(len(seq_t), dtype=np.int)
        s_ids_map_mask = np.zeros(len(seq_s), dtype=np.int)
        t_ids_map_mask[t_ids] = 1
        s_ids_map_mask[s_ids] = 1
        
        res.append([seq_t, seq_s, t_ids_map_mask.tolist(), s_ids_map_mask.tolist()])
        
    with open(shard_path, 'wb') as f:
        pickle.dump(res, f)
    logger.info(f'End processing {shard_path}.')
    return res


if __name__ == '__main__':
    args = get_args()
    ct = 0
    processes = []
    for file in os.listdir(f'{args.binarized_data_folder}'):
        p = mp.Process(target=find_matches, args=(args.binarized_data_folder, file, args.teacher_tokenizer, args.student_tokenizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
