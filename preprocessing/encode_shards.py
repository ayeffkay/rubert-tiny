from argparse import ArgumentParser
import multiprocessing as mp
import os
import logging
import pickle
import time
import glob
import numpy as np
import itertools
from transformers import BertTokenizer, DistilBertTokenizer
import progressbar
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data/shards")
    parser.add_argument("--teacher_tokenizer_name")
    parser.add_argument("--student_tokenizer_name")
    parser.add_argument("--mapping_file")
    parser.add_argument("--dump_folder")
    args, _ = parser.parse_known_args()
    Path(args.dump_folder).mkdir(parents=True, exist_ok=True)
    return args

def encode(tokenizer_name, data, distil=False):
    tok_type = BertTokenizer if not distil else DistilBertTokenizer
    tokenizer = tok_type.from_pretrained(tokenizer_name)
    bos = tokenizer.special_tokens_map['cls_token']
    sep = tokenizer.special_tokens_map['sep_token']
        
    logger.info("Start encoding")
    logger.info(f"{len(data)} examples to process.")

    rslt = []
    interval = 100000
    start = time.time()
    bar = progressbar.ProgressBar(max_value=len(data))
    for i, text in enumerate(data):
        text = f"{bos} {text.strip()} {sep}"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        rslt.append(token_ids)

        if (i + 1) % interval == 0:
            end = time.time()
            logger.info(f"{i} examples processed. - {(end-start):.2f}s/{interval}expl")
            start = time.time()
            bar.update(i)
    #rslt_ = convert_res(tokenizer.vocab_size, rslt)
    return rslt

def convert_res(vocab_size, res):
    if vocab_size < (1 << 16):
        rslt_ = [np.uint16(d) for d in res]
    else:
        rslt_ = [np.int32(d) for d in res]
    return rslt_

def load_mapping(mapping_file):
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    return mapping

def teacher2student(mapping_dict, seq):
    res = []
    #lengths = []
    res = itertools.chain.from_iterable([mapping_dict[idx] for idx in seq])
    return list(res)
    
def wrapper(data_file, binarized_folder, teacher_tokenizer_name, student_tokenizer_name, mapping_dict, max_seq_len=512):
    with open(data_file) as f:
        data = f.readlines()
    rslt1 = encode(teacher_tokenizer_name, data)
    rslt2 = encode(student_tokenizer_name, data, distil=True)
    stu_vocab_size = DistilBertTokenizer.from_pretrained(student_tokenizer_name).vocab_size
    
    t_tokens, st_tokens = [], []
    for seq_t, seq_s in zip(rslt1, rslt2):
        ids = teacher2student(mapping_dict, seq_t)
        if len(ids) >= len(seq_t) and len(ids) < max_seq_len:
            t_tokens.append(seq_t)
            st_tokens.append(seq_s)
            
    total = list(zip(t_tokens, st_tokens))
    #total = list(zip(rslt1, rslt2))
    output_file = os.path.join(binarized_folder, os.path.split(data_file)[-1]) + '.pickle'
    logger.info(f"Dump to {output_file}")
    with open(output_file, "wb") as handle:
        pickle.dump(total, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    logger.info("Finished binarization")
    logger.info(f"{len(total)} examples processed.")
         
        
def main():
    args = get_args()
    processes = []
    mapping_dict = load_mapping(args.mapping_file)
    Path(args.dump_folder).mkdir(parents=True, exist_ok=True)
    for file in glob.glob(args.root_dir+'/*'):
        p = mp.Process(target=wrapper, args=(file, args.dump_folder, 
                                             args.teacher_tokenizer_name, 
                                             args.student_tokenizer_name, 
                                             mapping_dict
                                            ))
        p.start()
        processes.append(p)
                
    for p in processes:
        p.join()        

if __name__ == '__main__':
    main()
