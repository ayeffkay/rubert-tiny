import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

from utils import logger
import time

from setup_logger import setup_logger

class LmSeqsDataset(Dataset):
    def __init__(self, params, all_tokens):
        self.params = params
        self.teacher_tok = [t[0] for t in all_tokens]
        self.student_tok = [t[1] for t in all_tokens]
        self.t_len = np.array([len(t) for t in self.teacher_tok])
        self.st_len = np.array([len(t) for t in self.student_tok])
                    
    
    def __getitem__(self, i):
        return (self.teacher_tok[i], self.t_len[i], 
                self.student_tok[i], self.st_len[i])

    def __len__(self):
        return len(self.teacher_tok)


    @staticmethod
    def pad2d(batch, pad_idx, max_seq_len=None):
        if not max_seq_len:
            max_seq_len = max(len(seq) for seq in batch)
        padded = [seq.tolist() + (max_seq_len - len(seq)) * [pad_idx] if not isinstance(seq, list) else seq + (max_seq_len - len(seq)) * [pad_idx] for seq in batch]
        return padded
    
    @staticmethod
    def pad3d(batch, pad_idx, pad_by_batch=False):
        max_seq_len = max(len(s) for seq in batch for s in seq)
        padded = [[s.tolist() + (max_seq_len - len(s)) * [pad_idx] if not isinstance(s, list) else s + (max_seq_len - len(s)) * [pad_idx] for s in seq] for seq in batch]
        if pad_by_batch:
            padded = LmSeqsDataset.pad2d(padded, [pad_idx] * max_seq_len)
        return padded
    
    @staticmethod
    def gen_t2s_mapping(teacher_tokens, teacher_mapping):
        t2s = []
        idxs = []
        dx = 0
        for t in teacher_tokens:
            mapping = teacher_mapping[t]
            t2s.extend(mapping)
            idxs.append(list(range(dx, dx + len(mapping))))
            dx += len(mapping)
        return t2s, idxs
    
    @staticmethod
    def gen_batch_t2s_mapping(teacher_batch, teacher_mapping, pad_tok):
        t2s = []
        t2s_lengths = []
        idxs = []
        for seq in teacher_batch:
            tokens, i = LmSeqsDataset.gen_t2s_mapping(seq, teacher_mapping)
            t2s.append(tokens)
            t2s_lengths.append(len(tokens))
            idxs.append(i)
        t2s = LmSeqsDataset.pad2d(t2s, pad_tok)
        pad_id = max(max(j) for idx in idxs for j in idx) + 1
        idxs = LmSeqsDataset.pad3d(idxs, pad_id, pad_by_batch=True)
        
        return t2s, t2s_lengths, idxs
       
    def batch_sequences(self, batch):
        """
        Do the padding and transform into torch.tensor.
        """

        # Pad token ids
        if self.params.mlm:
            pad_teacher_idx = self.params.teacher_tok_ids['pad_token']
            pad_student_idx = self.params.student_tok_ids['pad_token']
        else:
            pad_teacher_idx = self.params.teacher_tok_ids['unk_token']
            pad_student_idx = self.params.student_tok_ids['unk_token']
        
        teacher_ids = [b[0] for b in batch]
        teacher_lengths = torch.tensor([b[1] for b in batch])

        t2s = LmSeqsDataset.gen_batch_t2s_mapping(teacher_ids, self.params.teacher_mapping, pad_student_idx)
        
        student_ids = LmSeqsDataset.pad2d([b[2] for b in batch], pad_student_idx)
        student_ids = torch.tensor(student_ids)
        student_lengths = torch.tensor([b[3] for b in batch])
        
        teacher_ids = LmSeqsDataset.pad2d(teacher_ids, pad_teacher_idx)
        teacher_ids = torch.tensor(teacher_ids)
        
        t2s_ids = torch.tensor(t2s[0])
        
        return (teacher_ids, teacher_lengths, 
                student_ids, student_lengths, 
                t2s_ids, torch.tensor(t2s[1]), torch.tensor(t2s[2]))

