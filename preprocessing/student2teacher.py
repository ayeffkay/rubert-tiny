import sys
import pickle
import numpy as np
from transformers import DistilBertTokenizer, BertModel
from collections import defaultdict


"""
python preprocessing/student2teacher.py distilrubert_tiny_cased_convers ru_convers teacher2student.pickle teacher_counts.pickle 30 42
"""

student_tokenizer_name = sys.argv[1]
teacher_tokenizer_name = sys.argv[2]
t2s_mapping_file = sys.argv[3]
teacher_counts_file = sys.argv[4]
cut_len = int(sys.argv[5])
seed = int(sys.argv[6])

np.random.seed(seed)

# dict t_id: [st_ids, ...], without paddind and teacher counts
with open(t2s_mapping_file, 'rb') as f, open(teacher_counts_file, 'rb') as g:
    t2s_mapping = pickle.load(f)
    teacher_counts = pickle.load(g)

teacher_counts = np.array(teacher_counts)
teacher_probs = teacher_counts / np.sum(teacher_counts)
teacher_idxs = np.arange(len(teacher_probs))

student_vocab_size = DistilBertTokenizer.from_pretrained(sys.argv[1]).vocab_size
teacher_vocab_size = BertModel.from_pretrained(sys.argv[2]).config.vocab_size

s2t_vocab = defaultdict(list)
s2t_idxs = defaultdict(list)

for t_id in t2s_mapping:
    for j, s_id in enumerate(t2s_mapping[t_id]):
        s2t_vocab[s_id].append(t_id)
        s2t_idxs[s_id].append(j)

s2t_vocab_cut = dict.fromkeys(s2t_vocab.keys())
s2t_idxs_cut = dict.fromkeys(s2t_vocab.keys())
for s_id in s2t_vocab.keys():
    if len(s2t_vocab[s_id]) <= cut_len:
        s2t_vocab_cut[s_id] = s2t_vocab[s_id]
        s2t_idxs_cut[s_id] = s2t_idxs[s_id]
    else:
        probs = teacher_probs[s2t_vocab[s_id]]
        normalized = np.nan_to_num(probs / np.sum(probs), copy=True, nan=1e-20, posinf=None, neginf=None)
        normalized /= np.sum(normalized)
        idxs = np.arange(len(probs))

        sample_size = min(np.sum(normalized != 0), cut_len)
        idxs_slct = np.random.choice(idxs, size=sample_size, replace=False, p=normalized).tolist()
        s2t_vocab_cut[s_id] = np.array(s2t_vocab[s_id])[idxs_slct]
        s2t_idxs_cut[s_id] = np.array(s2t_idxs[s_id])[idxs_slct]


max_len = max(len(t_ids) for t_ids in s2t_vocab_cut.values())
assert max_len == cut_len

pad_token = teacher_vocab_size
s2t_vocab_cut_padded = teacher_vocab_size * np.ones((student_vocab_size + 1, max_len), dtype=np.int64)
s2t_idxs_cut_padded = np.zeros_like(s2t_vocab_cut_padded, dtype=np.int64)

for s_id in s2t_vocab_cut.keys():
    s2t_vocab_cut_padded[s_id, :len(s2t_vocab_cut[s_id])] = s2t_vocab_cut[s_id]
    s2t_idxs_cut_padded[s_id, :len(s2t_idxs_cut[s_id])] = s2t_idxs_cut[s_id]


with open('s2t_padded.pickle', 'wb') as f:
    pickle.dump(s2t_vocab_cut_padded.tolist(), f)
with open('s2t_idxs_padded.pickle', 'wb') as f:
    pickle.dump(s2t_idxs_cut_padded.tolist(), f)