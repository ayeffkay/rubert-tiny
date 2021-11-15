import pickle
import sys
from collections import defaultdict
from transformers import DistilBertTokenizer, BertModel
import numpy as np

# dict without padding
with open(sys.argv[1], 'rb') as f:
    t2s = pickle.load(f)
    
s2t_map = defaultdict(list)
student_tokenizer = DistilBertTokenizer.from_pretrained(sys.argv[2])
teacher_vocab_size = BertModel.from_pretrained(sys.argv[3]).config.vocab_size

for t_idx, st_idxs in t2s.items():
    for st in st_idxs:
        s2t_map[st].append(t_idx)
with open('student2teacher.pickle', 'wb') as f:
    pickle.dump(s2t_map, f)

s2t_map_cut = defaultdict(list)
with open('teacher_counts.pickle', 'rb') as f:
    teacher_counts = pickle.load(f)
    
teacher_counts = np.array(teacher_counts)
teacher_probs = teacher_counts / np.sum(teacher_counts)

for st_id, t_idxs in s2t_map.items():
    if len(t_idxs) <= 30:
        s2t_map_cut[st_id] = t_idxs
    
    else:
        probs = teacher_probs[t_idxs]
        normalized = np.nan_to_num(probs / np.sum(probs), copy=True, nan=1e-20, posinf=None, neginf=None)
        normalized /= np.sum(normalized)
        sample_size = min(np.sum(normalized != 0), 30)
        s2t_map_cut[st_id] = np.random.choice(t_idxs, size=sample_size, replace=False, p=normalized).tolist()
    

max_seq_len = max(len(v) for v in s2t_map_cut.values())
fill_token = teacher_vocab_size
s2t_padded = fill_token * np.ones((student_tokenizer.vocab_size + 1, max_seq_len), dtype=int)
for i in range(s2t_padded.shape[0]):
    if i in s2t_map_cut:
        s2t_padded[i, :len(s2t_map_cut[i])] = s2t_map_cut[i]
        s2t_padded[i, len(s2t_map_cut[i]):] = teacher_vocab_size * np.ones(max_seq_len - len(s2t_map_cut[i]))

s2t_padded = s2t_padded.tolist()
with open('s2t_padded.pickle', 'wb') as f:
    pickle.dump(s2t_padded, f)
