import pickle
import sys
from collections import defaultdict
from transformers import DistilBertTokenizer
import numpy as np

# dict without padding
with open(sys.argv[1], 'rb') as f:
    t2s = pickle.load(f)
    
s2t_map = defaultdict(list)
# path to trained tokenizer
student_tokenizer = DistilBertTokenizer.from_pretrained(sys.argv[2])
for t_idx, st_idxs in t2s.items():
    for st in st_idxs:
        #if t_idx not in s2t_map[st]:
        s2t_map[st].append(t_idx)
with open('student2teacher.pickle', 'wb') as f:
    pickle.dump(s2t_map, f)

s2t_map_cut = defaultdict(list)
# file with counts of teacher tokens (by processes corpus)
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
        s2t_map_cut[st_id] = np.random.choice(t_idxs, size=50, replace=True, p=normalized).tolist()
    

max_seq_len = max(len(v) for v in s2t_map_cut.values())
pad_token = student_tokenizer.convert_tokens_to_ids(['[PADS2T]'])[0]
unk_token = student_tokenizer.convert_tokens_to_ids(['[UNK]'])[0]
s2t_padded = unk_token * np.ones((student_tokenizer.vocab_size, max_seq_len), dtype=int)
for i in range(s2t_padded.shape[0]):
    if i in s2t_map_cut:
        s2t_padded[i, :len(s2t_map_cut[i])] = s2t_map_cut[i]
        s2t_padded[i, len(s2t_map_cut[i]):] = pad_token * np.ones(max_seq_len - len(s2t_map_cut[i]))

s2t_padded = s2t_padded.tolist()
with open('s2t_padded.pickle', 'wb') as f:
    pickle.dump(s2t_padded, f)
