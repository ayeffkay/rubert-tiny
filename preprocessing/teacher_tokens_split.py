import re

from collections import OrderedDict, defaultdict
from transformers import BertConfig, DistilBertTokenizer, BertTokenizer
import pickle
import progressbar
import numpy as np

student_tokenizer = DistilBertTokenizer.from_pretrained('distilrubert_tiny_cased_convers')
teacher_tokenizer = BertTokenizer.from_pretrained('ru_convers')
teacher_config = BertConfig.from_pretrained('ru_convers')

student_vocab = OrderedDict(sorted(list(student_tokenizer.vocab.items()), reverse=True))

teacher2student = defaultdict(list)
bar = progressbar.ProgressBar(max_value=len(teacher_tokenizer.vocab))
for k, (teacher_wordpiece, teacher_id) in enumerate(teacher_tokenizer.vocab.items()):
    student_wordpieces = []
    
    if re.search(r'unused|<S>|<T>|</S>', teacher_wordpiece):
        bar.update(k)
        continue
    if teacher_wordpiece in student_tokenizer.special_tokens_map.values():
        student_wordpieces.append(teacher_wordpiece)
    elif re.match(r'##.?', teacher_wordpiece) and len(teacher_wordpiece) <= 3 and teacher_wordpiece not in student_vocab:
        student_wordpieces.append('[UNK]')
    else:
        i = 0
        n = len(teacher_wordpiece)
        j = n
        while i < n:
            while (j > 0 and i < j and 
                   ((i > 0 and f'##{teacher_wordpiece[i:j]}' not in student_vocab) or
                   (i == 0 and teacher_wordpiece[i:j] not in student_vocab)
                   )
                  ):
                if re.match(r'##.?', teacher_wordpiece[i:j]) and len(teacher_wordpiece[i:j]) <= 3:
                    break
                j -= 1
            if i == 0 and re.match(r'##.?', teacher_wordpiece[i:j]) and len(teacher_wordpiece[i:j]) <= 3 and teacher_wordpiece not in student_vocab:
                student_wordpieces = ['[UNK]']
                break
            if i < j:
                current_wordpiece = f'##{teacher_wordpiece[i:j]}' if i > 0 else teacher_wordpiece[i:j]
            else:
                current_wordpiece = '[UNK]'
                
            student_wordpieces.append(current_wordpiece)
            if i == j:
                break
            i = j
            j = n
        
    bar.update(k)
    if len(student_wordpieces) == 0:
        student_wordpieces = ['[UNK]']
    
    if '[UNK]' in student_wordpieces:
        student_wordpices = ['UNK']
    teacher2student[teacher_id] = student_tokenizer.convert_tokens_to_ids(student_wordpieces)

with open('teacher2student.pickle', 'wb') as handle:
    pickle.dump(teacher2student, handle)

max_mapping_len = max(len(v) for v in teacher2student.values())
pad_id = student_tokenizer.vocab_size
#fill_id = student_tokenizer.convert_tokens_to_ids(['[UNK]'])[0]
fill_id = student_tokenizer.vocab_size
t2s_padded = (np.ones((teacher_config.vocab_size, max_mapping_len), dtype=int) * fill_id).tolist()
mapped = []
for i in range(len(t2s_padded)):
    if i in teacher2student:
        t2s = teacher2student[i]
        t2s_padded[i] = t2s + (max_mapping_len - len(t2s)) * [pad_id]
        mapped.append(i)
with open('t2s_padded.pickle', 'wb') as f1, open('t2s_mapped_ids.pickle', 'wb') as f2:
    pickle.dump(t2s_padded, f1)
    pickle.dump(mapped, f2)
    

