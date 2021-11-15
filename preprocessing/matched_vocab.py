import sys
from transformers import BertTokenizer, DistilBertTokenizer
import pickle

teacher_tokenizer = BertTokenizer.from_pretrained(sys.argv[1])
student_tokenizer = DistilBertTokenizer.from_pretrained(sys.argv[2])

teacher_voc = teacher_tokenizer.vocab
student_voc = student_tokenizer.vocab
matches = dict()

for token, idx in teacher_voc.items():
    if token in student_voc:
        matches[token] = [idx, student_voc[token]]

print('Matched tokens {}'.format(len(matches)))
with open('matched_tokens.pickle', 'wb') as f:
    pickle.dump(matches, f)
