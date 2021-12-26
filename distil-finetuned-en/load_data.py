from collections import defaultdict
from functools import partial

import datasets
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

def encode(examples, tokenizer, padding='longest', truncation=True):
    field_sets = [('question', 'sentence'), 
                  ('sentence', ), 
                  ('sentence1', 'sentence2'), 
                  ('question1', 'question2'), 
                  ('premise', 'hypothesis')
                ]
    tok_input = dict(text=None, text_pair=None, 
                    padding=padding, truncation=truncation)
    for field_set in field_sets:
        if all(field_name in examples for field_name in field_set if field_name in set(list(sum(field_sets, ())))):
            tok_input['text'] = examples[field_set[0]]
            if len(field_set) == 2:
                tok_input['text_pair'] = examples[field_set[1]]
            break
    return tokenizer(**tok_input)

def load_glue_dataset(dataset_name, tokenizer_name, padding='max_length', truncation=True, subset_type='train', split_prop=0, seed=42):
    dataset = datasets.load_dataset('glue', dataset_name, split=subset_type)
    dataset = dataset.rename_column('label', 'labels')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    partial_encode = partial(encode, tokenizer=tokenizer, padding=padding, truncation=truncation)
    dataset = dataset.map(partial_encode)
    columns = ['input_ids', 'attention_mask', 'labels']
    if 'token_type_ids' in dataset.features:
        columns.append('token_type_ids')
    dataset.set_format(type='torch', columns=columns)
    if split_prop > 0:
        train_test = dataset.train_test_split(test_size=split_prop, seed=seed)
        return train_test['train'], train_test['test']
    return dataset


def collate_fn(batch, pad_value=0):
    batch_dict = defaultdict(list)
    max_len = max(len(row['input_ids']) for row in batch)
    for row in batch:
        for k, v in row.items():
            if k != 'labels':
                d = max_len - v.shape[0]
                if d > 0:
                    pad_tensor = pad_value * torch.ones(d, dtype=torch.long, device=v.device)
                    v = torch.cat((v, pad_tensor))
            batch_dict[k].append(v)
    batch_dict = {k: torch.stack(v, dim=0) for k, v in batch_dict.items()}
    return batch_dict