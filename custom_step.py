import torch
from my_index import MyIndex

def reduce_seq(x, idxs_padded, s_pad_token):
    bs, _, stu_voc_size = x.shape
    fake_seq = s_pad_token * torch.ones(bs, 1, stu_voc_size).to(x.device)
    padded_seq = torch.cat((x, fake_seq), dim=1)
    batch_idx = torch.tensor([[[i]] for i in range(bs)], dtype=torch.long).to(x.device)
    reduced_seq = torch.sum(padded_seq[batch_idx, idxs_padded], dim=2)
    return reduced_seq
        
def map_seq(x, t2s_vocab_padded, s2t_vocab_padded=None):
    bs, seq_len, stu_voc_size = x.shape
    reshaped = x.reshape(-1, stu_voc_size)
    reshaped = torch.cat([reshaped, torch.zeros((bs * seq_len, 1), device=x.device)], dim=-1)
    # run without backward optimization
    if s2t_vocab_padded is None:
        mapped_seq = torch.sum(reshaped[:,t2s_vocab_padded], dim=-1)
    # apply backward optimization
    else:
        myindex= MyIndex.apply
        mapped_seq = torch.sum(myindex(reshaped, t2s_vocab_padded, 
                                                s2t_vocab_padded), dim=-1)
    mapped_seq = mapped_seq.reshape(bs, -1, len(t2s_vocab_padded))
    return mapped_seq
                                    

def map_step(student_repr, idxs_padded, s_pad_token, t2s_vocab_padded, s2t_vocab_padded=None):
    reduced_repr = reduce_seq(student_repr, idxs_padded, s_pad_token)
    mapped_repr = map_seq(reduced_repr, t2s_vocab_padded, s2t_vocab_padded)
    return mapped_repr


def match_step(x, mask, true_label, matched_voc_ids=None):
    mask = (mask==true_label).unsqueeze(-1).expand_as(x)
    last_dim = x.size(2)
    seq_subset = torch.masked_select(x, mask).reshape(-1, last_dim)
    if matched_voc_ids is not None:
        seq_subset = seq_subset[:, matched_voc_ids]
    return seq_subset
