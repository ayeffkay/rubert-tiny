import torch
from my_index import MyIndex

def reduce_seq(x, idxs_padded, s_pad_token):
    bs, _, stu_voc_size = x.shape
    fake_seq = s_pad_token * torch.ones(bs, 1, stu_voc_size).to(x.device)
    padded_seq = torch.cat((x, fake_seq), dim=1)
    batch_idx = torch.tensor([[[i]] for i in range(bs)], dtype=torch.long).to(x.device)
    reduced_seq = torch.sum(padded_seq[batch_idx, idxs_padded], dim=2)
    return reduced_seq
        
def map_seq(x, t2s_vocab_padded, s2t_vocab_padded=None, sum_probs=False):
    bs, seq_len, stu_voc_size = x.shape
    reshaped = x.reshape(-1, stu_voc_size)
    reshaped = torch.cat([reshaped, torch.zeros((bs * seq_len, 1), device=x.device)], dim=-1)
    # run without backward optimization
    if s2t_vocab_padded is None:
        if sum_probs:
            mapped_seq = torch.logsumexp(reshaped[:,t2s_vocab_padded], dim=-1)
        else:
            mapped_seq = torch.sum(reshaped[:,t2s_vocab_padded], dim=-1)
    # apply backward optimization
    else:
        myindex= MyIndex.apply
        if sum_probs:
            # todo: use torch.logsumexp, but how to optimize backward?
            # myindex works only with sum as prev operation:
            # logsumexp(myindex(reshaped)) ->
            # -> log(sum(myindex(exp(reshaped))))
            # z for numeric stability
            z, _ = torch.max(reshaped, dim=-1, keepdim=True)
            reshaped = reshaped - z
            mapped_seq = torch.log(torch.sum(myindex(torch.exp(reshaped), t2s_vocab_padded, s2t_vocab_padded), dim=-1))
            mapped_seq += z  # might skip this as result goes to softmax
        else:
            mapped_seq = torch.sum(myindex(reshaped, t2s_vocab_padded, s2t_vocab_padded), dim=-1)
    mapped_seq = mapped_seq.reshape(bs, -1, len(t2s_vocab_padded))
    return mapped_seq
                                    

def map_step(student_repr, idxs_padded, s_pad_token, t2s_vocab_padded, s2t_vocab_padded=None, sum_probs=False):
    reduced_repr = reduce_seq(student_repr, idxs_padded, s_pad_token)
    mapped_repr = map_seq(reduced_repr, t2s_vocab_padded, s2t_vocab_padded, sum_probs=sum_probs)
    return mapped_repr


def match_step(x, mask, true_label, matched_voc_ids=None):
    mask = (mask==true_label).unsqueeze(-1).expand_as(x)
    last_dim = x.size(2)
    seq_subset = torch.masked_select(x, mask).reshape(-1, last_dim)
    if matched_voc_ids is not None:
        seq_subset = seq_subset[:, matched_voc_ids]
    return seq_subset

def masked_select_reshape_2d(x, mask, reshape_last_dim):
    y = torch.masked_select(x, mask.unsqueeze(-1).expand_as(x)).reshape(-1, reshape_last_dim)
    return y

def average_by_layers(x, split_ids=None, pad_token=0, attn_mask=None):
    x_avg = torch.stack(x).mean(dim=0)
    x_avg = average_one(x_avg, split_ids, pad_token, attn_mask)
    return x_avg

def average_one(x, split_ids=None, pad_token=0, attn_mask=None):
    last_dim = x.size(-1)
    if split_ids is not None:
        x = reduce_seq(x, split_ids, pad_token)
    if attn_mask is not None:
        x = masked_select_reshape_2d(x, attn_mask, last_dim)
    return x


def cosine_similarity(teacher, student):
    teacher_norm = torch.norm(teacher, p=2, dim=1)
    student_norm = torch.norm(student, p=2, dim=1)
    prod = torch.bmm(teacher.unsqueeze(1), student.unsqueeze(2)).view(-1)
    sim = prod / (teacher_norm * student_norm)
    return sim
    
