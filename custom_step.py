import torch
from my_index import MyIndex, MyIndex_v1
import numpy as np
import torch.nn.functional as F


def reduce_seq(x, idxs_padded, s_pad_token):
    bs, _, stu_voc_size = x.shape
    fake_seq = s_pad_token * torch.ones(bs, 1, stu_voc_size).to(x.device)
    padded_seq = torch.cat((x, fake_seq), dim=1)
    batch_idx = torch.tensor([[[i]] for i in range(bs)], dtype=torch.long).to(x.device)
    reduced_seq = torch.sum(padded_seq[batch_idx, idxs_padded], dim=2)
    return reduced_seq


def map_seq(x, t2s_vocab_padded, s2t_vocab_padded=None, s2t_idxs_padded=None, sum_probs=False):
    bs, seq_len, stu_voc_size = x.shape
    dummy_value = 0.0
    if sum_probs:
        # we need -inf here
        dummy_value = -1e50

    reshaped = x.reshape(-1, stu_voc_size)
    reshaped = torch.cat([reshaped, dummy_value * torch.ones((bs * seq_len, 1), device=x.device)], dim=-1)

    # run without backward optimization
    if s2t_vocab_padded is None:
        if sum_probs:
            mapped_seq = torch.logsumexp(reshaped[:, t2s_vocab_padded], dim=-1)
        else:
            mapped_seq = torch.sum(reshaped[:, t2s_vocab_padded], dim=-1)
    # apply backward optimization
    else:
        if sum_probs:
            myindex = MyIndex_v1.apply
            mapped_seq = torch.logsumexp(myindex(reshaped, t2s_vocab_padded, s2t_vocab_padded, s2t_idxs_padded), dim=-1)
        else:
            myindex = MyIndex.apply
            mapped_seq = torch.sum(myindex(reshaped, t2s_vocab_padded, s2t_vocab_padded), dim=-1)
    mapped_seq = mapped_seq.reshape(bs, -1, len(t2s_vocab_padded))
    return mapped_seq


def map_step(student_repr, idxs_padded, s_pad_token, t2s_vocab_padded, s2t_vocab_padded=None, s2t_idxs_padded=None,
             sum_probs=False):
    reduced_repr = reduce_seq(student_repr, idxs_padded, s_pad_token)
    mapped_repr = map_seq(reduced_repr, t2s_vocab_padded, s2t_vocab_padded, s2t_idxs_padded, sum_probs=sum_probs)
    return mapped_repr


def match_step(x, mask, true_label, matched_voc_ids=None):
    mask = (mask == true_label).unsqueeze(-1).expand_as(x)
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


def cosine_similarity(student, teacher):
    teacher_norm = torch.norm(teacher, p=2, dim=0)
    student_norm = torch.norm(student, p=2, dim=1 if len(student.size()) > 1 else 0)
    prod = torch.matmul(student, teacher)
    sim = prod / (teacher_norm * student_norm)
    return sim


def ce_step(student_logits, teacher_logits, ce_loss_fct, temperature=1):
    b_size = student_logits.size(0)
    loss_ce = (
        ce_loss_fct(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
        ) * temperature ** 2
    ) / b_size
    return loss_ce


def mse_step(hid_projectors_mse_student, 
             hid_projectors_mse_teacher, 
             mse_loss_fct,
             s_hid, t_hid, s_mask, t_mask, true_label=1,
             proj_strategy=None, student_split_ids=None, 
             s_pad_token=0, t_s_layers_ids=None):
    loss_mse = 0.0
    for i, (s, t) in enumerate(get_t_s_hiddens(s_hid, t_hid, s_mask, t_mask, true_label,
                                               proj_strategy, student_split_ids, s_pad_token, 
                                               t_s_layers_ids)):
        b_size = t.size(0)
        if hid_projectors_mse_student is not None:
            s = hid_projectors_mse_student[i](s)
        if hid_projectors_mse_teacher is not None:
            t = hid_projectors_mse_teacher[i](t)
        loss_mse += mse_loss_fct(s, t) / b_size
    return loss_mse


def contrastive_step(train_cardinality, 
                     hid_projectors_contrastive_student, 
                     hid_projectors_contrastive_teacher, 
                     s_hid, t_hid,
                     s_mask, t_mask, false_label=0, true_label=1,
                     proj_strategy=None, student_split_ids=None,
                     s_pad_token=0,
                     negative_sampling_strategy='student', use_mismatched_ids=False,
                     n_negative_samples=-1, teacher_student_prop=0.5, temperature=1,
                     from_one_sample=False, add_neg_size_constant=False, t_s_layers_ids=None, 
                     similarity_metric=None, **kwargs):
    """"
        from_one_sample -- make sampling from current sample only, not from all batch

    """
    loss_contrastive = 0.0

    if from_one_sample:
        match_paddings = s_mask == true_label
        seq_len = match_paddings.sum(dim=1)
    if use_mismatched_ids:
        mismatches = get_t_s_hiddens(s_hid, t_hid,
                                     s_mask, t_mask,
                                     false_label, proj_strategy,
                                     student_split_ids, s_pad_token, 
                                     t_s_layers_ids=t_s_layers_ids)
    for i, (s, t) in enumerate(get_t_s_hiddens(s_hid, t_hid, s_mask, t_mask, true_label,
                                               proj_strategy, student_split_ids, s_pad_token, 
                                               t_s_layers_ids=t_s_layers_ids)):
        if hid_projectors_contrastive_student is not None:
            s = hid_projectors_contrastive_student[i](s)

        if hid_projectors_contrastive_teacher is not None:
            t = hid_projectors_contrastive_teacher[i](t)

        if kwargs.get('use_hyp_mapping_in_step', False):
            c = kwargs['c']
            s = kwargs['student_to_poincare'](s, c)
            t = kwargs['teacher_to_poincare'](t, c)

        # loop by b*seq_len, here t.size(0) = s.size(0) as get_t_s_hiddens returns aligned sequences
        b_seq_len = t.size(0)

        k = 0; offset = 0;
        s_mismatches_ct = 0; t_mismatches_ct = 0
        layer_contrastive_loss = 0.0

        if use_mismatched_ids:
            # TODO: fix teacher_and_student sampling weigths with option use_mismatched_ids
            layer_mismatches = next(mismatches)

            if 'student' in negative_sampling_strategy:
                s_mismatches = layer_mismatches[0]
                if hid_projectors_contrastive_student is not None:
                    s_mismatches = hid_projectors_contrastive_student[i](s_mismatches)
                stud_hid_proj = torch.cat((stud_hid_proj, s_mismatches), dim=0)
                s_mismatches_ct += s_mismatches.size(0)

            elif 'teacher' in negative_sampling_strategy: 
                t_mismatches = layer_mismatches[1]
                if hid_projectors_contrastive_teacher is not None:
                    t_mismatches = hid_projectors_contrastive_teacher[i](t_mismatches)
                t = torch.cat((t, t_mismatches), dim=0)
                t_mismatches_ct += t_mismatches.size(0)

        for j in range(b_seq_len):
            ct_mismatched = t_mismatches_ct + s_mismatches_ct
            weights = 1 / b_seq_len * torch.ones(b_seq_len + ct_mismatched)

            if use_mismatched_ids and t_mismatches_ct + s_mismatches_ct > 0:
                # ct_mismatched is less than b_seq_len, so mismatches obtain greater weights
                weights[:-ct_mismatched] = 1 / max(t_mismatches_ct, s_mismatches_ct)
                weights[len(weights) - ct_mismatched:] = 1 / b_seq_len

            if from_one_sample:
                if j >= offset + seq_len[k].item():
                    offset += seq_len[k].item()
                    k += 1
                j = j % offset if offset > 0 else j
                neg = negative_sampling(s[offset:offset + seq_len[k]],
                                        t[offset:offset + seq_len[k]], 
                                        j, n_negative_samples,
                                        weights[offset:offset + seq_len[k]].numpy(),
                                        teacher_student_prop,
                                        negative_sampling_strategy)
            else:
                neg = negative_sampling(s, t, j, n_negative_samples,
                                        weights.numpy(),
                                        teacher_student_prop,
                                        negative_sampling_strategy)

            if negative_sampling_strategy == 'teacher':
                pos_base = s[j]
                pos_twin = t[j]
            else:
                pos_base = t[j]
                pos_twin = s[j]
            num = torch.exp(similarity_metric(pos_base, pos_twin) / temperature)
            den = num + torch.exp(similarity_metric(neg, pos_base) / temperature).sum()

            if add_neg_size_constant:
                den += neg.size(0) / train_cardinality

            layer_contrastive_loss -= torch.log(num / den)
        # we should devide total loss on number of positive samples
        # currently num_positive_samples == b_seq_len
        loss_contrastive += layer_contrastive_loss / b_seq_len

    return loss_contrastive


def contrastive_step_v0(train_cardinality, 
                        hid_projectors_contrastive, 
                        s_hid, t_hid, s_mask, t_mask, false_label=0,
                        true_label=0, proj_strategy='average_by_layers',
                        s_pad_token=0, add_neg_size_constant=False):
    # v0: negative = mismatched
    loss_contrastive = 0.
    all_positive = get_t_s_hiddens(s_hid, t_hid, s_mask, t_mask, true_label, proj_strategy)
    all_negative = get_t_s_hiddens(s_hid, t_hid, s_mask, t_mask, false_label, proj_strategy)

    for i, ((s_p, t_p), (s_n, _)) in enumerate(zip(all_positive, all_negative)):
        layer_contrastive_loss = 0.0
        b_seq_len1 = t_p.size(0)
        b_seq_len2 = s_n.size(0)

        diff = b_seq_len1 - b_seq_len2
        # positive count is greater than negative
        if diff > 0:
            s_hid_dim = s_n.size(1)
            pad = s_pad_token * torch.ones(diff, s_hid_dim).to(s_n.device)
            s_n = torch.cat((s_n, pad), dim=0) if b_seq_len2 > 0 else pad

        student_hid_pos_proj = hid_projectors_contrastive[i](s_p)
        student_hid_neg_proj = hid_projectors_contrastive[i](s_n)
        num = torch.exp(F.cosine_similarity(student_hid_pos_proj, t_p, dim=1)).sum()
        den = num + torch.exp(F.cosine_similarity(student_hid_neg_proj, t_p, dim=1)).sum()
        if add_neg_size_constant:
            den += b_seq_len1 / train_cardinality
        layer_contrastive_loss -= torch.log(num / den)
        loss_contrastive += layer_contrastive_loss / b_seq_len1
    return loss_contrastive


def get_t_s_hiddens(s_hid, t_hid, student_mask, teacher_mask, true_label=1, 
                    proj_strategy=None, student_split_ids=None,
                    s_pad_token=0, t_s_layers_ids=None):
    s_hid_dim = s_hid[-1].size(-1)
    t_hid_dim = t_hid[-1].size(-1)

    if proj_strategy == 'average':
        for i in [0] + list(range(1, 13, 4)):
            s_hid_i = reduce_seq(s_hid[(i + 3) // 4], student_split_ids, s_pad_token) if student_split_ids is not None else s_hid[(i + 3) // 4]
            s_hid_i = masked_select_reshape_2d(s_hid[i], student_mask == true_label, s_hid_dim)
            t_hid_i = masked_select_reshape_2d(torch.stack(t_hid[i: (i + 4)]).mean(dim=0), teacher_mask == true_label, t_hid_dim)
            yield s_hid_i, t_hid_i

    elif proj_strategy == 'skip':
        for i in [0] + list(range(4, 13, 4)):
            s_hid_i = reduce_seq(s_hid[i // 4], student_split_ids, s_pad_token) if student_split_ids is not None else s_hid[i // 4]
            s_hid_i = masked_select_reshape_2d(s_hid_i, student_mask == true_label, s_hid_dim)
            t_hid_i = masked_select_reshape_2d(t_hid[i], teacher_mask == true_label, t_hid_dim)
            yield s_hid_i, t_hid_i
    elif proj_strategy == 'last':
        for i in range(9, 13):
            s_hid_i = reduce_seq(s_hid[i - 9], student_split_ids, s_pad_token) if student_split_ids is not None else s_hid[i - 9]
            s_hid_i = masked_select_reshape_2d(s_hid_i, student_mask == true_label, s_hid_dim)
            t_hid_i = masked_select_reshape_2d(t_hid[i], teacher_mask == true_label, t_hid_dim)
            yield s_hid_i, t_hid_i
    elif proj_strategy == 'average_by_layers':
        s_avg = average_by_layers(s_hid, student_split_ids, s_pad_token, student_mask == true_label)
        t_avg = average_by_layers(t_hid, attn_mask=teacher_mask == true_label)
        for _ in range(1):
            yield s_avg, t_avg
    elif proj_strategy == 'select_by_ids' and t_s_layers_ids is not None:
        s_id = t_s_layers_ids['student']
        t_id = t_s_layers_ids['teacher']
        s_i = reduce_seq(s_hid[s_id], student_split_ids, s_pad_token) if student_split_ids is not None else s_hid[s_id]
        s_i = masked_select_reshape_2d(s_i, student_mask == true_label, s_hid_dim)
        t_i = masked_select_reshape_2d(t_hid[t_id], teacher_mask == true_label, t_hid_dim)
        for _ in range(1):
            yield s_i, t_i


def negative_sampling(s=None, t=None, positive_idx=0, k=-1, weights=None, prop=0.5, sampling_strategy='teacher'):
    """
        Negative sampling generation for contrastive loss

        Args:
            t: teacher representation
            s: student representation
            positive_ids: index of positive sample that should be excluded
            k: number of negative samples (k=-1 to sample all except positive_idx)
            weights: sampling weights
            prop: proportion between teacher and student, ignored if sampling is performed from only teacher or only student
            sampling_strategy: ['teacher', 'student', 'teacher_and_student']
        Returns:
            Negative samples for contrastive loss

    """
    assert t is not None or s is not None
    b_seq_len = t.size(0) if sampling_strategy == 'teacher' else s.size(0)
    # get all if k == -1 or k is greater than (b_seq_len - 1)
    k = min(k, b_seq_len - 1) if k != -1 else b_seq_len - 1
    idxs = np.delete(np.arange(b_seq_len), positive_idx)
    weights = np.delete(weights, positive_idx)
    # normalize again for probs sum to 1
    weights /= np.sum(weights)
    idxs = torch.from_numpy(np.random.choice(idxs, size=k, replace=False, p=weights))
    if sampling_strategy == 'teacher' and t is not None:
        return t[idxs]
    if sampling_strategy == 'student' and s is not None:
        return s[idxs]
    if sampling_strategy == 'teacher_and_student' and t is not None and s is not None:
        n = len(idxs)
        l1 = int(prop * n)
        return torch.cat((t[idxs[:l1]], s[idxs[l1:]]), dim=0)
