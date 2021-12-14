import torch
import torch.nn.functional as F
from scipy.spatial import distance_matrix
import numpy as np
from tqdm import tqdm

import custom_step
import distiller

import hyptorch.delta as hypdelta



def calculate_c(delta, diam):
    rel_delta = (2 * delta) / diam
    return (0.144 / rel_delta) ** 2

def get_delta(model, dataloader, ids_field, lengths_field, 
              n_samples_slct, cuda_no, multi_gpu):
    features = []
    model.eval()

    if multi_gpu:
        torch.distributed.barrier()

    iter_bar = tqdm(dataloader, desc="-Iter", disable=cuda_no not in [-1, 0])
    total_samples = 0
    with torch.no_grad():
        for batch in iter_bar:
            batch_cuda = {name: value.to(f'cuda:{cuda_no}') for name, value in batch.items()}
            attn_mask = distiller.Distiller.generate_padding_mask(batch[ids_field].size(1), batch_cuda[lengths_field])
            out = model(batch_cuda[ids_field], attn_mask)
            last_dim = out.logits.size(-1)
            logits = custom_step.masked_select_reshape_2d(out.logits, attn_mask, last_dim)
            # if logits will not be casted, we need to cat all features (!this is too long)
            features.extend(logits.detach().cpu().tolist())
            total_samples += len(logits)
    idxs_slct = np.random.choice(total_samples, size=n_samples_slct, replace=False)   
    features = np.vstack([np.array(features[i]) for i in idxs_slct])
    dists = distance_matrix(features, features)
    delta = hypdelta.delta_hyp(dists)
    diam = np.max(dists)

    return delta, diam
