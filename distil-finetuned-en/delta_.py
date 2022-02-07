import torch
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')
import hyptorch.delta as hypdelta
import custom_step


def calculate_c(delta, diam):
    rel_delta = (2 * delta) / diam
    return (0.144 / rel_delta) ** 2

def get_delta(model, dataloader, cuda_no=0, n_samples_slct=1000, n_components=100, n_tries=3):
    features = []
    model.eval()

    iter_bar = tqdm(dataloader, desc="-Iter")
    total_samples = 0
    with torch.no_grad():
        for batch in iter_bar:
            batch_cuda = {name: value.to(f'cuda:{cuda_no}') for name, value in batch.items() if name in model.forward.__code__.co_varnames}
            out = model(**batch_cuda)
            last_dim = out.logits.size(-1)
            logits = custom_step.masked_select_reshape_2d(out.logits, batch_cuda['attention_mask'], last_dim)
            # if logits will not be casted, we need to cat all features (!this is too long)
            features.extend(logits.detach().cpu().tolist())
            total_samples += len(logits)
    deltas = 0; diams = 0
    for _ in n_tries:
        idxs_slct = np.random.choice(range(total_samples), size=min(total_samples, n_samples_slct), replace=False)   
        features_ = np.vstack([np.array(features[i]) for i in idxs_slct])
        if n_components > 0:
            pca = PCA(n_components=min((n_components, features_.shape[0], features_.shape[1])))
            features_ = pca.fit_transform(features_)
        dists = distance_matrix(features_, features_)
        delta = hypdelta.delta_hyp(dists)
        diam = np.max(dists)
        deltas += delta
        diams += diam

    deltas /= n_tries
    diams /= n_tries

    return deltas, diams