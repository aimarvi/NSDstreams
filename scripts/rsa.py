import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr

from tqdm import tqdm

import gen_utils as utils
test_ind = utils.shared_1000()


def rdm(model, fname):
    response_dir = os.path.join(response_root, f'{fname}.npy')
    mat = np.load(response_dir)[test_ind]
    norm1 = (mat - np.mean(mat, axis=1, keepdims=True)) / np.std(mat, axis=1, keepdims=True)
    diss1 = pdist(norm1, metric='correlation')
    rdm1 = squareform(diss1)

    return(rdm1)

response_root = '/om2/user/amarvi/NSDstreams/features'
# model_files = {'untrained-transformer': 'feat_dino-vit_base_patch16_224_final',
#               'simclr': 'feat_resnet50-simclr_in1k_avgpool',
#               'dino': 'feat_pret-dino-vit_base_patch16_224_final'}
# model_files = {'fb-dino': 'fb_dino_feats'}
# model_files = {f'dino-layer{_l}': f'dino-layers/dino-layer{_l}_features' for _l in range(12)}
model_files = {f'swin-layer{_l:02d}': f'swin-layers/swin{_l:02d}_features' for _l in range(24)}

for model, fname in tqdm(model_files.items()):
    _rdm = rdm(model, fname)
    np.save(f'/om2/user/amarvi/NSDstreams/figs/rdms/model/{model}_rdm.npy', _rdm)
