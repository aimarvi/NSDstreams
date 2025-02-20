import numpy as np
import pandas as pd

import rot2d_utils as utils

from tqdm import tqdm
from sklearn.decomposition import PCA

m, n, k, sparsity_level = 10, 5, 2, 0.9
X, _, _ = utils.generate_synthetic_data(m, n, k, sparsity_level)
_X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# rads = [_d for _d in range(1,10)] + [_d for _d in range(10, 100, 10)]
rads = [_d for _d in range(1,21)]
rads = rads[::-1]

# ===================
# rotate pca pc's
# ===================
pca = PCA(n_components=3)
_fit = pca.fit(_X)
og_pc = _fit.components_

cols = ['method', 'factor', 'similarity']
df = pd.DataFrame(columns=cols)

df.loc[len(df)] = {'method': 'pca', 'factor': '0', 'similarity': 1}
df.loc[len(df)] = {'method': 'pca', 'factor': '0', 'similarity': 1}  

for rad in rads: # pi=180
    rotator = utils.Representation(dim=_X.shape[1], factor=rad)
    final, seq = rotator.get_matrix(); seq.append(final)
    all_rotations = np.array([s.numpy() for s in seq])
    for rot_mat in all_rotations:
        _Xr = _X@rot_mat
        _rfit = pca.fit(_Xr)
        
        rot_pc = _rfit.components_
        rog_pc = og_pc@rot_mat
        
        metrics = []
        for idx in range(len(rog_pc)):
            metric = np.corrcoef(rot_pc[idx], rog_pc[idx])
            metrics.append(metric[0,1])
        metrics = np.array(metrics)
        df.loc[len(df)] = {'method': 'pca', 'factor': f'{rad}', 'similarity': np.mean(metrics)}   


# ===================
# do the same for nmf 
# ===================
_fit = utils.get_nmf(_X, n_components=3)
W = _fit.basis()
H = _fit.coef()

og_pc = W.T

df.loc[len(df)] = {'method': 'nmf', 'factor': '0', 'similarity': 1}  
df.loc[len(df)] = {'method': 'nmf', 'factor': '0', 'similarity': 1} 

for rad in tqdm(rads): # pi=180
    rotator = utils.Representation(dim=_X.shape[1], factor=rad)
    final, seq = rotator.get_matrix(); seq.append(final)
    all_rotations = np.array([s.numpy() for s in seq])
    for rot_mat in all_rotations:
        _Xr = _X@rot_mat
        _rfit = utils.get_nmf(_Xr, n_components=3)
        
        rot_pc = _rfit.basis().T
        rog_pc = og_pc@rot_mat
        
        metrics = []
        for idx in range(len(rog_pc)):
            metric = np.corrcoef(rot_pc[idx], rog_pc[idx])
            metrics.append(metric[0,1])
        metrics = np.array(metrics)
        df.loc[len(df)] = {'method': 'nmf', 'factor': f'{rad}', 'similarity': np.mean(metrics)}
