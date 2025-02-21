import nimfa
import scipy.sparse as sparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

import rot2d_utils as utils

def simulate(factor=20):
    m, n, k = 10, 4, 2
    sparsity_level = 0.9
    X, L, A = utils.generate_synthetic_data(m, n, k, sparsity_level)

    cols = ['method', 'factor', 'frac_planes', 'similarity']
    df = pd.DataFrame(columns=cols)

    for rot_seed in tqdm(range(5)):
        if factor == 0:
            mat = utils.Representation(dim = X.shape[1])
        else: 
            mat = utils.Representation(dim = X.shape[1], factor = factor)
        final, seq = mat.get_matrix() 
        seq.append(final)
        rot_matrices = np.array([s.numpy() for s in seq])

        for idx, a in tqdm(enumerate(rot_matrices),total=len(rot_matrices)): 
            X_r = X @ a if factor != 0 else X.copy()

            pc_score = utils.angle(X, X_r, 'pca')
            nmf_score = utils.angle(X, X_r, 'nmf')
            df.loc[len(df)] = {'method': 'pca', 'factor': factor, 'frac_planes': idx, 'similarity': pc_score}
            df.loc[len(df)] = {'method': 'nmf', 'factor': factor, 'frac_planes': idx, 'similarity': nmf_score} 

    return df

for factor in range(0,100, 20):
    sims = simulate(factor)
    sims.to_pickle(f'/om2/user/amarvi/NSDstreams/figs/rotation_sensitivity/angle{factor:02d}.pkl')
