import nimfa
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.decomposition import PCA

def evar(V, V_hat):
    V_cent = V - np.mean(V, axis=0)
    V_hat_cent = V_hat - np.mean(V_hat, axis=0)

    TSS = np.sum(np.square(V_cent))
    RSS_k = np.sum(np.square(V_cent - V_hat_cent))

    explained_variance_ratio_k = 1 - (RSS_k / TSS)
    return explained_variance_ratio_k

df = pd.DataFrame(columns = ['method', 'evar', 'subj', 'iter', 'stream'])

for subj in [1,2,5,7]:
    for stream in ['lateral', 'ventral', 'dorsal']:
        data_dir = f'NSD_processed/{stream}_visual_data/'
        n_components = 20

        data = np.load('%s/subj%d.npy' % (data_dir,subj))
        _dat = (data - data.min(0)).T
        for itr in tqdm(range(50)):

            # bayesian nmf
            bnmf = nimfa.Bd(_dat, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((_dat.shape[0], n_components)),
                  beta=np.zeros((n_components, _dat.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
                  n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
            bnmf_fit = bnmf()
            W = bnmf_fit.basis()
            H = bnmf_fit.coef()
            V_hat_bnmf = np.dot(W, H)
            
            evar_bnmf = evar(_dat, V_hat_bnmf)
            df.loc[len(df)] = {'method': 'bnmf', 'evar': evar_bnmf, 'subj': subj, 'iter': itr, 'stream': stream}

            # standard nmf
            nmf = nimfa.Nmf(_dat, seed="random_c", rank=n_components, max_iter=12) 
            nmf_fit = nmf()
            W = nmf_fit.basis()
            H = nmf_fit.coef()
            V_hat_nmf = np.dot(W, H)
            
            evar_nmf = evar(_dat, V_hat_nmf)
            df.loc[len(df)] = {'method': 'nmf', 'evar': evar_nmf, 'subj': subj, 'iter': itr, 'stream': stream}

            # pca explained variance
            pca = PCA(n_components=n_components)
            pca.fit(_dat)
            trans = pca.fit_transform(_dat)
            recon = pca.inverse_transform(trans)
            
            evar_pca = evar(_dat, recon) 
            df.loc[len(df)] = {'method': 'pca_fixed', 'evar': evar_pca, 'subj': subj, 'iter': itr, 'stream': stream}

df.to_pickle('evar_rss.pkl')

