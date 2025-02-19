import nimfa
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import kurtosis,skew

def hoyer_sparsity(row):
    if len(row.shape) > 1:
        n = row.shape[1]
    else:
        n = len(row)
        
    sabs = np.sum(np.abs(row))
    sssq = np.sqrt(np.sum(np.square(row)))
        
    s = (np.sqrt(n) - sabs/sssq) / (np.sqrt(n) - 1)
    return s

# df = pd.DataFrame(columns = ['method', 'sparsity_W', 'kurtosis_W', 'skew_W', 'sparsity_H', 'kurtosis_H', 'skew_H', 'subj', 'iter', 'stream'])
df = pd.read_pickle('/om2/user/amarvi/NSDstreams/figs/skew.pkl')

for subj in [1,2,5,7]:
    for stream in ['lateral', 'ventral', 'dorsal']:
        data_dir = f'/mindhive/nklab4/shared/datasets/NSD_processed/{stream}_visual_data/'
        n_components = 20

        data = np.load('%s/subj%d.npy' % (data_dir,subj))
        V = (data - data.min(0)).T
        total_variance = np.var(V)
        
        for itr in tqdm(range(50)):

            # bayesian nmf
            bnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
                  beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
                  n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
            bnmf_fit = bnmf()
            W = bnmf_fit.basis()
            H = bnmf_fit.coef()
            V_hat_bnmf = np.dot(W, H)
            
            W_sparsity = np.apply_along_axis(hoyer_sparsity, 0, W)
            H_sparsity = np.apply_along_axis(hoyer_sparsity, 0, H)
            
            W_kurtosis = np.mean(kurtosis(W, axis=0))
            H_kurtosis = np.mean(kurtosis(H, axis=0))
            
            W_skew = np.mean(skew(W, axis=0))
            H_skew = np.mean(skew(H, axis=0))
            
            df.loc[len(df)] = {'method': 'bnmf', 
                               'sparsity_W': W_sparsity, 'kurtosis_W': W_kurtosis, 'skew_W': W_skew,
                               'sparsity_H': H_sparsity, 'kurtosis_H': H_kurtosis, 'skew_H': H_skew,
                               'subj': subj, 'iter': itr, 'stream': stream}

            # standard nmf
            nmf = nimfa.Nmf(V, seed="random_c", rank=n_components, max_iter=12) 
            nmf_fit = nmf()
            W = nmf_fit.basis()
            H = nmf_fit.coef()
            V_hat_nmf = np.dot(W, H)
            
            W_sparsity = np.apply_along_axis(hoyer_sparsity, 0, W)
            H_sparsity = np.apply_along_axis(hoyer_sparsity, 0, H)
            
            W_kurtosis = np.mean(kurtosis(W, axis=0))
            H_kurtosis = np.mean(kurtosis(H, axis=0))
            
            W_skew = np.mean(skew(W, axis=0))
            H_skew = np.mean(skew(H, axis=0))
            
            df.loc[len(df)] = {'method': 'nmf', 
                               'sparsity_W': W_sparsity, 'kurtosis_W': W_kurtosis, 'skew_W': W_skew,
                               'sparsity_H': H_sparsity, 'kurtosis_H': H_kurtosis, 'skew_H': H_skew,
                               'subj': subj, 'iter': itr, 'stream': stream}

            # pca explained variance
            pca = PCA(n_components=n_components)
            pca.fit(V)
            
            evar_pca = pca.explained_variance_ratio_.sum()
            
            W = pca.transform(V)
            H = pca.components_
            
            W_sparsity = np.apply_along_axis(hoyer_sparsity, 0, W)
            H_sparsity = np.apply_along_axis(hoyer_sparsity, 0, H)
            
            W_kurtosis = np.mean(kurtosis(W, axis=0))
            H_kurtosis = np.mean(kurtosis(H, axis=0))
            
            W_skew = np.mean(skew(W, axis=0))
            H_skew = np.mean(skew(H, axis=0))
            
            df.loc[len(df)] = {'method': 'pca', 
                               'sparsity_W': W_sparsity, 'kurtosis_W': W_kurtosis, 'skew_W': W_skew,
                               'sparsity_H': H_sparsity, 'kurtosis_H': H_kurtosis, 'skew_H': H_skew,
                               'subj': subj, 'iter': itr, 'stream': stream}
            
df.to_pickle('/om2/user/amarvi/NSDstreams/figs/skew2.pkl')
