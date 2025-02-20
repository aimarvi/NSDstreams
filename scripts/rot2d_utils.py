import torch
import numpy as np
import scipy.sparse as sparse
import scipy.spatial.distance as dist

import nimfa
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import argparse 
import sys
import os

def get_dim(subj = 1, roi = 5):
    mask = np.isin(np.load('./masks/subj%d/roi_1d_mask_subj%02d_streams.npy'%(subj, subj)), [roi])
    ys = np.load('./voxel_data/subj%02d_test_singletrial_all.npy' % subj)[:,:,mask]
    
    ys = np.moveaxis(ys, 2, 0)
    sn2 = np.var(ys, 2, ddof = 1).mean(1)#sn = np.sqrt(sn2)
    ss2 = 1 - sn2
    ss2[ss2<0] = 0
    nc = np.sqrt((ss2/(ss2+sn2/3)))
    print('Number of streams voxels', nc.shape)
    return np.count_nonzero(nc > 0)

class Representation():

    def __init__(self, dim=4, factor = 50):
        self.dim = dim
        self.params = dim*(dim-1)//2
        #self.thetas = torch.autograd.Variable(np.pi*(2*torch.rand(self.params)-1)/dim, requires_grad=False)
        self.thetas = torch.autograd.Variable(np.pi*(2*torch.rand(self.params)-1)/factor, requires_grad=False)
        
        self.__matrix = None
    
    def set_thetas(self, thetas):
        self.thetas = thetas
        self.thetas.requires_grad = True
        self.clear_matrix()
    
    def clear_matrix(self):
        self.__matrix = None
        
    def get_matrix(self):
        if self.__matrix is None:
            k = 0
            save_idx = np.linspace(0, self.dim*(self.dim - 1)/2, 11).astype('int32')
            mats = []
            final_mat = torch.eye(self.dim, self.dim).cuda()
            final_mat.requires_grad = False
             
            mats.append(final_mat.cpu())
            with torch.no_grad():    
                for i in range(self.dim-1):
                    
                    for j in range(self.dim-1-i):
                        
                        theta_ij = self.thetas[k]
                        
                        c, s = torch.cos(theta_ij), torch.sin(theta_ij)

                        B = torch.clone(final_mat)
                        final_mat[:,i] = B[:,i]*c - s*B[:,j+i+1]
                        final_mat[:,j+i+1] = B[:,i]*s + c*B[:,j+i+1]
                        
                        if k in save_idx[1:]: 
                           
                            mats.append(final_mat.cpu())
                        
                        torch.cuda.empty_cache()
                        k+=1
                        
            self.__matrix = final_mat 
                                    
        return self.__matrix.cpu(), mats  

def generate_synthetic_data(m, n, k, sparsity_level, noise_level=0.01):
    """
    Generates a synthetic data matrix X = L @ A.T + Noise, where L has sparse latent components.
    
    Args:
    - m: Number of rows (samples)
    - n: Number of columns (features)
    - k: Number of latent components
    - sparsity_level: Proportion of non-zero elements in L (0 to 1)
    - noise_level: Standard deviation of Gaussian noise
    
    Returns:
    - X: The synthetic data matrix
    - L: The true sparse latent components
    - A: The mixing matrix
    """
    L = sparse.random(m, k, density=sparsity_level, data_rvs=lambda s: np.random.uniform(0, 1, size=s)) #sparse.random(m, k, density=sparsity_level, data_rvs=np.random.randn).toarray()
    # A = sparse.random(n, k, density=sparsity_level, data_rvs=lambda s: np.random.uniform(0, 1, size=s))
    A = np.random.uniform(size = (n, k))
    
    X = L @ A.T
    
    noise = noise_level * np.random.randn(m, n)
    X += noise
    
    return X, L, A

def get_pc(data, n_components=2):
    
    V = np.asarray((data-data.min(0)).T.copy())
    pca_fit = PCA(n_components=n_components).fit(V)
        
    return pca_fit.components_

def get_nmf(data, n_components = 2):
    V = (data-data.min(0)).T.copy()
    lsnmf = nimfa.Bd(V, seed="nndsvd", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
              beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
              n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)),track_factor = True, n_run = 50, n_sigma=False) 
    lsnmf_fit = lsnmf()
    
    return lsnmf_fit

def align(X, X_r, method):
    if method=='pca':
        c1 = get_pc(X)
        c2 = get_pc(X_r)
    elif method=='nmf':
        c1 = get_conn(X)
        c2 = get_conn(X_r)
    
    alignment = np.corrcoef(c1.flatten(), c2.flatten())
    return alignment[0,1]

def angle(X, X_r, method):
    if method=='pca':
        c1 = get_pc(X)
        c2 = get_pc(X_r)
    elif method=='nmf':
        c1 = np.asarray(get_conn(X)).squeeze()
        c2 = np.asarray(get_conn(X_r)).squeeze()
    
    dist1 = dist.cosine(c1[0], c1[1])
    dist2 = dist.cosine(c2[0], c2[1])
    return np.abs(dist1-dist2)
