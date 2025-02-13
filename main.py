from bayesnmf import BayesianNMF
import numpy as np
import matplotlib.pyplot as plt
import os
import nimfa
from utils import *
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import argparse 
from get_consensus import get_consensus_response, get_consensus_weights


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Complete NMF pipeline for a single subject')
    parser.add_argument('--subj', default=1, type=int, help='Subject number')
    parser.add_argument('--bic_save_dir', default='bic', type=str, help='BIC save directory')
    parser.add_argument('--run_save_dir', default='saved_runs', type=str, help='Runs save directory')
    parser.add_argument('--num_runs', default=50, type=int, help='Number of runs')
    parser.add_argument('--data_dir', default='/mindhive/nklab4/shared/datasets/NSD_processed/lateral_visual_data/', type=str, help='Data directory')
    parser.add_argument('--consensus_dir', default='consensus_results', type=str, help='Data directory')
    parser.add_argument('--iteration', default=1, type=int, help='Iteration Number')    
   
    args = parser.parse_args()
    
    subj = args.subj
    data_dir = args.data_dir
    save_dir = args.bic_save_dir
    
   if not os.path.isdir(save_dir): 
       os.mkdir(save_dir)
   
   data = np.load('%s/subj%d.npy' % (data_dir,subj)) 
 
   V = (data - data.min(0))
   search_list = range(5,35,5)
   bics = np.zeros_like(search_list)

   for n, n_components in enumerate(search_list):
       
       bd = BayesianNMF(n_components = n_components, mode = 'gibbs', mean_only = True,  max_iter = 20)
       res = bd.fit(V)
       bics[n] = res.bic(V) 
   np.save('%s/subj%d.npy' % (save_dir, subj), bics)
    
    
    ### Perform individual runs with optimal number of components
    
    n_components = 20
   
    iters = args.iteration 
    save_dir = args.run_save_dir
    if not os.path.isdir(save_dir): 
        os.mkdir(save_dir)
    num_runs = args.num_runs
    
    data = np.load('%s/subj%d.npy' % (data_dir,subj))  
    V = (data - data.min(0)).T
        
        
    for itr in range(1, iters+1):
        for r_ in range(num_runs):   
            try:
                _ = np.load('%s/subj%d_iter%d_run%d.npy' % (save_dir, subj, itr, r_), allow_pickle = True)
            except:
                bnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
                              beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
                              n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
                bnmf_fit = bnmf()
                W = bnmf_fit.basis()
                H = bnmf_fit.coef()
                data_transformed = np.asarray(H.T).copy()

                np.save('%s/subj%d_iter%d_run%d.npy' % (save_dir, subj, itr, r_), {'data_transformed': data_transformed, 'W': np.asarray(W)})
                del data_transformed, W

        ##### Perform consensus
        consensus_dir = args.consensus_dir
        if not os.path.isdir(consensus_dir): 
            os.mkdir(consensus_dir)

        resp = []
        for r_ in range(num_runs):
            data = np.load('%s/subj%d_iter%d_run%d.npy' %  (save_dir, subj, itr, r_), allow_pickle = True).item()['data_transformed'] 
            resp.append(data.T)
        resp = np.asarray(resp)
        consensus_response = get_consensus_response(resp)
        np.save('%s/subj%d_iter%d_response.npy' % (consensus_dir, subj, itr), consensus_response)

        weights = []
        for r_ in range(num_runs):
            data = np.load('%s/subj%d_iter%d_run%d.npy' %  (save_dir, subj, itr,  r_), allow_pickle = True).item()['W'] 
            weights.append(data)
        weights = np.asarray(weights)
        consensus_weights = get_consensus_weights(resp, weights, consensus_response)
        np.save('%s/subj%d_iter%d_weights.npy' % (consensus_dir, subj, itr), consensus_weights)
