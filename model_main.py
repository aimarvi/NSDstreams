from bayesnmf import BayesianNMF
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import nimfa
from utils import *
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import argparse 
from get_consensus import get_consensus_response, get_consensus_weights

model_root = '/om2/user/amarvi/NSDstreams/features'
# model_files = {'vit_resnet': 'feat_vit_base_resnet50_384_final.npy'}
#               #'untrained': 'feat_alexnet_in1k_pool_random.npy', 
#               #'moco': 'feat_resnet50_in1k_avgpool_moco.npy', 
#               #'alexnet': 'feat_alexnet_in1k_pool.npy', 
#               #'resnet50': 'feat_resnet50_in1k_avgpool.npy', 
#               #'vit': 'feat_vit_base_patch16_224_final.npy'}
# model_files = {'dino': 'feat_dino-vit_base_patch16_224_final.npy',
#               'simclr': 'feat_resnet50-simclr_in1k_avgpool.npy'}
# model_files = {f'dino-layer{_l}': f'dino-layers/dino-layer{_l}_features.npy' for _l in range(12)}
model_files = {f'swin-layer{_l:02d}': f'swin-layers/swin{_l:02d}_features.npy' for _l in range(24)}

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Complete NMF pipeline for a single model')
    parser.add_argument('--num_runs', default=50, type=int, help='Number of runs')
    parser.add_argument('--feat_dir', default='/mindhive/nklab4/shared/datasets/NSDstreams/features/', type=str, help='Feature directory')
    parser.add_argument('--iteration', default=1, type=int, help='Iteration Number')    
    args = parser.parse_args()
    
    ids_all = []
    for sb in [1,2,5,7]: 
        ids_all.extend(np.load('/mindhive/nklab4/users/mkhosla/NSD/data/coco_ID_of_repeats_subj%02d.npy'%sb))
    unique_ids = np.unique(ids_all)
    
    for sb in [1, 2, 5, 7]:
        coco = np.load('/mindhive/nklab4/users/mkhosla/NSD/data/coco_ID_of_repeats_subj%02d.npy'%sb)
        match = []
        for i, id in enumerate(coco):
            match.append(np.where(unique_ids == id)[0][0])
        match = np.asarray(match)

        for model, file in model_files.items():
            top_dir = '/om2/user/amarvi/NSDstreams/model_nmf_results/%s' % (model)
            if not os.path.isdir(top_dir):
                os.mkdir(top_dir)
            feat_dir = args.feat_dir

            # ========================================
            # ===== set num of components ============
            # ========================================
            n_components = 20

            iters = args.iteration 
            save_dir = '/om2/user/amarvi/NSDstreams/model_nmf_results/%s/%s_s%d_saved_runs' % (model,model,sb)
            if not os.path.isdir(save_dir): 
                os.mkdir(save_dir)
            num_runs = args.num_runs

            data = np.load(os.path.join(feat_dir, file))[match] 
            V = (data - data.min(0)).T

            # ========================================
            # ===== run n iterations of nmf ==========
            # ========================================
            for itr in range(1, iters+1):
                for r_ in range(num_runs):   
                    try:
                        _ = np.load('%s/%s_iter%d_run%d.npy' % (save_dir, model, itr, r_), allow_pickle = True)
                    except:
                        bnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
                                      beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
                                      n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
                        bnmf_fit = bnmf()
                        W = bnmf_fit.basis()
                        H = bnmf_fit.coef()
                        data_transformed = np.asarray(H.T).copy()

                        np.save('%s/%s_iter%d_run%d.npy' % (save_dir, model, itr, r_), {'data_transformed': data_transformed, 'W': np.asarray(W)})
                        del data_transformed, W

                # =============================
                # ==== consensus results ======
                # =============================
                consensus_dir = '/om2/user/amarvi/NSDstreams/model_nmf_results/%s/%s_s%d_consensus_results' % (model,model,sb)
                if not os.path.isdir(consensus_dir): 
                    os.mkdir(consensus_dir)

                resp = []
                for r_ in range(num_runs):
                    data = np.load('%s/%s_iter%d_run%d.npy' %  (save_dir, model, itr, r_), allow_pickle = True).item()['data_transformed'] 
                    resp.append(data.T)
                resp = np.asarray(resp)
                consensus_response = get_consensus_response(resp)
                np.save('%s/%s_iter%d_response.npy' % (consensus_dir, model, itr), consensus_response)

                weights = []
                for r_ in range(num_runs):
                    data = np.load('%s/%s_iter%d_run%d.npy' %  (save_dir, model, itr,  r_), allow_pickle = True).item()['W'] 
                    weights.append(data)
                weights = np.asarray(weights)
                consensus_weights = get_consensus_weights(resp, weights, consensus_response)
                np.save('%s/%s_iter%d_weights.npy' % (consensus_dir, model, itr), consensus_weights)
