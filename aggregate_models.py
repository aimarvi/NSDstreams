import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

def get_data(subj, stream, shared = False, consensus_dir = 'consensus_results'):
    splits = np.load('./data/splits_subj%02d.npy'%subj, allow_pickle = True).item()
    comps = np.load('%s/%s_iter1_response.npy' % (consensus_dir, stream))
    if shared:
        return comps[splits['test']]
    else:
        return comps

def aggregate_consistency(subjs = [1,2,5,7], plot_dir = 'figures', agg_dir = 'aggregate_data/model/', stream = 'ventral', n_comp=10):
    
    data_all = []
    for subj in subjs:
        consensus_dir = 'model_nmf_results/%s/%s_s%d_consensus_results' % (stream,stream,subj)
        data_all.append(get_data(subj, stream, shared = True, consensus_dir = consensus_dir))


    data_ = []
    for subj in subjs:
        consensus_dir = 'model_nmf_results/%s/%s_s%d_consensus_results' % (stream,stream,subj)
        data_.append(get_data(subj, stream, shared = False, consensus_dir = consensus_dir))
    n_components = data_[0].shape[1]
    
    subj_corr = np.zeros((n_components, n_components, n_components, n_components))
    subj_corr_all = np.zeros((n_components, n_components, n_components, n_components,6))

    count = 0
    for i in range(n_components):
        for j in range(n_components):
            for k in range(n_components):
                for l in range(n_components):
                    #### Stack data for all subjects
                    stacked = np.vstack((data_all[0][:,i], data_all[1][:,j], data_all[2][:,k], data_all[3][:,l]))
                    stacked_corr = np.corrcoef(stacked)
                    subj_corr[i,j,k,l] = np.nanmin(stacked_corr[np.triu_indices(4,k=1)]) #
                    subj_corr_all[i,j,k,l] = stacked_corr[np.triu_indices(4,k=1)]
                    count += 1

    all_ixs = np.dstack(np.unravel_index(np.argsort(subj_corr.ravel()), subj_corr.shape))[0,::-1]
    covered = []
    curr_ix = 0
    n_des = n_comp ### How many top consistent components to consider? 
    while len(covered) <= n_des:
        if curr_ix >= len(all_ixs):  # Check if curr_ix is out of bounds
            print("Ran out of indices to consider.")
            break
        best_ix = all_ixs[curr_ix]
        found = True
        for el in covered:
            if np.any(best_ix == el): 
                curr_ix += 1
                found = False

        if found == True:
            covered.append(best_ix)
            
    if not os.path.isdir(agg_dir): 
        os.mkdir(agg_dir)
    np.save('%s/%s_consistent_comp_ids.npy' % (agg_dir, stream), covered)
    print('its working !')
    
    return covered

for model in tqdm(['untrained', 'alexnet', 'resnet50', 'moco', 'vit']):
    aggregate_consistency(stream=model,n_comp=20)
