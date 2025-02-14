import os
import numpy as np
import pandas as pd
import seaborn as sns
import pickle 
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr, sem
import numpy as np

from gen_utils import shared_1000, shared_515, get_data

def normalize_datasets(dataset, mean=None, std_dev=None):
    
    if mean is None:
        mean = np.mean(dataset, axis=0)
    if std_dev is None:
        std_dev = np.std(dataset, axis=0)
        std_dev[std_dev == 0] = 1.0
    
    normalized_dataset = (dataset - mean) / std_dev

    return normalized_dataset, mean, std_dev

models = ['untrained', 'moco', 'alexnet', 'resnet50', 'vit', 'vit_resnet']
streams = ['ventral', 'lateral', 'dorsal']
rois = {'ventral': 5,
       'lateral': 6,
       'dorsal': 7}
subjs = [1, 2, 5, 7]
model_files = {'vit_resnet': 'feat_vit_base_resnet50_384_final.npy'}
              #'untrained': 'feat_alexnet_in1k_pool_random.npy', 
              #'moco': 'feat_resnet50_in1k_avgpool_moco.npy', 
              #'alexnet': 'feat_alexnet_in1k_pool.npy', 
              #'resnet50': 'feat_resnet50_in1k_avgpool.npy', 
              #'vit': 'feat_vit_base_patch16_224_final.npy'}

feat_dir = 'datasets/NSDstreams/features'
stimuli_dir = 'datasets/NSD_processed'
index_dir = '../COCO'

means = {model: [] for model in models}
errs = {model: [] for model in models}
all_dat = {model: [] for model in models}

ids_all = []
for sb in [1,2,5,7]: 
    ids_all.extend(np.load('../COCO/coco_ID_of_repeats_subj%02d.npy'%sb))
unique_ids = np.unique(ids_all)

for model, fname in model_files.items():
    for stream in streams:
        for sb in subjs:
            # get noise ceiling of NSD voxels
            _, nc = get_data(sb, rois[stream])
            
            # get subj's 10,000 images
            coco = np.load('../COCO//coco_ID_of_repeats_subj%02d.npy'%sb)
            match = []
            for i, id in enumerate(coco):
                match.append(np.where(unique_ids == id)[0][0])
            match = np.asarray(match)
    
            # load in ANN and brain data (voxel/unit x ???)
            voxel_dir = '%s/%s_visual_data/subj%d.npy' % (stimuli_dir, stream, sb)
            feature_dir = '%s/%s' % (feat_dir,fname)
            s_data = np.load(voxel_dir)
            m_data = np.load(feature_dir)[match]
                        
            test_idx = np.load('%s/splits_subj%02d.npy'%(index_dir,sb), allow_pickle = True).item()['test']
            train_idx = np.load('%s/splits_subj%02d.npy'%(index_dir,sb), allow_pickle = True).item()['train']
            
            # split img indices into train (8500) and test (1000)
            s_train = s_data[train_idx]
            m_train = m_data[train_idx] 
            s_test = s_data[test_idx]
            m_test = m_data[test_idx]
            
            predictor = RidgeCV(alphas=np.logspace(-4,4,9))
            predictor.fit(m_train, s_train)
            test_pred = predictor.predict(m_test)
                        
            vals1 = [pearsonr(test_pred[:,i], s_test[:,i])[0] for i in range(test_pred.shape[1])]
            vals2 = [pearsonr(test_pred[:,i], s_test[:,i])[0]/nc[i] for i in range(test_pred.shape[1])]
            save_path = '../encodings/linpred'
            np.save('%s/%s_subj%d_%s_decoding.npy'%(save_path,model,sb,stream), np.asarray(vals1))
            np.save('%s/%s_subj%d_%s_decoding_nc.npy'%(save_path,model,sb,stream), np.asarray(vals2))

