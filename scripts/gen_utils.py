import numpy as np


def get_data(subj = 1, roi = 5):
    splits = np.load('/mindhive/nklab4/users/mkhosla/NSD/prf_models/data/splits_subj%02d.npy'%subj, allow_pickle = True).item()
    data = np.load('/mindhive/nklab4/shared/datasets/NSD_processed/voxel_data/averaged_cortical_responses_zscored_by_run_subj%02d.npy'%subj, mmap_mode = 'r')
    mask = np.isin(np.load('/mindhive/nklab4/shared/datasets/NSD_processed/masks/subj%d/roi_1d_mask_subj%02d_streams.npy'%(subj, subj)), [roi])
    if subj not in [1,2,5,7]:
        return data[splits['test']][:,mask]
    else:
        ys = np.load('/mindhive/nklab4/shared/datasets/NSD_processed/voxel_data/subj%02d_test_singletrial_all.npy' % subj)[:,:,mask]
        ys = np.moveaxis(ys, 2, 0)
        sn2 = np.var(ys, 2, ddof = 1).mean(1)
        ss2 = 1 - sn2
        ss2[ss2<0] = 0
        nc = np.sqrt((ss2/(ss2+sn2/3)))
        
        data = data[splits['test']][:,mask] #
        # is returning negative nc ok ??
        #return data[:, nc > 0], nc[nc > 0]
        return data, nc
    
def rsquared(predicted, actual):
    '''
    coefficient of determination (NOT correlation coefficient)
    R-squared metric (from finzi)
    '''
    a_mean = actual.mean()
    num = np.linalg.norm(actual - predicted)**2
    denom = np.linalg.norm(actual - a_mean)**2
    return 1 - num / denom

def reject_outliers(data, m=2.):
    '''
    also from finzi
    '''
    data = np.asarray(data)
    d = np.abs(data - np.nanmean(data))
    mdev = np.nanmean(d)
    s = d / (mdev if mdev else 1.)
    return data[s<m]

def unique_indices():
    '''
    Return 37000 unique indices of the NSD images
    '''
    ids_all = []
    for sb in [1,2,5,7]: 
        ids_all.extend(np.load('/mindhive/nklab4/users/mkhosla/NSD/data/coco_ID_of_repeats_subj%02d.npy'%sb))
    unique_ids = np.unique(ids_all)
    
    return unique_ids

def shared_1000():
    '''
    Return the indices of the 37000 NSD images that are among the 1000 shared
    '''
    unique_ids = unique_indices()

    # coco contains the id of the shared 1000 images
    splits = np.load('/mindhive/nklab4/users/mkhosla/NSD/prf_models/data/splits_subj01.npy', allow_pickle = True).item()
    coco = np.load('/mindhive/nklab4/users/mkhosla/NSD/data/coco_ID_of_repeats_subj01.npy')[splits['test']] 
    match = []
    for i, id in enumerate(coco):
        match.append(np.where(unique_ids == id)[0][0])
    # match contains the indices in unique_ids that are the shared 1000 images
    match = np.asarray(match)
    
    return match

def shared_515():
    '''
    Reutrn the indices of the 37000 NSD images that are among the 515 shared
    '''
    splits_1 = np.load('/mindhive/nklab4/users/mkhosla/NSD/prf_models/data/splits_subj01.npy', allow_pickle = True).item()   
    coco_1 = np.load('/mindhive/nklab4/users/mkhosla/NSD/data/coco_ID_of_repeats_subj01.npy' )[splits_1['test']]
    
    splits_3 = np.load('/mindhive/nklab4/users/mkhosla/NSD/prf_models/data/splits_subj03.npy', allow_pickle = True).item()   
    coco_3 = np.load('/mindhive/nklab4/users/mkhosla/NSD/data/coco_ID_of_repeats_subj03.npy')[splits_3['test']]

    shared515 = []
    for i,id in enumerate(coco_3): 
        idx = np.where(coco_1==id)[0][0] 
        shared515.append(idx)
    shared515 = np.asarray(shared515)
    
    return shared515