import numpy as np
from tqdm import tqdm

def connectivity_matrix(model='moco', num_runs=50):
    for sb in [1,2,5,7]:
        for idx in tqdm(range(num_runs)):
            # load in run data
            data = np.load('model_nmf_results/%s/%s_s%d_saved_runs/%s_iter1_run%d.npy' % (model, model, sb, model, idx), allow_pickle=True).item()

            # get component responses
            # resp = data['data_transformed'][match]
            
            coco = np.load('common_indices/common_indices%d.npy'%sb)
            resp = data['data_transformed'][coco]

            # Calculate the top component for each image
            top_component = np.argmax(resp, axis=1)

            if(idx == 0):
                connectivity = np.zeros((top_component.shape[0], top_component.shape[0]), dtype=int)

            # Iterate through each image
            for i in range(top_component.shape[0]):
                # Iterate through other images
                for j in range(i, top_component.shape[0]):
                    connectivity[i, j] += (top_component[i] == top_component[j])
                    connectivity[j, i] += (top_component[i] == top_component[j])

        # save averaged connectivity matrix
        connectivity = connectivity / num_runs
        np.save('figs/model_conn/%s_s%d_icm.npy' % (model, sb), connectivity)

# layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
models = ['untrained', 'moco', 'alexnet', 'resnet50', 'vit']
# models = ['dino', 'simclr']
# models = [f'dino-layer{_l}' for _l in range(12)]
# models = [f'swin-layer{_l:02d}' for _l in range(24)]
for model in tqdm(models):
    connectivity_matrix(model=model)
