import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gen_utils
import pickle 
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from simclr_converter import resnet_wider

class DAT(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure it's in RGB format
        if self.transform:
            image = self.transform(image)
        return image

# Set transform function
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

all_coco_ids = gen_utils.unique_indices()
stimuli_dir = "./NSD_processed/images/" 

IMG_PATHS = [os.path.join(stimuli_dir, f'{id}.jpg') for id in all_coco_ids] 
dataset = DAT(IMG_PATHS, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # changed bs

def extract(model, model_name, output_dir):
    model.to(device)

    # hook function to extract features
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    hook_list = []
    for name, layer in model.named_modules():
        if '.mlp.4' in name: # relu for simclr, fc2 for dino, mlp.4 for swin
            hook_handle = layer.register_forward_hook(get_features(name))
            hook_list.append(hook_handle)

    # =================================
    # actually extract the features now
    # =================================
    FEATS = {}
    features = {}  
            
    model.eval()
    for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device)
        preds = model(inputs)
        
        for layer, feat in features.items():
            # =================================
            # how to extract the features (different for CNN vs ViT vs swin)
            # =================================
            # pooled = F.adaptive_avg_pool2d(features[layer], (1,1)).cpu().numpy()
            # pooled = feat[:,0,:].cpu().numpy()
            feats = features[layer].permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(feats, (1,1)).cpu().numpy()

            if idx==0:
                FEATS[layer] = pooled
            else:
                FEATS[layer] = np.concatenate((FEATS[layer], pooled),axis=0)
        if idx==0:
            PREDS = preds.detach().cpu().numpy()
        else:
            PREDS = np.concatenate((PREDS, preds.detach().cpu().numpy()),axis=0)
            
    # clean up
    for hook in hook_list:
        hook.remove()

    with open(os.path.join(output_dir, f'{model_name}_features.pkl'), 'wb') as f:
        pickle.dump(FEATS, f)

# ================
# pretrained dino
# ================
# model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
# state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth')
# state_dict = torch.load('/om2/user/amarvi/NSDstreams/dino_vitbase16_pretrain_full_checkpoint.pth', map_location='cpu')
# state_dict = state_dict['student']
# state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
# msg = model.load_state_dict(state_dict, strict=False)
# print(msg)

# ================
# pratrained simclr
# ================
# model = resnet_wider.resnet50x1()
# sd = '/om2/user/amarvi/NSDstreams/simclr_converter/resnet50-1x.pth'
# sd = torch.load(sd, map_location='cpu')
# model.load_state_dict(sd['state_dict'])

# ================
# pratrained swin
# ================
model = torchvision.models.get_model("swin_b", weights='DEFAULT')

print('model successfully loaded !! ')

output_dir = f'./features'
# os.makedirs(output_dir, exist_ok=True)
extract(model, 'swin', output_dir)
