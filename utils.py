import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from nimfa.utils.linalg import *

def save_df_to_npz(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

def save_df_to_text(obj, filename):
    obj.to_csv(filename, sep='\t')

def load_df_from_npz(filename):
    with np.load(filename, allow_pickle=True) as f:
        obj = pd.DataFrame(**f)
    return obj

def cophcorr(A):
    # upper diagonal elements of consensus
    avec = np.array([A[i, j] for i in range(A.shape[0] - 1)
                    for j in range(i + 1, A.shape[1])])
    # consensus entries are similarities, conversion to distances
    Y = 1 - avec
    Z = linkage(Y, method='average')
    #print(Z, Z.max(), Z.min())
    #print('Computed z')
    # cophenetic correlation coefficient of a hierarchical clustering
    # defined by the linkage matrix Z and matrix Y from which Z was
    # generated
    return cophenet(Z, Y)[0]

def sparseness(x):
            eps = np.finfo(x.dtype).eps if 'int' not in str(x.dtype) else 1e-9
            x1 = sqrt(x.shape[0]) - (abs(x).sum() + eps) / \
                (sqrt(multiply(x, x).sum()) + eps)
            x2 = sqrt(x.shape[0]) - 1
            return x1 / x2
        
def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img

def frame_image_blue(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.ones((b+ny+b, b+nx+b, img.shape[2]))*0.5
    elif img.ndim == 2: # grayscale image
        framed_img = np.ones((b+ny+b, b+nx+b))*0.5
    framed_img[b:-b, b:-b] = img
    return framed_img