import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt

def saliency(heatmap):
    plt.close('all')
    fig = plt.figure(figsize=(9, 7))
    a = plt.imshow(heatmap, cmap="Greens", aspect='auto', interpolation='nearest')
    _ = plt.xlabel("time steps")
    _ = plt.ylabel("nodes")
    _ = plt.tight_layout()
    plt.colorbar()
    
def threshold_saliency(heatmap):
    cmap = plt.cm.colors.ListedColormap(['lightpink', '#00CCFF'])
    bounds = [0, 0.5, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    plt.close('all')
    fig = plt.figure(figsize=(9, 7))
    a = plt.imshow(heatmap, cmap=cmap, aspect='auto', norm=norm, interpolation='nearest')
    _ = plt.xlabel("time steps")
    _ = plt.ylabel("nodes")
    _ = plt.tight_layout()
    cbar = plt.colorbar(a, ticks=[0.25, 0.75])
    cbar.set_ticklabels(['False', 'True'])
    
    
def normalize_tensor(data):
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def load_saliency(path):
    mask = np.load(path)['arr_0']
    mask= mask[0, 0, :, :]
    return mask