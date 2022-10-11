import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from scipy import io, ndimage
from glob import glob
from os import path


class DataSet(object):
    def __init__(self, data_path):
        self._data_path = data_path
        
    def __len__(self):
        return 20#len(glob(path.join(self._data_path, '*.mat')))
        
    def __getitem__(self, i):
        input_np = io.loadmat(path.join(self._data_path, f'data{i}.mat'))['input']
        input = torch.from_numpy(input_np).float()
        tag_np = io.loadmat(path.join(self._data_path, f'data{i}.mat'))['tag']
        tag = torch.from_numpy(tag_np).long()
        normalize = transforms.Normalize((torch.mean(input),), (torch.std(input),))
        return normalize(torch.stack([input], dim=0)), tag[1024-128:1024+128,1024-128:1024+128]
