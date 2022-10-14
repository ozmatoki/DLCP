import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from scipy import io, ndimage
from glob import glob
from os import path
import torch.nn as nn


class DataSet(object):
    def __init__(self, data_path):
        self._data_path = data_path
        
    def __len__(self):
        return 22#len(glob(path.join(self._data_path, '*.mat')))
        
    def __getitem__(self, i):
        input_np = io.loadmat(path.join(self._data_path, f'data{i}.mat'))['input']
        input = torch.from_numpy(input_np).float()
        input[input == float("-Inf")] = -6
        input2a_np = io.loadmat(path.join(self._data_path, f'data{i}.mat'))['input2a']
        input2a = nn.ReLU(torch.from_numpy(input2a_np).float()+1.5).inplace.data
        #input2a[input2a == float("-Inf")] = -15
        input2b_np = io.loadmat(path.join(self._data_path, f'data{i}.mat'))['input2b']
        input2b = torch.from_numpy(input2b_np).float()
        tag_np = io.loadmat(path.join(self._data_path, f'data{i}.mat'))['tag']
        tag = torch.from_numpy(tag_np).long()
        normalize = transforms.Normalize((torch.mean(input),), (torch.std(input),))
        '''
        print(torch.min(torch.min(input[ :, :], 0)[0], 0)[0])
        print(torch.min(torch.min(input2a[ :, :], 0)[0], 0)[0])
        print(torch.min(torch.min(input2b[ :, :], 0)[0], 0)[0])
        print(torch.min(torch.min(tag[ :, :], 0)[0], 0)[0])
        '''
        return normalize(torch.stack([input, input2a, input2b], dim=0)), tag[1024-84:1024+85,1024-84:1024+85]
