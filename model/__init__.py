# -- coding: utf-8 --
import torch.nn as nn
import torch
data = torch.rand([3, 3])
data1 = torch.unsqueeze(data, dim=1)
print("data_size: ", data.shape)
print("data1_size: ", data1.shape)
print("data1: ", data1)