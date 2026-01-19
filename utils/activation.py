import torch.nn.functional as F
import torch.nn as nn


def getActivation(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")