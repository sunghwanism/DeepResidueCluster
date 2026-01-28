import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from utils.activation import getActivation

class DenseGCN(nn.Module):
    def __init__(self, in_ft, out_ft_list, activation):
        super(DenseGCN, self).__init__()
        self.proj = nn.Linear(in_ft, out_ft_list[0])
        self.conv1 = SAGEConv(in_ft, out_ft_list[0], aggr='mean')
        self.conv2 = SAGEConv(out_ft_list[0] + out_ft_list[0], out_ft_list[1], aggr='mean')
        self.conv3 = SAGEConv(out_ft_list[0] + out_ft_list[0] + out_ft_list[1], out_ft_list[2], aggr='mean')

        self.act = getActivation(activation)

    def forward(self, x, edge_index):
        x_p = self.act(self.proj(x)) # (N, out_ft_list[0])
        
        # Layer 1
        h1 = self.act(self.conv1(x, edge_index))# (N, out_ft_list[0])
        
        # Layer 2
        in2 = torch.cat([x_p, h1], dim=1) # (N, out_ft_list[0] + out_ft_list[0])
        h2 = self.act(self.conv2(in2, edge_index))
        
        # Layer 3
        in3 = torch.cat([x_p, h1, h2], dim=1) # (N, out_ft_list[0] + out_ft_list[0] + out_ft_list[1])
        h3 = self.act(self.conv3(in3, edge_index))
        
        return h3
