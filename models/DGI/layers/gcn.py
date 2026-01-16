import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DenseGCN(nn.Module):
    def __init__(self, in_ft, out_ft, activation):
        super(DenseGCN, self).__init__()
        self.act = activation
        
        # Projection for skip connections
        self.proj = nn.Linear(in_ft, out_ft)
        
        # Layer 1
        self.conv1 = GCNConv(in_ft, out_ft)
        
        # Layer 2: Input is concatenation of projected X (out_ft) and H1 (out_ft)
        self.conv2 = GCNConv(out_ft + out_ft, out_ft)
        
        # Layer 3: Input is concatenation of projected X (out_ft), H1 (out_ft), and H2 (out_ft)
        self.conv3 = GCNConv(out_ft + 2 * out_ft, out_ft)

    def forward(self, x, edge_index):
        # Projected input for skip connections
        x_p = self.act(self.proj(x))
        
        # Layer 1
        h1 = self.act(self.conv1(x, edge_index))
        
        # Layer 2
        in2 = torch.cat([x_p, h1], dim=1)
        h2 = self.act(self.conv2(in2, edge_index))
        
        # Layer 3
        in3 = torch.cat([x_p, h1, h2], dim=1)
        h3 = self.act(self.conv3(in3, edge_index))
        
        return h3
