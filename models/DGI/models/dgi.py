import torch
import torch.nn as nn
from layers import DenseGCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = DenseGCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, x, shuf_x, edge_index, batch=None, samp_bias1=None, samp_bias2=None):
        h_1 = self.gcn(x, edge_index)

        c = self.read(h_1, batch)
        c = self.sigm(c)

        h_2 = self.gcn(shuf_x, edge_index)

        # Expand c to match h_1 and h_2 shapes
        if batch is None:
            # Assuming single graph, c is (1, n_h), h_1 is (N, n_h)
            c_expanded = c.expand_as(h_1)
        else:
            # c is (num_graphs, n_h), map to nodes using batch from PyG
            c_expanded = c[batch]

        # Pass expanded c to discriminator
        ret = self.disc(c_expanded, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, x, edge_index, batch=None):
        h_1 = self.gcn(x, edge_index)
        c = self.read(h_1, batch)

        return h_1.detach(), c.detach()
