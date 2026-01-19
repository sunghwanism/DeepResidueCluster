import torch
import torch.nn as nn
from ..layers import DenseGCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, agg_method):
        super(DGI, self).__init__()
        self.gcn = DenseGCN(n_in, n_h, activation, agg_method)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, x, shuf_x, edge_index, batch=None, samp_bias1=None, samp_bias2=None):
        h_1 = self.gcn(x, edge_index)

        c = self.read(h_1, batch)
        c = self.sigm(c)

        h_2 = self.gcn(shuf_x, edge_index)

        if batch is None:
            c_expanded = c.expand_as(h_1)
        else:
            c_expanded = c[batch]

        ret = self.disc(c_expanded, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    def embed(self, x, edge_index, batch=None):
        h_1 = self.gcn(x, edge_index)
        c = self.read(h_1, batch)

        return h_1.detach(), c.detach()
