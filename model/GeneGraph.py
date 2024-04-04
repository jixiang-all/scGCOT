import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv


class GeneGraph(nn.Module):
    def __init__(self, X, hidden_dim=128, latent_dim=15, adj_dim=32):
        super(GeneGraph, self).__init__()
        self.cell_dim = X.shape[1]  # genes number
        self.gene_dim = X.shape[0]  # cells number
        self.conv3 = TransformerConv(self.gene_dim, hidden_dim)
        self.conv4 = TransformerConv(hidden_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, adj_dim)
        self.bilinear2 = nn.Bilinear(adj_dim, adj_dim, self.cell_dim)

    def forward(self, GeneX, GeneEdgeIndex):
        GeneX = F.dropout(GeneX, p=0.2, training=self.training)
        GeneX = F.relu(self.conv3(GeneX, GeneEdgeIndex))
        z_mean2 = self.conv4(GeneX, GeneEdgeIndex)
        h2 = F.dropout(self.linear2(z_mean2), p=0.2, training=self.training)
        CellDecoder = torch.sigmoid(self.bilinear2(h2, h2))
        return CellDecoder, z_mean2
