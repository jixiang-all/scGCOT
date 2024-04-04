import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, GCNConv, GATConv


class CellGraph(nn.Module):
    def __init__(self, X, hidden_dim=128, latent_dim=15, adj_dim=32):
        super(CellGraph, self).__init__()
        self.cell_dim = X.shape[1]  # genes number
        self.gene_dim = X.shape[0]  # cells number
        self.conv1 = TransformerConv(self.cell_dim, hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, latent_dim)
        self.linear1 = nn.Linear(latent_dim, adj_dim)
        self.bilinear1 = nn.Bilinear(adj_dim, adj_dim, self.gene_dim)

    def forward(self, CellX, CellEdgeIndex):
        CellX = F.dropout(CellX, p=0.2, training=self.training)
        CellX = F.relu(self.conv1(CellX, CellEdgeIndex))
        z_mean = self.conv2(CellX, CellEdgeIndex)
        h = F.dropout(self.linear1(z_mean), p=0.2, training=self.training)
        CellDecoder = torch.sigmoid(self.bilinear1(h, h))
        return CellDecoder, z_mean
