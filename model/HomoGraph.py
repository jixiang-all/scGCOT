import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.CellGraph import CellGraph
from model.GeneGraph import GeneGraph
from module.ClusteringLayer import ClusteringLayer


class HomoGraph(nn.Module):
    def __init__(self, X, Cell, n_clusters, hidden_dim=128, latent_dim=15):
        super(HomoGraph, self).__init__()
        self.cell_dim = X.shape[1]  # genes number
        self.gene_dim = X.shape[0]  # cells number
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder1 = CellGraph(X, hidden_dim, latent_dim)
        self.encoder2 = GeneGraph(X, hidden_dim, latent_dim)
        self.classify = nn.Linear(latent_dim, 2)
        self.pi_decoder = nn.Linear(self.cell_dim, self.cell_dim)
        self.disp_decoder = nn.Linear(self.cell_dim, self.cell_dim)
        self.mean_decoder = nn.Linear(self.cell_dim, self.cell_dim)
        self.cluster_model = ClusteringLayer(n_clusters, latent_dim).to(self.device)

    def forward(self, Cell, Gene):
        # self.encoder2 = self.encoder2.to(self.device)
        CellDecoder, z_mean = self.encoder1(Cell.x, Cell.edge_index)
        GeneDecoder, z_mean2 = self.encoder2(Gene.x, Gene.edge_index)
        Matrix = torch.matmul(z_mean, z_mean2.transpose(0, 1))
        decx = Matrix
        pi = torch.sigmoid(self.pi_decoder(decx))
        disp = torch.clamp(F.softplus(self.disp_decoder(decx)), min=1e-4, max=1e4)
        mean = torch.clamp(torch.exp(self.mean_decoder(decx)), min=1e-5, max=1e6)
        return CellDecoder, GeneDecoder, Matrix, z_mean, pi, disp, mean

    def fintune(self, Cell, Gene, epoch, p, n_update=8):
        # self.encoder2 = self.encoder2.to(self.device)
        CellDecoder, z_mean = self.encoder1(Cell.x, Cell.edge_index)
        GeneDecoder, z_mean2 = self.encoder2(Gene.x, Gene.edge_index)
        Matrix = torch.matmul(z_mean, z_mean2.transpose(0, 1))
        decx = Matrix
        pi = torch.sigmoid(self.pi_decoder(decx))
        disp = torch.clamp(F.softplus(self.disp_decoder(decx)), min=1e-4, max=1e4)
        mean = torch.clamp(torch.exp(self.mean_decoder(decx)), min=1e-5, max=1e6)
        if epoch % n_update == 0:
            q = self.cluster_model(z_mean)
            p = self.target_distribution(q).detach()
        q_out = self.cluster_model(z_mean)
        return CellDecoder, GeneDecoder, Matrix, q_out, p, pi, disp, mean

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

