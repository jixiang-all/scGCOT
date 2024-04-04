import torch
from ot.bregman import empirical_sinkhorn_divergence
from torch import nn

from module.ZINBLoss import ZINBLoss

Train_weight = [1, 0.3, 0.3, 1]
Finetune_weight =  [0.3, 0, 0, 1.5, 2]


def Train_loss(C_out, G_out, Cell_X, CellAdj, GeneAdj, count, Mat, criterion, pi, disp, mean):
    Cell_loss = criterion(CellAdj, C_out)
    Gene_loss = criterion(GeneAdj, G_out)
    Mat_loss = criterion(Mat.cpu(), torch.tensor(count, dtype=torch.float))
    zinb = ZINBLoss()
    zinbloss = zinb(Cell_X.x, mean, disp, pi)
    W_a, W_x, W_d, W_z = Train_weight
    loss = W_a * Cell_loss + W_x * Gene_loss + W_d * Mat_loss + W_z * zinbloss
    Cell_loss = Cell_loss.cpu().detach().numpy()
    Gene_loss = Gene_loss.cpu().detach().numpy()
    Mat_loss = Mat_loss.cpu().detach().numpy()
    result = [Cell_loss, Gene_loss, Mat_loss, zinbloss, loss]
    return result


def Finetune_loss(C_out, G_out, Cell_X, CellAdj, GeneAdj, count, Mat, p, q, criterion, pi, disp, mean):
    res = Train_loss(C_out, G_out, Cell_X, CellAdj, GeneAdj, count, Mat, criterion,
                             pi, disp, mean)
    [Cell_loss, Gene_loss, Mat_loss, zinbloss, loss] = res
    W_a, W_x, W_d, W_z, W_s = Finetune_weight
    W_z = torch.tensor(W_z,dtype=torch.float)
    W_s = torch.tensor(W_s,dtype=torch.float)

    sinkhorn_divergence = empirical_sinkhorn_divergence(q, p, 1, numIterMax=2000)[0]

    loss = W_a * Cell_loss + W_x * Gene_loss + W_d * Mat_loss + W_z * zinbloss  + W_s * sinkhorn_divergence \
        + torch.log(torch.sqrt(1/(2*W_z))* torch.sqrt(1/(2*W_s)))
    result = [Cell_loss, Gene_loss, Mat_loss, zinbloss, sinkhorn_divergence, loss]
    return result, Finetune_weight
