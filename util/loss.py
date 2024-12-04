import torch
import numpy as np
import random

def mixup_data(x, nodes, y, alpha=1, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_nodes, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a = torch.argmax(y_a, dim=1)
    y_b = torch.argmax(y_b, dim=1)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):
    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_0 = torch.sum(y_1[:,0])
    sum_1 = torch.sum(y_1[:,1])
    loss = 0.0

    if sum_0.item() > 0:
        center_0 = torch.matmul(y_1[:,0], matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_1[:,0], diff_0)/(sum_0*roi_num*roi_num)
    if sum_1.item() > 0:
        center_1 = torch.matmul(y_1[:,1], matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1[:,1], diff_1)/(sum_1*roi_num*roi_num)
    if sum_0.item() > 0 and sum_1.item() > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss

def multi_mixup_cluster_loss(matrixs, y_a, y_b, lam, intra_weight=2):
    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()

    bz, roi_num, _ = matrixs.shape
    loss = 0.0
    matrixs = matrixs.reshape((bz, -1))
    center_list = []
    for i in range(y_1.shape[1]):
        sum = torch.sum(y_1[:,i])
        if sum.item() > 0:
            center = torch.matmul(y_1[:,0], matrixs)/sum
            diff = torch.norm(matrixs-center, p=1, dim=1)
            loss += torch.matmul(y_1[:,0], diff)/(sum*roi_num*roi_num)
            center_list.append(center)


    return loss
