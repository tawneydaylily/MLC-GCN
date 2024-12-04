import torch
import numpy as np
import random

def mixup_data(x, nodes, y, alpha=1, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
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

