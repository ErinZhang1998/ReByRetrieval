import torch 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOSSES = {
    "l1_mean": nn.L1Loss(reduction="mean"), 
    "mse_mean": nn.MSELoss(reduction="mean"), 
    "ce" : nn.CrossEntropyLoss(),
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

def pariwise_distances(embeddings, squared=False):
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))
    squared_norm = torch.diagonal(dot_product)
    distances = torch.unsqueeze(squared_norm, 0) - 2.0 * dot_product + torch.unsqueeze(squared_norm, 1)
    # distances = torch.maximum(distances, torch.zeros_like(distances))
    distances = torch.max(distances, torch.zeros_like(distances))
    if not squared:
        mask = torch.isclose(distances, torch.zeros_like(distances), rtol=0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    
    return distances

def generate_mask(labels):
    '''
    labels: (N,1)
    mask: (N,N,N)

    False if any:
    1) label i != label j
    2) label i == label k
    3) i == j
    '''
    labels = labels + 1
    N = len(labels)
    la_not_lp = labels[None, :] != labels[:, None]
    la_is_ln = labels[None, :] == labels[:, None]
    la_not_lp = la_not_lp.view((N,N))
    la_is_ln = la_is_ln.view((N,N))
    mask1 = la_not_lp[:, :,None] + la_is_ln[:, None, :]

    ind_vec = torch.arange(N).view((-1,1))
    a_eq_p = (ind_vec[None, :] == ind_vec[:, None]).view((N,N))
    a_eq_p = a_eq_p[:,:,None]
    all_false = (torch.zeros(N) > 0).view((1,-1))
    all_false = all_false[None,:,:]
    mask2 = a_eq_p + all_false
    mask2 = mask2.to(mask1.device)

    mask = torch.logical_not(mask1 + mask2)
    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dist = pariwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist[:, :, None] #torch.unsqueeze(pairwise_dist, dim=2)
    anchor_negative_dist = pairwise_dist[:, None, :] #torch.unsqueeze(pairwise_dist, dim=1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = generate_mask(labels)
    # triplet_loss = mask * triplet_loss
    triplet_loss = F.relu(triplet_loss) * mask

    return mask, torch.sum(triplet_loss) / torch.sum(mask).item()