import torch 
import numpy as np


def pariwise_distances(embeddings, squared=False):
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))
    squared_norm = torch.diagonal(dot_product)
    distances = torch.unsqueeze(squared_norm, 0) - 2.0 * dot_product + torch.unsqueeze(squared_norm, 1)
    distances = torch.maximum(distances, torch.zeros_like(distances))
    if not squared:
        mask = torch.isclose(distances, torch.zeros_like(distances), rtol=0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    
    return distances



def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dist = pariwise_distances(embeddings, squared=squared)

    anchor_positive_dist = torch.unsqueeze(pairwise_dist, dim=2)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, dim=1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    la_not_lp = labels != labels.T
    la_is_ln = labels == labels.T
    a = torch.arange(len(embeddings)).view(-1,1).to(embeddings.device)
    a_is_p = a == a.T
    la_not_lp = torch.unsqueeze(la_not_lp, dim=2)
    la_is_ln = torch.unsqueeze(la_is_ln, dim=1)
    mask_1 = torch.logical_or(la_not_lp, la_is_ln)
    mask_2 = torch.stack(len(embeddings)*[a_is_p], dim=0) 
    mask = torch.logical_not(torch.logical_or(mask_1,mask_2))
    mask = mask.float()

    triplet_loss = mask * triplet_loss
    triplet_loss = torch.maximum(triplet_loss, torch.zeros_like(triplet_loss))

    valid_triplets = torch.gt(triplet_loss, torch.ones_like(triplet_loss)*1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16) 
    # triplet_loss = torch.sum(triplet_loss)

    return triplet_loss


def pdist_torch(emb1, emb2):

    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)

    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


class BatchHardTripletSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def __call__(self, embeds, labels):
        dist_mtx = pdist_torch(embeds, embeds).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis = 1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis = 1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg