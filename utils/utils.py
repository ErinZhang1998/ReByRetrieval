import torch 
import numpy as np
from incat_dataset import *
from torch.utils.data import DataLoader
from torch import linalg as LA

class ARGS(object):
    batch_size = 32
    test_batch_size=32
    epochs = 15
    lr = 0.001
    gamma = 0.7
    log_every = 100
    val_every = 100
    save_at_end = False
    save_freq=-1
    use_cuda = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert '__' not in k and hasattr(self, k), "invalid attribute!"
            assert k != 'device', "device property cannot be modified"
            setattr(self, k, v)
        
    def __repr__(self):
        repr_str = ''
        for attr in dir(self):
            if '__' not in attr and attr !='use_cuda':
                repr_str += 'args.{} = {}\n'.format(attr, getattr(self, attr))
        return repr_str
    
    @property
    def device(self):
        return torch.device("cuda" if self.use_cuda else "cpu")


def get_data_loader(dir_root, shape_categories_file_path = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/taxonomy_tabletop_small_keys.txt', \
    train=True, batch_size=64, split='train', dataset_size = 227):
    dataset = InCategoryClutterDataset(split, dataset_size, dir_root, shape_categories_file_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
    )
    return loader

def eval_dataset(model, device, test_loader):
    gt, pred, valid = [],[],[]
    scale_loss, orient_loss, pixel_loss = [],[],[]
    
    
    with torch.no_grad():
        
        for image, scale_info, orient_info, pixel_info, cat_info in test_loader:
            model = model.to(device)
            # image, scale_info, orient_info, pixel_info, cat_info = image.to(args.device), scale_info.to(args.device), orient_info.to(args.device), pixel_info.to(args.device), cat_info.to(args.device)
            image = image.to(device)
            img_embed, pose_pred = model(image)
            img_embed = img_embed.cpu()
            pose_pred = pose_pred.cpu()
            pose_pred = pose_pred.float().detach()

            scale_loss.append(torch.square(LA.norm(pose_pred[:,:1] - scale_info)).item())
            orient_loss.append(torch.square(LA.norm(pose_pred[:,1:5] - orient_info, axis=1)).numpy())
            pixel_loss.append(torch.square(LA.norm(pose_pred[:,5:] - pixel_info, axis=1)).numpy())
            torch.cuda.empty_cache()
    
    total_samples = len(test_loader.dataset)
    scale_loss = np.sum(scale_loss) / total_samples
    orient_loss = np.sum(np.hstack(orient_loss)) / total_samples
    pixel_loss = np.sum(np.hstack(pixel_loss)) / total_samples
    return scale_loss, orient_loss, pixel_loss


def pdist_torch(emb1, emb2):

    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    # dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    
    # dist_mtx = torch.addmm(dist_mtx, emb1, emb2.t(), *, beta=1, alpha=-2)

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