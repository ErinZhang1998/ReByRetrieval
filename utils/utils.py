import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wandb
import io 
from PIL import Image
import PIL
import copy

class Struct:
    '''The recursive class for building and representing objects with.'''
    def __init__(self, obj):
        self.obj_dict = obj
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)
    
    def __getitem__(self, val):
        return self.__dict__[val]
    
    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))


def masked_image(img, mask):
    mask = mask > 0
    for i in range(3):
        img[:,:,i] = img[:,:,i] * mask
    return img

def plt_to_image(fig_obj):
    buf = io.BytesIO()
    fig_obj.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

def plot_image_with_mask(epoch, cnt, idx_in_dataset, dataset, mask_name):
    idx_in_dataset = idx_in_dataset.cpu().numpy().astype(int)
    fig, axs = plt.subplots(1, 3, figsize=(20,20))  
    
    for i in range(3):
        sample = dataset.idx_to_data_dict[idx_in_dataset[i]]
        rgb_all = mpimg.imread(sample['rgb_all_path'])
        mask = mpimg.imread(sample['mask_path'])
        valid = np.sum(mask > 0)
        masked_img = masked_image(rgb_all, mask)
        axs[i].imshow(masked_img)
        axs[i].set_title('{}_{}'.format(sample['obj_cat'], sample['obj_id']))
    
    # buf = io.BytesIO()
    # fig.savefig(buf)
    # buf.seek(0)
    # final_img = Image.open(buf)
    final_img = plt_to_image(fig)
    log_key = 'triplet_pairs/{}_{}_{}_{}-{}-{}'.format(mask_name, epoch, cnt, idx_in_dataset[0], idx_in_dataset[1], idx_in_dataset[2])
    wandb.log({log_key: wandb.Image(final_img)}, step=cnt)
    plt.close()

def plot_predicted_image(pixel_pred, batch_idx, test_loader, scale_pred, scale_info, cnt):
    batch_row = test_loader.batch_indices[batch_idx]
    j_idx = np.random.choice(len(batch_row),1)[0]
    j =  batch_row[j_idx]
    sample = test_loader.dataset.idx_to_data_dict[j]
    img = mpimg.imread(sample['rgb_all_path'])

    center = copy.deepcopy(sample['object_center'].reshape(-1,))
    center[0] = img.shape[1] - center[0]

    pixel_pred = pixel_pred[j_idx]
    center_pred0 = pixel_pred[0] * test_loader.dataset.img_w
    center_pred1 = pixel_pred[1] * test_loader.dataset.img_h

    fig = plt.figure(figsize=(10, 4))
    # ax1 = fig.add_subplot(111)
    plt.imshow(img)
    plt.scatter(center[0], center[1],   marker=".", c='b', s=30)
    plt.scatter(center_pred0, center_pred1,   marker=".", c='r', s=30)
    plt.title('gt: {}, pred: {}'.format(scale_info[j_idx].item(), scale_pred[j_idx].item()))
    # print(pixel_pred, center_pred0, center_pred1)

    final_img = plt_to_image(fig)
    
    wandb.log({'image/test_image_{}'.format(sample['sample_id']): wandb.Image(final_img)}, step=cnt)
    plt.close()
    # wandb.log({"image/test_image": [wandb.Image(image, caption=rgb_all_path)]})
    # wandb.log({"test/test_scale_pred": scale_pred[j].item(), 'test/test_scale_gt': scale_info[j].item()})
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # image = PIL.Image.open(buf)
    # image = ToTensor()(image) #.unsqueeze(0)
    # return image, j_idx, sample['rgb_all_path']






