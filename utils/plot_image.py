import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wandb
import io 
from PIL import Image
import PIL
import copy

def denormalize_image(imgs, mean, std):
    for i in range(3):
        meani = mean[i]
        stdi = std[i]
        imgs[:,:,:,i] = (imgs[:,:,:,i] * stdi) + meani
    return imgs


def plt_to_image(fig_obj):
    buf = io.BytesIO()
    fig_obj.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

def masked_image(img, mask):
    mask = mask > 0
    for i in range(3):
        img[:,:,i] = img[:,:,i] * mask
    return img

def return_image_and_masked(dataset, idx):
    sample = dataset.idx_to_data_dict[idx]
    img = mpimg.imread(sample['rgb_all_path'])
    img_all = copy.deepcopy(img)
    mask = mpimg.imread(sample['mask_path'])
    masked_img = masked_image(img, mask)
    return img_all, masked_img


def plot_predicted_image(cnt, img_plot, pixel_pred_idx, pixel_gt_idx, panel_name='unknown', sample_id='unknown', scale_pred_idx = None, scale_gt_idx = None):    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_plot)
    plt.scatter(pixel_gt_idx[0], pixel_gt_idx[1],   marker=".", c='b', s=30)
    plt.scatter(pixel_pred_idx[0], pixel_pred_idx[1],   marker=".", c='r', s=30)
    if (scale_pred_idx is not None) and (scale_gt_idx is not None):
        plt.title('gt: {}, pred: {}'.format(scale_gt_idx, scale_pred_idx))

    if cnt >= 0:
        final_img = plt_to_image(fig)
        wandb.log({'{}/{}'.format(panel_name, sample_id): wandb.Image(final_img)}, step=cnt)
    plt.close()

'''
def plot_closest_shapes(dataset, q_ind, distance, indices, top_n = 10):
    order = torch.argsort(distance)
    i = 0
    q_dataset = indices[q_ind]
    qimg, qmasked_img = return_image_and_masked(dataset, q_dataset)
    qsample = dataset.idx_to_data_dict[q_dataset]
    
    fig, axs = plt.subplots(top_n+1, 2, figsize=(top_n * 2, 15))
    axs[0, 0].imshow(qimg)
    axs[0, 1].imshow(qmasked_img)
    axs[0, 0].set_title('{} {}'.format(qsample['obj_cat'], qsample['obj_id']))
    
    for j in range(len(order)):
        if i >= top_n:
            break 
        t_ind = order[j].item()
        if t_ind == q_ind:
            continue 
        t_dataset = indices[t_ind]
        timg, tmasked_img = return_image_and_masked(dataset, t_dataset)
        tsample = dataset.idx_to_data_dict[t_dataset]

        axs[i+1, 0].imshow(timg)
        axs[i+1, 1].imshow(tmasked_img)
        axs[i+1, 0].set_title('{} {}'.format(tsample['obj_cat'], tsample['obj_id']))

        i+=1
    
    plt.show()
    plt.close()
'''




