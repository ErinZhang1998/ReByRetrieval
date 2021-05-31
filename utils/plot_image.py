import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wandb
import io 
from PIL import Image
import PIL
import os 
import copy

def plt_to_image(fig_obj):
    buf = io.BytesIO()
    fig_obj.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

def masked_image(img, mask):
    img_all = copy.deepcopy(img)
    mask = mask > 0
    for i in range(3):
        img_all[:,:,i] = img_all[:,:,i] * mask
    return img_all

def plot_predicted_image(cnt, img_plot, pixel_pred_idx, pixel_gt_idx, enable_wandb = False, image_type_name='unknown', image_dir = None, sample_id='unknown', scale_pred_idx = None, scale_gt_idx = None):    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(np.asarray(img_plot))
    plt.scatter(pixel_gt_idx[0], pixel_gt_idx[1],   marker=".", c='b', s=30)
    plt.scatter(pixel_pred_idx[0], pixel_pred_idx[1],   marker=".", c='r', s=30)
    if (scale_pred_idx is not None) and (scale_gt_idx is not None):
        plt.title('gt: {}, pred: {}'.format(scale_gt_idx, scale_pred_idx))

    if cnt >= 0:
        image_name = '{}_{}'.format(sample_id, cnt)
        if enable_wandb:
            final_img = plt_to_image(fig)
            wandb.log({'{}/{}'.format(image_type_name, image_name): wandb.Image(final_img)}, step=cnt)
        else:
            image_path = os.path.join(image_dir, "{}_{}.png".format(image_type_name, image_name))
            plt.savefig(image_path)
    plt.close()



