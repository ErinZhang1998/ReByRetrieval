import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wandb
import io 
from PIL import Image
import PIL
import copy

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

def plot_image_with_mask(epoch, cnt, idx_in_dataset, dataset, panel_name, key_name):
    idx_in_dataset = idx_in_dataset.astype(int)
    leng = len(idx_in_dataset)
    fig, axs = plt.subplots(1, leng, figsize=(leng * 10,20))  
    
    for i in range(leng):
        sample = dataset.idx_to_data_dict[idx_in_dataset[i]]
        rgb_all = mpimg.imread(sample['rgb_all_path'])
        mask = mpimg.imread(sample['mask_path'])
        masked_img = masked_image(rgb_all, mask)
        axs[i].imshow(masked_img)
        axs[i].set_title('{}_{}'.format(sample['obj_cat'], sample['obj_id']))

    final_img = plt_to_image(fig)
    log_key = '{}/{}_{}_{}'.format(panel_name, epoch, cnt, key_name)
    wandb.log({log_key: wandb.Image(final_img)}, step=cnt)
    plt.close()

def show_image_with_center(img, center):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(img)
    ax1.scatter(center[0], center[1],   marker=".", c='b', s=30)
    plt.show()
    plt.close()

def plot_predicted_image(cnt, test_loader, dataset_idx, pixel_pred, panel_name, scale_pred = None, scale_info = None):
    sample = test_loader.dataset.idx_to_data_dict[dataset_idx]
    img = mpimg.imread(sample['rgb_all_path'])
    center = copy.deepcopy(sample['object_center'].reshape(-1,))
    center[0] = img.shape[1] - center[0]

    center_pred0 = pixel_pred[0] * test_loader.dataset.img_w
    center_pred1 = pixel_pred[1] * test_loader.dataset.img_h

    fig = plt.figure(figsize=(10, 4))
    plt.imshow(img)
    plt.scatter(center[0], center[1],   marker=".", c='b', s=30)
    plt.scatter(center_pred0, center_pred1,   marker=".", c='r', s=30)
    if (scale_pred is not None) and (scale_info is not None):
        plt.title('gt: {}, pred: {}'.format(scale_info, scale_pred))

    final_img = plt_to_image(fig)
    
    if cnt < 0:
        wandb.log({'{}/{}'.format(panel_name, sample['sample_id']): wandb.Image(final_img)})
    else:
        wandb.log({'{}/{}'.format(panel_name, sample['sample_id']): wandb.Image(final_img)}, step=cnt)
    plt.close()

# def plot_predicted_image(pixel_pred, batch_idx, test_loader, scale_pred, scale_info, cnt):
#     batch_row = test_loader.batch_indices[batch_idx]
#     j_idx = np.random.choice(len(batch_row),1)[0]
#     j =  batch_row[j_idx]
#     sample = test_loader.dataset.idx_to_data_dict[j]
#     img = mpimg.imread(sample['rgb_all_path'])
#     center = copy.deepcopy(sample['object_center'].reshape(-1,))
#     center[0] = img.shape[1] - center[0]

#     pixel_pred = pixel_pred[j_idx]
#     center_pred0 = pixel_pred[0] * test_loader.dataset.img_w
#     center_pred1 = pixel_pred[1] * test_loader.dataset.img_h

#     fig = plt.figure(figsize=(10, 4))
#     # ax1 = fig.add_subplot(111)
#     plt.imshow(img)
#     plt.scatter(center[0], center[1],   marker=".", c='b', s=30)
#     plt.scatter(center_pred0, center_pred1,   marker=".", c='r', s=30)
#     plt.title('gt: {}, pred: {}'.format(scale_info[j_idx].item(), scale_pred[j_idx].item()))

#     final_img = plt_to_image(fig)
    
#     wandb.log({'image/test_image_{}'.format(sample['sample_id']): wandb.Image(final_img)}, step=cnt)
#     plt.close()

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





