import numpy as np 
import copy 
import utils.transforms as utrans
import PIL 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils.plot_image as uplot 

def sample_ids_to_cat_and_id(dataset, sample_ids):
    obj_cat = []
    obj_id = []
    for sample_id in list(sample_ids):
        idx = dataset.sample_id_to_idx[sample_id]
        sample = dataset.idx_to_data_dict[idx]
        obj_cat.append(sample['obj_cat'])
        obj_id.append(sample['obj_id'])
    
    return np.array(obj_cat), np.array(obj_id)

def sample_ids_to_pixel_left_ratio(dataset, sample_ids):
    vec = np.zeros(len(sample_ids))
    for i,sample_id in enumerate(list(sample_ids)):
        idx = dataset.sample_id_to_idx[sample_id]
        sample = dataset.idx_to_data_dict[idx]
        vec[i] = sample['pix_left_ratio']
    return vec

def transform_pixels_in_crop_img(dataset, pixel_pred, pixel_gt, sample_ids, area_types):
    pixel_pred_scaled = copy.deepcopy(pixel_pred) 
    pixel_gt_scaled = copy.deepcopy(pixel_gt) 
    for i,(pred,gt,sample_id,area_type) in enumerate(zip(copy.deepcopy(pixel_pred), copy.deepcopy(pixel_gt), sample_ids, area_types)):
        pred *= dataset.size 
        gt *= dataset.size
        
        idx = dataset.sample_id_to_idx[sample_id]
        sample = dataset.idx_to_data_dict[idx]
        corners = copy.deepcopy(sample['scene_corners'])
        crop_trans = utrans.CropArea(corners)
        cropped_w = crop_trans.x1 - crop_trans.x0
        cropped_h = crop_trans.y1 - crop_trans.y0
        
        patch_size = int(dataset.size * 0.5)
        if cropped_w > cropped_h:
            patch_w = patch_size
            patch_h = patch_size * (cropped_h / cropped_w)
        else:
            patch_h = patch_size
            patch_w = patch_size * (cropped_w / cropped_h)

        patch_w = int(patch_w)
        patch_h = int(patch_h)
        # area_type = sample['area_type']
        # area_type = hash(sample_id) % 4
        area_x, area_y = area_type#dataset.determine_patch_x_y(area_type, patch_w, patch_h)
        
        pred_x, pred_y = pred
        pred_x -= area_x
        pred_y -= area_y
        pred_x *= cropped_w / patch_w
        pred_y *= cropped_h / patch_h
        pred_x += crop_trans.x0
        pred_y += crop_trans.y0
        pixel_pred_scaled[i][0] = pred_x
        pixel_pred_scaled[i][1] = pred_y
        
        gt_x, gt_y = gt
        gt_x -= area_x
        gt_y -= area_y
        gt_x *= cropped_w / patch_w
        gt_y *= cropped_h / patch_h
        gt_x += crop_trans.x0
        gt_y += crop_trans.y0
        pixel_gt_scaled[i][0] = gt_x
        pixel_gt_scaled[i][1] = gt_y
        
    return pixel_pred_scaled, pixel_gt_scaled

def cdf_plot(plot_list, title):
    for pred_var, pred_values in plot_list.items():
        df = pd.DataFrame(pred_values, columns=[pred_var])
        df['cdf'] = df.rank(method = 'average', pct = True)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)
        
        n, bins, patches = ax.hist(pred_values, bins=20)    
        ax2 = ax.twinx()
        df.sort_values(pred_var).plot(x=pred_var, y = 'cdf', grid = True, 
                                    ax=ax2, colormap='spring', linewidth=4)
        
        ax.set_title(f"{title}: {pred_var}")
    plt.show()

def cdf_plots(dataset,pixel_pred, pixel_gt, pixel_pred_scaled, pixel_gt_scaled):

    gt_center_position = {'x' : pixel_gt[:, 0] * dataset.size, 
                    'y' : pixel_gt[:, 1]* dataset.size}
    cdf_plot(gt_center_position, 'GT')

    abs_error_list = {'x' : np.abs(pixel_pred_scaled[:, 0] - pixel_gt_scaled[:, 0]), 
                    'y' : np.abs(pixel_pred_scaled[:, 1] - pixel_gt_scaled[:, 1])}
    cdf_plot(abs_error_list, "Abs Diff")

    error_all = np.linalg.norm(pixel_pred_scaled - pixel_gt_scaled, axis=1)
    error_x = np.sqrt((pixel_pred_scaled[:,0] - pixel_gt_scaled[:,0]) ** 2)
    error_y = np.sqrt((pixel_pred_scaled[:,1] - pixel_gt_scaled[:,1]) ** 2)

    mse_error_list = {'x and y' : error_all, 
                    'only x' : error_x,
                    'only y' : error_y}

    cdf_plot(mse_error_list, "MSE")

def bar_plot(pred_labels, label_name_dict, bar_title):
    labels, counts = np.unique(pred_labels, return_counts=True)
    freq_series = pd.Series(counts)
    
    x_ticks = [label_name_dict[l] for l in labels]
    x_labels = x_ticks
    # Plot the figure.
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_title(bar_title)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    ax.set_xticklabels(x_labels)

    rects = ax.patches
    for rect, label in zip(rects, counts):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')
    plt.show()

def display_img(dataset, sample_id, center, center_pred):
    idx = dataset.sample_id_to_idx[sample_id]
    sample = dataset.idx_to_data_dict[idx]

    img = PIL.Image.open(sample['rgb_all_path'])
    img = np.asarray(img)
    mask = mpimg.imread(sample['mask_path'])
    
    fig, axs = plt.subplots(1, 2, figsize=(20,20))
    mark_size = 60
    axs[0].imshow(img)
    axs[0].scatter(center[0], center[1],   marker=".", c='b', s=mark_size, label='gt')
    axs[0].scatter(center_pred[0], center_pred[1],   marker=".", c='r', s=mark_size, label='pred')

    masked_img = uplot.masked_image(img, mask)
    axs[1].imshow(masked_img)
    axs[1].scatter(center[0], center[1],   marker=".", c='b', s=mark_size, label='gt')
    axs[1].scatter(center_pred[0], center_pred[1],   marker=".", c='r', s=mark_size, label='pred')
    # plt.title('Object center prediction with big MSE error', fontsize=20)
    plt.legend(bbox_to_anchor=(1.3, 0.9))
    plt.show()
    plt.close()

def acc_topk_with_dist(labels, arg_sorted_dist, k):
    order = arg_sorted_dist[:,1:][:,:k]
    mask = labels.reshape((-1,1)) == labels[order]
    i = np.where(mask)[0].reshape((-1,1))
    j = np.where(mask)[1].reshape((-1,1))
    correct = np.concatenate([i,j],axis=1)
    
    perc = np.any(mask, axis=1).sum() / len(order)
    perc_k = np.sum(mask, axis=1) / k
    # np.sum(np.sum(mask, axis=1) >= (k // 2 + 1)) / len(order)
            
    return correct, perc, perc_k

def return_not_correct(N, correct):
    '''
    correct: (M,2)
    '''
    incorrect_idx = np.ones(N).astype(bool)
    correct_idx = np.unique(correct[:,0])
    incorrect_idx[list(correct_idx)] = False 
    
    return np.where(incorrect_idx)[0]