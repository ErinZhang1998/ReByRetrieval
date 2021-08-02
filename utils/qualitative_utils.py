import os 
import json 
import torch
import numpy as np 
import copy 
import utils.transforms as utrans
import PIL 

import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils.plot_image as uplot 
import utils.metric as metric 

def pariwise_distances(X, Y, squared=False):
    XX = torch.matmul(X, torch.transpose(X, 0, 1)) #mxm
    YY = torch.matmul(Y, torch.transpose(Y, 0, 1)) #nxn
    YX = torch.matmul(Y, torch.transpose(X, 0, 1)) #nxm
    
    XX = torch.diagonal(XX)
    XX = torch.unsqueeze(XX, 0) #1xm
    
    YY = torch.diagonal(YY)
    YY = torch.unsqueeze(YY, 1) #nx1
    
    distances = XX - 2.0 * YX + YY #nxm
    distances = torch.max(distances, torch.zeros_like(distances))
    if not squared:
        mask = torch.isclose(distances, torch.zeros_like(distances), rtol=0).float()
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    
    return distances.T #mxn

def torchify(feat):
    feat = torch.FloatTensor(feat)
    feat = feat.cuda()
    return feat

def get_arg_sorted_dist(feat1, feat2):
    pairwise_dist = pariwise_distances(feat1, feat2, squared=False).cpu()
    arg_sorted_dist = np.argsort(pairwise_dist.numpy(), axis=1)
    return arg_sorted_dist

def sample_ids_to_cat_and_id(dataset, sample_ids):
    obj_cat = []
    obj_id = []
    for sample_id in list(sample_ids):
        idx = dataset.sample_id_to_idx[sample_id]
        sample = dataset.all_data_dict[idx]
        obj_cat.append(sample['obj_cat'])
        obj_id.append(sample['obj_id'])
    
    return np.array(obj_cat), np.array(obj_id)

def sample_ids_to_pixel_left_ratio(dataset, sample_ids):
    vec = np.zeros(len(sample_ids))
    for i,sample_id in enumerate(list(sample_ids)):
        idx = dataset.sample_id_to_idx[sample_id]
        sample = dataset.all_data_dict[idx]
        vec[i] = sample['pix_left_ratio']
    return vec

def get_count_list(L,  all_unique_values, ratio=False):
    vals, counts = np.unique(L, return_counts=True)
    L_dict = dict(zip(vals, counts))
    count_L = []
    for lab in all_unique_values:
        new_val = L_dict.get(lab, 0)
        if ratio:
            new_val = new_val / len(L)
        count_L.append(new_val)
    return count_L

def general_comparison_bar_plot(
    x,
    xname,
    data_list,
    name_list
):
    x = list(x)
    ratio_leng = [len(L) for L in data_list]
    labs = []
    for i,leng in enumerate(ratio_leng):
        labs += [name_list[i]] * leng

    data = []
    for L in data_list:
        data += L
    #print(len(data))
    df_all_ratios = pd.DataFrame(zip(x*len(data_list), labs, data), columns=[xname, "kind", "data"])
    plt.figure(figsize=(20, 12))
    splot = sns.barplot(x=xname, hue="kind", y="data", data=df_all_ratios)

    splot.set_xticklabels(splot.get_xticklabels(), rotation=45, horizontalalignment='right')

    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.show()

def csv_count(csv_fname):
    df = pd.read_csv(csv_fname)
    category_list = []
    object_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        category_list.append(row['catId'])
        object_list.append(row['objId'])
    return category_list, object_list


def majority_voting_prediction(lab_pred, k):
    l = lab_pred[:,:k]
    prediction = []
    for li in l:
        l_val, l_counts = np.unique(li, return_counts=True)
        #print(l_val, l_counts, np.argsort(l_counts)[::-1])
        #print(l_val[list(np.argsort(l_counts)[::-1])])
        prediction.append(l_val[list(np.argsort(l_counts)[::-1])][0])
    return np.asarray(prediction)


def report_metric(
    nearest_image,
    label,
    k = 1,
    maj_num = 5,
):
    '''
    Args:
        nearest_image: (N, N-1), each row ranked by "closest" to the image to "furthest"
        label: (N,) ground truth label (category \ object)
        k: mAP@k
        maj_num: number of highest ranking images to do majority voting 
    '''
    # Majority with 5 increase slightly
    maj_pred = majority_voting_prediction(nearest_image, maj_num)

    mAP_1 = metric.mapk(label.reshape(-1,1), maj_pred.reshape(-1,1), k=k)
    print("Majority mAP@{}: ".format(k), mAP_1)

    mAP_1_non_maj = metric.mapk(label.reshape(-1,1), nearest_image, k=k)
    print("mAP@{}: ".format(k), mAP_1_non_maj)
    
    return maj_pred

def report_metric_filtered(
    ranked_retrieval,
    target_label,
    query_label,
    filter_idx = None,
    k = 1,
    maj_num = 5,
):
    if filter_idx is not None:
        ranked_retrieval_filtered = ranked_retrieval[filter_idx]
        query_label_filtered = query_label[filter_idx]
    else:
        ranked_retrieval_filtered = ranked_retrieval
        query_label_filtered = query_label

    return report_metric(
        target_label[ranked_retrieval_filtered],
        query_label_filtered, 
        k = k,
        maj_num = maj_num,
    )


def process_csv_cat(csv_file):
    df = pd.read_csv(csv_file)
    unique_category_ids = df['catId'].unique()
    model_cat_info_dict = dict()

    for i in range(len(df)):
        row = df.iloc[i]
        cat_dict = model_cat_info_dict.get(row['catId'], dict())
        name = row['name']
        if name == 'bag,traveling bag,travelling bag,grip,suitcase':
            name = 'bag'
        if name == 'can,tin,tin can':
            name = 'can'
        cat_dict['name'] = name 
        model_list = cat_dict.get('shapenet_object_id', [])
        model_list.append((row['ShapeNetModelId'], row['objId']))
        cat_dict['shapenet_object_id'] = model_list
        model_cat_info_dict[row['catId']] = cat_dict
    unique_category_names = ['{}_{}'.format(str(xi), model_cat_info_dict[xi]['name']) for xi in unique_category_ids]

    return df, unique_category_ids, unique_category_names


def process_csv_obj(csv_file):
    df = pd.read_csv(csv_file)

    unique_object_ids = df['objId'].unique()
    model_object_info_dict = dict()

    for i in range(len(df)):
        row = df.iloc[i]
        name = row['name']
        if name == 'bag,traveling bag,travelling bag,grip,suitcase':
            name = 'bag'
        if name == 'can,tin,tin can':
            name = 'can'
        object_dict = model_object_info_dict.get(row['objId'], dict())
        object_dict['name'] = name
        model_list = object_dict.get('shapenet_object_id', [])
        model_list.append(row['ShapeNetModelId'])
        object_dict['shapenet_object_id'] = model_list
        model_object_info_dict[row['objId']] = object_dict
        
    unique_object_names = ['{}_{}'.format(str(xi), model_object_info_dict[xi]['name']) for xi in unique_object_ids]

    return df, unique_object_ids, unique_object_names


def get_csv_unique(csv_file, key):
    df = pd.read_csv(csv_file)
    unique_vals = df[key].unique()
    name_info = dict()
    for i in range(len(df)):
        row = df.iloc[i]
        name = row['name']
        if name == 'bag,traveling bag,travelling bag,grip,suitcase':
            name = 'bag'
        if name == 'can,tin,tin can':
            name = 'can'
        name_info[row[key]] = name
    unique_names = ['{}_{}'.format(str(xi), name_info[xi]) for xi in unique_vals]
    return unique_vals, unique_names

def read_npy(experiment_dir, epoch):
    features = np.load(os.path.join(experiment_dir, f'{epoch}_embedding.npy'))
    print("features.shape: ", features.shape)
    sample_id = np.load(os.path.join(experiment_dir, f'{epoch}_sample_id.npy')) 
    sample_id_res = []
    for L in sample_id:
        sample_id_res.append('-'.join([str(int(item)) for item in L]))
    return features, np.asarray(sample_id_res)

def return_json_annotations(root_dir, scene_num):
    dir_path = os.path.join(root_dir, f'scene_{scene_num:06}')
    annotations = json.load(open(os.path.join(dir_path, 'annotations.json')))
    return annotations

class IndexManagement(object):
    def __init__(self, dataset, test_batch_indices, samples_ids):
        self.dataset = dataset
        self.samples_ids = np.asarray(samples_ids)
        self.test_batch_indices = test_batch_indices
    
    def get_sample_ids(self, test_batch_idx):
        return self.samples_ids[list(test_batch_idx)]
    
    def get_dataset_idx(self, test_batch_idx):
        sample_id_idx = self.get_sample_ids(test_batch_idx)
        return np.asarray([
            self.dataset.sample_id_to_idx[item] for item in sample_id_idx.flatten()
        ])


####################################################################################
def aggregate_values(dataset, key, used=True):
    agg_L = []
    if not used:
        for _,v in dataset.all_data_dict.items():
            agg_L.append(v[key])
    else:
        for _,v in dataset.object_id_to_dict_idx.items():
            for idx in v:
                agg_L.append(dataset.all_data_dict[idx][key])

    return agg_L

####################################################################################
# def transform_pixels_in_crop_img(dataset, pixel_pred, pixel_gt, sample_ids, area_types):
#     pixel_pred_scaled = copy.deepcopy(pixel_pred) 
#     pixel_gt_scaled = copy.deepcopy(pixel_gt) 
#     for i,(pred,gt,sample_id,area_type) in enumerate(zip(copy.deepcopy(pixel_pred), copy.deepcopy(pixel_gt), sample_ids, area_types)):
#         pred *= dataset.size 
#         gt *= dataset.size
        
#         idx = dataset.sample_id_to_idx[sample_id]
#         sample = dataset.all_data_dict[idx]
#         corners = copy.deepcopy(sample['scene_corners'])
#         crop_trans = utrans.CropArea(corners)
#         cropped_w = crop_trans.x1 - crop_trans.x0
#         cropped_h = crop_trans.y1 - crop_trans.y0
        
#         patch_size = int(dataset.size * 0.5)
#         if cropped_w > cropped_h:
#             patch_w = patch_size
#             patch_h = patch_size * (cropped_h / cropped_w)
#         else:
#             patch_h = patch_size
#             patch_w = patch_size * (cropped_w / cropped_h)

#         patch_w = int(patch_w)
#         patch_h = int(patch_h)
#         # area_type = sample['area_type']
#         # area_type = hash(sample_id) % 4
#         area_x, area_y = area_type#dataset.determine_patch_x_y(area_type, patch_w, patch_h)
        
#         pred_x, pred_y = pred
#         pred_x -= area_x
#         pred_y -= area_y
#         pred_x *= cropped_w / patch_w
#         pred_y *= cropped_h / patch_h
#         pred_x += crop_trans.x0
#         pred_y += crop_trans.y0
#         pixel_pred_scaled[i][0] = pred_x
#         pixel_pred_scaled[i][1] = pred_y
        
#         gt_x, gt_y = gt
#         gt_x -= area_x
#         gt_y -= area_y
#         gt_x *= cropped_w / patch_w
#         gt_y *= cropped_h / patch_h
#         gt_x += crop_trans.x0
#         gt_y += crop_trans.y0
#         pixel_gt_scaled[i][0] = gt_x
#         pixel_gt_scaled[i][1] = gt_y
        
#     return pixel_pred_scaled, pixel_gt_scaled


# def cdf_plot(plot_list, title):
#     for pred_var, pred_values in plot_list.items():
#         df = pd.DataFrame(pred_values, columns=[pred_var])
#         df['cdf'] = df.rank(method = 'average', pct = True)
#         fig = plt.figure(figsize=(16, 8))
#         ax = fig.add_subplot(121)
        
#         n, bins, patches = ax.hist(pred_values, bins=20)    
#         ax2 = ax.twinx()
#         df.sort_values(pred_var).plot(x=pred_var, y = 'cdf', grid = True, 
#                                     ax=ax2, colormap='spring', linewidth=4)
        
#         ax.set_title(f"{title}: {pred_var}")
#     plt.show()


# def cdf_plots(dataset,pixel_pred, pixel_gt, pixel_pred_scaled, pixel_gt_scaled):

#     gt_center_position = {'x' : pixel_gt[:, 0] * dataset.size, 
#                     'y' : pixel_gt[:, 1]* dataset.size}
#     cdf_plot(gt_center_position, 'GT')

#     abs_error_list = {'x' : np.abs(pixel_pred_scaled[:, 0] - pixel_gt_scaled[:, 0]), 
#                     'y' : np.abs(pixel_pred_scaled[:, 1] - pixel_gt_scaled[:, 1])}
#     cdf_plot(abs_error_list, "Abs Diff")

#     error_all = np.linalg.norm(pixel_pred_scaled - pixel_gt_scaled, axis=1)
#     error_x = np.sqrt((pixel_pred_scaled[:,0] - pixel_gt_scaled[:,0]) ** 2)
#     error_y = np.sqrt((pixel_pred_scaled[:,1] - pixel_gt_scaled[:,1]) ** 2)

#     mse_error_list = {'x and y' : error_all, 
#                     'only x' : error_x,
#                     'only y' : error_y}

#     cdf_plot(mse_error_list, "MSE")


# def bar_plot(pred_labels, label_name_dict, bar_title):
#     labels, counts = np.unique(pred_labels, return_counts=True)
#     freq_series = pd.Series(counts)
    
#     x_ticks = [label_name_dict[l] for l in labels]
#     x_labels = x_ticks
#     # Plot the figure.
#     plt.figure(figsize=(12, 8))
#     ax = freq_series.plot(kind='bar')
#     ax.set_title(bar_title)
#     ax.set_xlabel('Labels')
#     ax.set_ylabel('Counts')
#     ax.set_xticklabels(x_labels)

#     rects = ax.patches
#     for rect, label in zip(rects, counts):
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
#                 ha='center', va='bottom')
#     plt.show()


# def display_img(dataset, sample_id, center, center_pred):
#     idx = dataset.sample_id_to_idx[sample_id]
#     sample = dataset.all_data_dict[idx]

#     img = PIL.Image.open(sample['rgb_all_path'])
#     img = np.asarray(img)
#     mask = mpimg.imread(sample['mask_path'])
    
#     fig, axs = plt.subplots(1, 2, figsize=(20,20))
#     mark_size = 60
#     axs[0].imshow(img)
#     axs[0].scatter(center[0], center[1],   marker=".", c='b', s=mark_size, label='gt')
#     axs[0].scatter(center_pred[0], center_pred[1],   marker=".", c='r', s=mark_size, label='pred')

#     masked_img = uplot.masked_image(img, mask)
#     axs[1].imshow(masked_img)
#     axs[1].scatter(center[0], center[1],   marker=".", c='b', s=mark_size, label='gt')
#     axs[1].scatter(center_pred[0], center_pred[1],   marker=".", c='r', s=mark_size, label='pred')
#     # plt.title('Object center prediction with big MSE error', fontsize=20)
#     plt.legend(bbox_to_anchor=(1.3, 0.9))
#     plt.show()
#     plt.close()


# def return_not_correct(N, correct):
#     '''
#     correct: (M,2)
#     '''
#     incorrect_idx = np.ones(N).astype(bool)
#     correct_idx = np.unique(correct[:,0])
#     incorrect_idx[list(correct_idx)] = False 
    
#     return np.where(incorrect_idx)[0]