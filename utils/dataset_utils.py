import os 
import numpy as np
import matplotlib.image as mpimg


def compile_mask_files(dir_path):
    '''
    Dictionary mapping from camera index (in str form, e.g. '00039') to files of individual
    object segmentation capture by the given camera
    '''
    mask_all_d = dict()
    for seg_path in os.listdir(dir_path):
        seg_path_pre = seg_path.split('.')[0]
        l = seg_path_pre.rsplit('_')
        if len(l) == 3 and l[0] == 'segmentation':
            other_obj_masks = mask_all_d.get(l[1], [])
            other_obj_masks.append(os.path.join(dir_path, seg_path))
            mask_all_d[l[1]] = other_obj_masks
    return mask_all_d

def data_dir_list(root_dir):
    l = []
    for subdir in os.listdir(root_dir):
        if subdir.startswith('scene_'):
            subdir_path = os.path.join(root_dir, subdir)
            scene_description_dir = os.path.join(subdir_path, 'scene_description.p')
            if not os.path.exists(scene_description_dir):
                continue 
            scene_name = subdir_path.split("/")[-1]
            # if int(subdir_path[-3] ) < 2:
            #     continue
            l.append(subdir_path)

    return l 

def compile_mask(mask_path_lists):
    masks = []
    for mask_path in mask_path_lists:
        mask = mpimg.imread(mask_path)
        masks.append(mask)
    
    return np.sum(np.stack(masks), axis=0)