import os 
import pickle
import numpy as np
from tqdm import tqdm 
import utils.pointcloud as pc
import PIL
import matplotlib.image as mpimg

from optparse import OptionParser

import utils.dataset_utils as data_utils
import utils.transforms as utrans

parser = OptionParser()
parser.add_option("--scene_dir", dest="scene_dir")

def process_ptc(dir_path):
    img_mean = [0.5,0.5,0.5]
    img_std = [0.5,0.5,0.5]
    scene_name = dir_path.split("/")[-1]
    scene_description_path = os.path.join(dir_path, 'scene_description.p')
    if not os.path.exists(scene_description_path):
        return
    object_descriptions = pickle.load(open(scene_description_path, 'rb'))

    mask_all_d = data_utils.compile_mask_files(dir_path)
    
    for object_idx in object_descriptions.keys():
        if not isinstance(object_idx, int):
            continue
        object_description = object_descriptions[object_idx]

        object_cam_d = object_description['object_cam_d']
        for cam_num, object_camera_info_i in object_cam_d.items():
            pix_left_ratio = object_camera_info_i['pix_left_ratio'] 
                
            if pix_left_ratio < 0.3:
                continue
            obj_pt_fname = os.path.join(dir_path, f'pc_{(cam_num):05}_{object_idx}.npy')
            obj_pt_feat_fname = os.path.join(dir_path, f'pc_feat_{(cam_num):05}_{object_idx}.npy')

            if os.path.exists(obj_pt_fname) and os.path.exists(obj_pt_feat_fname):
                continue

            rgb_all_path = object_camera_info_i['rgb_all_path'].split('/')[-1:]
            depth_all_path = object_camera_info_i['depth_all_path'].split('/')[-1:]
            mask_path = object_camera_info_i['mask_path'].split('/')[-1:]
            rgb_all_path = os.path.join(dir_path, *rgb_all_path)
            mask_path = os.path.join(dir_path, *mask_path)
            depth_all_path = os.path.join(dir_path, *depth_all_path)
            mask_all_path = os.path.join(dir_path, f'segmentation_{(cam_num):05}.png')
            mask_all_objs = mask_all_d[f'{(cam_num):05}']

            rgb_all = PIL.Image.open(rgb_all_path)
            depth_all = PIL.Image.open(depth_all_path)

            mask = mpimg.imread(mask_path)
            mask = utrans.mask_to_PIL(mask)
            mask_all = data_utils.compile_mask(mask_all_objs)
            mask_all = utrans.mask_to_PIL(mask_all)
            
            _, rot = pc.from_world_to_camera_mat_to_tf(object_camera_info_i['world_to_camera_mat'])
            obj_pt, obj_pt_features = pc.get_pointcloud(rgb_all, depth_all, mask, mask_all, rot, img_mean, img_std)
            
            with open(obj_pt_fname, 'wb+') as f:
                np.save(f, obj_pt)
            
            with open(obj_pt_feat_fname, 'wb+') as f:
                np.save(f, obj_pt_features)
            
        
def main():
    (options, _) = parser.parse_args()
    scene_dir = options.scene_dir
    dir_list = data_utils.data_dir_list(scene_dir)
    for dir_path_i in tqdm(range(len(dir_list))):
        dir_path = dir_list[dir_path_i]
        process_ptc(dir_path)

if __name__ == "__main__":
    main()