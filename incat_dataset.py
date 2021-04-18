import pickle 
import os 
import numpy as np
import torch
import cv2

import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import torch
import torch.nn
# from torchvision import transforms
import utils.transforms as utrans

from torch.utils.data import Dataset
import copy

import json
import pandas as pd
import numpy as np
from torch.utils.data.sampler import Sampler


class InCategoryClutterDataset(Dataset):
    
    def __init__(self, split, size, scene_dir, csv_file_path, shapenet_filepath):

        self.split = split 
        self.size = size 
        self.scene_dir = scene_dir
        self.csv_file_path = csv_file_path
        self.shapenet_filepath = shapenet_filepath
        self.img_mean = [0.5,0.5,0.5]
        self.img_std = [0.5,0.5,0.5]

        self.dir_list = self.data_dir_list(self.scene_dir)
        
        self.object_id_to_dict_idx = {}
        self.generate_object_category_information()

        self.idx_to_data_dict = dict()
        idx = 0
        for dir_path in self.dir_list:
            idx_to_data_dicti, idx = self.load_sample(dir_path, idx)
            self.idx_to_data_dict.update(idx_to_data_dicti)
        
        self.idx_to_sample_id = [[]]*len(self.idx_to_data_dict)
        for k,v in self.idx_to_data_dict.items():
            self.idx_to_sample_id[k] = [v['sample_id']]
        self.idx_to_sample_id = np.asarray(self.idx_to_sample_id).reshape(-1,)
        
        self.sample_id_to_idx = {}
        for k,v in self.idx_to_data_dict.items():
            self.sample_id_to_idx[v['sample_id']] = k
        
        self.keep_even()

        self.total = []
        for k,v in self.object_id_to_dict_idx.items():
            self.total.append(len(v))
        
        # self.determine_imge_dim()


    def determine_imge_dim(self):
        sample = self.idx_to_data_dict[0]
        rgb_all = mpimg.imread(sample['rgb_all_path'])
        self.img_h, self.img_w, _ = rgb_all.shape

    def data_dir_list(self, root_dir):
        l = []
        for subdir in os.listdir(root_dir):
            if subdir.startswith('scene_'):
                subdir_path = os.path.join(root_dir, subdir)
                scene_description_dir = os.path.join(subdir_path, 'scene_description.p')
                if not os.path.exists(scene_description_dir):
                    continue 
                l.append(subdir_path)
        
        if self.split == 'test':
            return l[:300]

        return l 
    
    def generate_object_category_information(self):
        df = pd.read_csv(self.csv_file_path)

        cat_ids = set()
        object_ids = set()
        cat_names = set()
        for idx in range(len(df)):
            sample = df.iloc[idx]
            cat_ids.add(sample['synsetId'])
            cat_names.add(sample['name'])
            object_ids.add(sample['objId']) 
        cat_ids = list(cat_ids)
        object_ids = list(object_ids)
        cat_names = list(cat_names)
        
        self.cat_ids = cat_ids
        self.cat_id_to_label = dict(zip(self.cat_ids, range(len(self.cat_ids))))
        self.object_ids = object_ids
        self.object_id_to_label = dict(zip(self.object_ids, range(len(self.object_ids))))
        
        self.cat_names = cat_names
        self.cat_names_to_cat_id = dict(zip(self.cat_names, self.cat_ids))
        self.cat_id_to_cat_names = dict(zip(self.cat_ids, self.cat_names))

    def keep_even(self):
        self.discarded_idx = []
        for k,v in self.object_id_to_dict_idx.items():
            if len(v) %2 == 0:
                continue 
            j = np.random.randint(0, len(v),1)[0]
            self.discarded_idx.append(v[j])
            v.remove(v[j])
            self.object_id_to_dict_idx[k] = v
            assert len(v)%2 == 0 

    def parse_sample_id(self,sample_id):
        scene_name, cam_num, object_idx = sample_id.rsplit('_',2)
        object_idx = int(object_idx)
        cam_num = int(cam_num)
        return scene_name, cam_num, object_idx
    
    def load_sample_img_mask(self, sample_id):
        # scene_000953_00022_0
        scene_name, cam_num, object_idx = self.parse_sample_id(sample_id)
        dir_path = os.path.join(self.scene_dir, scene_name)

        root_name = f'_{(cam_num):05}'
        obj_name = f'_{(cam_num):05}_{object_idx}'

        rgb_all_path = os.path.join(dir_path, 'rgb{}.png'.format(root_name))
        segmentation_filename = os.path.join(dir_path, 'segmentation{}.png'.format(obj_name))
        
        rgb_all = mpimg.imread(rgb_all_path)
        mask = mpimg.imread(segmentation_filename)

        object_descriptions = pickle.load(open(os.path.join(dir_path, 'scene_description.p'), 'rb'))
        center = object_descriptions[object_idx][cam_num]["object_center"]
        center_rev = copy.deepcopy(center.reshape(-1,))
        center_rev[0] = rgb_all.shape[1] - center_rev[0]
    
        return rgb_all, mask, center_rev
    
    
    def load_sample(self, dir_path, idx):
        scene_name = dir_path.split("/")[-1]
        scene_description_path = os.path.join(dir_path, 'scene_description.p')
        object_descriptions = pickle.load(open(scene_description_path, 'rb'))
        
        samples = {}
        idx_i = idx
        for object_idx in object_descriptions.keys():
            object_description = object_descriptions[object_idx]
            
            position = object_description['position']
            scale = object_description['scale']
            orientation = object_description['orientation']
            mesh_filename = object_description['mesh_filename']
            object_cat_id = self.cat_id_to_label[object_description['obj_cat']]
            object_obj_id = self.object_id_to_label[object_description['obj_id']]
            object_shapenet_id = object_description['obj_shapenet_id']

            Ai = self.object_id_to_dict_idx.get(object_obj_id, [])

            object_cam_d = object_description['object_cam_d']
            for cam_num, object_camera_info_i in object_cam_d.items():
                pix_left_ratio = object_camera_info_i['pix_left_ratio'] 
                if pix_left_ratio < 0.4:
                    continue
                
                root_name = f'_{(cam_num):05}'
                obj_name = f'_{(cam_num):05}_{object_idx}'
                sample_id = scene_name + f'_{cam_num}_{object_idx}'
                sample = {'sample_id': sample_id}
                sample['position'] = position
                sample['scale'] = scale
                sample['orientation'] = orientation
                sample['obj_cat'] = object_cat_id
                sample['obj_id'] = object_obj_id
                sample['obj_shapenet_id'] = object_shapenet_id
                sample['object_center'] = object_camera_info_i["object_center"]

                sample['rgb_all_path'] = object_camera_info_i['rgb_all_path']
                sample['mask_path'] = object_camera_info_i['mask_path'] 
                samples[idx_i] = sample

                Ai.append(idx_i)
                idx_i += 1

            self.object_id_to_dict_idx[object_obj_id] = Ai

        return samples, idx_i
    
    def __len__(self):
        # return len(self.idx_to_data_dict)
        return np.sum(self.total)

    def __getitem__(self, idx):
        sample = self.idx_to_data_dict[idx]
        if self.split == 'train':
            trans = utrans.Compose([utrans.Resized(width = self.size, height = self.size),
                    utrans.RandomHorizontalFlip(),
                ])
        else:
            trans = utrans.Compose([utrans.Resized(width = self.size, height = self.size)
                ])

        rgb_all = mpimg.imread(sample['rgb_all_path'])
        mask = mpimg.imread(sample['mask_path'])
        center = copy.deepcopy(sample['object_center'].reshape(-1,))
        center[0] = rgb_all.shape[1] - center[0]


        img_rgb, img_mask, center_trans = trans(rgb_all, mask, center)
        # img_mask, center_trans = trans(mask , center)

        img_mask = np.expand_dims(img_mask, axis=2)
        img_rgb = utrans.normalize(utrans.to_tensor(img_rgb), self.img_mean, self.img_std)
        img_mask = utrans.to_tensor(img_mask)


        img = torch.cat((img_rgb, img_mask), 0)
        image = torch.FloatTensor(img)

        #pose_info = np.concatenate((np.array([sample['scale']]).reshape(-1,), sample['orientation'].reshape(-1,), sample['object_center'].reshape(-1,)))

        scale_info = torch.FloatTensor(np.array([sample['scale']]).reshape(-1,))
        orient_info = torch.FloatTensor(sample['orientation'].reshape(-1,))
        pixel_info = torch.FloatTensor(center_trans.reshape(-1,) / self.size)
        cat_info = torch.FloatTensor(np.array([sample['obj_cat']]).reshape(-1,))
        id_info = torch.FloatTensor(np.array([sample['obj_id']]).reshape(-1,))
        idx_info = torch.FloatTensor(np.array([idx]).reshape(-1,))

        #return image, torch.FloatTensor(pose_info), torch.FloatTensor([sample['obj_cat']])
        return [image, scale_info, pixel_info, cat_info, id_info, idx_info]


