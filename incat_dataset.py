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
    
    def __init__(self, split, size, scene_dir, model_filepath, shape_categories_filepath, shapenet_filepath):

        self.split = split 
        self.size = size 
        self.scene_dir = scene_dir
        self.model_filepath = model_filepath
        self.shape_categories_filepath = shape_categories_filepath
        self.shapenet_filepath = shapenet_filepath
        self.img_mean = [0.5,0.5,0.5]
        self.img_std = [0.5,0.5,0.5]

        self.dir_list = self.data_dir_list(self.scene_dir)
        
        self.object_id_to_dict_idx = {}
        self.object_ids, self.object_id_to_label, self.cat_ids, self.cat_id_to_label = self.object_cat()

        self.idx_to_data_dict = dict()
        idx = 0
        for dir_path in self.dir_list:
            idx_to_data_dicti, idx = self.load_sample(dir_path, idx)
            self.idx_to_data_dict.update(idx_to_data_dicti)
        
        self.idx_to_sample_id = {}
        for k,v in self.idx_to_data_dict.items():
            self.idx_to_sample_id[k] = v['sample_id']

        self.idx_to_sample_id = {}
        acc = 0
        for k,v in self.idx_to_data_dict.items():
            if acc == k
        
        self.sample_id_to_idx = {}
        for k,v in self.idx_to_data_dict.items():
            self.sample_id_to_idx[v['sample_id']] = k
        
        self.keep_even()

        self.total = []
        for k,v in self.object_id_to_dict_idx.items():
            self.total.append(len(v))
        
        self.determine_imge_dim()


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
    
    def object_cat(self):
        # Automatically containing only object ids for the split
        shapenet_models = json.load(open(self.model_filepath))

        temp = json.load(open(os.path.join(self.shapenet_filepath, 'taxonomy.json'))) 
        taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}
        synset_ids_in_dir = os.listdir(self.shapenet_filepath)
        synset_ids_in_dir.remove('taxonomy.json')
        taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synset_ids_in_dir}
        
        # All categories that we are considering 
        table_top_categories = []
        with open(self.shape_categories_filepath) as shape_categories_file:
            for line in shape_categories_file:
                if line.strip() == '':
                    continue
                table_top_categories.append(line.strip())
        
        cat_ids = []
        object_ids = set()
        for cat_name in table_top_categories:
            for obj_id in shapenet_models[cat_name]:
                object_ids.add(obj_id)
            cat_ids.append(taxonomy_dict[cat_name])
        object_ids = list(object_ids)

        cat_id_to_label = {}
        for i in range(len(cat_ids)):
            cat_id_to_label[cat_ids[i]] =  i
        
        object_id_to_label = {}
        for i in range(len(object_ids)):
            object_id_to_label[object_ids[i]] =  i

        self.taxonomy_dict = taxonomy_dict
        return object_ids, object_id_to_label, cat_ids, cat_id_to_label

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

    
    def load_sample(self, dir_path, idx):
        scene_name = dir_path.split("/")[-1]
        scene_description_dir = os.path.join(dir_path, 'scene_description.p')
        scene_description = pickle.load(open(scene_description_dir, 'rb'))
        num_views = scene_description['camera_pos'].shape[0]
        num_objects = len(scene_description['object_descriptions'])

        samples = {}
        idx_i = idx
        for i in range(num_objects):
            object_description = scene_description['object_descriptions'][i]
            position = object_description['position']
            scale = object_description['scale']
            orientation = object_description['orientation']
            mesh_filename = object_description['mesh_filename']
            object_cat_id = object_description['obj_cat']
            object_obj_id = object_description['obj_id']#
            
            Ai = self.object_id_to_dict_idx.get(object_obj_id, [])
            
            for cam_num in range(num_views):

                
                root_name = f'_{(cam_num):05}'
                obj_name = f'_{(cam_num):05}_{i}'
                segmentation_filename = os.path.join(dir_path, 'segmentation'+obj_name+'.png')
                if not os.path.exists(segmentation_filename):
                    continue
                sample_id = scene_name + obj_name
                sample = {'sample_id': sample_id}
                sample['depth_all_path'] = os.path.join(dir_path, 'depth'+root_name+'.png')
                sample['rgb_all_path'] = os.path.join(dir_path, 'rgb'+root_name+'.png')
                sample['mask_path'] = segmentation_filename

                sample['position'] = position
                sample['scale'] = scale
                sample['orientation'] = orientation
                sample['mesh_filename'] = mesh_filename
                sample['object_center'] = object_description["object_center_{}".format(cam_num)]
                sample['obj_cat'] = self.cat_id_to_label[object_cat_id]
                sample['obj_id'] = self.object_id_to_label[object_obj_id]
                # 

                samples[idx_i] = sample
                # print(idx_i, sample_id)
                Ai.append(idx_i)
                idx_i += 1

            self.object_id_to_dict_idx[object_obj_id] = Ai


        return samples, idx_i
    
    def __len__(self):
        # return len(self.idx_to_data_dict)
        return np.sum(self.total)
    
    # def denormalize_image(self):
    #     for i in range(3):
    #         meani = mean[i]
    #         stdi = std[i]
    #         img[:,:,i] = (img[:,:,i] * stdi) + meani
    #     return img

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


        img_rgb, center_trans = trans(rgb_all, center)
        img_mask, center_trans = trans(mask , center)

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


