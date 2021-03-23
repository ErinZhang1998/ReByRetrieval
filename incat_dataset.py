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
from PIL import Image
# from torchvision import transforms
import utils.transforms as utrans

from torch.utils.data import Dataset
import copy

import json
import pandas as pd
import numpy as np
from torch.utils.data.sampler import Sampler




# train_dir = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set/scene_000000'
# scene_description_dir = os.path.join(train_dir, 'scene_description.p')
# scene_description = pickle.load(open(scene_description_dir, 'rb'))


# cam_num = 0
# i = 0
# root_name = f'_{(cam_num):05}'
# obj_name = f'_{(cam_num):05}_{i}'
# path = os.path.join(train_dir, 'rgb'+root_name+'.png')
# img = mpimg.imread(os.path.join(train_dir, 'segmentation'+obj_name+'.png'))
# plt.figure()
# plt.imshow(img)
# plt.show()


class InCategoryClutterDataset(Dataset):
    shapenet_filepath = '/media/xiaoyuz1/hdd5/xiaoyuz1/ShapeNetCore.v2'
    
    def __init__(self, split, size, dir_root, shape_categories_file_path):

        self.split = split 
        self.size = size 
        self.dir_root = dir_root
        self.shape_categories_file_path = shape_categories_file_path

        self.dir_list = self.data_dir_list(self.dir_root)
        # self.cat_ids, self.cat_id_to_label = self.object_cat()
        self.object_ids, self.object_id_to_label = self.object_cat()

        self.idx_to_data_dict = dict()
        idx = 0
        for dir_path in self.dir_list:
            idx_to_data_dicti, idx = self.load_sample(dir_path, idx)
            self.idx_to_data_dict.update(idx_to_data_dicti)

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
        if self.split == 'train':
            json_file_path = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/tabletop_small_training_instances.json'
        else:
            json_file_path = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/tabletop_small_testing_instances.json'

        shapenet_models = json.load(open(json_file_path))

        temp = json.load(open(os.path.join(self.shapenet_filepath, 'taxonomy.json'))) 
        taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}
        synset_ids_in_dir = os.listdir(self.shapenet_filepath)
        synset_ids_in_dir.remove('taxonomy.json')
        taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synset_ids_in_dir}
        
        table_top_categories = []
        with open(self.shape_categories_file_path) as shape_categories_file:
            for line in shape_categories_file:
                if line.strip() == '':
                    continue
                table_top_categories.append(line.strip())
        
        cat_ids = []
        cat_id_to_label = {}
        
        object_ids = []
        object_id_to_label = {}
        for cat_name in table_top_categories:
            for obj_id in shapenet_models[cat_name]:
                object_ids.append(obj_id)
            
            cat_ids.append(taxonomy_dict[cat_name])

        for i in range(len(cat_ids)):
            cat_id_to_label[cat_ids[i]] =  i
        
        for i in range(len(object_ids)):
            object_id_to_label[object_ids[i]] =  i

        return object_ids, object_id_to_label, cat_ids, cat_id_to_label

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
            object_cat_id = object_description['obj_id']#object_description['obj_cat']
            
            
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
                sample['obj_cat'] = self.object_id_to_label[object_cat_id]
                # self.cat_id_to_label[object_cat_id]

                samples[idx_i] = sample
                idx_i += 1

        return samples, idx_i
    
    def __len__(self):
        return len(self.idx_to_data_dict)
    
    def __getitem__(self, idx):
        sample = self.idx_to_data_dict[idx]
        
        # if self.split == 'train' or self.split == 'trainval':
        #     trans = transforms.Compose([
        #             transforms.ToPILImage(),
        #             transforms.Resize((int(self.size),int(self.size))),
        #             transforms.ToTensor(),
        #             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        # else:
        #     trans = transforms.Compose([
        #             transforms.ToPILImage(),
        #             transforms.Resize((int(self.size),int(self.size))),
        #             transforms.ToTensor(),
        #             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])    
        
        # trans = utrans.Compose([utrans.Resized(width = int(self.size * 1.5), height = int(self.size * 1.5)),
        #         utrans.RandomCrop(size = int(self.size)),
        #         utrans.RandomHorizontalFlip(),
        #     ])
        trans = utrans.Compose([utrans.Resized(width = self.size, height = self.size),
                utrans.RandomHorizontalFlip(),
            ])

        rgb_all = mpimg.imread(sample['rgb_all_path'])
        # mask = (mpimg.imread(sample['mask_path']) > 0).astype('int')
        # mask = np.stack([mask,mask,mask],axis=2) #np.expand_dims(mpimg.imread(sample['mask_path']), axis=0)
        mask = mpimg.imread(sample['mask_path'])
        center = copy.deepcopy(sample['object_center'].reshape(-1,))
        center[0] = rgb_all.shape[1] - center[0]


        img_rgb, center_trans = trans(rgb_all, center)
        img_mask, center_trans = trans(mask , center)

        img_mask = np.expand_dims(img_mask, axis=2)
        img_rgb = utrans.normalize(utrans.to_tensor(img_rgb), [0.5,0.5,0.5], [0.5,0.5,0.5])
        img_mask = utrans.to_tensor(img_mask)

        img = torch.cat((img_rgb, img_mask), 0)
        image = torch.FloatTensor(img)

        #pose_info = np.concatenate((np.array([sample['scale']]).reshape(-1,), sample['orientation'].reshape(-1,), sample['object_center'].reshape(-1,)))

        scale_info = torch.FloatTensor(np.array([sample['scale']]).reshape(-1,))
        orient_info = torch.FloatTensor(sample['orientation'].reshape(-1,))
        pixel_info = torch.FloatTensor(center_trans.reshape(-1,) / self.size)
        cat_info = torch.FloatTensor(np.array([sample['obj_cat']]).reshape(-1,))

        #return image, torch.FloatTensor(pose_info), torch.FloatTensor([sample['obj_cat']])
        return image, scale_info, orient_info, pixel_info, cat_info


class DummySampler(Sampler):
    def __init__(self, data):
        self.num_samples = len(data)

    def __iter__(self):
        print ('\tcalling Sampler:__iter__')
        return iter(range(self.num_samples))

    def __len__(self):
        print ('\tcalling Sampler:__len__')
        return self.num_samples