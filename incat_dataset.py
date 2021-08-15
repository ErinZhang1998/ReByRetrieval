import pickle 
import os 
import copy
import json 
import numpy as np
import pandas as pd

import torch
import torch.nn
import torch.distributed as dist
from torch.utils.data import Dataset

from PIL import Image
import cv2
import matplotlib.image as mpimg
import utils.transforms as utrans
import PIL
import torchvision
import matplotlib.pyplot as plt


import utils.utils as uu
import utils.plot_image as plot_utils

class InCategoryClutterDataset(Dataset):
    
    def __init__(self, split, args):

        self.split = split 
        self.args = args
        # self.size = args.dataset_config.size 
        self.size_w = args.dataset_config.size_w
        self.size_h = args.dataset_config.size_h
        self.max_num_pixel_in_final_tensor = args.dataset_config.max_num_pixel_in_final_tensor
        self.min_num_pixel_in_final_tensor = args.dataset_config.min_num_pixel_in_final_tensor
        self.superimpose = args.dataset_config.superimpose
        self.num_area_range =  args.dataset_config.superimpose_num_area_range
        if split == 'train':
            self.scene_dir = args.files.training_scene_dir
        elif split == 'base':
            self.scene_dir = args.files.base_scene_dir
        else:
            self.scene_dir = args.files.testing_scene_dir
        # In order to work well with perch's image annotation, use e.g.
        # /raid/xiaoyuz1/perch + image_annotation_path   for getting the images
        # /raid/xiaoyuz1/perch/training_set + scene_dir  for getting the annotations.json
        self.scene_dir_parent = os.path.abspath(os.path.join(self.scene_dir, os.pardir))
        self.canvas_file_path = args.files.canvas_file_path
        self.csv_file_path = args.files.csv_file_path
        self.shapenet_filepath = args.files.shapenet_filepath
        self.img_mean = args.dataset_config.img_mean#[0.5,0.5,0.5]
        self.img_std = args.dataset_config.img_std#[0.5,0.5,0.5]

        self.dir_list = uu.data_dir_list(self.scene_dir)
        if self.superimpose:
            file_ptr = open(self.canvas_file_path, 'r')
            self.all_canvas_path = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
        
        self.sample_id_to_idx = dict()
        self.all_data_dict = dict()
        self.all_scene_cam_dict = dict()
        idx = 0
        for dir_path in self.dir_list:
            data_dict, scene_dict, idx = self.load_annotations(dir_path, idx)
            self.all_data_dict.update(data_dict)
            self.all_scene_cam_dict.update(scene_dict)

        self.reset()
    
    def reset(self, seed=0):
        np.random.seed(seed)

        self.object_id_to_dict_idx = dict()
        for k,v in self.all_scene_cam_dict.items():
            l1,l2 = v
            for idx in l1:
                idx_obj_id = self.all_data_dict[idx]['obj_id']
                L = self.object_id_to_dict_idx.get(idx_obj_id, [])
                L.append(idx)
                self.object_id_to_dict_idx[idx_obj_id] = L
            if len(l2) > 0:
                idx2 = np.random.choice(l2, 1, replace=False)[0]
                idx2_obj_id = self.all_data_dict[idx2]['obj_id']
                L = self.object_id_to_dict_idx.get(idx2_obj_id, [])
                L.append(idx2)
                self.object_id_to_dict_idx[idx2_obj_id] = L

        self.total_ele = 0
        for k,v in self.object_id_to_dict_idx.items():
            if len(v) %2 == 0:
                self.total_ele += len(v)
                continue 
            if self.split == 'train':
                j = np.random.randint(0, len(v),1)[0]
            else:
                j = 0
            v.remove(v[j])
            self.object_id_to_dict_idx[k] = v
            assert len(v)%2 == 0 
            self.total_ele += len(v)


    def determine_imge_dim(self):
        sample = self.all_data_dict[0]
        rgb_all = mpimg.imread(sample['rgb_file'])
        self.img_h, self.img_w, _ = rgb_all.shape 

    def display_sample(self, idx, rgb=False, seg=True):
        sample = self.all_data_dict[idx]
        rgb_all = PIL.Image.open(sample['rgb_file'])
        mask = mpimg.imread(sample['object_mask_path'])
        img_masked = plot_utils.masked_image(np.asarray(rgb_all), mask)
        sample
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(img_masked)
        plt.show()
        if rgb:
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(np.asarray(rgb_all))
            plt.show()
    
    def load_annotations_blender_proc(self, dir_path, idx):
        annotations = json.load(open(os.path.join(dir_path, 'annotations.json')))
        scene_num = annotations['info']['scene_num']

        image_id_to_image_fname = {}
        for v in annotations['images']:
            image_id_to_image_fname[v['id']] = v
        
        category_id_to_model = {}
        for v in annotations['categories']:
            category_id_to_model[v['id']] = v
        
        data_dict = dict()
        scene_dict = dict()
        idx_i = idx

        for ann in annotations['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            sample_id_int = [scene_num, image_id, category_id]
            sample_id = '-'.join([str(item) for item in sample_id_int])
                        
            rgb_file = image_id_to_image_fname[image_id]['file_name']
            object_mask_path = ann['mask_file_path']
            all_object_mask_path = image_id_to_image_fname[image_id]['all_object_segmentation_path']
            all_object_with_table_mask_path = image_id_to_image_fname[image_id]['all_object_with_table_segmentation_path']
            corners = image_id_to_image_fname[image_id]['all_object_bbox']
            cmin, rmin, cleng, rleng = ann['bbox']
            rmax = rmin + rleng
            cmax = cmin + cleng
            bbox_2d = np.array([[cmin, rmin],[cmax, rmax]])
            
            sample = {
                'sample_id' : sample_id,
                'sample_id_int' : sample_id_int,
                'position' : np.asarray(ann['location']),
                'scale' : category_id_to_model[category_id]["size"][0],
                'obj_cat' : category_id_to_model[category_id]["shapenet_category_id"],
                'obj_id' : category_id_to_model[category_id]["shapenet_object_id"],
                'rgb_file' : os.path.join(self.scene_dir_parent, rgb_file),
                'object_mask_path' : os.path.join(self.scene_dir_parent, object_mask_path),
                'all_object_mask_path' : os.path.join(self.scene_dir_parent, all_object_mask_path),
                'all_object_with_table_mask_path' : os.path.join(self.scene_dir_parent, all_object_with_table_mask_path),
                'object_position_2d' : np.asarray(ann['center']),
                'scene_bounds' : np.asarray(corners),
                'total_pixel_in_scene' : ann['number_pixels'],
                'pix_left_ratio' : percentage_not_occluded,
                'area_type' : area_type,
                'object_bbox_world_frame_2d' : bbox_2d,
            }
            scene_dict_l_must,  scene_dict_l_one = scene_dict.get((scene_num, image_id), ([],[]))
            if self.split == "train":
                if (percentage_not_occluded <= 0.95): #or (cam_d['occlusion_target'] == object_idx):
                    scene_dict_l_must.append(idx_i)
                else:
                    scene_dict_l_one.append(idx_i)
            else:
                scene_dict_l_must.append(idx_i)
            
            scene_dict[(scene_num, image_id)] = (scene_dict_l_must, scene_dict_l_one)
            data_dict[idx_i] = sample 
            self.sample_id_to_idx[sample_id] = idx_i
            idx_i += 1
        
        return data_dict, scene_dict, idx_i

    
    def load_annotations(self, dir_path, idx):
        '''
        Given abs-path of the scene directory, which contains images and annotaitons, return processed samples
        to store in self.all_data_dict

        Args:
            dir_path: abs-path of the scene directory, str
            idx: next idx in self.all_data_dict to store the new processed data
        
        Returns:
            data_dict: dict of idx --> sample-dict

        '''
        # self.scene_dir
        annotations = json.load(open(os.path.join(dir_path, 'annotations.json')))
        scene_num = annotations['info']['scene_num']

        image_id_to_image_fname = {}
        for v in annotations['images']:
            image_id_to_image_fname[v['id']] = v
        
        category_id_to_model = {}
        for v in annotations['categories']:
            category_id_to_model[v['id']] = v
        
        data_dict = dict()
        scene_dict = dict()
        idx_i = idx

        for ann in annotations['annotations']:
            percentage_not_occluded = ann['percentage_not_occluded']
            if percentage_not_occluded < self.args.dataset_config.ignore_input_ratio:
                continue 

            image_id = ann['image_id']
            category_id = ann['category_id']
            sample_id_int = [scene_num, image_id, category_id]
            sample_id = '-'.join([str(item) for item in sample_id_int])
            
            #f'scene_{scene_num:06}_{image_id}_{category_id}'
            
            if self.split != 'train' and self.args.dataset_config.test_cropped_area_position > 3:
                area_type = hash(f'scene_{scene_num:06}_{image_id}_{category_id}') % 4
            else:
                area_type = self.args.dataset_config.test_cropped_area_position
            # 
            rgb_file = image_id_to_image_fname[image_id]['file_name']
            object_mask_path = ann['mask_file_path']
            all_object_mask_path = image_id_to_image_fname[image_id]['all_object_segmentation_path']
            all_object_with_table_mask_path = image_id_to_image_fname[image_id]['all_object_with_table_segmentation_path']
            corners = image_id_to_image_fname[image_id]['all_object_bbox']
            cmin, rmin, cleng, rleng = ann['bbox']
            rmax = rmin + rleng
            cmax = cmin + cleng
            bbox_2d = np.array([[cmin, rmin],[cmax, rmax]])
            
            sample = {
                'sample_id' : sample_id,
                'sample_id_int' : sample_id_int,
                'position' : np.asarray(ann['location']),
                'scale' : category_id_to_model[category_id]["size"][0],
                'obj_cat' : category_id_to_model[category_id]["shapenet_category_id"],
                'obj_id' : category_id_to_model[category_id]["shapenet_object_id"],
                'rgb_file' : os.path.join(self.scene_dir_parent, rgb_file),
                'object_mask_path' : os.path.join(self.scene_dir_parent, object_mask_path),
                'all_object_mask_path' : os.path.join(self.scene_dir_parent, all_object_mask_path),
                'all_object_with_table_mask_path' : os.path.join(self.scene_dir_parent, all_object_with_table_mask_path),
                'object_position_2d' : np.asarray(ann['center']),
                'scene_bounds' : np.asarray(corners),
                'total_pixel_in_scene' : ann['number_pixels'],
                'pix_left_ratio' : percentage_not_occluded,
                'area_type' : area_type,
                'object_bbox_world_frame_2d' : bbox_2d,
            }
            scene_dict_l_must,  scene_dict_l_one = scene_dict.get((scene_num, image_id), ([],[]))
            if self.split == "train":
                if (percentage_not_occluded <= 0.95): #or (cam_d['occlusion_target'] == object_idx):
                    scene_dict_l_must.append(idx_i)
                else:
                    scene_dict_l_one.append(idx_i)
            else:
                scene_dict_l_must.append(idx_i)
            
            scene_dict[(scene_num, image_id)] = (scene_dict_l_must, scene_dict_l_one)
            data_dict[idx_i] = sample 
            self.sample_id_to_idx[sample_id] = idx_i
            idx_i += 1
        
        return data_dict, scene_dict, idx_i


    def __len__(self):
        # return len(self.idx_to_data_dict)
        return self.total_ele

    def determine_patch_x_y(self, area_type, area_x_range, area_y_range):
        xmin,xmax = area_x_range
        ymin,ymax = area_y_range
        if area_type == 0:
            area_x = xmin + (xmax - xmin)//4
            area_y = ymin + (ymax - ymin)//4
        elif area_type == 1:
            area_x = xmin + 3*(xmax - xmin)//4
            area_y = ymin + (ymax - ymin)//4
        elif area_type == 2:
            area_x = xmin + (xmax - xmin)//4
            area_y = ymin + 3*(ymax - ymin)//4
        else:
            area_x = xmin + 3*(xmax - xmin)//4
            area_y = ymin + 3*(ymax - ymin)//4
    
        return int(area_x), int(area_y)
    
    def process_input(self, sample):

        rgb_all = PIL.Image.open(sample['rgb_file'])
        mask = mpimg.imread(sample['object_mask_path'])
        mask = utrans.mask_to_PIL(mask)
        # mask_all = mpimg.imread(sample['all_object_with_table_mask_path'])
        # mask_all = utrans.mask_to_PIL(mask_all)
        mask_all = mpimg.imread(sample['all_object_mask_path'])
        mask_all = utrans.mask_to_PIL(mask_all)

        center = copy.deepcopy(sample['object_position_2d'].reshape(-1,))
        corners = copy.deepcopy(sample['scene_bounds'])
        
        if not self.superimpose:
            if self.split == 'train':
                trans = utrans.Compose([utrans.PILResized(width = self.size_w, height = self.size_h),
                        utrans.PILRandomHorizontalFlip(),
                    ])
            else:
                trans = utrans.Compose([utrans.PILResized(width = self.size_w, height = self.size_h),
                    ])
            img_rgb, img_mask, center_trans = trans(rgb_all, mask, center)
        else:
            canvas_path = np.random.choice(self.all_canvas_path,1)[0]
            canvas = PIL.Image.open(canvas_path)
            canvas = canvas.resize((self.size_w, self.size_h))
            cropped_obj_transform = utrans.PILCropArea(corners)
            cropped_obj_img, cropped_mask, cropped_center = cropped_obj_transform(rgb_all, mask_all, center)
            cropped_object_mask = mask.crop(cropped_obj_transform.area)
            cropped_w, cropped_h = cropped_obj_img.size
            
            cropped_object_bbox_2d = cropped_obj_transform.modify_center(sample['object_bbox_world_frame_2d'])
            a0,b0 = np.min(cropped_object_bbox_2d, axis=0)
            a1,b1 = np.max(cropped_object_bbox_2d, axis=0)
            shape_ratio = (a1-a0) * (b1-b0) / (cropped_h * cropped_w)
            
            if self.split == "train":
                num_pixels_final = np.random.uniform(self.min_num_pixel_in_final_tensor,self.max_num_pixel_in_final_tensor,1)[0]
            else:
                num_pixels_final = int(0.5*(self.min_num_pixel_in_final_tensor+self.max_num_pixel_in_final_tensor))
            patch_ratio = num_pixels_final * 900 / shape_ratio
            # patch_ratio = (sampled_ratio * (self.size_w * self.size_h)) #/ (shape_ratio)
            if cropped_w > cropped_h:
                patch_w = np.sqrt(patch_ratio * (cropped_w / cropped_h))
                patch_h = patch_w * (cropped_h / cropped_w)
            else:
                patch_h = np.sqrt(patch_ratio * (cropped_h / cropped_w))
                patch_w = patch_h * (cropped_w / cropped_h)
            patch_w, patch_h = int(patch_w), int(patch_h)

            patch_object_bbox_2d = copy.deepcopy(cropped_object_bbox_2d)
            patch_object_bbox_2d[:,0] = patch_object_bbox_2d[:,0] * float(patch_w / cropped_w)
            patch_object_bbox_2d[:,1] = patch_object_bbox_2d[:,1] * float(patch_h / cropped_h)

            a0,b0 = np.min(patch_object_bbox_2d, axis=0)
            a1,b1 = np.max(patch_object_bbox_2d, axis=0)
            area_x_range = (-a0, self.size_w - a1)
            area_y_range = (-b0, self.size_h - b1)
            
            if self.split == 'train':
                area_range_w, area_range_h = np.random.choice(np.arange(self.num_area_range),2)
                area_step_w = (area_x_range[1] - area_x_range[0]) / self.num_area_range 
                area_step_h = (area_y_range[1] - area_y_range[0]) / self.num_area_range 

                area_x = int(np.random.uniform(area_x_range[0] + area_step_w*area_range_w, 
                                            area_x_range[0] + area_step_w*(area_range_w+1),1)[0])
                area_y = int(np.random.uniform(area_y_range[0] + area_step_h*area_range_h, 
                                            area_y_range[0] + area_step_h*(area_range_h+1),1)[0])
            else:
                area_type = sample['area_type']
                area_x, area_y = self.determine_patch_x_y(area_type, area_x_range, area_y_range)
            area = (area_x, area_y, area_x+patch_w, area_y+patch_h)
            
            # On the canvas, but mask showing the place that the objects will be
            # Produce input image
            cropped_mask_L_resized = cropped_mask.convert('L').resize((patch_w,patch_h))
            canvas_mask = Image.new("L", canvas.size, 255)
            canvas_mask.paste(cropped_mask_L_resized, area)

            # On the canvas, but mask showing the place that the sample object will be
            object_canvas_mask = Image.new("L", canvas.size, 255)
            object_canvas_mask.paste(cropped_object_mask.resize((patch_w,patch_h)), area)

            object_canvas_mask = PIL.ImageOps.invert(object_canvas_mask)

            obj_background = Image.new("RGB", canvas.size, 0)
            cropped_obj_img_resized = cropped_obj_img.resize((patch_w,patch_h))
            obj_background.paste(cropped_obj_img_resized, area)
            superimposed_img = Image.composite(canvas, obj_background, canvas_mask)

            patch_object_bbox_2d[:,0] = patch_object_bbox_2d[:,0] + area_x
            patch_object_bbox_2d[:,1] = patch_object_bbox_2d[:,1] + area_y
            cx,cy = cropped_center
            cx *= (patch_w) / cropped_w
            cy *= (patch_h) / cropped_h
            cx += area_x
            cy += area_y
            canvas_center = np.array([cx,cy])

            if self.split == 'train':
                flip_trans = utrans.PILRandomHorizontalFlip()
                img_rgb, img_mask, center_trans = flip_trans(superimposed_img, object_canvas_mask, canvas_center)
            else:
                img_rgb, img_mask, center_trans = superimposed_img, object_canvas_mask, canvas_center
    
        img_rgb = utrans.normalize(torchvision.transforms.ToTensor()(img_rgb), self.img_mean, self.img_std)
        img_mask = torchvision.transforms.ToTensor()(img_mask)

        img = torch.cat((img_rgb, img_mask), 0)
        image = torch.FloatTensor(img)

        cx,cy = center_trans.reshape(-1,)
        cx /= self.size_w
        cy /= self.size_h

        position = torch.FloatTensor(sample['position'].reshape(-1,))
        scale = torch.FloatTensor(np.array([sample['scale']]).reshape(-1,))
        # orientation = torch.FloatTensor(sample['orientation'].reshape(-1,))
        center = torch.FloatTensor(np.array([cx,cy]))
        category = torch.FloatTensor(np.array([sample['obj_cat']]).reshape(-1,))
        obj_id = torch.FloatTensor(np.array([sample['obj_id']]).reshape(-1,))
        sample_id = torch.FloatTensor(np.array(sample['sample_id_int']))

        data = {
            "image": image,
            "position" : position,
            "scale": scale,
            # "orientation":orientation,
            "center" : center,
            "obj_category":category,
            "obj_id":obj_id,
            "sample_id":sample_id,
            "area_type": torch.FloatTensor(np.array([area_x, area_y]).reshape(-1,2)),
        }

        return data


    def __getitem__(self, idx):
        
        sample = self.all_data_dict[idx]
        basic_data = self.process_input(sample)
        

        if self.args.use_pc:
            pts, xind, yind = None,None,None
            with open(sample["object_pc_fname"], 'rb') as f:
                pts, xind, yind = pickle.load(f)
            obj_pt = torch.FloatTensor(pts)

            img_rgb = utrans.normalize(torchvision.transforms.ToTensor()(rgb_all), self.img_mean, self.img_std)
            obj_points_features = img_rgb.permute(1,2,0)[xind, yind].float().T
            pc_data = {
                "obj_points": obj_pt,
                "obj_points_features" : obj_points_features,
            }
            basic_data.update(pc_data)
        # elif self.split == 'train' and self.args.loss.use_consistency_loss:
        #         pair_idx_arr = self.idx_to_same_object_dict[idx]
        #         pair_idx = np.random.choice(pair_idx_arr, 1)[0]
        #         pair_data = self.process_input(self.all_data_dict[pair_idx])
        #         pair_data_add = {
        #             'pair_image' : pair_data["image"],
        #         }
                

        return basic_data


