import pickle 
import os 
import copy
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
import utils.pointcloud as pc

import utils.dataset_utils as data_utils

class InCategoryClutterDataset(Dataset):
    
    def __init__(self, split, args):

        self.split = split 
        self.args = args
        # self.size = args.dataset_config.size 
        self.size_w = args.dataset_config.size_w
        self.size_h = args.dataset_config.size_h
        self.area_ratio_max = args.dataset_config.area_ratio_max
        self.area_ratio_min = args.dataset_config.area_ratio_min
        self.superimpose = args.dataset_config.superimpose
        self.num_area_range =  args.dataset_config.superimpose_num_area_range
        if split == 'train':
            self.scene_dir = args.files.training_scene_dir
        else:
            self.scene_dir = args.files.testing_scene_dir
        self.canvas_file_path = args.files.canvas_file_path
        self.csv_file_path = args.files.csv_file_path
        self.shapenet_filepath = args.files.shapenet_filepath
        self.img_mean = args.dataset_config.img_mean#[0.5,0.5,0.5]
        self.img_std = args.dataset_config.img_std#[0.5,0.5,0.5]

        self.dir_list = data_utils.data_dir_list(self.scene_dir)
        if self.superimpose:
            file_ptr = open(self.canvas_file_path, 'r')
            self.all_canvas_path = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
        
        
        self.generate_object_category_information()

        self.reset()
        self.determine_imge_dim()
    
    def reset(self, seed=0):
        np.random.seed(seed)
        self.object_id_to_dict_idx = {}
        self.idx_to_data_dict = dict()
        idx = 0
        for dir_path in self.dir_list:
            idx_to_data_dicti, idx = self.load_sample(dir_path, idx)
            self.idx_to_data_dict.update(idx_to_data_dicti)

        # 
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

    def determine_imge_dim(self):
        sample = self.idx_to_data_dict[0]
        rgb_all = mpimg.imread(sample['rgb_all_path'])
        self.img_h, self.img_w, _ = rgb_all.shape 
    
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
        self.label_to_cat_id = dict(zip(range(len(self.cat_ids)), self.cat_ids))
        
        self.object_ids = object_ids
        self.object_id_to_label = dict(zip(self.object_ids, range(len(self.object_ids))))
        self.object_label_to_id = dict(zip(range(len(self.object_ids)), self.object_ids))
        
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
    
    def load_sample(self, dir_path, idx):
        scene_name = dir_path.split("/")[-1]
        scene_description_path = os.path.join(dir_path, 'scene_description.p')
        if not os.path.exists(scene_description_path):
            return {}, idx
        object_descriptions = pickle.load(open(scene_description_path, 'rb'))

        mask_all_d = data_utils.compile_mask_files(dir_path)
        
        samples = {}
        idx_i = idx
        all_object_idx = object_descriptions["object_indices"]
        cam_num_to_occlusion_target = object_descriptions["cam_num_to_occlusion_target"]
        cam_num_to_selected_objects = dict()
        if self.split == "train":
            for cam_num, occlude_object_idx in cam_num_to_occlusion_target.items():
                b = all_object_idx[:]
                b.remove(occlude_object_idx)
                cam_num_to_selected_objects[cam_num] = [occlude_object_idx, np.random.choice(b,1,replace=False)[0]]
        else:
            for cam_num, _ in cam_num_to_occlusion_target.items():
                cam_num_to_selected_objects[cam_num] = all_object_idx
        
        for object_idx in object_descriptions.keys():
            if not isinstance(object_idx, int):
                continue
            object_description = object_descriptions[object_idx]
            
            position = object_description['position']
            scale = object_description['scale']
            orientation = object_description['orientation']
            mesh_filename = object_description['mesh_filename']
            object_cat_id = self.cat_id_to_label[object_description['obj_cat']]
            object_obj_id = self.object_id_to_label[object_description['obj_id']]
            object_shapenet_id = object_description['obj_shapenet_id']
            cam_height = object_description['cam_height'] 
            cam_width = object_description['cam_width']

            Ai = self.object_id_to_dict_idx.get(object_obj_id, [])

            object_cam_d = object_description['object_cam_d']
            for cam_num, object_camera_info_i in object_cam_d.items():
                pix_left_ratio = object_camera_info_i['pix_left_ratio'] 
                
                if pix_left_ratio < self.args.dataset_config.ignore_input_ratio:
                    continue
                selected_objs = cam_num_to_selected_objects.get(cam_num, all_object_idx)
                if (not object_idx in selected_objs) and (pix_left_ratio > 0.9):
                    continue
                    
                obj_pt_fname = os.path.join(dir_path, f'pc_{(cam_num):05}_{object_idx}.npy')
                obj_pt_feat_fname = os.path.join(dir_path, f'pc_feat_{(cam_num):05}_{object_idx}.npy')
                if not (os.path.exists(obj_pt_fname) and os.path.exists(obj_pt_feat_fname)):
                    continue

                center = copy.deepcopy(object_camera_info_i["object_center"].reshape(-1,))
                center[0] = cam_width - center[0]

                corners = copy.deepcopy(object_description["scene_bounds_{}".format(cam_num)])
                corners[:,0] = cam_width - corners[:,0]

                x0,y0 = np.min(corners, axis=0).astype(int)
                x1,y1 = np.max(corners, axis=0).astype(int)
                if not(center[0] >= x0 and center[0] <= x1) or not(center[1] >= y0 and center[1] <= y1):
                    continue
                
                root_name = f'_{(cam_num):05}'
                obj_name = f'_{(cam_num):05}_{object_idx}'
                sample_id = scene_name + f'_{cam_num}_{object_idx}'
                sample = {'sample_id': sample_id, 'sample_id_int': [int(scene_name.split("_")[-1]),cam_num, object_idx]}
                sample['object_center'] = center
                sample['scene_corners'] = corners

                sample['position'] = position
                sample['scale'] = scale
                sample['orientation'] = orientation
                sample['obj_cat'] = object_cat_id
                sample['obj_id'] = object_obj_id
                sample['obj_shapenet_id'] = object_shapenet_id
                sample['pix_left_ratio'] = pix_left_ratio
                sample['total_pixel_in_scene'] = object_camera_info_i['total_pixel_in_scene']
                sample['world_to_camera_mat'] = object_camera_info_i['world_to_camera_mat']

                rgb_all_path = object_camera_info_i['rgb_all_path'].split('/')[-1:]
                depth_all_path = object_camera_info_i['depth_all_path'].split('/')[-1:]
                mask_path = object_camera_info_i['mask_path'].split('/')[-1:]
                sample['rgb_all_path'] = os.path.join(dir_path, *rgb_all_path)
                sample['mask_path'] = os.path.join(dir_path, *mask_path)
                sample['depth_all_path'] = os.path.join(dir_path, *depth_all_path)
                sample['mask_all_path'] = os.path.join(dir_path, f'segmentation_{(cam_num):05}.png')
                sample['mask_all_objs'] = mask_all_d[f'{(cam_num):05}']
                sample['obj_pt'] = obj_pt_fname 
                sample['obj_pt_features'] = obj_pt_feat_fname 

                if self.split == 'test' and self.args.dataset_config.test_cropped_area_position > 3:
                    # for i in [0,1]:
                    #     sample_cp = copy.deepcopy(sample)
                    #     sample_cp['area_type'] = i
                    #     samples[idx_i] = sample_cp
                    #     Ai.append(idx_i)
                    #     idx_i += 1
                    sample['area_type'] = hash(sample_id) % 4
                    samples[idx_i] = sample
                    Ai.append(idx_i)
                    idx_i += 1
                else:
                    sample['area_type'] = self.args.dataset_config.test_cropped_area_position
                    samples[idx_i] = sample
                    Ai.append(idx_i)
                    idx_i += 1

            self.object_id_to_dict_idx[object_obj_id] = Ai

        return samples, idx_i
    
    def __len__(self):
        # return len(self.idx_to_data_dict)
        return np.sum(self.total)

    def determine_patch_x_y(self, area_type, patch_w, patch_h):
        if area_type == 0:
            area_x,area_y = 0,0
        elif area_type == 1:
            area_x = min(int(self.size_w//2), self.size_w-patch_w)
            area_y = 0
        elif area_type == 2:
            area_x = 0
            area_y = min(int(self.size_h//2), self.size_h-patch_h)
        else:
            area_x = min(int(self.size_w//2), self.size_w-patch_w)
            area_y = min(int(self.size_h//2), self.size_h-patch_h)
        return area_x,area_y
    
    def __getitem__(self, idx):
        sample = self.idx_to_data_dict[idx]
        rgb_all = PIL.Image.open(sample['rgb_all_path'])
        mask = mpimg.imread(sample['mask_path'])
        original_h, original_w = mask.shape
        mask = utrans.mask_to_PIL(mask)
        mask_all = data_utils.compile_mask(sample['mask_all_objs'])
        mask_all = utrans.mask_to_PIL(mask_all)

        # mask_all = mpimg.imread(sample['mask_all_path'])
        # mask_all = utrans.mask_to_PIL(mask_all)

        center = copy.deepcopy(sample['object_center'].reshape(-1,))
        corners = copy.deepcopy(sample['scene_corners'])
        
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

            shape_ratio = sample['total_pixel_in_scene'] / (cropped_h * cropped_w)
            
            if self.split == "train":
                # sampled_ratio = np.random.uniform(0.0015, 0.005, 1)[0]
                sampled_ratio = np.random.uniform(1/9, 1/4, 1)[0]
            else:
                sampled_ratio = 1/5

            patch_ratio = (sampled_ratio * (self.size_w * self.size_h)) #/ (shape_ratio)
            if cropped_w > cropped_h:
                patch_w = np.sqrt(patch_ratio * (cropped_w / cropped_h))
                patch_h = patch_w * (cropped_h / cropped_w)
            else:
                patch_h = np.sqrt(patch_ratio * (cropped_h / cropped_w))
                patch_w = patch_h * (cropped_w / cropped_h)
            patch_w, patch_h = int(patch_w), int(patch_h)

            if self.split == 'train':
                area_range_w,area_range_h = np.random.choice(np.arange(self.num_area_range),2)
                area_step_w, area_step_h = (self.size_w - patch_w)/ self.num_area_range , (self.size_h - patch_h)/ self.num_area_range 
                area_x = int(np.random.uniform(area_step_w*area_range_w, area_step_w*(area_range_w+1),1)[0])
                area_y = int(np.random.uniform(area_step_h*area_range_h, area_step_h*(area_range_h+1),1)[0])
            else:
                area_type = sample['area_type']
                area_x, area_y = self.determine_patch_x_y(area_type, patch_w, patch_h)
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

        scale = torch.FloatTensor(np.array([sample['scale']]).reshape(-1,))
        orientation = torch.FloatTensor(sample['orientation'].reshape(-1,))
        center = torch.FloatTensor(np.array([cx,cy]))
        category = torch.FloatTensor(np.array([sample['obj_cat']]).reshape(-1,))
        obj_id = torch.FloatTensor(np.array([sample['obj_id']]).reshape(-1,))
        idx_tensor = torch.FloatTensor(np.array([idx]).reshape(-1,))
        sample_id = torch.FloatTensor(np.array(sample['sample_id_int']))

        if self.args.use_pc:
            # depth_all = PIL.Image.open(sample['depth_all_path'])
            # _, rot = pc.from_world_to_camera_mat_to_tf(sample['world_to_camera_mat'])
            # obj_pt, obj_pt_features = self.get_pointcloud(rgb_all, depth_all, mask, mask_all, rot)
            obj_pt = np.load(sample["obj_pt"]) 
            obj_pt_features = np.load(sample["obj_pt_features"]) 
            obj_pt = torch.FloatTensor(obj_pt)
            obj_pt_features = torch.FloatTensor(obj_pt_features).transpose(0,1)
            data = {
                "image": image,
                "scale": scale,
                "orientation":orientation,
                "center":center,
                "obj_category":category,
                "obj_id":obj_id,
                "idx":idx_tensor,
                "sample_id":sample_id,
                "area_type": torch.FloatTensor(np.array([area_x, area_y]).reshape(-1,2)),
                "obj_points": obj_pt,
                "obj_points_features" : obj_pt_features,
            }
        else:
            data = {
                "image": image,
                "scale": scale,
                "orientation":orientation,
                "center":center,
                "obj_category":category,
                "obj_id":obj_id,
                "idx":idx_tensor,
                "sample_id":sample_id,
                "area_type": torch.FloatTensor(np.array([area_x, area_y]).reshape(-1,2)),
            }

        return data


