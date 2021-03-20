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
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset




train_dir = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set/scene_000000'
scene_description_dir = os.path.join(train_dir, 'scene_description.p')
scene_description = pickle.load(open(scene_description_dir, 'rb'))


cam_num = 0
i = 0
root_name = f'_{(cam_num):05}'
obj_name = f'_{(cam_num):05}_{i}'
path = os.path.join(train_dir, 'rgb'+root_name+'.png')
img = mpimg.imread(os.path.join(train_dir, 'segmentation'+obj_name+'.png'))
plt.figure()
plt.imshow(img)
plt.show()


class InCategoryClutterDataset(Dataset):
    def __init__(self, split, size, dir_root):

        self.split = split 
        self.size = size 
        self.dir_root = dir_root

        self.dir_list = self.data_dir_list(self.dir_root)
        # self.test_dir_list = self.data_dir_list(self.test_dir_root)

        self.idx_to_data_dict = dict()
        idx = 0
        for dir_path in self.dir_list:
            idx_to_data_dicti, idx = self.load_sample(dir_path, idx)
            self.idx_to_data_dict.update(idx_to_data_dicti)

    def data_dir_list(self, root_dir):
        l = []
        # 'scene_{scene_num:06}'
        for dir in os.listdir(root_dir):
            if dir.startswith('scene_'):
                l.append(os.path.join(root_dir, dir))
        
        return l 

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
            
            for cam_num in range(num_views):
                
                root_name = f'_{(cam_num):05}'
                obj_name = f'_{(cam_num):05}_{i}'
                segmentation_filename = os.path.join(dir_path, 'segmentation'+obj_name+'.png')
                if not os.path.exists(segmentation_filename):
                    continue
                sample_id = scene_name + obj_name
                sample = {'sample_id': sample_id}
                
                sample['depth_all'] = mpimg.imread(os.path.join(dir_path, 'depth'+root_name+'.png'))
                sample['rgb_all'] = mpimg.imread(os.path.join(dir_path, 'rgb'+root_name+'.png'))
                sample['mask'] = (mpimg.imread(segmentation_filename) > 0).astype('int')
                sample['mask'] = np.expand_dims(sample['mask'], axis=0)
                
                sample['depth_all_path'] = os.path.join(dir_path, 'depth'+root_name+'.png')
                sample['rgb_all_path'] = os.path.join(dir_path, 'rgb'+root_name+'.png')
                sample['mask_path'] = segmentation_filename
                
                # sample['depth'] = sample['mask'] * sample['depth_all']
                # sample['rgb'] = np.concatenate([np.expand_dims(img[:,:,i] * mask, axis=2) for i in range(3)], axis=2)
                sample['position'] = position
                sample['scale'] = scale
                sample['orientation'] = orientation
                sample['mesh_filename'] = mesh_filename
                sample['object_center'] = object_description["object_center_{}".format(cam_num)]
                

                samples[idx_i] = sample
                idx_i += 1

        return samples, idx_i
    
    def __getitem__(self, idx):
        sample = self.idx_to_data_dict[idx]
        
        if self.split == 'train' or self.split == 'trainval':
            trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((int(self.size),int(self.size))),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        else:
            trans = transforms.Compose([
                    transforms.Resize((int(self.size),int(self.size))),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])        
        
        # mask_arr = (mpimg.imread(sample['mask']) > 0).astype('int')
        

        # img_rgb = trans(Image.open(sample['rgb_all_path']))
        # img_mask = trans(Image.open(sample['mask_path']))

        img_rgb = trans(np.uint8(sample['rgb_all']) * 255)
        img_mask = trans(np.uint8(sample['mask']) * 255)
        
        img = torch.cat((img_rgb, img_mask[:1,:,:]), 0)
        image = torch.FloatTensor(img)
        return image, torch.FloatTensor([sample['scale']]), torch.FloatTensor(sample['orientation']), torch.FloatTensor(sample['object_center'])


    # def load_mesh_with_pose(self):
    #   mesh=trimesh.load(obj_mesh_filenames[added_object_ind], force='mesh')
    #   scale_mat=np.eye(4)
    #   scale_mat=scale_mat*obj_scales[added_object_ind]
    #   scale_mat[3,3]=1.0
    #   mesh.apply_transform(scale_mat)

    #   transform=np.eye(4)
    #   transform[0:3,0:3]=Quaternion(object_description['orientation']).rotation_matrix
    #   mesh.apply_transform(transform)

    #   transform=np.eye(4)
    #   transform[0:3,3]=object_description['position']
    #   mesh.apply_transform(transform)


'''
def collate_batch_data_to_tensors(proc_batch_dict):
    # Collate processed batch into tensors.
    # Now collate the batch data together
    x_tensor_dict = {}
    x_dict = proc_batch_dict
    device = config.get_device()
    args = config.args

    x_tensor_dict['batch_img'] = torch.stack(x_dict['batch_img_list']).to(device)
    
    x_tensor_dict['batch_obj_bb_list'] = torch.Tensor(
        x_dict['batch_obj_bb_list']).to(device) / 256.0
    
    x_tensor_dict['batch_action_list'] = torch.Tensor(
        x_dict['batch_action_list']).to(device)
    
    x_tensor_dict['batch_other_bb_pred_list'] = torch.Tensor(
        x_dict['batch_other_bb_pred_list']).to(device) / 256.0

    return x_tensor_dict
'''