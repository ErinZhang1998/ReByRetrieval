import numpy as np
import os 
import time 
import json
import torch
import csv
import datetime
from collections import OrderedDict
import utils.logging as logging

import open3d as o3d

logger = logging.get_logger(__name__)

def write_to_csv(csv_file, dict_data, csv_columns):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

class Struct:
    '''The recursive class for building and representing objects with.'''
    def __init__(self, obj):
        self.obj_dict = obj
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)
    
    def __getitem__(self, val):
        return self.__dict__[val]
    
    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def fill_in_args_from_default(my_dict, default_dict):
    filled_dict = {}
    for k,v in my_dict.items():
        filled_dict[k] = v
    for k,v in default_dict.items():
        try:
            myv = my_dict[k]
            if isinstance(myv, dict):
                assert isinstance(v, dict)
                subdict = fill_in_args_from_default(myv, v)
                filled_dict[k] = subdict
            else:
                filled_dict[k] = myv
        except:
            filled_dict[k] = v
    return filled_dict

def get_timestamp():                                                                                          
    ts = time.time()                                                                                            
    timenow = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')                             
    return timenow 

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

'''Load model from'''
def remove_module_key_transform(key):
    parts = key.split(".")
    if parts[0] == 'module':
        return ".".join(parts[1:])
    return key

def rename_state_dict_keys(ckp_path, key_transformation):
    state_dict = torch.load(ckp_path)['model_state_dict']
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    return new_state_dict

def load_model_from(args, model, data_parallel=False):
    ms = model.module if data_parallel else model
    if args.model_config.model_path is not None:
        logger.info("=> Loading model file from: {}".format(args.model_config.model_path))
        ckp_path = os.path.join(args.model_config.model_path)
        checkpoint = torch.load(ckp_path)
        try:
            ms.load_state_dict(checkpoint['model_state_dict'])
        except:
            state_dict = rename_state_dict_keys(ckp_path, remove_module_key_transform)
            ms.load_state_dict(state_dict)

def save_model(epoch, model, model_dir):
    model_path = os.path.join(model_dir, '{}.pth'.format(epoch))
    try:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, model_path)
    except:
        logger.info(f"ERROR: Cannot save model at {model_path}")

'''Training'''
def create_experiment_dirs(args, wandb_run_name):
    all_dir_in_experiment = []
    dir_dict = {}
    if args.experiment_save_dir is None:
        experiment_save_dir_default = args.experiment_save_dir_default
        create_dir(experiment_save_dir_default)
        this_experiment_dir = os.path.join(experiment_save_dir_default, wandb_run_name)
    else:
        this_experiment_dir = args.experiment_save_dir
    
    model_dir = os.path.join(this_experiment_dir, "models")            
    image_dir = os.path.join(this_experiment_dir, "images")
    prediction_dir = os.path.join(this_experiment_dir, "predictions")
    all_dir_in_experiment += [
        this_experiment_dir,
        model_dir,
        image_dir,
        prediction_dir,
    ]
    
    for d in all_dir_in_experiment:
        create_dir(d)
    
    return this_experiment_dir, image_dir, model_dir, prediction_dir
    # {
    #     'this_experiment_dir' : this_experiment_dir,
    #     'model_dir' : model_dir,
    #     'image_dir' : image_dir,
    #     'prediction_dir' : prediction_dir
    # }


def data_dir_list(root_dir, must_contain_file = ['annotations.json']):
    l = []
    for subdir in os.listdir(root_dir):
        if subdir.startswith('scene_'):
            subdir_path = os.path.join(root_dir, subdir)
            contain_all = True 
            for file in must_contain_file:
                scene_description_dir = os.path.join(subdir_path, file)
                if not os.path.exists(scene_description_dir):
                    contain_all = False 
                    break 
            if contain_all:
                l.append(subdir_path)

    return l 


"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def generateCroppedPointCloud(depth_img, intrinsic, camera_to_world_mat, img_width, img_height):
    od_cammat = cammat2o3d(intrinsic, img_width, img_height)
    od_depth = o3d.geometry.Image(depth_img)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
    transformed_cloud = o3d_cloud.transform(camera_to_world_mat)

    return transformed_cloud

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


class COCOAnnotation(object):

    def __init__(self, json_path):
        '''
        Unit = json 
        '''
        annotations = json.load(open(json_path))

        
        image_id_to_ann = dict()
        for ann in annotations['images']:
            image_id_to_ann[ann['id']] = ann 
        
        category_id_to_model_names = {}
        category_id_to_ann = dict()
        for ann in annotations['categories']:
            category_id_to_ann[ann['id']] = ann
            model_name = ann['name']
            category_id_to_model_names[ann['id']] = model_name
    
        self.category_id_to_model_names = category_id_to_model_names
        
        ann_id_to_ann = dict()
        image_category_id_to_ann = dict()
        for ann in annotations['annotations']:
            ann_id_to_ann[ann['id']] = ann

            D = image_category_id_to_ann.get(ann['image_id'], {})
            D[ann['category_id']] = ann
            image_category_id_to_ann[ann['image_id']] = D

        total_ann_dict = {
            'images' : image_id_to_ann,
            'categories' : category_id_to_ann,
            'annotations' : ann_id_to_ann,
            'annotations2' : image_category_id_to_ann,
        } 
        
        self.json_path = json_path
        self.total_ann_dict = total_ann_dict
        
    def get_ann(self, key, key_id):
        assert key in ['images', 'categories', 'annotations']
        ann_dict = self.total_ann_dict[key]
        return ann_dict[key_id]
    
    def get_ann_with_image_category_id(self, image_id, category_id=None):
        if category_id is None:
            return self.total_ann_dict['annotations2'][image_id]

        return self.total_ann_dict['annotations2'][image_id][category_id]
    
    def get_image_anns(self):
        return self.total_ann_dict['images']
  