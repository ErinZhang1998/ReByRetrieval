import numpy as np
import os 
import time 
import torch
import json
import datetime
from collections import OrderedDict
import utils.logging as logging

logger = logging.get_logger(__name__)

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

def create_experiment_dirs(args, wandb_run_name):
    all_dir_in_experiment = []
    dir_dict = {}
    if args.experiment_save_dir is None:
        experiment_save_dir_default = args.experiment_save_dir_default
        this_experiment_dir = os.path.join(experiment_save_dir_default, wandb_run_name)
        model_dir = os.path.join(this_experiment_dir, "models")
        image_dir = os.path.join(this_experiment_dir, "images")
        prediction_dir = os.path.join(this_experiment_dir, "predictions")
        
        all_dir_in_experiment += [
            experiment_save_dir_default,
            this_experiment_dir,
            model_dir,
            image_dir,
            prediction_dir,
        ]
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
    
    dir_dict = {
        'this_experiment_dir' : this_experiment_dir,
        'model_dir' : model_dir,
        'image_dir' : image_dir,
        'prediction_dir' : prediction_dir
    }
    
    return dir_dict


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

class COCOAnnotation(object):

    def __init__(self, json_path):
        '''
        Unit = json 
        '''
        self.model_name_to_model_full_name = {}
        annotations = json.load(open(json_path))

        image_id_to_ann = dict()
        for ann in annotations['images']:
            image_id_to_ann[ann['id']] = ann 
        
        category_id_to_ann = dict()
        for ann in annotations['categories']:
            category_id_to_ann[ann['id']] = ann
            model_name = ann['name']
            cat = ann['synset_id']
            model_id = ann['model_id']
            shapenet_category_id = ann['shapenet_category_id']
            shapenet_object_id = ann['shapenet_object_id']
    
            # category_name = category_dict[f'{cat}_{model_id}']
            self.model_name_to_model_full_name[model_name] = f'{shapenet_category_id}_{shapenet_object_id}' #f'{cat}_{model_id}'
        
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
        
        # self.scene_dir = scene_dir
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
  

class COCOAnnotationScene(object):

    def __init__(self, root_data_dir):
        '''
        Unit = json 
        '''
        self.root_data_dir = root_data_dir
        self.annotations_bank = {}
        self.model_name_to_model_full_name = {}
    
    def add_scene(self, scene_num):
        scene_dir = os.path.join(self.root_data_dir, f'scene_{scene_num:06}')
        json_path = os.path.join(scene_dir, 'annotations.json')
        new_coco_anno = COCOAnnotation(json_path)
        self.annotations_bank[scene_num] = (scene_dir, new_coco_anno)
        self.model_name_to_model_full_name.update(new_coco_anno.model_name_to_model_full_name)
    
    def get_ann(self, scene_num, key, key_id):
        if not scene_num in self.annotations_bank:
            self.add_scene(scene_num)
        
        return self.annotations_bank[scene_num][1].get_ann(key, key_id)
    
    def get_ann_with_image_category_id(self, scene_num, image_id, category_id=None):
        if not scene_num in self.annotations_bank:
            self.add_scene(scene_num)
        
        return self.annotations_bank[scene_num][1].get_ann_with_image_category_id(image_id, category_id)
    
    def get_image_anns(self, scene_num):
        if not scene_num in self.annotations_bank:
            self.add_scene(scene_num)
        
        return self.annotations_bank[scene_num][1].get_image_anns()
  