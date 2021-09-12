import tqdm
import os
import yaml 
import json
import pandas as pd
import numpy as np
import pycocotools.mask as coco_mask

import utils.utils as uu
import utils.blender_proc_utils as bp_utils
import utils.perch_utils as p_utils

def load_annotations_real_world(scene_num_to_image_path, anno_path, scene_num_start, scene_num_end):
    # Assumes that /[...]/scene_xxxxxx/yyy.png
    coco_anno = p_utils.COCOSelf(anno_path)
    image_id_to_anns = {}
    for ann_id, ann in coco_anno.ann_id_to_ann.items():
        L = image_id_to_anns.get(ann['image_id'], [])
        L.append(ann)
        image_id_to_anns[ann['image_id']] = L
    
    scene_num_to_sample = {}
    for image_id, image_ann in coco_anno.image_id_to_ann.items():
        
        scene_num_idx = image_ann['file_name'].index('_00')
        scene_num = int(image_ann['file_name'][scene_num_idx+1:scene_num_idx+5])
        if scene_num < scene_num_start or scene_num > scene_num_end:
            continue

        image_file_name_full = scene_num_to_image_path[scene_num]
        height = image_ann['height']
        width = image_ann['width']
        anns = image_id_to_anns[image_id]
        polygons = []
        category_name = None
        for ann in anns:
            polygons += ann['segmentation']
            if category_name is None:
                category_name = coco_anno.category_id_to_ann[ann['category_id']]['name']
            else:
                assert category_name == coco_anno.category_id_to_ann[ann['category_id']]['name']

        # Calculate bounding box
        rles = coco_mask.frPyObjects(polygons, height, width)
        rle = coco_mask.merge(rles)
        mask = coco_mask.decode(rle)

        y_min, x_min  = np.min(np.argwhere(mask), axis=0)
        y_max, x_max = np.max(np.argwhere(mask), axis=0)
        x_len = x_max - x_min + 1
        y_len = y_max - y_min + 1
        bbox = [x_min, y_min, x_len, y_len]
        
        annotation = {
            'bbox' : [int(elem) for elem in bbox],
            'category_id' : None,
            'category_name' : category_name,
            'segmentation' : polygons,
            'polygon' : True,
        }
        
        image_sample = scene_num_to_sample.get(scene_num, None)
        if image_sample is None:
            scene_dir_parts = os.path.normpath(os.path.join(image_file_name_full, os.pardir)).split('/')
            image_id_across_dataset = '-'.join(scene_dir_parts[-2:])
            image_sample = {
                'file_name' : image_file_name_full,
                'width' : int(width),
                'height' : int(height),
                'image_id' : image_id_across_dataset,
                'annotations' : [annotation],
            }
        else:
            annotations_new = image_sample['annotations'] + [annotation]
            image_sample['annotations'] = annotations_new
        scene_num_to_sample[scene_num] = image_sample
    
    json_paths = []
    for scene_num, image_sample in scene_num_to_sample.items():
        
        scene_dir = os.path.normpath(os.path.join(scene_num_to_image_path[scene_num], os.pardir))
        json_path = os.path.join(scene_dir, 'detectron_annotations.json')
        print("json_path: ", json_path)
        json_string = json.dumps([image_sample])
        json_file = open(json_path, 'w+')
        json_file.write(json_string)
        json_file.close()
        json_paths.append(json_path)
    
    return json_paths
    
def process_data_real_world(args, split):
    if split == 'train':
        scene_dir = args.real_world.train.data_path
        scene_num_start = args.real_world.train.scene_num_start
        scene_num_end = args.real_world.train.scene_num_end
    elif split == 'test':
        scene_dir = args.real_world.test.data_path
        scene_num_start = args.real_world.test.scene_num_start
        scene_num_end = args.real_world.test.scene_num_end
    else:
        raise
    
    scene_num_to_image_path = {}
    for scene_num in range(scene_num_start, scene_num_end+1):
        scene_num_to_image_path[scene_num] = os.path.join(scene_dir, f'scene_{scene_num:06}', f'rgb_{scene_num:04}.png')
    
    processed_json_paths = []
    for anno_path in args.real_world.annotation_paths:
        json_paths = load_annotations_real_world(scene_num_to_image_path, anno_path, scene_num_start, scene_num_end)
        processed_json_paths += json_paths
    
    return processed_json_paths



def load_annotations_detectron(args, split, df, one_scene_dir, image_id_prefix=None):
    scene_num = int(one_scene_dir.split('/')[-1].split('_')[-1])

    yaml_file_prefix = '_'.join(one_scene_dir.split('/')[-2:])
    if split == 'train':
        yaml_file = os.path.join(args.files.training_yaml_file_dir, '{}.yaml'.format(yaml_file_prefix))
    else:
        yaml_file = os.path.join(args.files.testing_yaml_file_dir, '{}.yaml'.format(yaml_file_prefix))

    yaml_file_obj = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
    datagen_yaml = bp_utils.from_yaml_to_object_information(yaml_file_obj, df)
    coco_fname = os.path.join(one_scene_dir, 'coco_data', 'coco_annotations.json')
    coco = p_utils.COCOSelf(coco_fname)

    data_dict_list = []
    for image_id, image_ann in coco.image_id_to_ann.items():        
        image_id_across_dataset = '-'.join([str(scene_num), str(image_id)])
        if image_id_prefix is not None:
            image_id_across_dataset = image_id_prefix + "-" + image_id_across_dataset
        image_file_name_full = os.path.join(one_scene_dir, image_ann['file_name'])

        annotations = []
        for category_id, ann in coco.image_id_to_category_id_to_ann[image_id].items():
            assert image_id == ann['image_id']
            
            if category_id == 0:
                continue
            if ann['area'] < args.dataset_config.ignore_num_pixels:
                continue
            if category_id not in datagen_yaml:
                continue
            if 'scale' not in datagen_yaml[category_id]:
                continue
            
            annotation = {
                'bbox' : ann['bbox'],
                'category_id' : int(datagen_yaml[category_id]['obj_cat']),
                'segmentation' : ann['segmentation'],
                'polygon' : False,
            }
            annotations += [annotation]
        
        image_sample = {
            'file_name' : image_file_name_full,
            'width' : image_ann['width'],
            'height' : image_ann['height'],
            'image_id' : image_id_across_dataset,
            'annotations' : annotations,
        }
        data_dict_list += [image_sample]
    return data_dict_list

                
def get_data_detectron(args, split):
    df = pd.read_csv(args.files.csv_file_path)
    
    if split == 'train':
        scene_dir = args.files.training_scene_dir
    elif split == 'test':
        scene_dir = args.files.testing_scene_dir
    else:
        raise

    image_id_prefix = scene_dir.split('/')[-1]
    dir_list = uu.data_dir_list(
        scene_dir, 
        must_contain_file = ['0.hdf5', 'coco_data/coco_annotations.json']
    )

    if args.dataset_config.only_load < 0:
        dir_list_load = dir_list
    else:
        dir_list_load = dir_list[:args.dataset_config.only_load]
    
    data_dict_lists = []
    dir_list_load_len = len(dir_list_load)
    for i in tqdm.tqdm(range(dir_list_load_len)):
        dir_path = dir_list_load[i]
        data_dict_list = load_annotations_detectron(args, split, df, dir_path, image_id_prefix = image_id_prefix)
        json_string = json.dumps(data_dict_list)
        json_file = open(os.path.join(dir_path, 'detectron_annotations.json'), 'w+')
        json_file.write(json_string)
        json_file.close()
        # data_dict_lists += data_dict_list
    
    # return data_dict_lists