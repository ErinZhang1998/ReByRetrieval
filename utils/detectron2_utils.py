import tqdm
import os
import yaml 
import json
import pandas as pd

import utils.utils as uu
import utils.blender_proc_utils as bp_utils
import utils.perch_utils as p_utils


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
            rle = ann['segmentation']
            annotation = {
                'bbox' : ann['bbox'],
                'category_id' : int(datagen_yaml[category_id]['obj_cat']),
                'segmentation' : ann['segmentation'],
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