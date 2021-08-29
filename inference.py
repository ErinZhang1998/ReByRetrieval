from __future__ import print_function

import pickle
import yaml
import json
import copy
import numpy as np
import os
import argparse
import shutil
import trimesh
import open3d as o3d
import pandas as pd 

import utils.qualitative_utils as q_utils
import utils.perch_utils as p_utils
import utils.utils as uu
import incat_dataset

import utils.datagen_utils as datagen_utils
from scipy.spatial.transform import Rotation as R, rotation        


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", dest="config_file")
parser.add_argument("--k", dest="k", type=int, default=5)

parser.add_argument("--start_scene_idx", dest="start_scene_idx", type=int, default=0)
parser.add_argument("--end_scene_idx", dest="end_scene_idx", type=int, default=100000)
parser.add_argument("--scene_indices", nargs="+", default=[])
parser.add_argument("--clean_up", dest="clean_up", action='store_true')

parser.add_argument("--run_gt", dest="run_gt", action='store_true')
parser.add_argument("--run_in_cat_random", dest="run_in_cat_random", action='store_true')
parser.add_argument("--run_random", dest="run_random", action='store_true')
parser.add_argument("--run_pred", dest="run_pred", action='store_true')
parser.add_argument("--run_cuboid", dest="run_cuboid", action='store_true')

parser.add_argument("--query_epochs", dest="query_epochs", type=int, default=30)
parser.add_argument("--query_save_dir", dest="query_save_dir", type=str)
parser.add_argument("--query_data_dir", dest="query_data_dir", type=str)

parser.add_argument("--target_epochs", dest="target_epochs", type=int, default=30)
parser.add_argument("--target_save_dir", dest="target_save_dir", type=str)
parser.add_argument("--target_data_dir", dest="target_data_dir", type=str)


parser.add_argument("--yaml_file_root_dir", dest="yaml_file_root_dir", help='Yaml files for blender proc')

parser.add_argument("--csv_file_path", dest="csv_file_path", help='CSV file of Shapenet object model annotations')
parser.add_argument("--shapenet_filepath", dest="shapenet_filepath")
parser.add_argument("--model_save_root_dir", dest="model_save_root_dir")


parser.add_argument("--result_save_dir", dest="result_save_dir", type=str)


r = R.from_euler('xyz', [(1/2)*np.pi, 0, 0], degrees=False)
UPRIGHT_MAT = np.eye(4)
UPRIGHT_MAT[0:3, 0:3] = r.as_matrix()
SHAPENET_PATH = '/raid/xiaoyuz1/ShapeNetCore.v2'
MODEL_SAVE_ROOT_DIR = '/raid/xiaoyuz1/perch/perch_balance/models'


def save_cuboid_of_model_size(model_save_root_dir, model_name, actual_size):
    '''
    Args:
        model_save_root_dir: 
            the "model_dir" directory used in Perch 
        model_name: 
            name of model used by Perch 
        actual_size: 
            the size (x,y,z) that will be used to generate cuboid 
        shapenet_file_name: 
            path to shapenet model_normalized.obj file 
    
    Return:
        mesh_scale:
            list of 3 numbers representing scale
    '''
    model_save_dir = os.path.join(model_save_root_dir, model_name)

    if not os.path.exists(model_save_dir):
        assert len(model_name.split("-")) > 1
        os.mkdir(model_save_dir)
    
    model_fname = os.path.join(model_save_dir, 'textured.obj')
    model_ply_fname = os.path.join(model_save_dir, 'textured.ply')

    lx,ly,lz = actual_size
    cuboid_mesh = trimesh.creation.box((lx,ly,lz))
    cuboid_mesh.export(model_fname)

    copy_textured_mesh = o3d.io.read_triangle_mesh(model_fname)
    o3d.io.write_triangle_mesh(model_ply_fname, copy_textured_mesh)




def update_model_with_size(synset_id, model_id, model_name, actual_size):
    '''
    Args:
        synset_id:
            Shapnet synset id used to access the shapnet model file
        model_id:
            Shapenet model id used to access the shapenet model file
        model_name:
            name of model used by Perch 
        actual_size: 
            (3,) size of the mesh bounding box after it is turned "upright" according to perch_scene.
         
    Return:
        object_mesh:
            scaled object mesh, according to actual_size
        mesh_scale:
            list of 3 numbers representing scale
    '''
    shapenet_file_name = os.path.join(
            options.shapenet_filepath,
            '{}/{}/models/model_normalized.obj'.format(synset_id, model_id),
        )
    return save_correct_size_model(options.model_save_root_dir, model_name, actual_size, shapenet_file_name)


def update_category_annotation(category_anno_query, category_anno_target):
    '''
    Args:
        category_anno_query: 
            the original annotation in "categories" field of COCO json file.
        category_anno_target:
            the updated/predicted annotation in "categories" field of another COCO json file.
            OR
            just a mock dictionary with fields: 
            shapenet_category_id, shapenet_object_id, synset_id, model_id

    Returns:
        category_anno:
            new category annotation with 
            shapenet_category_id, shapenet_object_id, synset_id, model_id, size 
            updated
            id, name, actual_size, position, quat 
            remains the same
    '''
    # category_anno = copy.deepcopy(category_anno_query)
    category_anno = category_anno_query
    category_anno['shapenet_category_id'] = category_anno_target['shapenet_category_id']
    category_anno['shapenet_object_id'] = category_anno_target['shapenet_object_id']
    category_anno['synset_id'] = category_anno_target['synset_id']
    category_anno['model_id'] = category_anno_target['model_id']
    _, scale = update_model_with_size(
        category_anno['synset_id'], 
        category_anno['model_id'], 
        category_anno['name'], 
        category_anno['actual_size'],
    )
    category_anno['size'] = [float(item) for item in scale]
    return category_anno


def run_cuboid(scene_num):
    scene_dir = os.path.join(options.query_data_dir, f'scene_{scene_num:06}')
    all_json_file = os.path.join(scene_dir, 'annotations.json')
    anno = json.load(open(all_json_file))

    new_categories = []
    old_name_to_new_name = {}
    for category_ann in anno['categories']:
        ann = copy.deepcopy(category_ann)
        new_name = '-'.join([ann['name'], "cuboid"])
        old_name_to_new_name[ann['name']] = new_name
        ann['name'] = new_name
        save_cuboid_of_model_size(options.model_save_root_dir, ann['name'], ann['actual_size'] )
        new_categories.append(ann)
    
    new_annotations = []
    for ann in anno['annotations']:
        if ann['percentage_not_occluded'] < 0.1:
            continue
        ann['model_name'] = old_name_to_new_name[ann['model_name']]
        new_annotations.append(ann)
        
    image_json_path = os.path.join(scene_dir, 'annotations_cuboid.json')
    shutil.copyfile(all_json_file, image_json_path)
    annotations_cuboid = json.load(open(image_json_path))
    annotations_cuboid['categories'] = new_categories
    annotations_cuboid['annotations'] = new_annotations

    json_string = json.dumps(annotations_cuboid)
    json_file = open(image_json_path, "w+")
    json_file.write(json_string)
    json_file.close()
    
    return [image_json_path]


def run_gt(scene_num):
    print("run_gt: ", scene_num)
    scene_dir = os.path.join(options.query_data_dir, f'scene_{scene_num:06}')
    json_annotations = json.load(open(os.path.join(scene_dir, "annotations.json")))
    for category_ann in json_annotations["categories"]:
        update_model_with_size(
            category_ann['synset_id'], 
            category_ann['model_id'], 
            category_ann['name'], 
            np.asarray(category_ann['actual_size']),
        )


def run_pred_scene(scene_num, image_id2category_id2sample_id, target_coco_annos, new_name_root = 'pred_majority'):
    scene_dir = os.path.join(options.query_data_dir, f'scene_{scene_num:06}')
    all_json_file = os.path.join(scene_dir, 'annotations.json')
    coco_anno = p_utils.COCOAnnotation(all_json_file)
    
    image_json_paths = []
    for image_id in coco_anno.total_ann_dict['images'].keys():
        image_json_path = os.path.join(scene_dir, 'annotations_{}_{}.json'.format(new_name_root, image_id))
        if image_id not in image_id2category_id2sample_id:
            continue 
        category_id2sample_id = image_id2category_id2sample_id[image_id]

        image_annotation = copy.deepcopy(coco_anno.total_ann_dict['images'][image_id])
        this_image_annotations = list(coco_anno.get_ann_with_image_category_id(image_id).values())
        
        category_annotations = []
        old_name_to_new_name = {}
        for category_id, category_ann in coco_anno.total_ann_dict['categories'].items():
            if category_id not in category_id2sample_id:
                continue 
            ann = copy.deepcopy(category_ann)
            target_scene_num, target_image_id, target_category_id = category_id2sample_id[category_id]
            target_annotation = target_coco_annos.get_ann(target_scene_num, 'categories', target_category_id)

            # Change annotation model name 
            new_name = '-'.join([ann['name'], 'image_{}_{}_{}'.format(image_id, new_name_root, options.k)])
            old_name_to_new_name[ann['name']] = new_name
            ann['name'] = new_name
            
            category_anno_target = {
                'shapenet_category_id': int(target_annotation['shapenet_category_id']),
                'shapenet_object_id' : int(target_annotation['shapenet_object_id']),
                'synset_id' : target_annotation['synset_id'],
                'model_id' : target_annotation['model_id'],
            }
            new_ann = update_category_annotation(ann, category_anno_target)
            category_annotations.append(new_ann)
        
        new_this_image_annotations = []
        for ann in this_image_annotations:
            if ann['percentage_not_occluded'] < 0.1:
                continue
            if ann['model_name'] not in old_name_to_new_name:
                continue 
            ann['model_name'] = old_name_to_new_name[ann['model_name']]
            new_this_image_annotations.append(ann)
        
        
        shutil.copyfile(all_json_file, image_json_path)
        annotations_image_id = json.load(open(image_json_path))
        annotations_image_id['images'] = [image_annotation]
        annotations_image_id['categories'] = category_annotations
        annotations_image_id['annotations'] = new_this_image_annotations

        json_string = json.dumps(annotations_image_id)
        json_file = open(image_json_path, "w+")
        json_file.write(json_string)
        json_file.close()

        image_json_paths.append(image_json_path)
    return image_json_paths

# def get_features(args, epochs, save_dir, data_dir, feature_file_template = '{}_embedding.npy'):
#     prediction_dir = os.path.join(save_dir, 'predictions')
#     args.files.testing_scene_dir = data_dir
#     test_dataset = incat_dataset.InCategoryClutterDataset('test', args)
#     assert os.path.exists(prediction_dir)
#     features = np.load(os.path.join(prediction_dir, feature_file_template.format(epoch)))
#     feats, sample_ids = q_utils.read_npy(prediction_dir, epochs)
#     return test_dataset, feats, sample_ids

def get_features(save_dir, epoch, fname_template = '{}_embedding.npy'):
    prediction_dir = os.path.join(save_dir, 'predictions')
    features = np.load(os.path.join(prediction_dir, fname_template.format(epoch)))
    return features

def get_sample_ids(save_dir, epoch, fname_template = '{}_sample_id.npy'):
    prediction_dir = os.path.join(save_dir, 'predictions')
    sample_id = np.load(os.path.join(prediction_dir, fname_template.format(epoch))) 
    sample_id_res = []
    for L in sample_id:
        sample_id_res.append('-'.join([str(int(item)) for item in L]))
    return sample_id_res

# Goal is to run pose estimation for one object with predicted other object

def run_pred(args):
    # test_dataset, query_feats, query_sample_ids = get_features(args, options.query_epochs, options.query_save_dir, options.query_data_dir)
    # target_dataset, target_feats, target_sample_ids = get_features(args, options.target_epochs, options.target_save_dir, options.target_data_dir)
    
    target_dataset = incat_dataset.InCategoryClutterDataset('test', args)
    
    query_feats = get_features(options.query_save_dir, options.query_epochs, feature_file_template = '{}_img_embed.npy')
    query_sample_ids = get_sample_ids(options.query_save_dir, options.query_epochs, fname_template = '{}_sample_id.npy')
    
    target_feats = get_features(options.target_save_dir, options.target_epochs, feature_file_template = '{}_img_embed.npy')
    target_sample_ids = get_sample_ids(options.target_save_dir, options.target_epochs, fname_template = '{}_sample_id.npy')
    
    query_feats = q_utils.torchify(query_feats)
    target_feats = q_utils.torchify(target_feats)
    arg_sorted_dist = q_utils.get_arg_sorted_dist(query_feats, target_feats)

    _, target_object_id = q_utils.sample_ids_to_cat_and_id(target_dataset, target_sample_ids)

    top_k_arg_sorted_dist = arg_sorted_dist[:,:options.k]

    selected_target_sample_id = []
    for row, test_batch_idx_row in zip(target_object_id[top_k_arg_sorted_dist], top_k_arg_sorted_dist):
        res = np.unique(row, return_counts=True)
        val, count = res
        max_idx = np.argsort(count)[::-1][0]
        max_val_idx = np.where(row == val[max_idx])[0]
        max_val_idx = max_val_idx[0]
        max_test_batch_idx = test_batch_idx_row[max_val_idx]
        selected_target_sample_id.append(target_sample_ids[max_test_batch_idx])
    
    # for each category id --> predicted category
    scene_num2image_id2category_id2sample_id = {}
    for sample_id, target_sample_id in zip(query_sample_ids, selected_target_sample_id):
        scene_num, image_id, category_id = np.array(sample_id.split('-')).astype(int)    
        
        image_id2category_id2sample_id = scene_num2image_id2category_id2sample_id.get(scene_num, {})
        category_id2sample_id = image_id2category_id2sample_id.get(image_id, {})
        category_id2sample_id[category_id] = np.array(target_sample_id.split('-')).astype(int)
        
        image_id2category_id2sample_id.update({
            image_id : category_id2sample_id,
        })
        scene_num2image_id2category_id2sample_id.update({
            scene_num : image_id2category_id2sample_id,
        })
    
    target_coco_annos = p_utils.COCOAnnotationScenes(options.target_data_dir)
    image_json_paths_list = []
    error_scenes = []
    for scene_num,image_id2category_id2sample_id in scene_num2image_id2category_id2sample_id.items():
        
        if scene_num < options.start_scene_idx:
            continue
        if scene_num >= options.end_scene_idx:
            continue
        if options.scene_indices != []:
            if scene_num not in options.scene_indices:
                continue
        print("Processing: ", scene_num)
        try:
            image_json_paths = run_pred_scene(scene_num, image_id2category_id2sample_id, target_coco_annos)
        except:
            error_scenes.append(scene_num)
            image_json_paths = []
        image_json_paths_list += image_json_paths
        # break
    
    with open(os.path.join(options.result_save_dir, 'run_pred_processed_json.pkl'), 'wb+') as fp:
        pickle.dump([image_json_paths_list, error_scenes], fp)


def run_random(df_csv, scene_num, random_any=True):
    scene_dir = os.path.join(options.query_data_dir, f'scene_{scene_num:06}')
    all_json_file = os.path.join(scene_dir, 'annotations.json')
    json_annotations = json.load(open(all_json_file))
    coco_anno = p_utils.COCOAnnotation(all_json_file)
    image_json_paths = []
    
    for image_id, image_annotation in coco_anno.total_ann_dict['images'].items():
        image_annotation = copy.deepcopy(coco_anno.total_ann_dict['images'][image_id])
        
        this_image_annotations = list(coco_anno.get_ann_with_image_category_id(image_id).values())
        
        category_annotations = []
        old_name_to_new_name = {}
        for _, category_ann in coco_anno.total_ann_dict['categories'].items():
            ann = copy.deepcopy(category_ann)
            if random_any:
                random_row = df_csv.iloc[np.random.choice(len(df_csv))]

                # Change annotation model name 
                new_name = '-'.join([ann['name'], f'image_{image_id}_random'])
                old_name_to_new_name[ann['name']] = new_name
                ann['name'] = new_name

            else:
                # df_select = df_csv[df_csv['objId'] == ann['shapenet_object_id']]
                # random_row = df_select.iloc[np.random.choice(len(df_select))]
                shapnet_object_id = int(ann['shapenet_object_id'])
                if shapnet_object_id == 0 or \
                    shapnet_object_id == 1 or \
                    shapnet_object_id == 7 or \
                    shapnet_object_id == 10 or \
                    shapnet_object_id == 12 or \
                    shapnet_object_id == 13 or \
                    shapnet_object_id == 14 or \
                    shapnet_object_id == 15 or \
                    shapnet_object_id == 16:
                    df_select = df_csv[df_csv['objId'] == shapnet_object_id]
                elif shapnet_object_id == 2 or \
                    shapnet_object_id == 3 or \
                    shapnet_object_id == 4 or \
                    shapnet_object_id == 5 or \
                    shapnet_object_id == 6:
                    df_select = df_csv[df_csv['objId'].isin([2,3,4,5,6])]
                elif shapnet_object_id == 9 or \
                    shapnet_object_id == 11:
                    df_select = df_csv[df_csv['objId'].isin([9,11])]
                elif shapnet_object_id == 8 or \
                    shapnet_object_id == 17 or \
                    shapnet_object_id == 18 or \
                    shapnet_object_id == 19:
                    df_select = df_csv[df_csv['objId'].isin([8,17,18,19])]
                else:
                    print(shapnet_object_id)
                    df_select = df_csv[df_csv['objId'] == shapnet_object_id]
                    raise 
                
                random_row = df_select.iloc[np.random.choice(len(df_select))]

            
                # Change annotation model name 
                new_name = '-'.join([ann['name'], f'image_{image_id}_random_cat'])
                old_name_to_new_name[ann['name']] = new_name
                ann['name'] = new_name
            
            synset_id = random_row['synsetId']
            category_anno_target = {
                'shapenet_category_id': int(random_row['catId']),
                'shapenet_object_id' : int(random_row['objId']),
                'synset_id' : f'0{synset_id}',
                'model_id' : random_row['ShapeNetModelId'],
            }

            new_ann = update_category_annotation(ann, category_anno_target)
            category_annotations.append(new_ann)
        
        new_this_image_annotations = []
        for ann in this_image_annotations:
            if ann['percentage_not_occluded'] < 0.1:
                continue
            ann['model_name'] = old_name_to_new_name[ann['model_name']]
            new_this_image_annotations.append(ann)
        
        image_json_path = os.path.join(scene_dir, 'annotations_random_{}.json'.format(image_id))
        shutil.copyfile(all_json_file, image_json_path)
        annotations_image_id = json.load(open(image_json_path))
        annotations_image_id['images'] = [image_annotation]
        annotations_image_id['categories'] = category_annotations
        annotations_image_id['annotations'] = new_this_image_annotations

        json_string = json.dumps(annotations_image_id)
        json_file = open(image_json_path, "w+")
        json_file.write(json_string)
        json_file.close()

        image_json_paths.append(image_json_path)
    return image_json_paths


def clean_up(options):
    query_data_dir = options.query_data_dir
    dir_list = uu.data_dir_list(query_data_dir)
    for scene_dir in dir_list:
        scene_num = int(scene_dir.split('/')[-1].split('_')[-1])
        if scene_num < options.start_scene_idx:
            continue
        if scene_num >= options.end_scene_idx:
            continue
        for fname in os.listdir(scene_dir):
            if fname.startswith('annotations'): 
                fname_list = fname.split('.')[0].split('_')
                if len(fname_list) < 2:
                    continue 
                assert fname != "annotations.json"
                full_fname = os.path.join(scene_dir, fname)
                print("Remove: ", full_fname)
                os.remove(full_fname)


if __name__ == "__main__":
    options = parser.parse_args()
    if options.clean_up:
        clean_up(options)
    else:

        if options.run_pred:
            f =  open(options.config_file)
            args_dict = yaml.safe_load(f)
            default_args_dict = yaml.safe_load(open('configs/default.yaml'))
            args_dict_filled = uu.fill_in_args_from_default(args_dict, default_args_dict)
            args = uu.Struct(args_dict_filled)
            
            args.files.testing_scene_dir = options.target_data_dir
            args.files.yaml_file_root_dir = options.yaml_file_root_dir
            # '/raid/xiaoyuz1/blender_proc/testing_set_2/yaml_files'
            args.files.csv_file_path = options.csv_file_path
            # '/raid/xiaoyuz1/new_august_21/preselect_table_top.csv'

            run_pred(args)
        else:
            # if options.run_in_cat_random or options.run_random:
            #     clean_up(options)
            df_csv = pd.read_csv(options.csv_file_path)
            for subdir in os.listdir(options.query_data_dir):
                if not subdir.startswith('scene_'):
                    continue
                scene_num = int(subdir.split('_')[-1])
                if scene_num < options.start_scene_idx:
                    continue
                if scene_num >= options.end_scene_idx:
                    continue
                if options.scene_indices != []:
                    if scene_num not in options.scene_indices:
                        continue
                print("Processing: ", scene_num)
                if options.run_in_cat_random:
                    run_random(df_csv, scene_num, random_any=False)
                elif options.run_gt:
                    run_gt(scene_num)
                elif options.run_random:
                    run_random(df_csv, scene_num, random_any=True)
                elif options.run_cuboid:
                    run_cuboid(scene_num)
                
            
        # clean_up(options.query_data_dir)
        # # f =  open(options.config_file)
        # # args_dict = yaml.safe_load(f)
        # # default_args_dict = yaml.safe_load(open('configs/default.yaml'))
        # # args_dict_filled = uu.fill_in_args_from_default(args_dict, default_args_dict)
        # # args = uu.Struct(args_dict_filled)
        # # inference(args, options)
        # query_annos = p_utils.COCOAnnotationScene(options.query_data_dir)
        # category_bank = get_category_bank(options.target_data_dir)
        # selected_model_ids = {}
        
        # for subdir in os.listdir(options.query_data_dir):
        #     if not subdir.startswith('scene_'):
        #         continue
        #     scene_num = int(subdir.split('_')[-1])
        #     # scene_num = 38
        #     # image_id = 0
            
        #     for image_id, image_ann in query_annos.get_image_anns(scene_num).items():
        #         D = run_random(scene_num, image_id, random = options.ran_random)
        #         L = selected_model_ids.get(scene_num, {})
        #         L[image_id] = D
        #         selected_model_ids[scene_num] = L
                
            
        #     if options.ran_random:
        #         json_string = json.dumps(selected_model_ids)
        #         json_file = open(os.path.join(options.query_save_dir, 'random_chosen.json'), "w+")
        #         json_file.write(json_string)
        #         json_file.close()