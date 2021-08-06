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

import utils.qualitative_utils as q_utils
import utils.utils as uu
import incat_dataset
import utils.dataset_utils as data_utils
import data_gen.datagen_utils as datagen_utils
from scipy.spatial.transform import Rotation as R, rotation        


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", dest="config_file")
parser.add_argument("--k", dest="k", type=int, default=5)
parser.add_argument("--clean_up", dest="clean_up", action='store_true')
parser.add_argument("--ran_random", dest="ran_random", action='store_true')
parser.add_argument("--ran_pred", dest="ran_pred", action='store_true')

parser.add_argument("--query_epochs", dest="query_epochs", type=int, default=30)
parser.add_argument("--base_epochs", dest="base_epochs", type=int, default=30)
parser.add_argument("--query_save_dir", dest="query_save_dir", type=str)
parser.add_argument("--base_save_dir", dest="base_save_dir", type=str)
parser.add_argument("--query_data_dir", dest="query_data_dir", type=str)
parser.add_argument("--base_data_dir", dest="base_data_dir", type=str)

parser.add_argument("--result_save_dir", dest="result_save_dir", type=str)


r = R.from_euler('xyz', [(1/2)*np.pi, 0, 0], degrees=False)
UPRIGHT_MAT = np.eye(4)
UPRIGHT_MAT[0:3, 0:3] = r.as_matrix()
SHAPENET_PATH = '/raid/xiaoyuz1/ShapeNetCore.v2'
MODEL_SAVE_ROOT_DIR = '/raid/xiaoyuz1/perch/perch_balance/models'

def save_correct_size_model(model_save_root_dir, model_name, actual_size, shapenet_file_name):
    model_save_dir = os.path.join(model_save_root_dir, model_name)
    model_fname = os.path.join(model_save_dir, 'textured.obj')
    model_ply_fname = os.path.join(model_save_dir, 'textured.ply')

    model_fname_backup = os.path.join(model_save_dir, 'textured_backup.obj')
    model_ply_fname_backup = os.path.join(model_save_dir, 'textured_backup.ply')

    if os.path.exists(model_fname) and not os.path.exists(model_fname_backup):
        shutil.copyfile(model_fname, model_fname_backup)
    
    if os.path.exists(model_ply_fname) and not os.path.exists(model_ply_fname_backup):
        shutil.copyfile(model_ply_fname, model_ply_fname_backup)


    object_mesh = trimesh.load(shapenet_file_name, force='mesh')
    object_mesh.apply_transform(UPRIGHT_MAT)
    
    # scale the object_mesh to have the actual_size
    mesh_scale = actual_size / (object_mesh.bounds[1] - object_mesh.bounds[0])
    object_mesh = datagen_utils.apply_scale_to_mesh(object_mesh, list(mesh_scale))
    object_mesh.export(model_fname)

    copy_textured_mesh = o3d.io.read_triangle_mesh(model_fname)
    o3d.io.write_triangle_mesh(model_ply_fname, copy_textured_mesh)

    # ## DEBUG
    # from plyfile import PlyData, PlyElement
    # cloud = PlyData.read(model_ply_fname).elements[0].data
    # cloud = np.transpose(np.vstack((cloud['x'], cloud['y'], cloud['z'])))
    # if cloud.shape[0] > 10000:
    #     import pdb; pdb.set_trace()

    return object_mesh, list(mesh_scale)

def update_base_model_with_pred_pose(target_category_annotation, actual_size):
    '''
    actual_size: (3,) size of the mesh bounding box after it is turned "upright" according to perch_scene. 
    '''
    synset_id = target_category_annotation['synset_id']
    model_id = target_category_annotation['model_id']
    shapenet_file_name = os.path.join(
            '/raid/xiaoyuz1/ShapeNetCore.v2',
            '{}/{}/models/model_normalized.obj'.format(synset_id, model_id),
        )
    model_name = target_category_annotation['name']
    return save_correct_size_model(MODEL_SAVE_ROOT_DIR, model_name, actual_size, shapenet_file_name)
   
class CategoryBank(object):

    def __init__(self, json_path_list):
        category_bank = {}
        for json_path in json_path_list:
            original_annotations = json.load(open(json_path))
            for ann in original_annotations['categories']:
                shapenet_category_id = ann['shapenet_category_id']
                shapenet_object_id = ann['shapenet_object_id']
                
                D = category_bank.get(shapenet_category_id, {})
                L = D.get(shapenet_object_id, [])
                L.append(ann)
                D[shapenet_object_id] = L #(ann, scene_dir)
                category_bank[shapenet_category_id] = D
        
        self.category_bank = category_bank
    
    def get_random_ann(self, category_id, object_id):
        L = self.category_bank[category_id][object_id]
        idx = np.random.choice(np.arange(len(L)))
        return L[idx]

def get_category_bank(base_data_dir):
    base_json_dirs = uu.data_dir_list(base_data_dir, must_contain_file = ['annotations.json'])
    json_path_list = [os.path.join(scene_dir, 'annotations.json') for scene_dir in base_json_dirs]
    return CategoryBank(json_path_list)


def update_json_with(
    scene_dir, 
    scene_num, 
    image_id, 
    category_id, 
    target_category_annotation_orig = None,
    use_gt_scale = True,
    pred_scale = None,
):
    image_id = int(image_id)
    category_id = int(category_id)
    scene_num = int(scene_num)
    if target_category_annotation_orig is not None:
        target_category_annotation = copy.deepcopy(target_category_annotation_orig)
    else:
        target_category_annotation = query_annos.get_ann(scene_num, 'categories', category_id)

    image_json_path = os.path.join(scene_dir, 'annotations_{}.json'.format(image_id))
    not_existent_before = not os.path.exists(image_json_path)
    if not_existent_before:
        shutil.copyfile(os.path.join(scene_dir, 'annotations.json'), image_json_path)
    image_annotations = json.load(open(image_json_path))
    
    if not_existent_before:
        image_annotations['annotations'] = []
        image_annotations['categories'] = []
        query_image = query_annos.get_ann(scene_num, 'images', image_id)
        query_image["id"] = 0
        image_annotations['images'] = [query_image] 
    
    # Add the annotation to new file
    query_ann = query_annos.get_ann_with_image_category_id(scene_num, image_id, category_id)
    query_category_ann = query_annos.get_ann(scene_num, 'categories', query_ann['category_id'])
    target_category_annotation['name'] = query_ann['model_name']
    _, scale = update_base_model_with_pred_pose(target_category_annotation, query_category_ann['actual_size'])
    new_category_id = len(image_annotations['categories'])
    query_ann['category_id'] = new_category_id
    
    target_category_annotation['id'] = new_category_id
    if use_gt_scale:
        # if scale != query_category_ann['size']:
        #     import pdb; pdb.set_trace()
        target_category_annotation['size'] = scale
        target_category_annotation['actual_size'] = query_category_ann['actual_size']
    else:
        import pdb; pdb.set_trace()
    target_category_annotation['position'] = query_category_ann['position']
    target_category_annotation['quat'] = query_category_ann['quat']
    query_ann['id'] = len(image_annotations['annotations'])
    query_ann['image_id'] = 0
    
    image_annotations['categories'].append(target_category_annotation)
    image_annotations['annotations'].append(query_ann)
    if len(image_annotations['images']) > 1:
        import pdb; pdb.set_trace()

    json_string = json.dumps(image_annotations)
    json_file = open(image_json_path, "w+")
    json_file.write(json_string)
    json_file.close()
    return image_json_path
    


def update_json_with_predicted_category(query_data_dir, query_sample_id, target_data_dir, target_sample_id, pred_pose):
    
    scene_num, _, category_id = target_sample_id.split('-')
    category_id = int(category_id)
    scene_num = int(scene_num)
    dir_path = os.path.join(target_data_dir, f'scene_{scene_num:06}')
    json_path = os.path.join(dir_path, 'annotations.json')
    target_annotations = json.load(open(json_path))
    target_category_annotation = None 
    for ann in target_annotations['categories']:
        if ann['id'] == category_id:
            target_category_annotation = ann 
    assert target_category_annotation is not None 
    update_base_model_with_pred_pose(target_category_annotation, pred_pose)

    scene_num, image_id, category_id = query_sample_id.split('-')
    image_id = int(image_id)
    category_id = int(category_id)
    scene_num = int(scene_num)
    dir_path = os.path.join(query_data_dir, f'scene_{scene_num:06}')
    json_path = os.path.join(dir_path, 'annotations.json')
    annotations = json.load(open(json_path))
    
    old_categories = {}
    for ann in annotations['categories']:
        old_categories[ann['id']] = ann

    image_json_path = os.path.join(dir_path, 'annotations_{}.json'.format(image_id))
    not_existent_before = not os.path.exists(image_json_path)
    if not_existent_before:
        shutil.copyfile(json_path, image_json_path)
    image_annotations = json.load(open(image_json_path))
    
    if not_existent_before:
        image_annotations['annotations'] = []
        image_annotations['categories'] = []
        query_image = None
        for ann in annotations['images']:
            if ann['id'] == image_id:
                query_image = ann 
        assert query_image is not None
        query_image["id"] = 0
        image_annotations['images'] = [query_image] 
    
    # Add the annotation to new file
    query_ann = None
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id and ann['category_id'] == category_id:
            query_ann = ann 
    assert query_ann is not None
    old_category_id = query_ann['category_id']
    # Process the annotation
    new_category_id = len(image_annotations['categories'])
    query_ann['category_id'] = new_category_id
    query_ann['model_name'] = target_category_annotation['name']
    target_category_annotation['id'] = new_category_id
    target_category_annotation['size'] = [float(pred_pose)] * 3
    target_category_annotation['position'] = old_categories[old_category_id]['position']
    target_category_annotation['quat'] = old_categories[old_category_id]['quat']

    query_ann['id'] = len(image_annotations['annotations'])
    query_ann['image_id'] = 0
    
    image_annotations['categories'].append(target_category_annotation)
    image_annotations['annotations'].append(query_ann)
    if len(image_annotations['images']) > 1:
        import pdb; pdb.set_trace()

    json_string = json.dumps(image_annotations)
    json_file = open(image_json_path, "w+")
    json_file.write(json_string)
    json_file.close()
    return image_json_path


def get_features(args, epochs, save_dir, data_dir):
    prediction_dir = os.path.join(save_dir, 'predictions')
    args.files.testing_scene_dir = data_dir
    test_dataset = incat_dataset.InCategoryClutterDataset('test', args)
    assert os.path.exists(prediction_dir)
        
    # experiment_dir = '/raid/xiaoyuz1/retrieve_perch/perch_test/crimson-plasma-74/predictions'
    feats, sample_ids = q_utils.read_npy(prediction_dir, epochs)
    return test_dataset, feats, sample_ids


def inference(args, options):
    test_dataset, query_feats, query_sample_ids = get_features(args, options.query_epochs, options.query_save_dir, options.query_data_dir)
    base_dataset, base_feats, base_sample_ids = get_features(args, options.base_epochs, options.base_save_dir, options.base_data_dir)

    query_feats = q_utils.torchify(query_feats)
    base_feats = q_utils.torchify(base_feats)
    arg_sorted_dist = q_utils.get_arg_sorted_dist(query_feats, base_feats)

    obj_cat, obj_id = q_utils.sample_ids_to_cat_and_id(base_dataset, base_sample_ids)
    query_cat, query_id = q_utils.sample_ids_to_cat_and_id(test_dataset, query_sample_ids)
    
    top_k_arg_sorted_dist = arg_sorted_dist[:,:options.k]

    top_k_unique = []
    for row, test_batch_idx_row in zip(obj_id[top_k_arg_sorted_dist], top_k_arg_sorted_dist):
        res = np.unique(row, return_counts=True)
        val, count = res
        max_idx = np.argsort(count)[::-1][0]
        max_val_idx = np.where(row == val[max_idx])[0]
        max_val_idx = max_val_idx[0]
        max_test_batch_idx = test_batch_idx_row[max_val_idx]
        top_k_unique.append(base_sample_ids[max_test_batch_idx])
    
    uu.create_dir(options.result_save_dir)
    
    # Change each annotations.json to multiple, one for each image
    query_pose = np.load(os.path.join(options.query_save_dir, 'predictions', f'{options.query_epochs}_pose.npy'))
    query_pose = query_pose[:,0]
    all_new_path = []
    for pred_pose, query_sample_id, rev_sample_id in zip(query_pose, query_sample_ids, top_k_unique):
        if not(int(query_sample_id.split('-')[0]) == 25 and int(query_sample_id.split('-')[1]) == 5):
            continue
        new_path = update_json_with_predicted_category(options.query_data_dir, query_sample_id, options.base_data_dir, rev_sample_id, pred_pose)
        all_new_path.append(new_path)
        
    with open(os.path.join(options.result_save_dir, 'retrieved_sample_ids.pkl'), 'wb+') as fp:
        pickle.dump([top_k_unique, all_new_path], fp)


def clean_up(query_data_dir):
    dir_list = uu.data_dir_list(query_data_dir)
    for scene_dir in dir_list:
        for fname in os.listdir(scene_dir):
            if fname.startswith('annotations'): 
                fname_list = fname.split('.')[0].split('_')
                if len(fname_list) < 2:
                    continue 
                assert fname != "annotations.json"
                full_fname = os.path.join(scene_dir, fname)
                print("Remove: ", full_fname)
                os.remove(full_fname)


def run_random(scene_num, image_id, random = True):
    selected_model_ids = {}
    scene_dir = os.path.join(options.query_data_dir, f'scene_{scene_num:06}')
    query_image_anns = query_annos.get_ann_with_image_category_id(scene_num, image_id)
    
    for category_id, query_ann in query_image_anns.items():
        # category_id = 0
        # query_ann = query_annos.get_ann_with_image_category_id(scene_num, image_id, category_id)
        query_category_ann = query_annos.get_ann(scene_num, 'categories', query_ann['category_id'])
        shapenet_category_id = query_category_ann['shapenet_category_id']
        shapenet_object_id = query_category_ann['shapenet_object_id']
    
        if random:
            target_category_annotation_orig = category_bank.get_random_ann(shapenet_category_id, shapenet_object_id)
            selected_model_ids[category_id] = (
                target_category_annotation_orig['synset_id'],
                target_category_annotation_orig['model_id'],
            )
        else:
            target_category_annotation_orig = None
        update_json_with(
            scene_dir, 
            scene_num, 
            image_id, 
            category_id, 
            target_category_annotation_orig,
            use_gt_scale = True,
            pred_scale = None,
        )
    return selected_model_ids


if __name__ == "__main__":
    options = parser.parse_args()
    if options.clean_up:
        clean_up(options.query_data_dir)
    else:
        clean_up(options.query_data_dir)
        # f =  open(options.config_file)
        # args_dict = yaml.safe_load(f)
        # default_args_dict = yaml.safe_load(open('configs/default.yaml'))
        # args_dict_filled = uu.fill_in_args_from_default(args_dict, default_args_dict)
        # args = uu.Struct(args_dict_filled)
        # inference(args, options)
        query_annos = uu.COCOAnnotationScene(options.query_data_dir)
        category_bank = get_category_bank(options.base_data_dir)
        selected_model_ids = {}
        
        for subdir in os.listdir(options.query_data_dir):
            if not subdir.startswith('scene_'):
                continue
            scene_num = int(subdir.split('_')[-1])
            # scene_num = 38
            # image_id = 0
            
            for image_id, image_ann in query_annos.get_image_anns(scene_num).items():
                D = run_random(scene_num, image_id, random = options.ran_random)
                L = selected_model_ids.get(scene_num, {})
                L[image_id] = D
                selected_model_ids[scene_num] = L
                
            
            if options.ran_random:
                json_string = json.dumps(selected_model_ids)
                json_file = open(os.path.join(options.query_save_dir, 'random_chosen.json'), "w+")
                json_file.write(json_string)
                json_file.close()