import os 
import shutil
import json
import copy
import pickle
import tqdm

from PIL import Image 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.datagen_utils as datagen_utils
import utils.blender_proc_utils as bp_utils

def get_experiment_names(args):
    target_train_or_test = args.target.data_dir.split('/')[-1]
    experiment_name = f'{args.result_name}-{target_train_or_test}-{args.target.epoch}-{args.query.epoch}'
    return experiment_name

def create_new_annotation_file(source_anno_path, new_anno_path, image_ann_list = None, category_ann_list = None, annotations_ann_list = None):
    
    # try:
    #     if not os.path.samefile(source_anno_path, new_anno_path):
    #         shutil.copyfile(source_anno_path, new_anno_path)
    # except:
    #     print("EXCEPTION!", source_anno_path, new_anno_path)

    if source_anno_path != new_anno_path:
        shutil.copyfile(source_anno_path, new_anno_path)

    new_anno = json.load(open(new_anno_path))
    if image_ann_list is not None:
        new_anno['images'] = image_ann_list
    if category_ann_list is not None:
        new_anno['categories'] = category_ann_list
    if annotations_ann_list is not None:
        new_anno['annotations'] = annotations_ann_list

    json_string = json.dumps(new_anno)
    json_file = open(new_anno_path, "w+")
    json_file.write(json_string)
    json_file.close()


def paste_in_new_category_annotation_perch(
    original_anno_path, 
    new_anno_path,
    category_id, 
    new_annotation,
    new_model_name,
    model_save_dir,
    normalized_shapenet_model_dir,
    turn_upright_before_scale,
    turn_upright_after_scale,
    save_new_model,
    new_actual_size,
    new_scale,
    over_write_new_anno_path,
):
    coco_anno1 = COCOSelf(original_anno_path)
    
    if os.path.exists(new_anno_path) and not over_write_new_anno_path:
        new_coco_anno = json.load(open(new_anno_path))
        category_annotations = new_coco_anno['categories']
        new_annotations = new_coco_anno['annotations']
    else:
        new_coco_anno = json.load(open(original_anno_path))
        category_annotations = []
        new_annotations = []

    image_ann_list = new_coco_anno['images']
    old_name_to_new_name = {}
    '''Find the one that you want to insert into the new anno path'''
    already_replaced = False
    for category_id_orig, category_ann in coco_anno1.category_id_to_ann.items():
        if category_id_orig != category_id:
            continue 
        new_ann = copy.deepcopy(category_ann) 
        assert not already_replaced       
        
        old_name_to_new_name[category_ann['name']] = new_model_name
        new_ann['name'] = new_model_name
        new_ann['shapenet_category_id'] = int(new_annotation['shapenet_category_id'])
        new_ann['shapenet_object_id'] = int(new_annotation['shapenet_object_id'])
        new_ann['synset_id'] = new_annotation['synset_id']
        new_ann['model_id'] = new_annotation['model_id']

        # if save_new_model:
        # else:
        #     assert new_scale is not None
        #     scale_xyz = new_scale
        mesh_file_name = os.path.join(
            normalized_shapenet_model_dir, 
            new_annotation['synset_id'],
            new_annotation['model_id'],
            'models',
            'model_normalized.obj',
        )
        new_mesh, scale_xyz = datagen_utils.save_correct_size_model(
            model_save_dir, 
            new_model_name, 
            new_actual_size, 
            mesh_file_name, 
            scale=new_scale,
            turn_upright_before_scale = turn_upright_before_scale,
            turn_upright_after_scale = turn_upright_after_scale,
        )
        actual_size = new_mesh.bounds[1] - new_mesh.bounds[0]
        new_ann['size'] = [float(item) for item in scale_xyz]
        new_ann['actual_size'] = actual_size
        category_annotations.append(new_ann)
        already_replaced = True
    
    for ann_id, ann in coco_anno1.ann_id_to_ann.items():
        if 'percentage_not_occluded' in ann:
            if ann['percentage_not_occluded'] is not None and ann['percentage_not_occluded'] < 0.1:
                continue
        
        if ann['area'] < 1000:
            continue
        
        if ann['model_name'] not in old_name_to_new_name:
            continue 
        
        ann['model_name'] = old_name_to_new_name[ann['model_name']]
        new_annotations.append(ann)
    
    new_coco_anno['images'] = image_ann_list
    new_coco_anno['categories'] = category_annotations
    new_coco_anno['annotations'] = new_annotations
    
    json_string = json.dumps(new_coco_anno)
    json_file = open(new_anno_path, "w+")
    json_file.write(json_string)
    json_file.close()


def paste_in_new_category_annotation(
    original_anno_path, 
    new_anno_path,
    new_model_name,
    new_scale,
    image_id,
    category_id, 
    pred_synset_id,
    pred_model_id,
    pred_shapenet_category_id,
    pred_object_category_id,
    model_save_dir,
    normalized_shapenet_model_dir,
    turn_upright_before_scale,
    turn_upright_after_scale,
):
    coco_anno = COCOSelf(original_anno_path)
    image_ann = coco_anno.image_id_to_ann[image_id]
    
    new_coco_anno = json.load(open(original_anno_path))
    category_annotations = []
    
    old_name_to_new_name = {}
    '''Find the one that you want to insert into the new anno path'''
    already_replaced = False
    for category_id_orig, category_ann in coco_anno.category_id_to_ann.items():
        if category_id_orig != category_id:
            continue 
        new_ann = copy.deepcopy(category_ann) 
        assert not already_replaced       
        
        old_name_to_new_name[category_ann['name']] = new_model_name
        new_ann['name'] = new_model_name
        new_ann['shapenet_category_id'] = int(pred_shapenet_category_id)
        new_ann['shapenet_object_id'] = int(pred_object_category_id)
        new_ann['synset_id'] = pred_synset_id
        new_ann['model_id'] = pred_model_id

        mesh_file_name = os.path.join(
            normalized_shapenet_model_dir, 
            pred_synset_id,
            pred_model_id,
            'models',
            'model_normalized.obj',
        )
        new_mesh, scale_xyz = datagen_utils.save_correct_size_model(
            model_save_dir, 
            new_model_name, 
            [-1,-1,-1], 
            mesh_file_name, 
            scale = new_scale,
            turn_upright_before_scale = turn_upright_before_scale,
            turn_upright_after_scale = turn_upright_after_scale,
        )
        actual_size = new_mesh.bounds[1] - new_mesh.bounds[0]
        new_ann['size'] = [float(item) for item in scale_xyz]
        new_ann['actual_size'] = [float(item) for item in actual_size]
        category_annotations.append(new_ann)
        already_replaced = True
    
    new_annotations = []
    for category_id_search, ann in coco_anno.image_id_to_category_id_to_ann[image_id].items():
        if 'percentage_not_occluded' in ann:
            if ann['percentage_not_occluded'] is not None and ann['percentage_not_occluded'] < 0.1:
                continue
        
        if ann['area'] < 1000:
            continue
        
        if ann['model_name'] not in old_name_to_new_name:
            continue 
        
        if category_id_search != category_id:
            continue
        ann['model_name'] = old_name_to_new_name[ann['model_name']]
        new_annotations.append(ann)
    
    
    new_coco_anno['images'] = [image_ann]
    new_coco_anno['categories'] = category_annotations
    new_coco_anno['annotations'] = new_annotations
    
    json_string = json.dumps(new_coco_anno)
    json_file = open(new_anno_path, "w+")
    json_file.write(json_string)
    json_file.close()


def separate_annotation_into_images(
    orig_anno_model_save_root_dir,
    coco_anno_path, 
    new_fname_dir, 
    new_fname_template, 
    skip_image_ids = None,
    model_name_suffx_template = None,
    new_anno_model_save_root_dir = None,
):
    '''
    Separate the coco annotation file into individual annotations files, one for each image_id 
    Or just delete the other image annotations
    
    Returns:
        dict: image_id --> coco_anno_path
    '''
    coco_anno = COCOSelf(coco_anno_path)
    coco_anno_parent_path = '/'.join(coco_anno_path.split('/')[:-1])
    # os.path.join(*coco_anno.split('/')[:-1])
    image_json_paths = []
    all_category_ann = copy.deepcopy(list(coco_anno.category_id_to_ann.values()))

    if new_anno_model_save_root_dir is None:
        new_anno_model_save_root_dir = orig_anno_model_save_root_dir
    
    old_name_to_new_name = {}
    for image_id, image_ann in coco_anno.image_id_to_ann.items():
        if skip_image_ids is not None:
            if int(image_id) in skip_image_ids:
                continue
        
        image_json_path = os.path.join(new_fname_dir, new_fname_template.format(image_id))
        
        shutil.copyfile(coco_anno_path, image_json_path)
        annotations_image_id = json.load(open(image_json_path))
        annotations_image_id['images'] = [image_ann]
        
        category_annotations_new = []
        for category_ann_orig in all_category_ann:
            category_ann = copy.deepcopy(category_ann_orig)
            model_name = category_ann['name']
            if model_name_suffx_template is None:
                new_model_name = model_name + '_image_{}'.format(image_id)
            else:
                new_model_name = model_name + model_name_suffx_template.format(image_id)
            old_name_to_new_name[model_name] = new_model_name
            
            old_model_save_dir = os.path.join(orig_anno_model_save_root_dir, model_name)

            assert os.path.exists(old_model_save_dir)
            model_save_dir = os.path.join(new_anno_model_save_root_dir, new_model_name)
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)

            files_to_copy = [
                'textured.obj',
                'textured.ply',
                'sampled.ply',
            ]
            for fname in files_to_copy:
                old_file = os.path.join(old_model_save_dir, fname)
                new_file = os.path.join(model_save_dir, fname)
                shutil.copyfile(old_file, new_file)
            
            category_ann['name'] = new_model_name
            category_annotations_new += [category_ann]

        
        annotations_image_id['categories'] = category_annotations_new
        this_image_annotations = []
        for ann in list(coco_anno.image_id_to_category_id_to_ann[image_id].values()):
            old_name = ann['model_name']
            ann['model_name'] = old_name_to_new_name[old_name]
            this_image_annotations += [ann]
        annotations_image_id['annotations'] = this_image_annotations

        json_string = json.dumps(annotations_image_id)
        json_file = open(image_json_path, "w+")
        json_file.write(json_string)
        json_file.close()
        image_json_paths.append(image_json_path)
    
    return image_json_paths


class COCOSelf(object):

    def __init__(self, json_path):
        '''
        Unit = json 
        '''
        self.model_name_to_model_full_name = {}
        annotations = json.load(open(json_path))

        
        image_id_to_ann = dict()
        image_file_name_to_image_id = dict()
        for ann in annotations['images']:
            image_id_to_ann[ann['id']] = ann 
            image_file_name_to_image_id[ann['file_name']] = ann['id']
        
        category_id_to_ann = dict()
        model_name_to_category_id = dict()
        for ann in annotations['categories']:
            category_id_to_ann[ann['id']] = ann
            model_name = ann['name']
            model_name_to_category_id[model_name] = ann['id']
            
        ann_id_to_ann = dict()
        image_id_to_category_id_to_ann = dict()
        for ann in annotations['annotations']:
            ann_id_to_ann[ann['id']] = ann

            D = image_id_to_category_id_to_ann.get(ann['image_id'], {})
            D[ann['category_id']] = ann
            image_id_to_category_id_to_ann[ann['image_id']] = D
        

        self.image_id_to_ann = image_id_to_ann
        self.category_id_to_ann = category_id_to_ann
        self.ann_id_to_ann = ann_id_to_ann
        self.image_id_to_category_id_to_ann = image_id_to_category_id_to_ann

        self.image_file_name_to_image_id = image_file_name_to_image_id
        self.model_name_to_category_id = model_name_to_category_id

        

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
        model_name_to_category_ann = dict()
        category_id_to_model_names = {}
        for ann in annotations['categories']:
            category_id_to_ann[ann['id']] = ann
            model_name = ann['name']
            model_name_to_category_ann[model_name] = ann
            shapenet_category_id = ann['shapenet_category_id']
            shapenet_object_id = ann['shapenet_object_id']

            category_id_to_model_names[ann['id']] = model_name
    
            # category_name = category_dict[f'{cat}_{model_id}']
            self.model_name_to_model_full_name[model_name] = f'{shapenet_category_id}_{shapenet_object_id}' #f'{cat}_{model_id}'
        self.category_id_to_model_names = category_id_to_model_names
        
        ann_id_to_ann = dict()
        image_category_id_to_ann = dict()
        for ann in annotations['annotations']:
            ann_id_to_ann[ann['id']] = ann

            D = image_category_id_to_ann.get(ann['image_id'], {})
            D[ann['category_id']] = ann
            image_category_id_to_ann[ann['image_id']] = D

        self.model_name_to_category_ann = model_name_to_category_ann
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
  

class COCOAnnotationScenes(object):

    def __init__(self, root_data_dir):
        '''
        Unit = json 
        '''
        self.root_data_dir = root_data_dir
        self.annotations_bank = {}
        # self.model_name_to_model_full_name = {}
    
    def add_scene(self, scene_num):
        scene_dir = os.path.join(self.root_data_dir, f'scene_{scene_num:06}')
        json_path = os.path.join(scene_dir, 'annotations.json')
        new_coco_anno = COCOAnnotation(json_path)
        self.annotations_bank[scene_num] = (scene_dir, new_coco_anno)
        # self.model_name_to_model_full_name.update(new_coco_anno.model_name_to_model_full_name)
    
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
  

def process_one_scene_perch_result( 
        df, 
        threshold = 0.02,
        custom = True,
        model_name_to_category_list_name = None,
    ):

    def isnumber(x):
        try:
            float(x)
            return float(x)
        except:
            return -1

    def isNan(x):
            if np.isnan(x):
                return -1
            else:
                return x

    cleaned_df_col = {}
    for col in df:
        df_col = df[col] 
        df_col = df_col.map(isnumber).map(isNan)
        cleaned_df_col[col] = df_col
    
    # Calculate accuracy using threshold
    acc_dict = {}
    
    above_threshold_image_dict = {}
    missing_object_image_dict = {} # model_full_name --> [image_name]
    
    for col_name, df_col in cleaned_df_col.items():
        if col_name.split('-')[-1] != 'adds':
            continue
        model_name = col_name.split('-')[0]
        # print(model_name)
        if model_name_to_category_list_name is not None:
            model_full_name = model_name_to_category_list_name[model_name]
        else:
            model_full_name = model_name
        if custom:
                object_idx = model_name.split('_')[-1]
        col_arr = df_col.to_numpy().astype(float)
        
        for index, img_name in df['name'][df_col > threshold].iteritems():
            image_name_dir = above_threshold_image_dict.get(model_full_name, [])
            if custom:
                image_name_dir.append((img_name, df_col[index], object_idx))   
            else:
                image_name_dir.append((img_name, df_col[index]))
            above_threshold_image_dict[model_full_name] = image_name_dir
        
        for index, img_name in df['name'][df_col < 0].iteritems():
            image_name_dir = missing_object_image_dict.get(model_full_name, [])
            if custom:
                image_name_dir.append((img_name, object_idx))
            else:
                image_name_dir.append(img_name)
            missing_object_image_dict[model_full_name] = image_name_dir

        leng = np.sum((col_arr >= 0))
        if leng == 0:
            continue
        L, total_num = acc_dict.get(model_full_name, ([], 0))
        L += list(col_arr[col_arr >= 0])
        total_num += leng
        acc_dict[model_full_name] = (L, total_num)

    return above_threshold_image_dict, missing_object_image_dict, acc_dict, cleaned_df_col


def get_clean_name(img_fname):
    return img_fname.replace('.jpg', '').replace('.png', '').replace('/', '_').replace('.', '_')


def transform_df_values(df, single_image=False):
    '''
    -1: fail to predict
    -2: nan values, probably whole scene has failed
    '''
    def isnumber(x):
        try:
            float(x)
            if np.isnan(float(x)):
                return -1
            return float(x)
        except:
            return -2
    
    for col_name in df:
        # do not process the model name column
        if col_name == 'name':
            continue 
        # do not include 'add' metric
        if col_name.split('-')[-1] != 'adds':
            df = df.drop([col_name], axis=1)
            continue
        
        df[col_name] = df[col_name].apply(isnumber)
        adds_index = col_name.index('-adds')
        new_col_name = col_name[:adds_index]
        # if single_image:
        #     new_col_name = '-'.join(col_name.split("-")[:1])
        # else:
        #     new_col_name = col_name.split("-")[0]
        df = df.rename(columns={col_name: new_col_name})
    return df


def from_annotation_list_path_to_model_dict_list(
    perch_output_dir, 
    annotation_list_path, 
    root_dir,
):
    if type(annotation_list_path) is list:
        L_container = []
        for path in annotation_list_path:
            print("Loading: ", path)
            L_container += pickle.load(open(path, 'rb'))
    else:
        L_container = pickle.load(open(annotation_list_path, 'rb'))
    model_dict_list = []
    doesnt_exist = []
    
    for anno_idx in tqdm.tqdm(range(len(L_container))):
        anno_name, anno_fname = L_container[anno_idx]
        anno_fname = anno_fname.replace('/data/custom_dataset', root_dir)
        perch_dir = os.path.join(perch_output_dir, anno_name)
        if not os.path.exists(perch_dir):
            # print(perch_dir, "does not exist.")
            doesnt_exist += [perch_dir]
            continue
        acc_fname = os.path.join(perch_dir, 'accuracy_6d.txt')
        df_acc = pd.read_csv(acc_fname)
        df_acc = transform_df_values(df_acc, single_image = True)
        
        for row in df_acc.to_dict('records'):
            image_name = row['name']
            perch_name = get_clean_name(image_name)

            in_image_idx = 0
            for k,v in row.items():
                if k == 'name':
                    continue
                
                sample_id, target_sample_id, rank = k.split('__')

                model_dict = {
                    'sample_id' : sample_id,
                    'target_sample_id' : target_sample_id,
                    'rank' : rank,
                    'model_name' : k,
                    'add-s' : v,
                    'image_file_name' : image_name,
                    'perch_dir' : perch_dir,
                    'perch_name' : perch_name,
                    'in_image_index' : in_image_idx,
                    'annotation_path' : anno_fname,
                }
                model_dict_list.append(model_dict)
                in_image_idx += 1
    df_final = pd.DataFrame.from_dict(
        dict(zip(range(len(model_dict_list)), model_dict_list)), 
        orient = 'index',
        columns = list(model_dict_list[0].keys()),
    )
    return {
        'df' : df_final,
        'model_dict_list' : model_dict_list,
        'anno_list' : L_container,
        'doesnt_exist' : doesnt_exist,
    }


def get_pred_annotations(perch_dir):
    fname = os.path.join(perch_dir, 'output_poses.txt')
    annotations = []
    f = open(fname, "r")
    lines = f.readlines()
    if len(lines) == 0:
        print("Invalid PERCH run : {}".format(len(lines)))
        return None

    for i in np.arange(0, len(lines), 13):
        location = list(map(float, lines[i+1].rstrip().split()[1:]))
        quaternion = list(map(float, lines[i+2].rstrip().split()[1:]))
        transform_matrix = np.zeros((4,4))
        preprocessing_transform_matrix = np.zeros((4,4))
        for l_t in range(4, 8) :
            transform_matrix[l_t - 4,:] = list(map(float, lines[i+l_t].rstrip().split()))
        for l_t in range(9, 13) :
            preprocessing_transform_matrix[l_t - 9,:] = list(map(float, lines[i+l_t].rstrip().split()))
        annotations.append({
                        'location' : [location[0], location[1], location[2]],
                        'quaternion_xyzw' : quaternion,
                        'model_name' : lines[i].rstrip(),
                        'transform_matrix' : transform_matrix,
                        'preprocessing_transform_matrix' : preprocessing_transform_matrix
                    })

    annotations_dict = {}
    for ann in annotations:
        annotations_dict[ann['model_name']] = ann

    return annotations_dict

class PerchOutput(object):
    def __init__(self, perch_root_dir, perch_name):
        self.perch_root_dir = perch_root_dir
        self.perch_name = perch_name
        self.perch_dir = os.path.join(perch_root_dir, perch_name)
        
        self.pred_img_gt_contour = 'gpu-graph_state-0-color-contour.png'
        self.pred_img_gt_pred_contour = 'gpu-graph_state-0-color-contour-predicted.png'
        self.pred_img = 'gpu-graph_state-0-color.png'

        self.pred_pose_dict = self.get_pred_annotations()
        model_names = list(self.pred_pose_dict.keys())
        self.model_object_idx_to_model_name = {}
        for model_name in model_names:
            object_idx = int(model_name.split('-')[0].split('_')[-1])
            self.model_object_idx_to_model_name[object_idx] = model_name
        
        image_list = []
        model_name_to_image_dict = {}
        for fname in os.listdir(self.perch_dir):
            # gpu-lowest_cost_state_0-0-color.png
            if fname.startswith('gpu-lowest_cost_state'):
                image_list.append(fname)
                object_idx = fname.split(".")[0].split("-")[1].split("_")[-1]
                object_idx = int(object_idx)
                # model_name = self.model_object_idx_to_model_name[object_idx]
                model_name_to_image_dict[object_idx] = fname
        self.image_list = image_list
        self.model_name_to_image_dict = model_name_to_image_dict
        
    def get_pred_annotations(self):
        return get_pred_annotations(self.perch_dir)
    
    def get_cost_dump(self):
        cost_dict = {}
        cost_json = json.load(open(os.path.join(self.perch_dir,'cost_dump.json') ))
        # for model_name in list(self.model_object_idx_to_model_name.values()):
        for pose_cost in cost_json['poses']:
            if 'model_name' not in pose_cost:
                return None
            L = cost_dict.get(pose_cost['model_name'], [])
            L.append(pose_cost)
            cost_dict[pose_cost['model_name']] = L
        
        for model_name, L in cost_dict.items():
            lowest_cost = 10000
            lowest_cost_id = -1
            id_to_cost_dict = {}
            for pose_cost in L:
                id_to_cost_dict[pose_cost['id']] = pose_cost
                if pose_cost['total_cost'] < lowest_cost:
                    lowest_cost_id = pose_cost['id']
                    lowest_cost = pose_cost['total_cost']
            
            cost_dict[model_name] = (lowest_cost, lowest_cost_id, id_to_cost_dict)
        
        return cost_dict

    def display_object_preds(self):
        for fname in self.image_list:
            fname = os.path.join(self.perch_dir, fname)
            img = Image.open(fname)
            
            plt.figure(figsize=(12, 12))
            plt.title(fname)
            plt.imshow(np.asarray(img))
            plt.show()

    def display_object_pred(self, model_name):
        fname = self.model_name_to_image_dict[model_name]
        fname = os.path.join(self.perch_dir, fname)
        img = Image.open(fname)
        plt.figure(figsize=(12, 12))
        plt.title(fname)
        plt.imshow(np.asarray(img))
        plt.show()
    
    def display_all_pred(self, gt=True, gt_contour=True, pred_contour=True):
        if gt:
            fname = os.path.join(self.perch_dir, 'input_color_image.png')
            img = Image.open(fname)
            plt.figure(figsize=(12, 12))
            plt.imshow(np.asarray(img))
            plt.show()

        if gt_contour:
            fname = os.path.join(self.perch_dir, self.pred_img_gt_contour)
            img = Image.open(fname)
            plt.figure(figsize=(12, 12))
            plt.imshow(np.asarray(img))
            plt.show()
        
        if pred_contour:
            fname = os.path.join(self.perch_dir, self.pred_img_gt_pred_contour)
            img = Image.open(fname)
            plt.figure(figsize=(12, 12))
            plt.imshow(np.asarray(img))
            plt.show()

        fname = os.path.join(self.perch_dir, self.pred_img)
        img = Image.open(fname)
        plt.figure(figsize=(12, 12))
        plt.imshow(np.asarray(img))
        plt.show()

