import numpy as np 
import os 
from PIL import Image 
import matplotlib.pyplot as plt
import json


class COCOAnnotation(object):

    def __init__(self, json_path):
        '''
        Unit = json 
        '''
        self.model_name_to_model_full_name = {}
        annotations = json.load(open(json_path))

        category_id_to_model_names = {}
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
        if col_name == 'name':
            continue 
        if col_name.split('-')[-1] != 'adds':
            df = df.drop([col_name], axis=1)
            continue
        
        df[col_name] = df[col_name].apply(isnumber)
        if single_image:
            new_col_name = '-'.join(col_name.split("-")[:2])
        else:
            new_col_name = col_name.split("-")[0]
        df = df.rename(columns={col_name: new_col_name})
    return df


def df_to_image_result(df, single_image=False):
    df = transform_df_values(df, single_image)
    df_rows = df.to_dict('records')
    perch_name_to_res_dict = dict()
    for row in df_rows:
        image_fname = row['name']
        perch_name = get_clean_name(image_fname)
        perch_name_to_res_dict[perch_name] = row 
    
    return df, perch_name_to_res_dict


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

