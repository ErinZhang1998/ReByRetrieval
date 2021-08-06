import numpy as np 
import os 
from PIL import Image 
import matplotlib.pyplot as plt


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


def transform_df_values(df):
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
        df = df.rename(columns={col_name: col_name.split("-")[0]})
    return df


def df_to_image_result(df):
    df = transform_df_values(df)
    df_rows = df.to_dict('records')
    perch_name_to_res_dict = dict()
    for row in df_rows:
        image_fname = row['name']
        perch_name = get_clean_name(image_fname)
        perch_name_to_res_dict[perch_name] = row 
    
    return df, perch_name_to_res_dict


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
        self.model_object_idx_to_model_name = dict(zip(range(len(model_names)), model_names))
        
        model_name_to_image_dict = {}
        for fname in os.listdir(self.perch_dir):
            # gpu-lowest_cost_state_0-0-color.png
            if fname.startswith('gpu-lowest_cost_state'):
                object_idx = fname.split(".")[0].split("-")[1].split("_")[-1]
                object_idx = int(object_idx)
                model_name = self.model_object_idx_to_model_name[object_idx]
                model_name_to_image_dict[model_name] = fname
        
        self.model_name_to_image_dict = model_name_to_image_dict
        
    
    def get_pred_annotations(self):
        fname = os.path.join(self.perch_dir, 'output_poses.txt')
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
    
    def display_object_pred(self, model_name):
        fname = self.model_name_to_image_dict[model_name]
        fname = os.path.join(self.perch_dir, fname)
        img = Image.open(fname)
        plt.figure(figsize=(12, 12))
        plt.imshow(np.asarray(img))
        plt.show()
    
    def display_all_pred(self, gt_contour=True, pred_contour=True):
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

