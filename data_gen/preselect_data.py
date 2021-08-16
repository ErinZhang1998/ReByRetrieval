import csv
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import trimesh
import numpy as np 
import shutil
import pickle
from optparse import OptionParser

import utils.blender_proc_utils as bp_utils

# from selected_shapenet_data_list import *

parser = OptionParser()
parser.add_option("--csv_file_dir", dest="csv_file_dir")
parser.add_option("--shapenet_filepath", dest="shapenet_filepath")
parser.add_option("--selected_filepath", dest="selected_filepath")
parser.add_option("--normalized_model_save_dir", dest="normalized_model_save_dir")

def check_too_many_faces(csv_fname, shapenet_dir):
    df = pd.read_csv(csv_fname)
    for idx in range(len(df)):
        row = df.iloc[idx]
        synset_category, shapenet_model_id = row['synsetId'], row['ShapeNetModelId']
        mesh_fname = os.path.join(
                shapenet_dir,
                '0{}/{}/models/model_normalized.obj'.format(synset_category, shapenet_model_id),
            )
        mesh = trimesh.load(mesh_fname, force='mesh')
        print(mesh.faces.shape[0])

def write_to_csv(csv_file, dict_data, csv_columns):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

def output_selected(csv_file, selected_l):
    csv_df = pd.read_csv(csv_file)
    res = []
    for idx in selected_l:
        res.append(csv_df.iloc[idx]["fullId"].split(".")[-1])
    return res

def get_dict_data(preselect):
    csv_columns = ['synsetId', 'catId', 'name', 'ShapeNetModelId', 'objId', 'half_or_whole', 'perch_rot_angle']
    dict_data = []
    obj_id = 0
    cat_id = 0
    for cat_name, cat_synset_id, cat_objects in preselect:
        for objs in cat_objects:
            for obj in objs:
                row = {
                    'synsetId': cat_synset_id,
                    'catId' : cat_id,
                    'name': cat_name,
                    'ShapeNetModelId': obj,
                    'objId': obj_id,
                    'half_or_whole' : 0,
                    'perch_rot_angle' : 0,
                }

                # obj_cat = int(row["synsetId"])
                # obj_model_id = row["ShapeNetModelId"]
                # obj_mesh_filename = os.path.join(args.shapenet_filepath,'0{}/{}/models/model_normalized.obj'.format(obj_cat, obj_model_id))
                # object_mesh = trimesh.load(obj_mesh_filename, force='mesh')
                # if object_mesh.faces.shape[0] >= 200000:
                #     continue
                dict_data.append(row)
            obj_id += 1
        cat_id += 1
    return dict_data, csv_columns

if __name__ == '__main__':
    # preselect = [bag, bottle, bowl, can, clock, jar, laptop, camera, mug, basket]
    # preselect = [bag, bottle, bowl, can, clock, jar, laptop, camera, mug]
    # test_only_ids = [2801938]
    (args, argss) = parser.parse_args()
    f = open(args.selected_filepath, 'rb')
    preselect = pickle.load(f)
    test_only_ids = []
    dict_data, csv_columns = get_dict_data(preselect)
    for row in dict_data:
        bp_utils.save_normalized_object_to_file(args.shapenet_filepath, args.normalized_model_save_dir, row)
    if not os.path.exists(args.csv_file_dir):
        os.mkdir(args.csv_file_dir)
    csv_file = os.path.join(args.csv_file_dir, "preselect_table_top.csv")
    write_to_csv(csv_file, dict_data, csv_columns)


    df_1 = pd.read_csv(csv_file)
    for test_id in test_only_ids:
        df_1 = df_1[(df_1['synsetId'] != test_id)]
    train_test_data = df_1.to_dict('records')

    df_2 = pd.read_csv(csv_file)
    for test_id in test_only_ids:
        df_2 = df_2[(df_2['synsetId'] == test_id)]
    test_data = df_2.to_dict('records')

    # train,test = train_test_split(train_test_data, test_size=0.3)
    # train_csv_file_path = os.path.join(args.csv_file_dir, "preselect_table_top_train.csv")
    # test_csv_file_path = os.path.join(args.csv_file_dir, "preselect_table_top_test.csv")

    # write_to_csv(train_csv_file_path, train, csv_columns)
    # write_to_csv(test_csv_file_path, test+test_data, csv_columns)

    train_csv_file_path = os.path.join(args.csv_file_dir, "preselect_table_top_train.csv")
    test_csv_file_path = os.path.join(args.csv_file_dir, "preselect_table_top_test.csv")

    df = pd.read_csv(csv_file)
    all_train, all_test = [],[]
    for synsetid in df['objId'].unique():
        synset_data = df[df['objId'] == synsetid].to_dict('records')
        if len(synset_data) == 1:
            train = []
            test = synset_data
        else:
            train,test = train_test_split(synset_data, test_size=0.3)
        print(synsetid, len(train), len(test))
        all_train += train
        all_test += test

    write_to_csv(train_csv_file_path, all_train, csv_columns)
    write_to_csv(test_csv_file_path, all_test, csv_columns)



    # cat_ids = set()
    # object_ids = set()
    # cat_names = set()
    # for idx in range(len(df)):
    #     sample = df.iloc[idx]
    #     cat_ids.add(sample['synsetId'])
    #     cat_names.add(sample['name'])
    #     object_ids.add(sample['objId']) 
    # cat_ids = list(cat_ids)
    # object_ids = list(object_ids)
    # cat_names = list(cat_names)

    # self.cat_ids = cat_ids
    # self.cat_id_to_label = dict(zip(self.cat_ids, range(len(self.cat_ids))))
    # self.label_to_cat_id = dict(zip(range(len(self.cat_ids)), self.cat_ids))

    # self.object_ids = object_ids
    # self.object_id_to_label = dict(zip(self.object_ids, range(len(self.object_ids))))
    # self.object_label_to_id = dict(zip(range(len(self.object_ids)), self.object_ids))

    # self.cat_names = cat_names
    # self.cat_names_to_cat_id = dict(zip(self.cat_names, self.cat_ids))
    # self.cat_id_to_cat_names = dict(zip(self.cat_ids, self.cat_names))



    '''
    Problematic meshes:
    b95ca4fa91a57394e4b68d3b17c43658
    6b78948484df58cdc664c3d4e2d59341
    1497a7a1871af20162360e5e854659a
    6d036fd1c70e5a5849493d905c02fa86
    9726bf2b38d817eab169d2793795b997
    '''

    '''
    # def output_selected(csv_file, selected_l):
    #     csv_df = pd.read_csv(csv_file)
    #     res = []
    #     for idx in selected_l:
    #         res.append(csv_df.iloc[idx]["fullId"].split(".")[-1])
    #     return res
    # triangle_body_strap_selected = [1,2,4,6,15,17,24,38,45,57,58,61,62,63,67,68,77,78,80]
    # square_body_strap_selected = [11,28,37,69]
    # triangle_body_straps = output_selected('/raid/xiaoyuz1/bag.csv', triangle_body_strap_selected)
    # square_body_straps = output_selected('/raid/xiaoyuz1/bag.csv', square_body_strap_selected)
    # bag = ['bag,traveling bag,travelling bag,grip,suitcase', '02773838', [bag1, bag2,bag3]]
    # square_basket_selected = [0,3,7,21,25,29,30,31,39,42,49,54,66,71,74,76,78,79,80,81,87]
    # round_basket_selected = [15,18,28,37,44,48,51,94]
    # square_baskets =  output_selected('/raid/xiaoyuz1/basket.csv', square_basket_selected)
    # round_baskets = output_selected('/raid/xiaoyuz1/basket.csv', round_basket_selected)
    # basket = ['basket,handbasket', '02801938', [square_baskets, round_baskets]]
    # bowl_df = pd.read_csv("/raid/xiaoyuz1/bowl.csv")
    # bowl_selected = [8,19,22,29,33,34,37,61,64,66,75,78,80,90,94, \
    #                  102,104,123,130,131,135,140,147,153,183]
    # bowls = output_selected("/raid/xiaoyuz1/bowl.csv", bowl_selected)

    # bowl = ['bowl', '02880940', [bowls]]
    # print(len(bowls))
    # wine_bottle_df = pd.read_csv("/raid/xiaoyuz1/wine_bottle.csv")
    # wine_bottle_selected= [0,1,2,4,10,11,17,22,23,24,35,37,38,48, \
    #                        51,54,60,64,67,69,74,77,83,84,95,99,112,115,137,156]
    # wine_bottles = output_selected("/raid/xiaoyuz1/wine_bottle.csv",wine_bottle_selected)


    # sprayers = ["cbc1cbc9cf65e9c2fd1d6016d24cc8d", 
    #             "9b9a4bb5550f00ea586350d6e78ecc7", 
    #            "d45bf1487b41d2f630612f5c0ef21eb8"]


    # round_bottles_selected = [18,31,42,58,72,75,87,93,102,113,135,142,158, \
    #                           159,183,188,196,212,223,229,233,244,247,256, \
    #                           128,143,206,218,220]
    # round_bottles = output_selected("/raid/xiaoyuz1/wine_bottle.csv",round_bottles_selected)


    # square_water_bottle_selected=[63,65,106,109,170]
    # square_water_bottles = output_selected("/raid/xiaoyuz1/wine_bottle.csv",square_water_bottle_selected)


    # strap_bottles_selected = [42,43,114]
    # strap_bottles = output_selected("/raid/xiaoyuz1/wine_bottle.csv",strap_bottles_selected)



    # bottle = ['bottle', '02876657', [wine_bottles, sprayers,round_bottles,square_water_bottles, \
    #                                 strap_bottles]]
    # print(np.sum([len(l) for l in bottle[2]]))
    # can_selected = list(np.arange(108))
    # can_selected.remove(10)
    # can_selected.remove(32)
    # can_selected.remove(71)
    # can_selected.remove(98)
    # cans = output_selected("/raid/xiaoyuz1/can.csv",can_selected)
    # can = ['can,tin,tin can', '02946921', [cans]]
    # print(np.sum([len(l) for l in can[2]]))
    # clock1=['3521751471b748ff2846fa729d90e125']
    # clock2=['1d5a354ee3e977d7ce57d3de4658a486', 'e8c8090792a48c08b045cbdf51c133cd', \
    # '37a995cd9a8a125743dbb6421d614c0d']
    # clock3 = ['253156f6fea2d869ff59f04994ef1f0c', \
    #     '57c8fe2fb023b648ae29bc118c70aa10']
    # clock4 = ['6d12c792767c7d46bf3c901830f323db']
    # clock = ['clock', '03046257', [clock1, clock2, clock3, clock4]]
    # jar1=['c1be3d580b4088bf4cc80585c0d3d970', 
    #       '1da5c02a928b8889dfd1be983f4bd279', 
    #       'a1fae2bdac896ab83a75e6d000e08290', 
    #       '8e5595181e9eef7d82ec48ff3a4fe07c', 
    #       'ee10db30b91b9683f9215f842248bc25',
    #       '386a6dffaef80143fa0d49b618d792ba']
    # jar2=['a18343c4b0a8026faca4186c3b7dd23d', '57693fd44e597bd8fed792cc021b4e66', \
    # 'cb451e92ce5a422c9095fe1213108032']
    # jar = ['jar', '03593526', [jar1,jar2]]
    # laptop_csv = pd.read_csv("/raid/xiaoyuz1/laptop.csv")
    # laptop_selected = [  1,   2,   3,   4,   6,   7,   8,   9,  11,  13,  15,  16,  19,
    #         20,  22,  25,  26,  30,  31,  32,  34,  35,  37,  38,  39,  41,
    #         43,  44,  45,  48,  49,  53,  55,  57,  58,  59,  60,  61,  63,
    #         66,  67,  68,  69,  70,  71,  73,  74,  75,  76,  77,  80,  81,
    #         83,  86,  88,  90,  92,  93,  97,  98,  99, 100, 101, 104, 105,
    #        110, 112, 113, 114, 115, 116, 117, 118, 120, 123, 124, 125, 127,
    #        128, 129, 130, 131, 134, 135, 136, 137, 138, 139, 140, 141, 142,
    #        145, 146, 148, 150, 158, 159, 161, 163, 165]
    # laptops = output_selected("/raid/xiaoyuz1/laptop.csv",laptop_selected)
    # cameras_selected = [1,4,5,6,13,14,17,21,22,27,31,37,40,50,61,77,81,93,95,101,103,104,109,111]
    # cameras = output_selected("/raid/xiaoyuz1/camera.csv",cameras_selected)
    # square_handle_mug_selected = [1,21,23,38,41,53,67,69,82,121,166,168,171,177,211]
    # round_hanle_mug_selected = [3,10,11,13,15,17,18,25,27,30,31,35,44,46,51,54,55,57, \
    #                             63,65,72,74,75,83,84,87,88,96,98,99,108,110,112,120,122, \
    #                             123,130,137,140,152,159,160,164,169,170,176,184,206]
    # square_handle_mugs = output_selected("/raid/xiaoyuz1/mug.csv",square_handle_mug_selected)
    # round_hanle_mugs = output_selected("/raid/xiaoyuz1/mug.csv",round_hanle_mug_selected)
    # cup_like_mug_selected = [61,86,139,151]
    # cup_like_mugs = output_selected("/raid/xiaoyuz1/mug.csv",cup_like_mug_selected)
    # mug = ['mug', '03797390', [square_handle_mugs, round_hanle_mugs,cup_like_mugs]]

    '''