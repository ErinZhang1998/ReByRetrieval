import os
import json
import copy
import shutil
import traceback
import pickle
import numpy as np
import pandas as pd

from datagen_args import *
from utils.datagen_utils import *
from blender_proc_scene import BlenderProcScene

def get_selected_objects(args):
    df = pd.read_csv(args.csv_file_path)
    unique_object_ids = df['objId'].unique()
    selected_object_indices = []
    # For each scene, choose object cateogry first
    for scene_num in range(args.num_scenes):
        num_object = np.random.randint(args.min_num_objects, args.max_num_objects+1, 1)[0]
        print("num_object: ", num_object)
        object_object_ids = np.random.choice(unique_object_ids, num_object, replace=True)
        selected_object_idx = []
        for obj_id in object_object_ids:
            obj_id_df = df[df['objId'] == obj_id]
            idx_in_obj_id_df = np.random.randint(0, len(obj_id_df), 1)[0]
            selected_object_idx.append(obj_id_df.iloc[idx_in_obj_id_df].name)
        selected_object_indices.append(selected_object_idx)
        
    selected_objects = []

    for selected_indices in selected_object_indices:
        selected_objects_i = []
        for idx in selected_indices:
            row = df.iloc[idx]
            selected_dict = {
                'synsetId' : row['synsetId'],
                'catId' : row['catId'],
                'ShapeNetModelId' : row['ShapeNetModelId'],
                'objId' : row['objId'],
                'half_or_whole' : row['half_or_whole'],
                'perch_rot_angle' : row['perch_rot_angle'],
            }
            selected_objects_i.append(selected_dict)
        selected_objects.append(selected_objects_i)
    return selected_objects

def generate_blender(args):
    if not args.single_object:
        selected_objects = get_selected_objects(args)
    else:
        selected_objects = []
        df = pd.read_csv(args.csv_file_path)
        synset_id_unique = [2876657, 2880940, 2946921, 3797390, 2942699, 3642806, 3593526, 3046257, 2773838]
        # df.synsetId.unique()
        # for synset_id in synset_id_unique:
        for synset_id in df['synsetId'].unique():
            df_select = df[df['synsetId'] == synset_id]
            for i in range(len(df_select)):
                row = df_select.iloc[i]
                selected_objects_i = [{
                    'synsetId' : row['synsetId'],
                    'catId' : row['catId'],
                    'ShapeNetModelId' : row['ShapeNetModelId'],
                    'objId' : row['objId'],
                    'half_or_whole' : row['half_or_whole'],
                    'perch_rot_angle' : row['perch_rot_angle'],
                }]
                selected_objects.append(selected_objects_i)

        # for i in range(len(df)):
        #     row = df.iloc[i]
        #     synset_id = '0{}'.format(row['synsetId'])
        #     if synset_id != '02876657':
        #         continue
        #     selected_objects_i = [{
        #         'synsetId' : row['synsetId'],
        #         'catId' : row['catId'],
        #         'ShapeNetModelId' : row['ShapeNetModelId'],
        #         'objId' : row['objId'],
        #         'half_or_whole' : row['half_or_whole'],
        #         'perch_rot_angle' : row['perch_rot_angle'],
        #     }]
        #     selected_objects.append(selected_objects_i)
        
    # print("len(selected_objects[0]): ", len(selected_objects[0]))
    all_yaml_file_name = []
    scene_num_to_selected = {}
    for scene_num in range(args.num_scenes):
        
        acc_scene_num = scene_num + args.start_scene_idx
        scene_folder_path = None
        if acc_scene_num >= len(selected_objects):
            continue
        try:
            scene_num_to_selected[acc_scene_num] = selected_objects[scene_num]
            blender_proc_scene = BlenderProcScene(acc_scene_num, selected_objects[scene_num], args)
            scene_folder_path = blender_proc_scene.scene_folder_path
            blender_proc_scene.output_yaml()
            if scene_num % 100 == 0:
                print(blender_proc_scene.yaml_fname)
            all_yaml_file_name.append(blender_proc_scene.yaml_fname)
        except:
            print('##################################### GEN Error!')
            if scene_folder_path is not None:
                shutil.rmtree(scene_folder_path)
            traceback.print_exc()

    # output to .sh file
    output_save_dir = os.path.join(args.scene_save_dir, args.train_or_test)
    sh_fname = os.path.join(output_save_dir, 'blender_proc.sh')
    output_fid = open(sh_fname, 'w+', encoding='utf-8')
    tmp_dir = os.path.join(args.scene_save_dir, 'tmp')
    for yaml_fname in all_yaml_file_name:
        line = f'python run.py {yaml_fname} --temp-dir {tmp_dir}\n'
        output_fid.write(line)
    output_fid.close()

    selected_object_file = os.path.join(
        output_save_dir, 
        'selected_objects_{}_{}.pkl'.format(args.start_scene_idx, args.num_scenes),
    )
    with open(selected_object_file, 'wb+') as fh:
        pickle.dump(scene_num_to_selected, fh)

if __name__ == '__main__':
    generate_blender(args)
    