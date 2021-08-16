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
from blender_proc_datagen import BlenderProcScene


def get_selected_objects(args):
    # unit_x, unit_y = args.wall_unit_x / 2, args.wall_unit_y / 2
    # axis_grid_x = np.linspace(-unit_x, unit_x, 3)
    # axis_grid_y = np.linspace(-unit_y, unit_y, 3)
    # x_pos, y_pos = np.meshgrid(axis_grid_x, axis_grid_y)
    # LOCATION_GRID = np.hstack([x_pos.reshape(-1,1), y_pos.reshape(-1,1)])
    
    df = pd.read_csv(args.csv_file_path)
    # TO CREATE A MORE BALANCED DATASET 
    unique_object_ids = df['objId'].unique()
    selected_object_indices = []

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
    category_list = []
    object_list = []

    for selected_indices in selected_object_indices:
        selected_objects_i = []
        # actual_size_choices_i = copy.deepcopy(actual_size_choices)
        # location_grid_choice = copy.deepcopy(LOCATION_GRID)
        
        for idx in selected_indices:
            # if len(location_grid_choice) < 1:
            #     break 
            
            # loc = np.random.choice(len(location_grid_choice))
            # x,y = location_grid_choice[loc]
            # location_grid_choice = np.delete(location_grid_choice, loc, 0)

            row = df.iloc[idx]
            # cat_actual_size_choice = actual_size_choices_i[idx]
            # if len(cat_actual_size_choice) == 0:
            #     continue
            # sampled_actual_size = np.random.choice(cat_actual_size_choice)
            # actual_size_choices_i[idx].remove(sampled_actual_size)
            selected_dict = {
                'synsetId' : row['synsetId'],
                'catId' : row['catId'],
                'ShapeNetModelId' : row['ShapeNetModelId'],
                'objId' : row['objId'],
                # 'size' : sampled_actual_size,
                # 'scale' : [sampled_actual_size / 0.9] * 3,
                'half_or_whole' : row['half_or_whole'],
                'perch_rot_angle' : row['perch_rot_angle'],
                # 'position' : np.asarray([x,y]),
            }
            selected_objects_i.append(selected_dict)
            category_list.append(row['catId'])
            object_list.append(row['objId'])

        selected_objects.append(selected_objects_i)
    return selected_objects

def generate_blender(args):
    selected_objects = get_selected_objects(args)
    print("len(selected_objects[0]): ", len(selected_objects[0]))
    all_yaml_file_name = []
    scene_num_to_selected = {}
    for scene_num in range(args.num_scenes):
        acc_scene_num = scene_num + args.start_scene_idx
        scene_folder_path = None
        try:
            scene_num_to_selected[acc_scene_num] = selected_objects[scene_num]
            blender_proc_scene = BlenderProcScene(acc_scene_num, selected_objects[scene_num], args)
            scene_folder_path = blender_proc_scene.scene_folder_path
            blender_proc_scene.output_yaml()
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
    