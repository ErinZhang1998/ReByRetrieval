import os
import json
import copy
import shutil
import traceback
import numpy as np
import pandas as pd

import trimesh
from mujoco_env import MujocoEnv
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from data_gen_args import *
from simple_clutter_utils import *
from perch_scene_utils import *

def create_one_6d_scene(scene_num, selected_objects, args):
    # perch_scene = PerchScene(scene_num, selected_objects, args)
    scene_folder_path = None
    try:
        perch_scene = PerchScene(scene_num, selected_objects, args)
        scene_folder_path = perch_scene.scene_folder_path
        perch_scene.create_convex_decomposed_scene()
        perch_scene.create_camera_scene()
        # import pdb; pdb.set_trace()
    except:
        print('##################################### GEN Error!')
        # if scene_folder_path is not None:
        #     shutil.rmtree(scene_folder_path)
        print(selected_objects)
        traceback.print_exc()

if __name__ == '__main__':
    # np.random.seed(129)
    df = pd.read_csv(args.csv_file_path)


    scale_choices = {}
    for i in range(len(df)):
        scale_choices[i] = [0.75, 0.85, 1.0]

    selected_object_indices = []
    for scene_idx in range(args.num_scenes):
        num_object = np.random.randint(args.min_num_objects, args.max_num_objects+1, 1)[0]
        selected_object_indices.append(np.random.randint(0, len(df), num_object))

    selected_objects = []
    for selected_indices in selected_object_indices:
        selected_objects_i = []
        scale_choices_i = copy.deepcopy(scale_choices)
        for idx in selected_indices:
            row = df.iloc[idx]
            cat_scale_choice = scale_choices_i[idx]
            if len(cat_scale_choice) == 0:
                continue
            sample_scale = np.random.choice(cat_scale_choice)
            scale_choices_i[idx].remove(sample_scale)

            selected_objects_i.append((
                row['synsetId'], 
                row['catId'], 
                row['ShapeNetModelId'], 
                row['objId'], 
                sample_scale,
                row['half_or_whole'],
                row['perch_rot_angle']
            ))
        print(selected_objects_i)
        selected_objects.append(selected_objects_i)

    for scene_num in range(args.num_scenes):
        acc_scene_num = scene_num + args.start_scene_idx
        # create_one_6d_scene(acc_scene_num, selected_objects[scene_num], args)
        create_one_6d_scene(acc_scene_num, selected_objects[scene_num], args)