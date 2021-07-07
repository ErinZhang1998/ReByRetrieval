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

    try:
        # selected_objects = [
        #     ('2876657',1,'3f91158956ad7db0322747720d7d37e8',2),
        #     ('2946921',3,'d44cec47dbdead7ca46192d8b30882',8),
        # ]
        perch_scene = PerchScene(scene_num, selected_objects, args)

        # import pdb; pdb.set_trace()
    except:
        print('##################################### GEN Error!')
        # shutil.rmtree(scene_folder_path)
        print(selected_objects)
        traceback.print_exc()
        # DANGER   

# def main():


if __name__ == '__main__':
    # np.random.seed(129)
    df = pd.read_csv(args.csv_file_path)

    selected_object_indices = []
    for scene_idx in range(args.num_scenes):
        num_object = np.random.randint(args.min_num_objects, args.max_num_objects+1, 1)[0]
        selected_object_indices.append(np.random.randint(0, len(df), num_object))

    selected_objects = []
    for selected_indices in selected_object_indices:
        selected_objects_i = []
        for idx in selected_indices:
            sample = df.iloc[idx]
            selected_objects_i.append((sample['synsetId'], sample['catId'], sample['ShapeNetModelId'], sample['objId']))
        selected_objects.append(selected_objects_i)

    for scene_num in range(args.num_scenes):
        acc_scene_num = scene_num + args.start_scene_idx
        # create_one_6d_scene(acc_scene_num, selected_objects[scene_num], args)
        create_one_6d_scene(acc_scene_num, selected_objects[scene_num], args)