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


def create_one_6d_scene(scene_num, selected_objects, args):

    shapenet_filepath, top_dir, train_or_test = args.shapenet_filepath, args.top_dir, args.train_or_test

    num_objects = len(selected_objects) 
    selected_colors = [ALL_COLORS[i] for i in np.random.choice(len(ALL_COLORS), num_objects+1, replace=False)]

    if not os.path.exists(args.scene_save_dir):
        os.mkdir(args.scene_save_dir)
    
    output_save_dir = os.path.join(args.scene_save_dir, train_or_test)
    if not os.path.exists(output_save_dir):
        os.mkdir(output_save_dir)
    
    scene_folder_path = os.path.join(args.scene_save_dir, f'{train_or_test}/scene_{scene_num:06}')
    if os.path.exists(scene_folder_path):
        shutil.rmtree(scene_folder_path)
    os.mkdir(scene_folder_path)

    asset_folder_path = os.path.join(scene_folder_path, 'assets')
    if not os.path.exists(asset_folder_path):
        os.mkdir(asset_folder_path)

    scene_xml_file = os.path.join(top_dir, f'base_scene.xml')
    xml_folder = os.path.join(args.scene_save_dir, f'{train_or_test}_xml')
    if not os.path.exists(xml_folder):
        os.mkdir(xml_folder)
    
    cam_temp_scene_xml_file = os.path.join(xml_folder, f'cam_temp_data_gen_scene_{scene_num}.xml')
    shutil.copyfile(scene_xml_file, cam_temp_scene_xml_file)

    try:
        # Add table
        table_id = '97b3dfb3af4487b2b7d2794d2db4b0e7'
        table_mesh_fname = os.path.join(shapenet_filepath, f'04379243/{table_id}/models/model_normalized.obj')
        transformed_table_mesh_fname = os.path.join(scene_folder_path, f'assets/table_{scene_num}.stl')
        
        table_info = add_table_in_scene(num_objects, selected_colors[0], table_mesh_fname, transformed_table_mesh_fname)
        table_info.update({
            'scene_name' : cam_temp_scene_xml_file,
        })

        table_bounds = table_info['bounds']
        table_r = R.from_euler('xyz', table_info['rot'], degrees=False)
        table_frame_to_world = autolab_core.RigidTransform(
            rotation=table_r.as_matrix(),
            translation=table_info['pos'],
            from_frame='table',
            to_frame='world',
        )
        table_height = table_bounds[1][2] - table_bounds[0][2]
        table_corners = bounds_xyz_to_corners(table_bounds)
        table_top_corners = table_corners[table_corners[:,2] == table_bounds[1,2]]
        table_top_corners = transform_3d_frame(table_frame_to_world.matrix, table_top_corners)
        table_min_x, table_min_y,_ = np.min(table_top_corners, axis=0)
        table_max_x, table_max_y,_ = np.max(table_top_corners, axis=0)
        table_info.update({
            'table_top_corners' : table_top_corners,
            'table_height' : table_height,
        })
        
        # Add objects
        object_idx_to_obj_info = dict()
        for object_idx in range(num_objects):
            synset_category, shapenet_model_id = selected_objects[object_idx][0], selected_objects[object_idx][2]
            obj_mesh_filename = os.path.join(
                shapenet_filepath,
                '0{}/{}/models/model_normalized.obj'.format(synset_category, shapenet_model_id),
            )
            upright_fname = os.path.join(
                scene_folder_path,
                f'assets/model_normalized_{scene_num}_{object_idx}.stl'
            )
            x_sample_range = [table_min_x+1, table_max_x-1]
            y_sample_range = [table_min_y+1, table_max_y-1]
           
            object_info = {
                'scene_name': cam_temp_scene_xml_file,
                'object_name': f'object_{object_idx}_{scene_num}',
                'mesh_names': [upright_fname],
                'synset_category' : synset_category,
                'shapenet_model_id' : shapenet_model_id,
                'mesh_fname' : obj_mesh_filename,
                'x_sample_range' : x_sample_range, 
                'y_sample_range' : y_sample_range,
                'table_height' : table_height,
                'color' : selected_colors[object_idx+1],
            }
            output_object_info = add_object_in_scene(object_info, scale_3d=False)
            object_info.update(output_object_info)
            object_idx_to_obj_info[object_idx] = object_info
        
        all_textures = os.listdir(os.path.join(top_dir, 'table_texture'))
        texture_file = np.random.choice(all_textures, 1)[0]
        texture_name = texture_file.split(".")[0]
        add_texture(cam_temp_scene_xml_file, texture_name, os.path.join(top_dir, 'table_texture', texture_file), texture_type="cube")
        add_objects(table_info, material_name=texture_name)
        
        # scene_name, object_name, mesh_names, pos, size, color, rot, run_id, contact_geom_list=None, add_ind=-1, add_contacts=True
        for object_idx in object_idx_to_obj_info.keys():
            object_info = object_idx_to_obj_info[object_idx]
            add_objects(object_info)
        
        light_position, light_direction = get_light_pos_and_dir(args.num_lights)
        ambients = np.random.uniform(0,0.05,args.num_lights*3).reshape(-1,3)
        diffuses = np.random.uniform(0.25,0.35,args.num_lights*3).reshape(-1,3)
        speculars = np.random.uniform(0.25,0.35,args.num_lights*3).reshape(-1,3)
       
        for light_id in range(args.num_lights):
            add_light(
                cam_temp_scene_xml_file,
                directional=True,
                ambient=ambients[light_id],
                diffuse=diffuses[light_id],
                specular=speculars[light_id],
                castshadow=False,
                pos=light_position[light_id],
                dir=light_direction[light_id],
                name=f'light{light_id}'
            )
        
        mujoco_env_pre_test = MujocoEnv(cam_temp_scene_xml_file, 1, has_robot=False)
        mujoco_env_pre_test.sim.physics.forward()
        
        for _ in object_idx_to_obj_info.keys():
            for _ in range(4000):
                mujoco_env_pre_test.model.step()
        all_poses = mujoco_env_pre_test.data.qpos.ravel().copy().reshape(-1,7)

        import pdb; pdb.set_trace()
    except:
        print('##################################### GEN Error!')
        shutil.rmtree(scene_folder_path)
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
        create_one_6d_scene(acc_scene_num, selected_objects[scene_num], args)