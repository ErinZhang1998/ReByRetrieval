import os
import json
import copy
import numpy as np
import multiprocessing as mp
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.image as mpimg

from mujoco_env import MujocoEnv
import trimesh
import shutil
import random
import cv2
from pyquaternion import Quaternion
import pickle

import traceback
import pybullet as p
from dm_control.mujoco.engine import Camera

from functools import partial
# from trajopt.mujoco_utils import add_objects
from scipy.spatial.transform import Rotation as R
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image 

from data_gen_args import *
from simple_clutter_utils import *
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

#Color of objects:
all_colors_dict = mcolors.CSS4_COLORS #TABLEAU_COLORS #
ALL_COLORS = []
excluded_color_names = ['black', 'midnightblue', 'darkslategray','darkslategrey','dimgray','dimgrey']
for name, color in all_colors_dict.items():
    c = mcolors.to_rgb(color)
    if c[0] > 0.8 and c[1] > 0.8 and c[2] > 0.8:
        continue 
    if name in excluded_color_names:
        continue
    ALL_COLORS.append(np.asarray(mcolors.to_rgb(color)))


REGION_LIMIT = 2*np.sqrt(0.5)
    
#@profile
def gen_data(scene_num, selected_objects, args):
    shapenet_filepath, shapenet_decomp_filepath, top_dir, train_or_test = args.shapenet_filepath, args.shapenet_decomp_filepath, args.top_dir, args.train_or_test

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

    try:
        

        '''
        Add table
        '''
        # MAGIC NUMBER 
        if num_objects < 10:
            camera_distance = 1.5 * 2.5
        else:
            camera_distance = 1.5 * 5 

        # Choose table and table scale and add to sim
        table_id = '97b3dfb3af4487b2b7d2794d2db4b0e7'
        table_mesh_filename = os.path.join(shapenet_filepath, f'04379243/{table_id}/models/model_normalized.obj')
        table_mesh=trimesh.load(table_mesh_filename, force='mesh')
        # Rotate table so that it appears upright in Mujoco
        scale_mat=np.eye(4)
        r = R.from_euler('x', 90, degrees=True)
        scale_mat[0:3,0:3] = r.as_matrix()
        table_mesh.apply_transform(scale_mat)
        table_bounds = table_mesh.bounds
        
        table_xyz_range = np.min(table_bounds[1,:] - table_bounds[0,:])
        table_size = (camera_distance*2)/table_xyz_range
        table_scale = [table_size,table_size,table_size]
        # Export table mesh as .stl file
        stl_table_mesh_filename=os.path.join(top_dir, f'assets/table_{scene_num}.stl')
        f = open(stl_table_mesh_filename, "w+")
        f.close()
        table_mesh.export(stl_table_mesh_filename)

        table_color = selected_colors[0] #np.random.uniform(size=3)
        table_bounds = table_bounds*np.array([table_scale,table_scale])
        table_bottom = -table_bounds[0][2]
        table_height = table_bounds[1][2] - table_bounds[0][2]
        # Move table above floor
        table_xyz = [0, 0, table_bottom]
        table_orientation = [0,0,0]
        
        '''
        Add objects
        '''
        
        object_max_height = -10
        object_position_region = None
        # probs = [0.15,0.15,0.15,0.15,0.1,0.1,0.1,0.1]
        probs = [1/8] * 8
        prev_bbox = []
        
        object_idx_to_obj_info = dict()
        

        
        for object_idx in range(num_objects):
            # sample['synsetId'], sample['catId'], sample['ShapeNetModelId'], sample['objId']
            obj_synset_cat, obj_shapenet_id = selected_objects[object_idx][0], selected_objects[object_idx][2]
            obj_info = dict()

            obj_mesh_filename = os.path.join(shapenet_filepath,'0{}/{}/models/model_normalized.obj'.format(obj_synset_cat, obj_shapenet_id))
            object_mesh = trimesh.load(obj_mesh_filename, force='mesh')
            old_bound = object_mesh.bounds 
            '''
            Determine object color
            '''
            object_color = selected_colors[object_idx+1] #np.random.uniform(size=3)
            '''
            Determine object size
            '''
            scale_vec, scale_matrix = determine_object_scale(obj_synset_cat, object_mesh)
            object_mesh.apply_transform(scale_matrix)
            object_bounds = object_mesh.bounds
            
            range_max = np.max(object_bounds[1] - object_bounds[0])
            object_size = 1 / range_max
            normalize_vec = [object_size, object_size, object_size]
            normalize_matrix = np.eye(4)
            normalize_matrix[:3, :3] *= normalize_vec
            object_mesh.apply_transform(normalize_matrix)
            obj_scale_vec = np.array(scale_vec) * np.array(normalize_vec)
            '''
            Determine object rotation
            '''
            rot_vec, upright_mat  = determine_object_rotation(object_mesh)
            object_mesh.apply_transform(upright_mat)
            z_rot = np.random.uniform(0,2*np.pi,1)[0]
            
            # Store the upright object in .stl file in assets
            stl_obj_mesh_filename = os.path.join(top_dir, f'assets/model_normalized_{scene_num}_{object_idx}.stl')
            f = open(stl_obj_mesh_filename, "w+")
            f.close()
            object_mesh.export(stl_obj_mesh_filename)
            object_rot = [0,0,z_rot]
            object_bounds = object_mesh.bounds
            object_bottom = -object_bounds[0][2]
            '''
            Determine object position
            '''
            object_z = table_height + object_bottom + 0.005
            if object_idx == 0:
                # To put at the center of the table
                object_x = (table_bounds[1,0] + table_bounds[0,0]) / 2
                object_y = (table_bounds[1,1] + table_bounds[0,1]) / 2
                object_xyz = [object_x, object_y, object_z]
                # Transforming object corners from object frame to world frame
                lower_x, upper_x = object_bounds[:,0]
                lower_y, upper_y = object_bounds[:,1]
                lower_z,_ = object_bounds[:,2]
                object_x_width, object_y_width, _ = object_bounds[1] - object_bounds[0]
                add_x = object_x_width * 0.1
                add_y = object_y_width * 0.1
                upper_x, upper_y = upper_x+add_x, upper_y+add_y
                lower_x, lower_y = lower_x-add_x, lower_y-add_y

                new_corners_3d = np.array([[lower_x, lower_y, lower_z], \
                    [upper_x, lower_y, lower_z], \
                    [upper_x, upper_y, lower_z], \
                    [lower_x, upper_y, lower_z]]) #(4,3) --> (3,4)
                r2 = R.from_rotvec(object_rot)
                object_tf = autolab_core.RigidTransform(rotation = r2.as_matrix(), translation = np.asarray(object_xyz), from_frame='object_{}'.format(object_idx), to_frame='world')
                
                pt_3d_homo = np.append(new_corners_3d.T, np.ones(4).astype('int').reshape(1,-1), axis=0) #(4,4)
                bounding_coord = object_tf.matrix @ pt_3d_homo #(4,4)
                bounding_coord = bounding_coord / bounding_coord[-1, :]
                bounding_coord = bounding_coord[:-1, :].T #(4,3)
                corners = bounding_coord[:,:2]
                prev_bbox.append(corners)
            else:
                object_x, object_y, probs, object_tf, corners = generate_object_xy(object_rot, object_z, object_bounds, prev_bbox, object_position_region, probs, scene_folder_path)
                prev_bbox.append(corners)
            
            object_xyz = [object_x, object_y, object_z]
            
            object_height = object_bounds[1][2] - object_bounds[0][2]            
            object_max_height = max(object_max_height, object_height)

            obj_info['xyz'] = np.asarray(object_xyz)
            obj_info['scale'] = obj_scale_vec
            obj_info['color'] = selected_colors[object_idx+1]
            obj_info['rotation'] = object_rot
            obj_info['obj_mesh_filename'] = '0{}/{}/models/model_normalized.obj'.format(obj_synset_cat, obj_shapenet_id)
            obj_info['object_height'] = object_height
            obj_info['object_tf'] = object_tf
            obj_info['object_bounds'] = object_bounds
            object_idx_to_obj_info[object_idx] = obj_info

        layout_filename = os.path.join(scene_folder_path, 'layout.png')
        # draw_boundary_points_rect(prev_bbox, layout_filename)

        
        scene_xml_file=os.path.join(top_dir, f'base_scene.xml')
        cam_temp_scene_xml_file=os.path.join(top_dir, f'{train_or_test}_xml/cam_temp_data_gen_scene_{scene_num}.xml')
        shutil.copyfile(scene_xml_file, cam_temp_scene_xml_file)

        all_textures = os.listdir(os.path.join(top_dir, 'table_texture'))
        texture_file = np.random.choice(all_textures, 1)[0]
        texture_name = texture_file.split(".")[0]
        add_texture(cam_temp_scene_xml_file, texture_name, os.path.join(top_dir, 'table_texture', texture_file), texture_type="cube")
        add_objects(cam_temp_scene_xml_file, 'table', [stl_table_mesh_filename], table_xyz, table_scale, table_color, table_orientation, scene_num, material_name=texture_name)
        
        # scene_name, object_name, mesh_names, pos, size, color, rot, run_id, contact_geom_list=None, add_ind=-1, add_contacts=True
        for object_idx in object_idx_to_obj_info.keys():
            obj_info = object_idx_to_obj_info[object_idx]
            mesh_names = [os.path.join(top_dir, f'assets/model_normalized_{scene_num}_{object_idx}.stl')]
            add_objects(cam_temp_scene_xml_file, \
                        f'object_{object_idx}_{scene_num}', \
                        mesh_names, \
                        obj_info['xyz'], \
                        [1,1,1], \
                        obj_info['color'], \
                        obj_info['rotation'], \
                        scene_num)
        
        light_position, light_direction = get_light_pos_and_dir(args.num_lights)
        ambients = np.random.uniform(0,0.05,args.num_lights*3).reshape(-1,3)
        diffuses = np.random.uniform(0.25,0.35,args.num_lights*3).reshape(-1,3)
        speculars = np.random.uniform(0.25,0.35,args.num_lights*3).reshape(-1,3)
       
        for light_id in range(args.num_lights):
            pos = light_position[light_id]
            light_dir = light_direction[light_id]
            ambient = ambients[light_id]#[0,0,0]
            diffuse = diffuses[light_id]#[0.3,0.3,0.3]
            specular = speculars[light_id]#[0.3,0.3,0.3]
            # print(ambient, diffuse, specular)
            add_light(cam_temp_scene_xml_file, directional=True, ambient=ambient, diffuse=diffuse, specular=specular, castshadow=False, pos=pos, dir=light_dir, name=f'light{light_id}')
        
        mujoco_env_pre_test = MujocoEnv(cam_temp_scene_xml_file, 1, has_robot=False)
        mujoco_env_pre_test.sim.physics.forward()
        
        for _ in object_idx_to_obj_info.keys():
            for _ in range(4000):
                mujoco_env_pre_test.model.step()
        

        all_poses = mujoco_env_pre_test.data.qpos.ravel().copy()
        # 
        original_obj_keys = list(object_idx_to_obj_info.keys())
        new_obj_keys = list(object_idx_to_obj_info.keys())
        for object_idx in original_obj_keys:
            current_xyz = all_poses[7+7*object_idx : 7+7*object_idx+3]
            original_xyz = object_idx_to_obj_info[object_idx]['xyz']

            if np.linalg.norm(current_xyz - original_xyz) > 0.2:
                print("WARNING: object location shifted more than 20cm: ", current_xyz, original_xyz, np.linalg.norm(current_xyz - original_xyz))
                print("WARNING: object location shifted more than 20cm (information): ",  object_idx_to_obj_info[object_idx]['obj_mesh_filename'])
                new_obj_keys.remove(object_idx)
                #del object_idx_to_obj_info[object_idx]  
        
        '''
        Generate camera position and target
        ''' 
        xys = []
        # for object_idx in object_idx_to_obj_info.keys(): 
        for object_idx in new_obj_keys:
            xyz = object_idx_to_obj_info[object_idx]['xyz']
            xys.append([xyz[0],xyz[1]])
        xys = np.asarray(xys)

        cam_width = 640
        cam_height = 480

        cam_num = 0
        cam_xyzs = dict()
        cam_targets = dict()
        cam_num_to_occlusion_target = dict()

        num_rotates = 1
        step_deg = 10
        # for object_i in object_idx_to_obj_info.keys(): 
        for object_i in new_obj_keys:
            xyz1 = object_idx_to_obj_info[object_i]['xyz']
            height1 = object_idx_to_obj_info[object_i]['object_height']
            
            pairwise_diff = xys - xyz1[:2].reshape((1,2))
            dist = np.linalg.norm(pairwise_diff, axis=1)
            max_dist = np.max(dist)
            
            # for object_j in object_idx_to_obj_info.keys():  
            for object_j in new_obj_keys:
                if object_i == object_j:
                    continue 
                xyz2 = object_idx_to_obj_info[object_j]['xyz']
                height2 = object_idx_to_obj_info[object_j]['object_height']
                
                for sign in [1,-1]:
                    keep_rotating = True
                    for deg_i in range(num_rotates):
                        if not keep_rotating:
                            break
                        low_deg = deg_i*step_deg*sign
                        high_deg = (deg_i+1)*step_deg*sign
                        cam_xyz1,cam_target,cam_xyz2 = get_camera_position_occluded_one_cam(table_height, xyz1,xyz2,height1,height2,max_dist,[low_deg,high_deg])
                        
                        for cam_i,cam_xyz in enumerate([cam_xyz1, cam_xyz2]):
                            if cam_i == 2:
                                if not np.random.binomial(n=1, p=(1/5), size=1)[0]:
                                    continue

                            add_camera(cam_temp_scene_xml_file, f'gen_cam_{cam_num}', cam_xyz, cam_target, cam_num)

                            cam_xyzs[cam_num] = cam_xyz
                            cam_targets[cam_num] = cam_target
                            cam_num_to_occlusion_target[cam_num] = object_i
                            cam_num += 1
        
        print(len(original_obj_keys), len(new_obj_keys), len(cam_xyzs))
        mujoco_env_test = MujocoEnv(cam_temp_scene_xml_file, 1, has_robot=False)
        mujoco_env_test.sim.physics.forward()


        cam_information = dict()
        cam_transformations = dict()

        for cam_num in cam_xyzs.keys():
            object_i = cam_num_to_occlusion_target[cam_num]
            camera = Camera(physics=mujoco_env_test.model, height=cam_height, width=cam_width, camera_id=cam_num)
            segs = camera.render(segmentation=True)[:,:,0] #(480, 640, 2)
            pix_left_ratio, onoccluded_pixel_num, segmentation = get_pixel_left_ratio(scene_num, camera, cam_num, mujoco_env_test, object_i, original_obj_keys, cam_width, cam_height)
            if pix_left_ratio >= 0.05 and onoccluded_pixel_num > 0:
                camera_id,camera_tf,camera_res = get_camera_matrix(camera)
                assert cam_num == camera_id
                
                cam_transformations[cam_num] = camera_tf
                
                this_cam_stats = dict()
                this_cam_mask_files = dict()
                this_cam_pc_files = dict()

                rgb=mujoco_env_test.model.render(height=cam_height, width=cam_width, camera_id=cam_num, depth=False, segmentation=False)
                cv2.imwrite(os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png'), rgb)
                # Depth image
                depth = mujoco_env_test.model.render(height=cam_height, width=cam_width, camera_id=cam_num, depth=True, segmentation=False)
                depth = (depth*1000).astype(np.uint16) #(height, width)
                cv2.imwrite(os.path.join(scene_folder_path, f'depth_{(cam_num):05}.png'), depth)
                cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}.png'), segs)

                # Point cloud of entire scene 
                all_ptcld = from_depth_img_to_pc(depth, cam_width/2, cam_height/2, camera_res['focal_scaling'], camera_res['focal_scaling'])
                all_inds = np.meshgrid(np.arange(cam_width), np.arange(cam_height))
                all_pt_fname = os.path.join(scene_folder_path, f'pc_all_{(cam_num):05}.pkl')
                with open(all_pt_fname, 'wb+') as f:
                    pickle.dump([all_ptcld, all_inds[0].flatten(), all_inds[1].flatten()], f)
                
                for object_idx in original_obj_keys: 
                    pix_left_ratio_idx, onoccluded_pixel_num_idx, segmentation_idx = get_pixel_left_ratio(scene_num, camera, cam_num, mujoco_env_test, object_idx, original_obj_keys, cam_width, cam_height)
                    this_cam_stats[object_idx] = [pix_left_ratio_idx, onoccluded_pixel_num_idx]
                    
                    if onoccluded_pixel_num_idx == 0:
                        continue 
                    segmentation_idx_path = os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}_{object_idx}.png')
                    obj_mask = segmentation_idx.astype(np.uint8)
                    cv2.imwrite(segmentation_idx_path, obj_mask)
                    this_cam_mask_files[object_idx] = f'segmentation_{(cam_num):05}_{object_idx}.png'

                    # Save pointcloud for each object 
                    object_output = get_pointcloud(obj_mask, depth, all_ptcld, camera_res['rot'], cam_height, cam_width)
                    if len(object_output) == 0:
                        continue
                    
                    
                    obj_pt_fname = os.path.join(scene_folder_path, f'pc_{(cam_num):05}_{object_idx}.pkl')
                    with open(obj_pt_fname, 'wb+') as f:
                        pickle.dump(object_output, f)
                    this_cam_pc_files[object_idx] = f'pc_{(cam_num):05}_{object_idx}.pkl'

                
                # Compile all masks, save
                mask_files_full = [os.path.join(scene_folder_path, partial_path)
                    for partial_path in list(this_cam_mask_files.values())
                ]
                all_objects_mask = compile_mask(mask_files_full).astype(bool).astype(np.uint8) #(height, width)
                cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_all_objects_{(cam_num):05}.png'), all_objects_mask)
                
                # Save pointcloud for the entire scene (without table)
                all_objects_output = get_pointcloud(all_objects_mask, depth, all_ptcld, camera_res['rot'], cam_height, cam_width)
                all_objects_pt_fname = os.path.join(scene_folder_path, f'pc_no_table_{(cam_num):05}.pkl')
                with open(all_objects_pt_fname, 'wb+') as f:
                    pickle.dump(all_objects_output, f)
                
                # Save pointcloud for the entire scene (with table)
                all_objects_with_table_mask = (segs > 0).astype(np.uint8)
                all_objects_output_table = get_pointcloud(all_objects_with_table_mask, depth, all_ptcld, camera_res['rot'], cam_height, cam_width)
                all_objects__table_pt_fname = os.path.join(scene_folder_path, f'pc_with_table_{(cam_num):05}.pkl')
                with open(all_objects__table_pt_fname, 'wb+') as f:
                    pickle.dump(all_objects_output_table, f)
                
                
                '''
                res = {
                    'P' : P,
                    'pos' : pos, 
                    'rot' : rot, 
                    'fov' : fov,
                    'focal_scaling' : focal_scaling,
                    'camera_frame_to_world_frame_mat' : camera_tf.matrix,
                    'world_frame_to_camera_frame_mat' : camera_tf.inverse().matrix,
                }
                '''
                camera_res.update({
                    'cam_height' : cam_height,
                    'cam_width' : cam_width,
                    'occlusion_target' : object_i,
                    'target' : cam_targets[cam_num],
                    'objects_left_ratio' : this_cam_stats,
                    'all_segmentation_file' : f'segmentation_{(cam_num):05}.png',
                    'all_object_segmentation_file' : f'segmentation_all_objects_{(cam_num):05}.png',
                    'object_segmentation_files' : this_cam_mask_files,
                    'all_pc_file' : f'pc_all_{(cam_num):05}.pkl',
                    'with_table_pc_file' : f'pc_with_table_{(cam_num):05}.pkl',
                    'no_table_pc_file' : f'pc_no_table_{(cam_num):05}.pkl',
                    'object_pc_files' : this_cam_pc_files,
                    'rgb_file' : f'rgb_{(cam_num):05}.png',
                    'depth_file' : f'depth_{(cam_num):05}.png',
                })
                cam_information[cam_num] = camera_res
                
        
        object_descriptions = dict()
        object_descriptions['scene_num'] = scene_num
        object_descriptions['light_position'] = light_position
        object_descriptions['light_direction'] = light_direction
        object_descriptions['ambients'] = ambients
        object_descriptions['speculars'] = speculars
        object_descriptions['diffuses'] = diffuses
        object_descriptions["object_indices"] = new_obj_keys #list(object_idx_to_obj_info.keys())
        object_descriptions["original_obj_keys"] = original_obj_keys
        

        plt_dict = dict()

        for object_idx in new_obj_keys:
            obj_info = object_idx_to_obj_info[object_idx]

            object_description = dict()
            object_description['mesh_filename'] = obj_info['obj_mesh_filename']
            object_description['position'] = mujoco_env_test.data.qpos.ravel()[7+7*object_idx:7+7*object_idx+3].copy()
            object_description['orientation'] = mujoco_env_test.data.qpos.ravel()[7+7*object_idx+3:7+7*object_idx+7].copy()
            object_quat = copy.deepcopy(object_description['orientation'])
            object_quat[:3] = object_description['orientation'][1:]
            object_quat[3] = object_description['orientation'][0]
            object_description['orientation_quat'] = object_quat

            object_description['scale'] = obj_info['scale']
            object_description['color'] = obj_info['color']
            
            object_description['obj_synset_cat'] = selected_objects[object_idx][0]
            object_description['obj_cat'] = selected_objects[object_idx][1]
            object_description['obj_shapenet_id'] = selected_objects[object_idx][-2]
            object_description['obj_id'] = selected_objects[object_idx][-1]

            object_description['table']={'mesh_filename': f'04379243/{table_id}/models/model_normalized.obj', \
                    'position': mujoco_env_test.data.qpos.ravel()[0:3].copy(), \
                    'orientation': mujoco_env_test.data.qpos.ravel()[3:7].copy(), \
                    'scale': table_scale}

            r2 = R.from_quat(object_quat)
            object_tf = autolab_core.RigidTransform(rotation = r2.as_matrix(), translation = object_description['position'], from_frame='object2_{}'.format(object_idx), to_frame='world')
            
            object_bounds = obj_info['object_bounds']
            object_description['object_bounds_self_frame'] = obj_info['object_bounds']
            object_bbox = bounds_xyz_to_corners(object_bounds) #(N,3)
            object_bbox_world = transform_3d_frame(object_tf.matrix, object_bbox)
            object_description['object_bbox_self_frame'] = object_bbox
            object_description['object_bbox_world_frame'] = object_bbox_world
            object_description['object_bounds_world_frame'] = transform_3d_frame(object_tf.matrix, obj_info['object_bounds'])
            object_description['object_frame_to_world_frame_mat'] = object_tf.matrix
            object_description['world_frame_to_object_frame_mat'] = object_tf.inverse().matrix
            
            cur_position = object_description['position']
            object_cam_d = dict()
            
            for cam_num in cam_information.keys():
                non_occluded_object_pixel_num = cam_information[cam_num]['objects_left_ratio'].get(object_idx, None)
                assert not non_occluded_object_pixel_num is None
                if non_occluded_object_pixel_num[-1] == 0:
                    continue

                P,camera_tf = cam_information[cam_num]['P'], cam_transformations[cam_num]

                pixel_coord = project_2d(P, camera_tf, np.array(cur_position).reshape(-1,3))
                pixel_coord = pixel_coord.reshape((-1,))

                object_bbox_world_frame_2d = project_2d(P, camera_tf, np.array(object_description['object_bbox_world_frame']).reshape(-1,3))
                object_bounds_world_frame_2d = project_2d(P, camera_tf, np.array(object_description['object_bounds_world_frame']).reshape(-1,3))

                # img_path = os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png')
                # img_plot = mpimg.imread(img_path)
                # fig, ax = plt.subplots(figsize=(10, 10))
                # ax.imshow(img_plot)
                # bounds_2d = copy.deepcopy(object_bbox_world_frame_2d)
                # bounds_2d[:,0] = cam_width -  bounds_2d[:,0]
                # for bound in bounds_2d:
                #     i,j = bound
                #     ax.scatter(i,j,   marker=".", c='r', s=100)
                # fig.savefig(os.path.join(scene_folder_path, f'rgb_{(cam_num):05}_{object_idx}.png'), dpi=fig.dpi)
                # plt.close()
                
                
                l1,l2 = plt_dict.get(cam_num, ([],[]))
                l1.append(pixel_coord)
                l2.append(object_bbox_world_frame_2d)
                plt_dict[cam_num] = (l1,l2)

                object_cam_d[cam_num] = {
                    'object_position_2d' : pixel_coord,
                    'object_bbox_world_frame_2d' : object_bbox_world_frame_2d,
                    'object_bounds_world_frame_2d' : object_bounds_world_frame_2d,
                }
            
            
            object_description['object_cam_d'] = object_cam_d
            object_descriptions[object_idx] = object_description

        

        for cam_num in cam_information.keys():
            cam_d = cam_information[cam_num]
            l1,l2 = plt_dict[cam_num]
            
            all_pixels = np.vstack(l1+l2)
            upper_x, upper_y = np.max(all_pixels, axis=0)
            lower_x, lower_y = np.min(all_pixels, axis=0)
            add_x = (upper_x - lower_x) * 0.12
            add_y = (upper_y - lower_y) * 0.12
            upper_x, upper_y = upper_x+add_x, upper_y+add_y
            lower_x, lower_y = lower_x-add_x, lower_y-add_y
            upper_x, lower_x = np.clip(upper_x, 0, cam_width), np.clip(lower_x, 0, cam_width)
            upper_y, lower_y = np.clip(upper_y, 0, cam_height), np.clip(lower_y, 0, cam_height)
            corners4 = np.array([[lower_x, lower_y], \
                [upper_x, lower_y], \
                [upper_x, upper_y], \
                [lower_x, upper_y]])
            
            cam_d.update({
                'scene_bounds' : corners4,
            })
            cam_information[cam_num] = cam_d
            # corners4_to_plot = np.array([[cam_width-lower_x, lower_y], \
            #     [cam_width-upper_x, lower_y], \
            #     [cam_width-upper_x, upper_y], \
            #     [cam_width-lower_x, upper_y]])
            # fig, ax = plt.subplots(figsize=(10, 10))
            # img_path = os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png')
            # img_plot = mpimg.imread(img_path)
            # ax.imshow(img_plot)
            # poly = patches.Polygon(corners4, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_artist(poly)
            # for bounds_2d_obj in l2:
            #     bounds_2d = copy.deepcopy(bounds_2d_obj)
            #     bounds_2d[:,0] = cam_width -  bounds_2d[:,0]
            #     for bound in bounds_2d:
            #         i,j = bound
            #         ax.scatter(i,j,   marker=".", c='r', s=100)
            # fig.savefig(img_path, dpi=fig.dpi)
            # plt.close()
        # import pdb; pdb.set_trace()
        object_descriptions['cam_information'] = cam_information
        
        
        with open(os.path.join(scene_folder_path, 'scene_description.p'), 'wb+') as save_file:
            pickle.dump(object_descriptions, save_file)  

        print("DONE: ", scene_num, len(cam_information))
    except:
        print('##################################### GEN Error!')
        shutil.rmtree(scene_folder_path)
        print(selected_objects)
        traceback.print_exc()
        # DANGER   

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        raise    

if __name__ == '__main__':
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
        gen_data(acc_scene_num, selected_objects[scene_num], args)
        
        
