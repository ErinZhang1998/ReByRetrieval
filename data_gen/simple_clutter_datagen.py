import json
import os
import numpy as np
import multiprocessing as mp
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool


from trajopt.envs.mujoco_env import MujocoEnv
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
from trajopt.mujoco_utils import add_camera, add_objects
from scipy.spatial.transform import Rotation as R
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simple_clutter_utils import *
from PIL import Image 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from data_gen_args import *
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

#Color of objects:
all_colors_dict = mcolors.CSS4_COLORS #TABLEAU_COLORS #
ALL_COLORS = []
for name, color in all_colors_dict.items():
    ALL_COLORS.append(np.asarray(mcolors.to_rgb(color)))




def move_object(e, ind, pos, rot):
    all_poses=e.data.qpos.ravel().copy()
    all_vels=e.data.qvel.ravel().copy()
    
    all_poses[7+7*ind : 7+7*ind+3]=pos
    all_poses[7+7*ind+3 : 7+7*ind+7]=rot
    
    all_vels[6+6*ind : 6+6*ind+6] = 0
    e.set_state(all_poses, all_vels)

REGION_LIMIT = 2*np.sqrt(0.5)


#@profile
def gen_data(scene_num, selected_objects, shapenet_filepath, shapenet_decomp_filepath, top_dir, train_or_test):

    num_objects = len(selected_objects) 
    selected_colors = [ALL_COLORS[i] for i in np.random.choice(len(ALL_COLORS), num_objects+1, replace=False)]

    try:
        # Make temporary scene xml file
        scene_xml_file=os.path.join(top_dir, f'base_scene.xml')

        

        temp_scene_xml_file=os.path.join(top_dir, f'{train_or_test}_xml/temp_data_gen_scene_{scene_num}.xml')
        shutil.copyfile(scene_xml_file, temp_scene_xml_file)

        scene_description={}

        '''
        Add table
        '''
        if num_objects < 10:
            camera_distance = 1.5 * 2.5
        else:
            camera_distance = 1.5 * 5 
        table_generated = False
        table_bounds = None
        table_xyz = None
        table_trans = None
        while not table_generated:
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
            # Export table mesh as .stl file
            stl_table_mesh_filename=os.path.join(top_dir, f'assets/table_{scene_num}.stl')
            f = open(stl_table_mesh_filename, "w+")
            f.close()
            table_mesh.export(stl_table_mesh_filename)

            table_color = selected_colors[0] #np.random.uniform(size=3)
            # Move table above floor
            table_bounds = table_bounds*table_size
            table_width = table_bounds[1,0] - table_bounds[0,0]
            table_length = table_bounds[1,1] - table_bounds[0,1]
            # 
            if min(table_width, table_length)/max(table_width, table_length) < 0.7:
                # We want roughly square-shaped table to ensure that objects like knife
                # will not fall off table
                continue
            table_bottom = -table_bounds[0][2]
            table_height = table_bounds[1][2] - table_bounds[0][2]
            table_xyz = [0, 0, table_bottom]
            table_orientation = [0,0,0]

            # Add table to the scene
            add_objects(temp_scene_xml_file, 'table', [stl_table_mesh_filename], table_xyz, table_size, table_color, table_orientation, scene_num, add_contacts=False)
            table_generated = True

            r = R.from_euler('xyz', table_orientation, degrees=False) 
            table_trans = autolab_core.RigidTransform(rotation = r.as_matrix(), translation = np.asarray(table_xyz), from_frame='table')    
        
        
        '''
        Add objects
        '''
        obj_xyzs=[]
        obj_rotations=[]
        obj_scales=[]
        object_max_height = -10
        obj_mesh_filenames = []

        prev_polys = []
        probs = [0.15,0.15,0.15,0.15,0.1,0.1,0.1,0.1]
        prev_bbox = []
        all_obj_bounds = []
        
        object_position_region = None
        for object_idx in range(num_objects):
            obj_cat, obj_id = selected_objects[object_idx]
            obj_mesh_filename = os.path.join(shapenet_filepath,'{}/{}/models/model_normalized.obj'.format(obj_cat, obj_id))
            object_mesh = trimesh.load(obj_mesh_filename, force='mesh')
            old_bound = object_mesh.bounds 
            '''
            Determine object rotation
            '''
            # 
            rot_vec, upright_mat  = determine_object_rotation(object_mesh)
            object_mesh.apply_transform(upright_mat)
            z_rot = np.random.uniform(0,2*np.pi,1)[0]
            
            # Store the upright object in .stl file in assets
            stl_obj_mesh_filename = os.path.join(top_dir, f'assets/model_normalized_{scene_num}_{object_idx}.stl')
            f = open(stl_obj_mesh_filename, "w+")
            f.close()
            object_mesh.export(stl_obj_mesh_filename)
            # Rotate object to face different directions
            object_rot = rot_vec 
            
            '''
            Determine object color
            '''
            object_color = selected_colors[object_idx+1] #np.random.uniform(size=3)
            '''
            Determine object size
            '''
            object_bounds = object_mesh.bounds
            range_max = np.max(object_bounds[1, :2] - object_bounds[0, :2])
            random_scale = np.random.uniform(0.6,1,1)[0]
            object_size = random_scale / range_max
            object_bounds = object_bounds*object_size
            object_bottom = -object_bounds[0][2]

            '''
            Determine object position
            '''
            object_z = table_height + object_bottom + 0.005
            if object_idx == 0:
                a,b,_ = object_bounds[1] - object_bounds[0]
                diag_length = np.sqrt(a **2 + b**2)
                left_x, right_x = -diag_length/2, diag_length/2
                down_y, up_y = -diag_length/2, diag_length/2
                
                # To put at the center of the table
                object_x = (table_bounds[1,0] + table_bounds[0,0]) / 2
                object_y = (table_bounds[1,1] + table_bounds[0,1]) / 2
                
                x_top, XMAX =  object_x+right_x, object_x+right_x+REGION_LIMIT
                x_bottom, XMIN = object_x+left_x, object_x+left_x-REGION_LIMIT
                y_top, YMAX = object_y+up_y, object_y+up_y+REGION_LIMIT
                y_bottom, YMIN = object_y+down_y, object_y+down_y-REGION_LIMIT
                
                object_position_region = {
                    0: [[x_top, XMAX],[y_bottom, y_top]],
                    1: [[x_bottom, x_top],[y_top,YMAX]],
                    2: [[XMIN, x_bottom],[y_bottom, y_top]],
                    3: [[x_bottom, x_top],[YMIN, y_bottom]],
                    4: [[x_top,XMAX],[YMIN, y_bottom]],
                    5: [[x_top,XMAX],[y_top,YMAX]],
                    6: [[XMIN, x_bottom],[y_top,YMAX]],
                    7: [[XMIN, x_bottom],[YMIN, y_bottom]],
                }
            else:
                object_x, object_y, probs = generate_object_xy_rect(object_bounds, prev_bbox, object_position_region, probs, obj_xyzs, all_obj_bounds)
            
            object_xyz = [object_x, object_y, object_z]
            corner = get_2d_diagonal_corners([object_xyz], [object_bounds])[0]
            prev_bbox.append(corner)
            
            object_height = object_bounds[1][2] - object_bounds[0][2]            
            object_max_height = max(object_max_height, object_height)

            obj_xyzs.append(object_xyz)
            obj_rotations.append([0,0,object_rot[-1]])
            obj_scales.append(object_size)
            all_obj_bounds.append(object_bounds)
            obj_mesh_filenames.append(obj_mesh_filename)

        
        # 
        scene_folder_path = os.path.join(top_dir, f'{train_or_test}/scene_{scene_num:06}')

        if os.path.exists(scene_folder_path):
            shutil.rmtree(scene_folder_path)
        os.mkdir(scene_folder_path)

        layout_filename = os.path.join(scene_folder_path, 'layout.png')
        draw_boundary_points_rect(prev_bbox, layout_filename)

        scene_xml_file=os.path.join(top_dir, f'base_scene.xml')
        cam_temp_scene_xml_file=os.path.join(top_dir, f'{train_or_test}_xml/cam_temp_data_gen_scene_{scene_num}.xml')
        shutil.copyfile(scene_xml_file, cam_temp_scene_xml_file)

        add_objects(cam_temp_scene_xml_file, 'table', [stl_table_mesh_filename], table_xyz, table_size, table_color, table_orientation, scene_num, add_contacts=False)
        
        # scene_name, object_name, mesh_names, pos, size, color, rot, run_id, contact_geom_list=None, add_ind=-1, add_contacts=True
        for object_idx in range(num_objects):
            mesh_names = [os.path.join(top_dir, f'assets/model_normalized_{scene_num}_{object_idx}.stl')]
            add_objects(cam_temp_scene_xml_file, f'object_{object_idx}_{scene_num}', mesh_names, obj_xyzs[object_idx], obj_scales[
                        object_idx], selected_colors[object_idx+1], [0,0,z_rot], scene_num, add_contacts=False)
        
        
        '''
        Generate camera position and target
        ''' 
        # Generate camera heights
        max_object_height = table_height + object_max_height
        camera_poss, cam_targets = get_camera_position(camera_distance, table_height, max_object_height, obj_xyzs)
        # camera_poss, cam_targets = get_fixed_camera_position(camera_distance, table_height+1, table_xyz)
        num_camera = len(camera_poss)

        for cam_num in range(num_camera):
            camera_pos = camera_poss[cam_num] #[camera_pos_x[cam_num], camera_pos_y[cam_num], camera_pos_z[cam_num]]
            cam_target = cam_targets[cam_num]
            add_camera(cam_temp_scene_xml_file, f'gen_cam_{cam_num}', camera_pos, cam_target, cam_num)
        
        e = MujocoEnv(cam_temp_scene_xml_file, 1, has_robot=False)
        e.sim.physics.forward()
        
        
        
        for added_object_ind in range(num_objects):
            # cam_num = 0
            '''
            # Render before any steps are taken in the scene. For example, an object before it fall down 
            # on the table, not necessary.
            for _ in range(num_camera):
                rgb=e.model.render(height=480, width=640, camera_id=cam_num, depth=False, segmentation=False)
                cv2.imwrite(os.path.join(scene_folder_path, f'before_rgb_{(cam_num):05}.png'), rgb)
                cam_num += 1
            '''
            for _ in range(4000):
                e.model.step()
        state = e.get_env_state().copy()

        cam_width = 640
        cam_height = 480
    
        for cam_num in range(num_camera):
            # 
            rgb=e.model.render(height=cam_height, width=cam_width, camera_id=cam_num, depth=False, segmentation=False)
            cv2.imwrite(os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png'), rgb)
            
            # Depth image
            depth = e.model.render(height=cam_height, width=cam_width, camera_id=cam_num, depth=True, segmentation=False)
            depth = (depth*1000).astype(np.uint16)
            cv2.imwrite(os.path.join(scene_folder_path, f'depth_{(cam_num):05}.png'), depth)
            
            camera = Camera(physics=e.model, height=cam_height, width=cam_width, camera_id=cam_num)
            segs = camera.render(segmentation=True)[:,:,0] #(480, 640, 2)
            occluded_geom_id_to_seg_id = {camera.scene.geoms[geom_ind][3]: camera.scene.geoms[geom_ind][8] for geom_ind in range(camera.scene.geoms.shape[0])}
            cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}.png'), segs)

            

            present_in_view_ind = 0
            for added_object_ind in range(num_objects):
                '''
                # Assume no object has fall off table, see code from old ARM repo if needed
                if added_object_ind in off_table_inds: 
                    continue
                '''

                target_id = e.model.model.name2id(f'gen_geom_object_{added_object_ind}_{scene_num}_0', "geom")
                segmentation = segs==occluded_geom_id_to_seg_id[target_id]
                
                target_obj_pix = np.argwhere(segmentation).shape[0] #(num_equal_target_id, 2)
                if target_obj_pix < 50:
                    continue
                # Move all other objects far away, except the table, so that we can capture
                # only one object in a scene.
                for move_obj_ind in range(num_objects):
                    if move_obj_ind != added_object_ind:
                        move_object(e, move_obj_ind, [20, 20, move_obj_ind], [0,0,0,0])
                e.sim.physics.forward()
                '''
                # Test code: only one object on table; other objects are far away
                rgb=e.model.render(height=480, width=640, camera_id=cam_num, depth=False, segmentation=False)
                cv2.imwrite(os.path.join(scene_folder_path, f'before_moving_things_back_{present_in_view_ind}_rgb_{(cam_num):05}.jpeg'), rgb)
                '''
                unocc_target_id = e.model.model.name2id(f'gen_geom_object_{added_object_ind}_{scene_num}_0', "geom")
                unoccluded_camera = Camera(physics=e.model, height=cam_height, width=cam_width, camera_id=cam_num)
                unoccluded_segs = unoccluded_camera.render(segmentation=True)
                # Move other objects back onto table 
                e.set_env_state(state)
                e.sim.physics.forward()

                unoccluded_geom_id_to_seg_id = {unoccluded_camera.scene.geoms[geom_ind][3]: unoccluded_camera.scene.geoms[geom_ind][8] for geom_ind in range(unoccluded_camera.scene.geoms.shape[0])}
                unoccluded_segs = np.concatenate((unoccluded_segs[:,:,0:1],unoccluded_segs[:,:,0:1],unoccluded_segs[:,:,0:1]), axis=2).astype(np.uint8)
                unoccluded_segmentation = unoccluded_segs[:,:,0]==unoccluded_geom_id_to_seg_id[unocc_target_id]
                # num_unoccluded_pix = np.argwhere(unoccluded_segmentation).shape[0]
                
                segmentation = np.logical_and(segmentation, unoccluded_segmentation)
                try:
                    pix_left_ratio = np.argwhere(segmentation).shape[0] / np.argwhere(unoccluded_segmentation).shape[0]
                except:
                    continue

                if pix_left_ratio > 0.3:
                    cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}_{present_in_view_ind}.png'), segmentation.astype(np.uint8))
                present_in_view_ind += 1 

        scene_description['camera_pos'] = camera_poss
        scene_description['cam_targets'] = cam_targets
        scene_description['object_descriptions']=[]

        scene_description['table']={'mesh_filename':table_mesh_filename, \
                'position': e.data.qpos.ravel()[0:3].copy(), \
                'orientation': e.data.qpos.ravel()[3:7].copy(), \
                'scale': table_size}
        
        scene_description['cam_height'] = cam_height
        scene_description['cam_width'] = cam_width

        object_descriptions = []
        for object_idx in range(num_objects):
            obj_mesh_filename = obj_mesh_filenames[object_idx]
        
            object_description={}
            object_description['mesh_filename'] = obj_mesh_filename
            object_description['position'] = e.data.qpos.ravel()[7+7*object_idx:7+7*object_idx+3].copy()
            object_description['orientation'] = e.data.qpos.ravel()[7+7*object_idx+3:7+7*object_idx+7].copy()
            object_description['scale'] = obj_scales[object_idx]
            object_description['obj_cat'], object_description['obj_id'] = selected_objects[object_idx]

            cur_position = object_description['position']
            # q = np.zeros(4)
            # q[0] = object_description['orientation'][1]
            # q[1] = object_description['orientation'][2]
            # q[2] = object_description['orientation'][3]
            # q[3] = object_description['orientation'][0]
            # r = R.from_quat(q)

            for cam_num in range(num_camera):
                camera = Camera(physics=e.model, height=480, width=640, camera_id=cam_num)
                P,camera_tf = get_camera_matrix(camera)
                world_to_camera_tf_mat = camera_tf.inverse().matrix

                pixel_coord = project_2d(P, camera_tf, np.array(cur_position.reshape(-1,3)))
                pixel_coord = pixel_coord.reshape((-1,))
                object_center = np.array([pixel_coord[0], pixel_coord[1]])
                object_description["object_center_{}".format(cam_num)] = pixel_coord
                object_description["intrinsics_{}".format(cam_num)] = P
                object_description["world_to_camera_mat_{}".format(cam_num)] = world_to_camera_tf_mat
            
            object_descriptions.append(object_description)
        
        scene_description['object_descriptions'] = object_descriptions
        with open(os.path.join(scene_folder_path, 'scene_description.p'), 'wb+') as save_file:
                pickle.dump(scene_description, save_file)  
        
    except:
        print('##################################### GEN Error!')
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
        num_object = np.random.randint(args.min_num_objects, args.max_num_objects, 1)[0]
        selected_object_indices.append(np.random.randint(0, len(df), num_object))

    selected_objects = []
    for selected_indices in selected_object_indices:
        selected_objects_i = []
        for idx in selected_indices:
            sample = df.iloc[idx]
            selected_objects_i.append((sample['synsetId'], sample['ShapeNetModelId']))
        selected_objects.append(selected_objects_i)

    for scene_num in range(args.num_scenes):
        acc_scene_num = scene_num + args.start_scene_idx
        gen_data(acc_scene_num, selected_objects[scene_num], args.shapenet_filepath, args.shapenet_decomp_filepath, args.top_dir, args.train_or_test)
    
        
        
