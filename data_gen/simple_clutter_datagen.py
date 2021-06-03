import json
import os
import json
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
for name, color in all_colors_dict.items():
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
        # Export table mesh as .stl file
        stl_table_mesh_filename=os.path.join(top_dir, f'assets/table_{scene_num}.stl')
        f = open(stl_table_mesh_filename, "w+")
        f.close()
        table_mesh.export(stl_table_mesh_filename)

        table_color = selected_colors[0] #np.random.uniform(size=3)
        table_bounds = table_bounds*table_size
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
            # print("=> OBJECT {}".format(object_idx), selected_objects[object_idx])
            obj_cat, obj_id, _ = selected_objects[object_idx]
            obj_info = dict()
            obj_info['obj_cat'] = obj_cat
            obj_info['obj_id'] = obj_id
            
            obj_mesh_filename = os.path.join(shapenet_filepath,'0{}/{}/models/model_normalized.obj'.format(obj_cat, obj_id))
            object_mesh = trimesh.load(obj_mesh_filename, force='mesh')
            old_bound = object_mesh.bounds 
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
            # Rotate object to face different directions
            z_rot = np.random.uniform(0,2*np.pi,1)[0]
            object_rot = [0,0,z_rot]
            '''
            Determine object color
            '''
            object_color = selected_colors[object_idx+1] #np.random.uniform(size=3)
            '''
            Determine object size
            '''
            object_bounds = object_mesh.bounds
            range_max = np.max(object_bounds[1] - object_bounds[0])
            random_scale = np.random.uniform(0.6,1,1)[0]
            object_size = random_scale / range_max
            object_bounds = object_bounds*object_size
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
            obj_info['scale'] = object_size
            obj_info['color'] = selected_colors[object_idx+1]
            obj_info['rotation'] = object_rot
            obj_info['obj_mesh_filename'] = obj_mesh_filename
            obj_info['object_height'] = object_height
            obj_info['object_tf'] = object_tf
            obj_info['object_bounds'] = object_bounds
            object_idx_to_obj_info[object_idx] = obj_info

        layout_filename = os.path.join(scene_folder_path, 'layout.png')
        # draw_boundary_points_rect(prev_bbox, layout_filename)

        
        scene_xml_file=os.path.join(top_dir, f'base_scene.xml')
        cam_temp_scene_xml_file=os.path.join(top_dir, f'{train_or_test}_xml/cam_temp_data_gen_scene_{scene_num}.xml')
        shutil.copyfile(scene_xml_file, cam_temp_scene_xml_file)

        add_objects(cam_temp_scene_xml_file, 'table', [stl_table_mesh_filename], table_xyz, table_size, table_color, table_orientation, scene_num)
        
        # scene_name, object_name, mesh_names, pos, size, color, rot, run_id, contact_geom_list=None, add_ind=-1, add_contacts=True
        for object_idx in object_idx_to_obj_info.keys():
            obj_info = object_idx_to_obj_info[object_idx]
            mesh_names = [os.path.join(top_dir, f'assets/model_normalized_{scene_num}_{object_idx}.stl')]
            add_objects(cam_temp_scene_xml_file, \
                        f'object_{object_idx}_{scene_num}', \
                        mesh_names, \
                        obj_info['xyz'], \
                        obj_info['scale'], \
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
        
        e = MujocoEnv(cam_temp_scene_xml_file, 1, has_robot=False)
        e.sim.physics.forward()
        
        for _ in object_idx_to_obj_info.keys():
            for _ in range(4000):
                e.model.step()
        
        state = e.get_env_state().copy()

        stable = True
        all_poses=e.data.qpos.ravel().copy()
        original_obj_keys = list(object_idx_to_obj_info.keys())
        for object_idx in original_obj_keys:
            current_xyz = all_poses[7+7*object_idx : 7+7*object_idx+3]
            original_xyz = object_idx_to_obj_info[object_idx]['xyz']

            if np.linalg.norm(current_xyz - original_xyz) > 0.15:
                print("WARNING: object location shifted more than 15cm: ", current_xyz, original_xyz, np.linalg.norm(current_xyz - original_xyz))
                print("WARNING: object location shifted more than 15cm (information): ",  object_idx_to_obj_info[object_idx]['obj_mesh_filename'])
                del object_idx_to_obj_info[object_idx]  
        
        '''
        Generate camera position and target
        ''' 
        xys = []
        for object_idx in object_idx_to_obj_info.keys(): 
            xyz = object_idx_to_obj_info[object_idx]['xyz']
            xys.append([xyz[0],xyz[1]])
        xys = np.asarray(xys)

        cam_width = 640
        cam_height = 480

        valid_cameras = []
        camera_stats = dict()

        cam_num = 0
        cam_xyzs = dict()
        cam_targets = dict()
        cam_num_to_occlusion_target = dict()
        camera_objects = dict()

        num_rotates = 3
        step_deg = 7
        for object_i in object_idx_to_obj_info.keys(): 
            xyz1 = object_idx_to_obj_info[object_i]['xyz']
            height1 = object_idx_to_obj_info[object_i]['object_height']
            
            pairwise_diff = xys - xyz1[:2].reshape((1,2))
            dist = np.linalg.norm(pairwise_diff, axis=1)
            max_dist = np.max(dist)
            
            for object_j in object_idx_to_obj_info.keys():  
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
                        cam_xyz,cam_target = get_camera_position_occluded_one_cam(table_height, xyz1,xyz2,height1,height2,max_dist,[low_deg,high_deg])
                        add_camera(cam_temp_scene_xml_file, f'gen_cam_{cam_num}', cam_xyz, cam_target, cam_num)

                        mujoco_env_test = MujocoEnv(cam_temp_scene_xml_file, 1, has_robot=False)
                        mujoco_env_test.sim.physics.forward()

                        camera = Camera(physics=mujoco_env_test.model, height=cam_height, width=cam_width, camera_id=cam_num)
                        segs = camera.render(segmentation=True)[:,:,0] #(480, 640, 2)
                        pix_left_ratio, onoccluded_pixel_num, segmentation = get_pixel_left_ratio(scene_num, camera, cam_num, mujoco_env_test, object_i, original_obj_keys, cam_width, cam_height)
                        if pix_left_ratio >= 0.05 and onoccluded_pixel_num > 0:

                            cam_xyzs[cam_num] = cam_xyz
                            cam_targets[cam_num] = cam_target
                            cam_num_to_occlusion_target[cam_num] = object_i
                            camera_objects[cam_num] = camera

                            valid_cameras.append(cam_num)
                            
                            this_cam_stats = dict()
                            this_cam_stats[object_i] = [pix_left_ratio, onoccluded_pixel_num]
                            cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}_{object_i}.png'), segmentation.astype(np.uint8))
                            for object_idx in object_idx_to_obj_info.keys(): 
                                if object_idx == object_i:
                                    continue 
                                pix_left_ratio_idx, onoccluded_pixel_num_idx, segmentation_idx = get_pixel_left_ratio(scene_num, camera, cam_num, mujoco_env_test, object_idx, original_obj_keys, cam_width, cam_height)
                                this_cam_stats[object_idx] = [pix_left_ratio_idx, onoccluded_pixel_num_idx]
                                if not segmentation_idx is None:
                                    cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}_{object_idx}.png'), segmentation_idx.astype(np.uint8))
                            rgb=mujoco_env_test.model.render(height=cam_height, width=cam_width, camera_id=cam_num, depth=False, segmentation=False)
                            cv2.imwrite(os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png'), rgb)

                            # Depth image
                            depth = mujoco_env_test.model.render(height=cam_height, width=cam_width, camera_id=cam_num, depth=True, segmentation=False)
                            depth = (depth*1000).astype(np.uint16)
                            cv2.imwrite(os.path.join(scene_folder_path, f'depth_{(cam_num):05}.png'), depth)

                            cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}.png'), segs)
                            valid_cameras.append(cam_num)
                            camera_stats[cam_num] = this_cam_stats
                        
                        cam_num += 1
                        if pix_left_ratio > 0.95:
                            keep_rotating = False

        
        object_descriptions = dict()
        object_descriptions['light_position'] = light_position
        object_descriptions['light_direction'] = light_direction
        object_descriptions['ambients'] = ambients
        object_descriptions['speculars'] = speculars
        object_descriptions['diffuses'] = diffuses
        object_descriptions["object_indices"] = list(object_idx_to_obj_info.keys())
        object_descriptions["cam_num_to_occlusion_target"] = cam_num_to_occlusion_target
        plt_dict = dict()
        for object_idx in object_idx_to_obj_info.keys():
            obj_info = object_idx_to_obj_info[object_idx]

            object_description = dict()
            object_description['mesh_filename'] = obj_info['obj_mesh_filename']
            object_description['position'] = e.data.qpos.ravel()[7+7*object_idx:7+7*object_idx+3].copy()
            object_description['orientation'] = e.data.qpos.ravel()[7+7*object_idx+3:7+7*object_idx+7].copy()
            object_description['scale'] = obj_info['scale']
            object_description['color'] = obj_info['color']
            object_description['obj_cat'], object_description['obj_shapenet_id'], object_description['obj_id'] = selected_objects[object_idx]
            object_description['camera_pos'] = cam_xyzs
            object_description['cam_targets'] = cam_targets
            object_description['table']={'mesh_filename':table_mesh_filename, \
                    'position': e.data.qpos.ravel()[0:3].copy(), \
                    'orientation': e.data.qpos.ravel()[3:7].copy(), \
                    'scale': table_size}
            object_description['cam_height'] = cam_height
            object_description['cam_width'] = cam_width

            cur_position = object_description['position']
            object_cam_d = dict()
            for cam_num in camera_objects.keys():
                if camera_stats[cam_num][object_idx][-1] == 0:
                    continue
                
                object_camera_info_i = dict()
                object_camera_info_i['pix_left_ratio'] = camera_stats[cam_num][object_idx][0]
                object_camera_info_i['total_pixel_in_scene'] = camera_stats[cam_num][object_idx][0] * camera_stats[cam_num][object_idx][1]
                camera = camera_objects[cam_num] #Camera(physics=e.model, height=cam_height, width=cam_width, camera_id=cam_num)
                P,camera_tf = get_camera_matrix(camera)
                world_to_camera_tf_mat = camera_tf.inverse().matrix

                pixel_coord = project_2d(P, camera_tf, np.array(cur_position).reshape(-1,3))
                pixel_coord = pixel_coord.reshape((-1,))

                pt_3d_homo = np.append(obj_info['object_bounds'].T, np.ones(2).astype('int').reshape(1,-1), axis=0) #(4,2)
                bounding_coord = obj_info['object_tf'].matrix @ pt_3d_homo #(4,2)
                bounding_coord = bounding_coord / bounding_coord[-1, :]
                bounding_coord = bounding_coord[:-1, :].T #(2,3)
                bounding_pixel_coord = project_2d(P, camera_tf, np.array(bounding_coord).reshape(-1,3))

                l1,l2 = plt_dict.get(cam_num, ([],[]))
                l1.append(pixel_coord)
                l2.append(bounding_pixel_coord)
                plt_dict[cam_num] = (l1,l2)

                object_camera_info_i["object_center"] = pixel_coord
                object_camera_info_i["object_bounds"] = bounding_pixel_coord
                object_camera_info_i["intrinsics"] = P
                object_camera_info_i["world_to_camera_mat"] = world_to_camera_tf_mat

                root_name = f'_{(cam_num):05}'
                object_camera_info_i['rgb_all_path'] = os.path.join(scene_folder_path, 'rgb'+root_name+'.png')
                object_camera_info_i['depth_all_path'] = os.path.join(scene_folder_path, 'depth'+root_name+'.png')
                
                obj_name = f'_{(cam_num):05}_{object_idx}'
                segmentation_filename = os.path.join(scene_folder_path, 'segmentation'+obj_name+'.png')
                object_camera_info_i['mask_path'] = segmentation_filename
                object_cam_d[cam_num] = object_camera_info_i
            object_description['object_cam_d'] = object_cam_d
            object_descriptions[object_idx] = object_description

        for cam_num in camera_objects.keys():
            l1,l2 = plt_dict[cam_num]
            # img_path = os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png')
            # img_plot = mpimg.imread(img_path)
            # fig, ax = plt.subplots(figsize=(10, 10))
            all_pixels = np.vstack(l1+l2)
            upper_x, upper_y = np.max(all_pixels, axis=0)
            lower_x, lower_y = np.min(all_pixels, axis=0)
            add_x = (upper_x - lower_x) * 0.15
            add_y = (upper_y - lower_y) * 0.15
            upper_x, upper_y = upper_x+add_x, upper_y+add_y
            lower_x, lower_y = lower_x-add_x, lower_y-add_y
            upper_x, lower_x = np.clip(upper_x, 0, cam_width), np.clip(lower_x, 0, cam_width)
            upper_y, lower_y = np.clip(upper_y, 0, cam_height), np.clip(lower_y, 0, cam_height)
            corners4 = np.array([[lower_x, lower_y], \
                [upper_x, lower_y], \
                [upper_x, upper_y], \
                [lower_x, upper_y]])
            
            for object_idx in object_idx_to_obj_info.keys():
                object_description = object_descriptions[object_idx]
                object_description["scene_bounds_{}".format(cam_num)] = corners4
                object_descriptions[object_idx] = object_description
            
            # corners4 = np.array([[cam_width-lower_x, lower_y], \
            #     [cam_width-upper_x, lower_y], \
            #     [cam_width-upper_x, upper_y], \
            #     [cam_width-lower_x, upper_y]])
            # ax.imshow(img_plot)
            # poly = patches.Polygon(corners4, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_artist(poly)
            # fig.savefig(img_path, dpi=fig.dpi)
            # plt.close()
        
        with open(os.path.join(scene_folder_path, 'scene_description.p'), 'wb+') as save_file:
            pickle.dump(object_descriptions, save_file)  

        print("DONE: ", scene_num, len(camera_objects))
    except:
        print('##################################### GEN Error!')
        shutil.rmtree(scene_folder_path)
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
            selected_objects_i.append((sample['synsetId'], sample['ShapeNetModelId'], sample['objId']))
        selected_objects.append(selected_objects_i)

    for scene_num in range(args.num_scenes):
        acc_scene_num = scene_num + args.start_scene_idx
        gen_data(acc_scene_num, selected_objects[scene_num], args)

    # for scene_num in range(len(df)):
    #     acc_scene_num = scene_num + args.start_scene_idx
    #     sample = df.iloc[scene_num]
    #     sample_input = [(sample['synsetId'], sample['ShapeNetModelId'], sample['objId'])]
    #     gen_data(acc_scene_num, sample_input, args.shapenet_filepath, args.shapenet_decomp_filepath, args.top_dir, args.train_or_test)
    
        
        
