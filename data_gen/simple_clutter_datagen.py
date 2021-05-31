import json
import os
import json
import numpy as np
import multiprocessing as mp
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.image as mpimg

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
from PIL import Image 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from data_gen_args import *
from simple_clutter_utils import *
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

#Color of objects:
all_colors_dict = mcolors.CSS4_COLORS #TABLEAU_COLORS #
ALL_COLORS = []
for name, color in all_colors_dict.items():
    ALL_COLORS.append(np.asarray(mcolors.to_rgb(color)))




def move_object(e, ind, pos, rot):
    # ASSUME THERE IS TABLE so 7+ and 6+
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
        scene_description=dict()
        scene_folder_path = os.path.join(top_dir, f'{train_or_test}/scene_{scene_num:06}')

        if os.path.exists(scene_folder_path):
            shutil.rmtree(scene_folder_path)
        os.mkdir(scene_folder_path)

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
            print("=> OBJECT {}".format(object_idx), selected_objects[object_idx])
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
            # if obj_cat == 2876657 or obj_cat == 3593526 or obj_cat == 2946921:
            #     random_scale = np.random.uniform(0.6,0.8,1)[0]
            #     object_size = random_scale / range_max
            #     print(obj_cat, random_scale, range_max)
            # elif obj_cat == 2773838 or obj_cat == 2880940:
            #     random_scale = np.random.uniform(0.8,1,1)[0]
            #     object_size = random_scale / range_max
            #     print(obj_cat, random_scale, range_max)
            # else:
            random_scale = np.random.uniform(0.6,1,1)[0]
            object_size = random_scale / range_max
            object_bounds = object_bounds*object_size
            object_bottom = -object_bounds[0][2]

            '''
            Determine object position
            '''
            object_z = table_height + object_bottom + 0.005
            if object_idx == 0:
                # a,b,_ = object_bounds[1] - object_bounds[0]
                # diag_length = np.sqrt(a **2 + b**2)
                # left_x, right_x = -diag_length/2, diag_length/2
                # down_y, up_y = -diag_length/2, diag_length/2
                
                # To put at the center of the table
                object_x = (table_bounds[1,0] + table_bounds[0,0]) / 2
                object_y = (table_bounds[1,1] + table_bounds[0,1]) / 2
                object_xyz = [object_x, object_y, object_z]
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
                # x_top, XMAX =  object_x+right_x, object_x+right_x+REGION_LIMIT
                # x_bottom, XMIN = object_x+left_x, object_x+left_x-REGION_LIMIT
                # y_top, YMAX = object_y+up_y, object_y+up_y+REGION_LIMIT
                # y_bottom, YMIN = object_y+down_y, object_y+down_y-REGION_LIMIT
                
                # object_position_region = {
                #     0: [[x_top, XMAX],[y_bottom, y_top]],
                #     1: [[x_bottom, x_top],[y_top,YMAX]],
                #     2: [[XMIN, x_bottom],[y_bottom, y_top]],
                #     3: [[x_bottom, x_top],[YMIN, y_bottom]],
                #     4: [[x_top,XMAX],[YMIN, y_bottom]],
                #     5: [[x_top,XMAX],[y_top,YMAX]],
                #     6: [[XMIN, x_bottom],[y_top,YMAX]],
                #     7: [[XMIN, x_bottom],[YMIN, y_bottom]],
                # }
                prev_bbox.append(corners)
            else:
                # all_corners = np.vstack(prev_bbox)
                # assert all_corners.shape[1] == 2
                # x_bottom, y_bottom = np.min(all_corners, axis=0)
                # x_top, y_top = np.max(all_corners, axis=0)
                # object_position_region = {
                #     0: [[x_top, 1],[y_bottom, 1]],
                #     1: [[x_bottom, 1],[y_top,1]],
                #     2: [[x_bottom, -1],[y_bottom, 1]],
                #     3: [[x_bottom, 1],[y_bottom, -1]],
                #     4: [[x_top,1],[y_bottom, -1]],
                #     6: [[x_bottom, -1],[y_top,1]],
                #     5: [[x_top,1],[y_top,1]],
                #     7: [[x_bottom, -1],[y_bottom, -1]],
                # }

                # object_x, object_y, probs = generate_object_xy_rect(object_bounds, prev_bbox, object_position_region, probs)
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

        add_objects(cam_temp_scene_xml_file, 'table', [stl_table_mesh_filename], table_xyz, table_size, table_color, table_orientation, scene_num, add_contacts=False)
        
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
                        scene_num, \
                        add_contacts=False)
        
        
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
                print("????????????????????????", current_xyz, original_xyz, np.linalg.norm(current_xyz - original_xyz))
                print("????????????????????????",  object_idx_to_obj_info[object_idx]['obj_mesh_filename'])
                del object_idx_to_obj_info[object_idx]  
        
        '''
        Generate camera position and target
        ''' 
        # Generate camera heights
        max_object_height = table_height + object_max_height
        xyzs = dict()
        heights = dict()
        for object_idx in object_idx_to_obj_info.keys(): 
            xyzs[object_idx] = object_idx_to_obj_info[object_idx]['xyz']
            heights[object_idx] = object_idx_to_obj_info[object_idx]['object_height']
        
        camera_poss, cam_targets, cam_num_to_occlusion_target = get_camera_position_occluded(camera_distance, table_height, max_object_height, xyzs, heights)
        num_camera = len(camera_poss)
        
        
        for cam_num in camera_poss.keys():
            camera_pos = camera_poss[cam_num] #[camera_pos_x[cam_num], camera_pos_y[cam_num], camera_pos_z[cam_num]]
            cam_target = cam_targets[cam_num]
            add_camera(cam_temp_scene_xml_file, f'gen_cam_{cam_num}', camera_pos, cam_target, cam_num)

        e = MujocoEnv(cam_temp_scene_xml_file, 1, has_robot=False)
        e.sim.physics.forward()
        
        for _ in range(num_objects):
            for _ in range(4000):
                e.model.step()
        
        state = e.get_env_state().copy()
        
        cam_width = 640
        cam_height = 480

        valid_cameras = []
        camera_stats = dict()
        
        for cam_num in camera_poss.keys():
            cam_pix_left_ratio_d = dict()
            this_cam_stats = dict()
            discard_cam = False
            
            camera = Camera(physics=e.model, height=cam_height, width=cam_width, camera_id=cam_num)
            segs = camera.render(segmentation=True)[:,:,0] #(480, 640, 2)
            occluded_geom_id_to_seg_id = {camera.scene.geoms[geom_ind][3]: camera.scene.geoms[geom_ind][8] for geom_ind in range(camera.scene.geoms.shape[0])}
        
            for object_idx in object_idx_to_obj_info.keys():
                if discard_cam:
                    break
                target_id = e.model.model.name2id(f'gen_geom_object_{object_idx}_{scene_num}_0', "geom")
                segmentation = segs == occluded_geom_id_to_seg_id[target_id]
                
                # Move all other objects far away, except the table, so that we can capture
                # only one object in a scene.
                for move_obj_ind in original_obj_keys:
                    if move_obj_ind != object_idx:
                        move_object(e, move_obj_ind, [20, 20, move_obj_ind], [0,0,0,0])

                e.sim.physics.forward()

                unocc_target_id = e.model.model.name2id(f'gen_geom_object_{object_idx}_{scene_num}_0', "geom")
                unoccluded_camera = Camera(physics=e.model, height=cam_height, width=cam_width, camera_id=cam_num)
                unoccluded_segs = unoccluded_camera.render(segmentation=True)
                
                # Move other objects back onto table 
                e.set_env_state(state)
                e.sim.physics.forward()

                unoccluded_geom_id_to_seg_id = {unoccluded_camera.scene.geoms[geom_ind][3]: unoccluded_camera.scene.geoms[geom_ind][8] for geom_ind in range(unoccluded_camera.scene.geoms.shape[0])}
                unoccluded_segmentation = unoccluded_segs[:,:,0] == unoccluded_geom_id_to_seg_id[unocc_target_id]
                
                # If the object is not in the scene of this object 
                if np.argwhere(unoccluded_segmentation).shape[0] == 0:
                    this_cam_stats[object_idx] = [-1, 0]
                    continue
                
                segmentation = np.logical_and(segmentation, unoccluded_segmentation)
                pix_left_ratio = np.argwhere(segmentation).shape[0] / np.argwhere(unoccluded_segmentation).shape[0]
                
                if object_idx == cam_num_to_occlusion_target[cam_num]:
                    if pix_left_ratio < 0.4:
                        discard_cam = True
                        continue
                
                this_cam_stats[object_idx] = [pix_left_ratio, np.argwhere(unoccluded_segmentation).shape[0]]
                cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}_{object_idx}.png'), segmentation.astype(np.uint8))

            if discard_cam:
                continue
            
            rgb=e.model.render(height=cam_height, width=cam_width, camera_id=cam_num, depth=False, segmentation=False)
            cv2.imwrite(os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png'), rgb)

            # Depth image
            depth = e.model.render(height=cam_height, width=cam_width, camera_id=cam_num, depth=True, segmentation=False)
            depth = (depth*1000).astype(np.uint16)
            cv2.imwrite(os.path.join(scene_folder_path, f'depth_{(cam_num):05}.png'), depth)

            cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}.png'), segs)
            valid_cameras.append(cam_num)
            camera_stats[cam_num] = this_cam_stats

        
        # for k,v in camera_stats.items():
        #     print("cam_num: ", k)
        #     print("cam_num_to_occlusion_target: ", cam_num_to_occlusion_target[k])
        #     print(v)
        
        original_keys = list(camera_poss.keys())
        for cam_num in original_keys:
            if not cam_num in valid_cameras:
                del camera_poss[cam_num]
                del cam_targets[cam_num]
        
        object_descriptions = dict()
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
            object_description['camera_pos'] = camera_poss
            object_description['cam_targets'] = cam_targets
            object_description['table']={'mesh_filename':table_mesh_filename, \
                    'position': e.data.qpos.ravel()[0:3].copy(), \
                    'orientation': e.data.qpos.ravel()[3:7].copy(), \
                    'scale': table_size}
            object_description['cam_height'] = cam_height
            object_description['cam_width'] = cam_width

            cur_position = object_description['position']
            object_cam_d = dict()
            for cam_num in camera_poss.keys():
                if camera_stats[cam_num][object_idx][-1] == 0:
                    continue
                
                object_camera_info_i = dict()
                object_camera_info_i['pix_left_ratio'] = camera_stats[cam_num][object_idx][0]
                object_camera_info_i['total_pixel_in_scene'] = camera_stats[cam_num][object_idx][0] * camera_stats[cam_num][object_idx][1]
                camera = Camera(physics=e.model, height=480, width=640, camera_id=cam_num)
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

        for cam_num in camera_poss.keys():
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
        gen_data(acc_scene_num, selected_objects[scene_num], args.shapenet_filepath, args.shapenet_decomp_filepath, args.top_dir, args.train_or_test)

    # for scene_num in range(len(df)):
    #     acc_scene_num = scene_num + args.start_scene_idx
    #     sample = df.iloc[scene_num]
    #     sample_input = [(sample['synsetId'], sample['ShapeNetModelId'], sample['objId'])]
    #     gen_data(acc_scene_num, sample_input, args.shapenet_filepath, args.shapenet_decomp_filepath, args.top_dir, args.train_or_test)
    
        
        
