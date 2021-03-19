import json
import os
# os.environ["OMP_NUM_THREADS"]="1"
# os.environ["MUJOCO_GL"]="osmesa"
import numpy as np
import multiprocessing as mp

from ARM_erin.trajopt.envs.mujoco_env import MujocoEnv
import trimesh
import shutil
import random
import cv2
from pyquaternion import Quaternion
import pickle
from optparse import OptionParser
import traceback
import pybullet as p
from dm_control.mujoco.engine import Camera
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from trajopt.mujoco_utils import add_camera, add_objects
from scipy.spatial.transform import Rotation as R
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from simple_clutter_utils import *
from simple_clutter_2d_projection import *
from PIL import Image 

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

parser = OptionParser()
#path to shapenet dataset
parser.add_option("--shapenet_filepath", dest="shapenet_filepath", default='/media/xiaoyuz1/hdd5/xiaoyuz1/ShapeNetCore.v2')
#filepath to convex decompositions of shapenet objects. I posted this in the slack channel
parser.add_option("--shapenet_decomp_filepath", dest="shapenet_decomp_filepath", default='/media/xiaoyuz1/hdd5/xiaoyuz1/data/shapenet_conv_decmops')
#root project dir
parser.add_option("--top_dir", dest="top_dir", default='/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/')
#roo project dir+/inhand_datagen
parser.add_option("--json_file_path", dest="json_file_path", default='/media/xiaoyuz1/hdd5/xiaoyuz1/data/training_instances.json')
parser.add_option("--shape_categories_file_path", dest="shape_categories_file_path", default='/media/xiaoyuz1/hdd5/xiaoyuz1/data/taxonomy_tabletop_small_keys.txt')

parser.add_option("--train_or_test", dest="train_or_test", default='training_set')
parser.add_option("--num_scenes", dest="num_scenes", type="int", default=5)
parser.add_option("--num_threads", dest="num_threads", type="int", default=6)
# parser.add_option('--single_object', action='store_true')
(options, args) = parser.parse_args()

np.set_printoptions(precision=4, suppress=True)




def move_object(e, ind, pos, rot):
    all_poses=e.data.qpos.ravel().copy()
    all_vels=e.data.qvel.ravel().copy()
    
    all_poses[7+7*ind : 7+7*ind+3]=pos
    all_poses[7+7*ind+3 : 7+7*ind+7]=rot
    
    all_vels[6+6*ind : 6+6*ind+6] = 0
    e.set_state(all_poses, all_vels)

REGION_LIMIT = 1


#@profile
def gen_data(scene_num, selected_objects, shapenet_filepath, shapenet_decomp_filepath, json_file_path, shape_categories_file_path, top_dir, obj_pos_dist_std, train_or_test):
    
    '''
    scene_num:
    selected_objects: [(obj_cat_synset_id, obj_id)]
    hapenet_filepath:
    shapenet_decomp_filepath:
    json_file_path, shape_categories_file_path:
    top_dir:
    obj_pos_dist_std:
    train_or_test
    '''
    # np.random.seed(scene_num)
    thread_num=scene_num
    num_objects = len(selected_objects) 
    # Color of objects:
    all_colors_dict = mcolors.CSS4_COLORS #TABLEAU_COLORS #
    all_colors = [] # a list of rgb colors
    for name, color in all_colors_dict.items():
        all_colors.append(np.asarray(mcolors.to_rgb(color)))
    selected_colors = [all_colors[i] for i in np.random.choice(len(all_colors), num_objects+1, replace=False)]
    
    '''
    '''
    objects_xy = np.array([[0,0],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
    objects_xy = objects_xy*1.3

    temp = json.load(open(os.path.join(shapenet_filepath, 'taxonomy.json'))) 
    taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}
    synset_ids_in_dir = os.listdir(shapenet_filepath)
    synset_ids_in_dir.remove('taxonomy.json')
    # Dictionary mapping category name --> category synset id
    taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synset_ids_in_dir}

    training_tables_filename = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/training_shapenet_tables.json'
    valid_tables = json.load(open(training_tables_filename))
    # valid_tables = train_tables if train_or_test == 'training_set' else test_tables
    
    
    try:
        # Make temporary scene xml file
        scene_xml_file=os.path.join(top_dir, f'base_scene.xml')
        temp_scene_xml_file=os.path.join(top_dir, f'temp_data_gen_scene_{thread_num}.xml')
        shutil.copyfile(scene_xml_file, temp_scene_xml_file)

        scene_description={}

        '''
        Add table
        '''
        camera_distance = 5#num_objects * 3
        table_generated = False
        table_bounds = None
        table_xyz = None
        table_trans = None
        while not table_generated:
            # Choose table and table scale and add to sim
            table_id = valid_tables[1027] #valid_tables[np.random.randint(0, len(valid_tables))]
            table_mesh_filename = os.path.join(shapenet_filepath, f'04379243/{table_id}/models/model_normalized.obj')
            table_mesh=trimesh.load(table_mesh_filename, force='mesh')
            # Rotate table so that it appears upright in Mujoco
            scale_mat=np.eye(4)
            r = R.from_euler('x', 90, degrees=True)
            scale_mat[0:3,0:3] = r.as_matrix()
            table_mesh.apply_transform(scale_mat)
            table_bounds = table_mesh.bounds
            
            table_xyz_range = np.max(table_bounds[1,:] - table_bounds[0,:])
            table_size = (camera_distance*1.5)/table_xyz_range
            # Export table mesh as .stl file
            stl_table_mesh_filename=os.path.join(top_dir, f'assets/table_{thread_num}.stl')
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
            add_objects(temp_scene_xml_file, 'table', [stl_table_mesh_filename], table_xyz, table_size, table_color, table_orientation, thread_num, add_contacts=False)
            table_generated = True

            print("table_bounds: ", table_bounds)

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
        all_obj_bounds = []
        obj_trans = []

        all_obj_corners = []

        prev_polys = []

        
        object_position_region = None
        for object_idx in range(num_objects):
            obj_cat, obj_id = selected_objects[object_idx]
            obj_mesh_filename = os.path.join(shapenet_filepath,'{}/{}/models/model_normalized.obj'.format(obj_cat, obj_id))
            object_mesh = trimesh.load(obj_mesh_filename, force='mesh')
            # if obj_cat in ['02992529', '03624134', '02773838','04401088']:
                # object_mesh.show()
                
            '''
            Determine object rotation
            '''
            # 
            rot_vec, upright_mat  = determine_object_rotation(object_mesh)

            print("Before transformation: ", object_mesh.bounds)

            object_mesh.apply_transform(upright_mat)

            print("After transformation, before scaling: ", object_mesh.bounds)

            
            # object_mesh.show()
            # Store the upright object in .stl file in assets
            stl_obj_mesh_filename = os.path.join(top_dir, f'assets/model_normalized_{thread_num}_{object_idx}.stl')
            f = open(stl_obj_mesh_filename, "w+")
            f.close()
            object_mesh.export(stl_obj_mesh_filename)
            # Rotate object to face different directions
            object_rot = rot_vec 
            object_rot_add = [0,0,0] #[0,0,np.random.uniform(0,2*np.pi,1)[0]]
            
            all_obj_corners.append(object_mesh.scene().bounds_corners)
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
            # 
            object_bounds = object_bounds*object_size
            object_bottom = -object_bounds[0][2]
            
            print("After transformation, after scaling: ", object_mesh.bounds)

            
            '''
            Determine object position
            '''
            object_z = table_height + object_bottom + 0.005
            if object_idx == 0:
                left_x, right_x = object_bounds[:,0]
                down_y, up_y = object_bounds[:,1]
                
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
                # 
                object_x, object_y = generate_object_xy(object_position_region, prev_polys, obj_trans, 
                    obj_rotations, 
                    object_rot, 
                    object_bounds, 
                    obj_xyzs, 
                    all_obj_bounds, 
                    object_z)
            
            object_xyz = [object_x, object_y, object_z]
            # lower_left = [object_x+object_bounds[0,0], object_y+object_bounds[0,1]]
            # upper_right = [object_x+object_bounds[1,0], object_y+object_bounds[1,1]]
            
            r = R.from_euler('xyz', [0,0,object_rot[-1]], degrees=False) 
            trans = autolab_core.RigidTransform(rotation = r.as_matrix(), translation = np.asarray(object_xyz), from_frame='object_{}'.format(object_idx))

            corners4 = get_bound_2d_4corners(trans, object_bounds)
            prev_polys.append(Polygon(corners4))
            
            # Determine the maximum object height
            object_height = object_bounds[1][2] - object_bounds[0][2]            
            object_max_height = max(object_max_height, object_height)
            
            # Load objects from .stl file
            object_mesh=trimesh.load(stl_obj_mesh_filename)
            if object_mesh.faces.shape[0]>200000:
                print('Too many mesh faces!')
                continue

            obj_trans.append(trans)
            obj_xyzs.append(object_xyz)
            obj_rotations.append([0,0,object_rot[-1]])
            obj_scales.append(object_size)
            all_obj_bounds.append(object_bounds)
            obj_mesh_filenames.append(obj_mesh_filename)
            # 
            
            mesh_names = load_mesh_convex_parts(shapenet_decomp_filepath, obj_cat, obj_id, upright_mat)
            
            add_objects(temp_scene_xml_file, f'object_{object_idx}_{thread_num}', mesh_names, object_xyz, object_size, object_color, [0,0,0], thread_num, add_contacts=False)
        
        # 
        scene_folder_path = os.path.join(top_dir, f'{train_or_test}/scene_{scene_num:06}')

        if os.path.exists(scene_folder_path):
            shutil.rmtree(scene_folder_path)
        os.mkdir(scene_folder_path)

        layout_filename = os.path.join(scene_folder_path, 'layout.png')
        draw_boundary_points([table_trans] + obj_trans, [table_xyz] + obj_xyzs, [table_bounds] + all_obj_bounds, layout_filename)

        
        # cam_temp_scene_xml_file=os.path.join(top_dir, f'cam_temp_data_gen_scene_{thread_num}.xml')
        # shutil.copyfile(scene_xml_file, cam_temp_scene_xml_file)

        scene_xml_file=os.path.join(top_dir, f'base_scene.xml')
        cam_temp_scene_xml_file=os.path.join(top_dir, f'cam_temp_data_gen_scene_{thread_num}.xml')
        shutil.copyfile(scene_xml_file, cam_temp_scene_xml_file)

        add_objects(cam_temp_scene_xml_file, 'table', [stl_table_mesh_filename], table_xyz, table_size, table_color, table_orientation, thread_num, add_contacts=False)
        
        # scene_name, object_name, mesh_names, pos, size, color, rot, run_id, contact_geom_list=None, add_ind=-1, add_contacts=True
        for object_idx in range(num_objects):
            mesh_names = [os.path.join(top_dir, f'assets/model_normalized_{thread_num}_{object_idx}.stl')]
            add_objects(cam_temp_scene_xml_file, f'object_{object_idx}_{thread_num}', mesh_names, obj_xyzs[object_idx], obj_scales[
                        object_idx], selected_colors[object_idx+1], [0,0,0], thread_num, add_contacts=False)
        
        
        '''
        Generate camera position and target
        ''' 
        # Generate camera heights
        max_object_height = table_height + object_max_height
        camera_poss, cam_targets = get_camera_position(camera_distance, table_height, max_object_height, obj_xyzs)
        # camera_poss, cam_targets = get_fixed_camera_position(camera_distance, table_height+1, table_xyz)
        num_camera = len(camera_poss)

        print(camera_poss)

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
    
        for cam_num in range(num_camera):
            # 
            rgb=e.model.render(height=480, width=640, camera_id=cam_num, depth=False, segmentation=False)
            cv2.imwrite(os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png'), rgb)
            
            # Depth image
            depth = e.model.render(height=480, width=640, camera_id=cam_num, depth=True, segmentation=False)
            depth = (depth*1000).astype(np.uint16)
            cv2.imwrite(os.path.join(scene_folder_path, f'depth_{(cam_num):05}.png'), depth)
            
            camera = Camera(physics=e.model, height=480, width=640, camera_id=cam_num)
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

                target_id = e.model.model.name2id(f'gen_geom_object_{added_object_ind}_{thread_num}_0', "geom")
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
                unocc_target_id = e.model.model.name2id(f'gen_geom_object_{added_object_ind}_{thread_num}_0', "geom")
                unoccluded_camera = Camera(physics=e.model, height=480, width=640, camera_id=cam_num)
                unoccluded_segs = unoccluded_camera.render(segmentation=True)
                # Move other objects back onto table
                e.set_env_state(state)
                e.sim.physics.forward()

                unoccluded_geom_id_to_seg_id = {unoccluded_camera.scene.geoms[geom_ind][3]: unoccluded_camera.scene.geoms[geom_ind][8] for geom_ind in range(unoccluded_camera.scene.geoms.shape[0])}
                unoccluded_segs = np.concatenate((unoccluded_segs[:,:,0:1],unoccluded_segs[:,:,0:1],unoccluded_segs[:,:,0:1]), axis=2).astype(np.uint8)
                unoccluded_segmentation = unoccluded_segs[:,:,0]==unoccluded_geom_id_to_seg_id[unocc_target_id]
                # num_unoccluded_pix = np.argwhere(unoccluded_segmentation).shape[0]
                segmentation = np.logical_and(segmentation, unoccluded_segmentation)
                if np.argwhere(segmentation).shape[0] > 0:
                    cv2.imwrite(os.path.join(scene_folder_path, f'segmentation_{(cam_num):05}_{present_in_view_ind}.png'), segmentation.astype(np.uint8))
                present_in_view_ind += 1 
        
        # import pdb; pdb.set_trace()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        
        # for object_idx, obj_mesh_filename in enumerate(obj_mesh_filenames):
        #     cur_position = e.data.qpos.ravel()[7+7*object_idx:7+7*object_idx+3].copy()
        #     corners8 = get_bound_corners(obj_trans[object_idx], all_obj_bounds[object_idx])
        #     ax.scatter(corners8[:,0], corners8[:,1], corners8[:,2], marker='o')

        # plt.show()

        #Draw bounding boxes:
        for cam_num in range(num_camera):
            fig = plt.figure(figsize=(16, 8))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            camera = Camera(physics=e.model, height=480, width=640, camera_id=cam_num)
            P,camera_tf = get_camera_matrix(camera)
            vertical_img = Image.open(os.path.join(scene_folder_path, f'rgb_{(cam_num):05}.png'))
            
            # vertical_img = vertical_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            ax1.imshow(vertical_img)
            ax2.imshow(vertical_img)

            table_center_coord = project_2d(P,camera_tf, np.array(table_xyz).reshape(-1,3))

            corners8 = get_bound_corners(table_trans, table_bounds)
            table_bound_coord = project_2d(P, camera_tf, np.asarray(corners8).reshape(-1,3))
            
            # for ax in [ax1, ax2]:
            #     ax.scatter(640 -table_center_coord[:,0], table_center_coord[:,1],   marker=".", c='b', s=30)
            #     ax.scatter(640 -table_bound_coord[:,0], table_bound_coord[:,1],   marker=".", c='r', s=30)
            
            for object_idx in range(num_objects):

                cur_position = e.data.qpos.ravel()[7+7*object_idx:7+7*object_idx+3].copy()
                # cur_orientation = 
                corners8 = get_bound_corners(obj_trans[object_idx], all_obj_bounds[object_idx])

                
                
                pixel_coord = project_2d(P,camera_tf, np.array(cur_position.reshape(-1,3)))
                bbox_pixel_coord = project_2d(P,camera_tf, corners8.reshape(-1,3))
                
                
                # vertical_img = vertical_img.transpose(Image.FLIP_TOP_BOTTOM)
                ax2.scatter(640 - pixel_coord[:,0], pixel_coord[:,1],   marker=".", c='b', s=30)
                ax1.scatter(pixel_coord[:,0], pixel_coord[:,1],   marker=".", c='g', s=30)
                ax2.scatter(640 - bbox_pixel_coord[:,0], bbox_pixel_coord[:,1],   marker=".", c='r', s=30)
                ax1.scatter(bbox_pixel_coord[:,0], bbox_pixel_coord[:,1],   marker=".", c='y', s=30)

            
            plt.savefig(os.path.join(scene_folder_path, f'rgb_{(cam_num):05}_center.png'), bbox_inches='tight')
            
            plt.close()
        # import pdb; pdb.set_trace()

        scene_description['camera_pos'] = camera_poss
        scene_description['cam_targets'] = cam_targets
        scene_description['object_descriptions']=[]
        # scene_description['table_2']={'mesh_filename':table_mesh_filename, 'position': table_xyz, 'orientation': table_orientation, 'scale':table_size}
        scene_description['table']={'mesh_filename':table_mesh_filename, \
                'position': e.data.qpos.ravel()[0:3].copy(), \
                'orientation': e.data.qpos.ravel()[3:7].copy(), \
                'scale': table_size}
        object_descriptions = []
        for object_idx in range(num_objects):
            obj_mesh_filename = obj_mesh_filenames[object_idx]
        
            object_description={}
            object_description['mesh_filename'] = obj_mesh_filename
            object_description['position'] = e.data.qpos.ravel()[7+7*object_idx:7+7*object_idx+3].copy()
            object_description['orientation'] = e.data.qpos.ravel()[7+7*object_idx+3:7+7*object_idx+7].copy()
            object_description['scale'] = obj_scales[object_idx]

            # q = np.zeros(4)
            # q[0] = object_description['orientation'][1]
            # q[1] = object_description['orientation'][2]
            # q[2] = object_description['orientation'][3]
            # q[3] = object_description['orientation'][0]
            # r = R.from_quat(q)

            cur_position = object_description['position']
               
            
            for cam_num in range(num_camera):
                # view_matrix = p.computeViewMatrix(camera_poss[cam_num], cam_targets[cam_num], np.array([0,1,0]))
                # view_matrix = np.array(view_matrix).reshape(4,4, order='F')
                camera = Camera(physics=e.model, height=480, width=640, camera_id=cam_num)
                camera_matrix = get_camera_matrix(camera)
                pixel_coord = project_2d(camera_matrix, np.eye(4), np.array(cur_position.reshape(-1,3)))
                object_description["object_center_{}".format(cam_num)] = pixel_coord
                object_description["camera_matrix_{}".format(cam_num)] = camera_matrix
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
    
    print("\n\n################################# DEBUG /home/xiaoyuz1/Documents/ARM/inhand_datagen/cluttered_scene_datagen.py:")
    print("options.shapenet_filepath: ", options.shapenet_filepath)
    print("options.shapenet_decomp_filepath: ", options.shapenet_decomp_filepath)
    print("options.json_file_path: ", options.json_file_path)
    print("options.top_dir: ", options.top_dir)
    print("options.num_scenes: ", options.num_scenes)
    print("obj_pos_dist_std: ", 0.1)
    print("options.train_or_test: ", options.train_or_test)
    print("#################################\n\n")
    
    np.random.seed(0)
    #Color of objects:
    all_colors_dict = mcolors.CSS4_COLORS #TABLEAU_COLORS #
    all_colors = []
    for name, color in all_colors_dict.items():
        all_colors.append(np.asarray(mcolors.to_rgb(color)))
    
    # 

    # Dictionary from object category names to object ids in the category
    shapenet_models = json.load(open(options.json_file_path))
    shapenet_models_test = json.load(open('/media/xiaoyuz1/hdd5/xiaoyuz1/data/training_instances.json'))
    
    temp = json.load(open(os.path.join(options.shapenet_filepath, 'taxonomy.json'))) 
    taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}
    synset_ids_in_dir = os.listdir(options.shapenet_filepath)
    synset_ids_in_dir.remove('taxonomy.json')
    taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synset_ids_in_dir}
    
    table_top_categories = []
    with open(options.shape_categories_file_path) as shape_categories_file:
        for line in shape_categories_file:
            if line.strip() == '':
                continue
            table_top_categories.append(line.strip())
    
    # List of (category_name, object_id) tuples in the training set
    objects_cat_id = []
    objects_cat_start_id = []
    start_idx = 0
    for cat_name in shapenet_models:
        if not cat_name in table_top_categories:
            continue
        print(cat_name, start_idx)
        objects_cat_start_id.append((start_idx, len(shapenet_models[cat_name])))
        start_idx += len(shapenet_models[cat_name])
        
        for obj_id in shapenet_models[cat_name]:
            objects_cat_id.append((cat_name, obj_id))
    
    for cat_name in shapenet_models_test:
        if not cat_name in table_top_categories:
            continue
        print(cat_name, start_idx)
        objects_cat_start_id.append((start_idx, len(shapenet_models[cat_name])))
        start_idx += len(shapenet_models[cat_name])
        
        for obj_id in shapenet_models[cat_name]:
            objects_cat_id.append((cat_name, obj_id))
    
    # 
    '''
    '''
    # test_selected_indices = [i+1 for i,_ in objects_cat_start_id]
    num_objects = 2
    selected_object_indices = np.random.randint(0, len(objects_cat_id), (options.num_scenes, num_objects))
    # selected_object_indices[:,0] = np.random.randint(1308, 1646, options.num_scenes)
    # selected_object_indices[:,1] = np.random.randint(0, 66, options.num_scenes)
    # selected_object_indices[:,2] = np.random.randint(2314, 3185, options.num_scenes)
    # selected_object_indices[:,2] = np.random.randint(3185, 3849, options.num_scenes)

    selected_objects = []
    for selected_indices in selected_object_indices:
        selected_objects_i = []
        for idx in selected_indices:
            selected_objects_i.append((taxonomy_dict[objects_cat_id[idx][0]] , objects_cat_id[idx][1]))
        selected_objects.append(selected_objects_i)

    for scene_num in range(options.num_scenes):
        gen_data(scene_num, selected_objects[scene_num], options.shapenet_filepath, options.shapenet_decomp_filepath, options.json_file_path, options.shape_categories_file_path, options.top_dir, 0.1, options.train_or_test)


    # num_processes=options.num_threads
    # pool = mp.Pool(processes=num_processes, maxtasksperchild=1)
    # num_objects = 1
    # selected_object_indices = np.arange(len(objects_cat_id)).reshape(-1,1)
    
    # for scene_num in range(options.num_scenes):
    #     abortable_func = partial(abortable_worker, gen_data, timeout=600)
    #     pool.apply_async(abortable_func, args=(scene_num, selected_object_indices[scene_num], options.shapenet_filepath, options.shapenet_decomp_filepath, options.json_file_path, options.shape_categories_file_path, options.top_dir, 0.1, options.train_or_test))
    
    # pool.close()
    # pool.join()
    
        
        
