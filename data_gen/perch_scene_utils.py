import numpy as np
import os
import copy
import json 
import math
import cv2
from PIL import Image
import pickle
from datetime import datetime
import shutil
import trimesh
from scipy.spatial.transform import Rotation as R, rotation

from mujoco_env import MujocoEnv


from data_gen_args import *
from simple_clutter_utils import *
from mujoco_object_utils import MujocoNonTable, MujocoTable
from perch_coco_utils import ImageAnnotation

class SceneCamera(object):
    def __init__(self, location, target, cam_num):
        self.pos  = location
        self.target = target 
        self.cam_num = cam_num
        self.cam_name = f'gen_cam_{cam_num}'

        self.rgb_image_annotations = []

    def add_camera_to_file(self, xml_fname):
        add_camera(xml_fname, self.cam_name, self.pos, self.target, self.cam_num)

    def get_o3d_intrinsics(self):
        cx = self.width / 2
        cy = self.height / 2

        return o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, cx, cy)
    
    def project_2d(self, pt_3d):
        '''
        pt_3d: (N,3)
        '''
        N = len(pt_3d)
        pt_3d_pad = np.append(pt_3d.T, np.ones(N).astype(
            'int').reshape(1, -1), axis=0)  # (4,N)
        pt_3d_camera = self.world_frame_to_camera_frame_mat @ pt_3d_pad  # (4,N)
        assert np.all(np.abs(pt_3d_camera[-1] - 1) < 1e-6)
        pixel_coord = self.intrinsics @ (pt_3d_camera[:-1, :])
        mult = pixel_coord[-1, :]
        pixel_coord = pixel_coord / pixel_coord[-1, :]
        pixel_coord = pixel_coord[:2, :]  # (2,N)
        pt_2d = pixel_coord.astype('int').T
        return pt_2d
    
    def set_camera_info_with_mujoco_env(self, mujoco_env, cam_height, cam_width):
        camera = Camera(physics=mujoco_env.model, height=cam_height, width=cam_width, camera_id=self.cam_num)
        camera_id = camera._render_camera.fixedcamid
        pos = camera._physics.data.cam_xpos[camera_id]
        cam_rot_matrix = camera._physics.data.cam_xmat[camera_id].reshape(3, 3)
        fov = camera._physics.model.cam_fovy[camera_id]

        # # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * camera.height / 2.0
        f = 0.5 * camera.height / math.tan(fov * np.pi / 360)
        assert np.abs(f - focal_scaling) < 1e-3

        P = np.array(((focal_scaling, 0, camera.width / 2),
                    (0, focal_scaling, camera.height / 2), (0, 0, 1)))
        camera_tf = autolab_core.RigidTransform(rotation=cam_rot_matrix, translation=np.asarray(
            pos), from_frame='camera_{}'.format(camera_id), to_frame='world')

        assert np.all(np.abs(camera_tf.matrix @ np.array([0, 0, 0, 1]).reshape(
            4, -1) - np.array([[pos[0], pos[1], pos[2], 1]]).reshape(4, -1)) < 1e-5)
        
        self.height = camera.height
        self.width = camera.width
        self.pos = pos #world frame
        self.intrinsics = P
        self.rot_quat = R.from_matrix(cam_rot_matrix).as_quat() #(xyzw) #world frame
        self.fov = fov
        self.fx = focal_scaling
        self.fy = focal_scaling
        self.camera_frame_to_world_frame_mat = camera_tf.matrix
        self.world_frame_to_camera_frame_mat = camera_tf.inverse().matrix

    def get_camera_annotation(self):
        return {
            'intrinsics_matrix': get_json_cleaned_matrix(self.intrinsics, type='float'),
            'pos' : get_json_cleaned_matrix(self.pos, type='float'), 
            'rot_quat' : get_json_cleaned_matrix(self.rot_quat, type='float'),
            'camera_frame_to_world_frame_mat' : get_json_cleaned_matrix(self.camera_frame_to_world_frame_mat, type='float'),
            'world_frame_to_camera_frame_mat' : get_json_cleaned_matrix(self.world_frame_to_camera_frame_mat, type='float'),
        }
    
    def get_annotations_dict(self):
        coco_annos = []
        for coco_anno in self.rgb_image_annotations:
            coco_annos.append(coco_anno.output_dict())
        return coco_annos

class PerchScene(object):
    
    def __init__(self, scene_num, selected_objects, args):
        self.args = args
        self.scene_num = scene_num
        self.selected_objects = selected_objects
        self.shapenet_filepath, top_dir, self.train_or_test = args.shapenet_filepath, args.top_dir, args.train_or_test
        self.num_lights = args.num_lights
        self.num_objects = len(selected_objects) 
        self.selected_colors = [ALL_COLORS[i] for i in np.random.choice(len(ALL_COLORS), self.num_objects+1, replace=False)]
        self.depth_factor = args.depth_factor
        self.width = args.width
        self.height = args.height 
        self.scene_save_dir = args.scene_save_dir
        self.num_meshes_before_object = 1#args.use_walls + 1
        if not os.path.exists(args.scene_save_dir):
            os.mkdir(args.scene_save_dir)
        
        output_save_dir = os.path.join(args.scene_save_dir, self.train_or_test)
        if not os.path.exists(output_save_dir):
            os.mkdir(output_save_dir)
        self.output_save_dir = output_save_dir
        
        scene_folder_path = os.path.join(args.scene_save_dir, f'{self.train_or_test}/scene_{scene_num:06}')
        if os.path.exists(scene_folder_path):
            shutil.rmtree(scene_folder_path)
        os.mkdir(scene_folder_path)
        self.scene_folder_path = scene_folder_path

        asset_folder_path = os.path.join(scene_folder_path, 'assets')
        if not os.path.exists(asset_folder_path):
            os.mkdir(asset_folder_path)
        self.asset_folder_path = asset_folder_path

        scene_xml_file = os.path.join(top_dir, f'base_scene.xml')
        xml_folder = os.path.join(args.scene_save_dir, f'{self.train_or_test}_xml')
        if not os.path.exists(xml_folder):
            os.mkdir(xml_folder)
        self.xml_folder = xml_folder
        
        self.convex_decomp_xml_file = os.path.join(xml_folder, f'convex_decomp_data_gen_scene_{scene_num}.xml')
        shutil.copyfile(scene_xml_file, self.convex_decomp_xml_file)
        self.cam_temp_scene_xml_file = os.path.join(xml_folder, f'cam_temp_data_gen_scene_{scene_num}.xml')
        shutil.copyfile(scene_xml_file, self.cam_temp_scene_xml_file)

        self.create_table()
        self.object_info_dict = dict()
        for object_idx in range(self.num_objects):
            self.create_object(object_idx)
        self.total_camera_num = 0
        self.camera_info_dict = dict()

    def create_table(self):
        table_id = '97b3dfb3af4487b2b7d2794d2db4b0e7'
        table_mesh_fname = os.path.join(self.shapenet_filepath, f'04379243/{table_id}/models/model_normalized.obj')
        transformed_mesh_fname = os.path.join(self.scene_folder_path, f'assets/table_{self.scene_num}.stl')
        self.table_info = MujocoTable(
            object_name='table',
            shapenet_file_name=table_mesh_fname,
            transformed_mesh_fname=transformed_mesh_fname,
            color=self.selected_colors[0],
            num_objects_in_scene=self.num_objects,
            table_size=self.args.table_size,
        )

    def create_object(self, object_idx):
        synset_category, shapenet_model_id = self.selected_objects[object_idx][0], self.selected_objects[object_idx][2]
        mesh_fname = os.path.join(
            self.shapenet_filepath,
            '0{}/{}/models/model_normalized.obj'.format(synset_category, shapenet_model_id),
        )
        transformed_fname = os.path.join(
            self.scene_folder_path,
            f'assets/model_normalized_{self.scene_num}_{object_idx}.stl'
        )
        object_info = MujocoNonTable(
            object_name=f'object_{object_idx}_{self.scene_num}',
            shapenet_file_name=mesh_fname,
            transformed_mesh_fname=transformed_fname,
            color=self.selected_colors[object_idx+1],
            num_objects_in_scene=self.num_objects,
            object_idx=object_idx,
            shapenet_convex_decomp_dir=self.args.shapenet_convex_decomp_dir,
            scale=self.selected_objects[object_idx][4],
            half_or_whole=self.selected_objects[object_idx][5],
            perch_rot_angle=self.selected_objects[object_idx][6],
        )
        self.object_info_dict[object_idx] = object_info
    
    def add_lights_to_scene(self, xml_fname):
        light_position, light_direction = get_light_pos_and_dir(self.num_lights)
        ambients = np.random.uniform(0,0.05,self.num_lights*3).reshape(-1,3)
        diffuses = np.random.uniform(0.25,0.35,self.num_lights*3).reshape(-1,3)
        speculars = np.random.uniform(0.25,0.35,self.num_lights*3).reshape(-1,3)
       
        for light_id in range(self.num_lights):
            add_light(
                xml_fname,
                directional=True,
                ambient=ambients[light_id],
                diffuse=diffuses[light_id],
                specular=speculars[light_id],
                castshadow=False,
                pos=light_position[light_id],
                dir=light_direction[light_id],
                name=f'light{light_id}'
            )
    
    def generate_cameras_fibonacci_sphere_grid(self, center_x=0, center_y=0, camera_z_above_table=1.5, num_angles=20, radius=5, upper_limit=0.5):
        # locations = []
        # phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
        # for i in range(num_angles*2):
        #     y = 1 - (i / float(num_angles*2 - 1)) * 2  # y goes from 1 to 0
        #     y_radius = math.sqrt(1 - y * y) * radius  # radius at y
        #     theta = phi * i  # golden angle increment
        #     x = math.cos(theta) * y_radius
        #     z = math.sin(theta) * y_radius
        #     if z <= 0:
        #         continue
        #     locations.append((x, y, z))
        num_pts = num_angles * 4
        indices = np.arange(0, num_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        locations = np.asarray([x, y, z]).T
        locations = locations[locations[:,2] >= 0]
        locations = locations[locations[:,2] <= upper_limit]
        locations = locations * radius
        locations[:,0] += center_x
        locations[:,1] += center_y
        locations[:,2] += self.table_info.height + camera_z_above_table
        targets = np.asarray([[center_x,center_y,self.table_info.height]] * len(locations))
        return locations, targets
        
    
    def generate_cameras_around(self, center_x=0, center_y=0, camera_z_above_table=1.5, num_angles=20, radius=5):
        quad = (2.0*math.pi) / num_angles
        normal_thetas = [i*quad for i in range(num_angles)]
        table_width = np.max(self.table_info.object_mesh.bounds[1,:2] - self.table_info.object_mesh.bounds[0,:2])
        
        locations = []
        targets = []
        for theta in normal_thetas:
            cam_x = np.cos(theta) * radius + center_x
            cam_y = np.sin(theta) * radius + center_y
            location = [cam_x, cam_y, self.table_info.height + camera_z_above_table]
            target = [center_x,center_y,self.table_info.height]

            locations.append(location)
            targets.append(target)

        return locations, targets
    
    def add_cameras(
        self, 
        sphere_sampling=True,
        center_x=0, 
        center_y=0, 
        camera_z_above_table=1.5, 
        num_angles=20, 
        radius=1,
        upper_limit=0.5,
    ):
        if sphere_sampling:
            locations, targets = self.generate_cameras_fibonacci_sphere_grid(
                center_x, 
                center_y, 
                camera_z_above_table, 
                num_angles, 
                radius,
                upper_limit,
            )
        else:
            locations, targets = self.generate_cameras_around(
                center_x, 
                center_y, 
                camera_z_above_table, 
                num_angles, 
                radius,
            )
        for location, target in zip(locations, targets):
            new_camera = SceneCamera(location, target, self.total_camera_num)
            new_camera.rgb_save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'rgb_{(self.total_camera_num):05}.png')
            new_camera.depth_save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'depth_{(self.total_camera_num):05}.png')
            self.camera_info_dict[self.total_camera_num] = new_camera
            self.total_camera_num += 1
    
    def generate_default_add_position(self, object_idx, distance_away=50):
        theta = ((2.0*math.pi) / self.num_objects) * object_idx
        start_x = np.cos(theta) * distance_away
        start_y = np.sin(theta) * distance_away
        start_z = object_idx + 1
        return [start_x, start_y, start_z]
    
    def create_walls_far_away(self, xml_fname):
        for wall_idx in range(4):
            pos = self.generate_default_add_position(wall_idx, distance_away=80)
            wall_mesh_file = os.path.join(self.scene_folder_path, f'assets/wall_{wall_idx}.stl')
            
            mujoco_add_dict = {
                'object_name': f'wall_{wall_idx}_{self.scene_num}',
                'mesh_names': [wall_mesh_file],
                'pos': pos,
                'size': [1,1,1],
                'color': [0,0,0],
                'quat': [0,0,0,0],
                'mocap' : True,
            }
            add_objects(
                xml_fname,
                mujoco_add_dict,
            )
    
    def create_walls_on_table(self, xml_fname):
        outer_pts = self.table_info.table_top_corners
        unit = self.object_info_dict[0].canonical_size
        low_x, low_y = -unit, -unit
        upper_x, upper_y = unit, unit
        inner_pts = [
            [low_x, low_y, self.table_info.height],
            [upper_x, low_y, self.table_info.height],
            [low_x, upper_y, self.table_info.height],
            [upper_x, upper_y, self.table_info.height],
        ]
        wall_infos = create_walls(inner_pts, outer_pts, bottom_height=self.table_info.height)
        for wall_idx, v in wall_infos.items():
            pos, quat, ls = v
            lx,ly,lz = ls
            wall_mesh = trimesh.creation.box((lx,ly,lz))
            wall_mesh_file = os.path.join(self.scene_folder_path, f'assets/wall_{wall_idx}.stl')
            f = open(wall_mesh_file, "w+")
            f.close()
            wall_mesh.export(wall_mesh_file)
            
            mujoco_add_dict = {
                'object_name': f'wall_{wall_idx}_{self.scene_num}',
                'mesh_names': [wall_mesh_file],
                'pos': pos,
                'size': [1,1,1],
                'color': [0,0,0,0],
                'quat': quat,
                'mocap' : True,
            }
            # mujoco_add_dict = {
            #     'object_name': f'wall_{wall_idx}_{self.scene_num}',
            #     'mesh_names': [wall_mesh_file],
            #     'site' : True,
            #     'site_sizes' : [lx/2,ly/2,0.1],
            #     'type' : 'box',
            #     'pos': pos,
            #     'size': [1,1,1],
            #     'color': [0,0,0,0.3],
            #     'quat': quat,
            # }
            add_objects(
                xml_fname,
                mujoco_add_dict,
            )

    def create_walls_around_table(self, xml_fname):
        table_corners = self.table_info.table_top_corners
        table_height= self.table_info.height
        
        comb = [[0,1,1,-1],[2,3,1,1],[0,2,0,-1],[1,3,0,1]]
        r = R.from_euler('xyz', [0, (1/2)*np.pi, 0], degrees=False)
        quat = r.as_quat()
        quat = quat_xyzw_to_wxyz(quat)
        for wall_idx,(idx1,idx2,change_idx,change_sign) in enumerate(comb):
            if wall_idx >= self.num_meshes_before_object-1:
                break
            # Create wall mesh
            if wall_idx < 2:
                wall_length = np.linalg.norm(table_corners[idx1] - table_corners[idx2]) * 0.9
            else:
                wall_length = np.linalg.norm(table_corners[idx1] - table_corners[idx2]) * 5
            wall_mesh = trimesh.creation.box((table_height+1, wall_length, wall_length))
            wall_mesh_file = os.path.join(self.scene_folder_path, f'assets/wall_{wall_idx}.stl')
            f = open(wall_mesh_file, "w+")
            f.close()
            wall_mesh.export(wall_mesh_file)
            
            middle = np.mean(table_corners[[idx1,idx2]], axis=0)
            middle[change_idx] += change_sign * (wall_length * 0.5 + 0.005)
            mujoco_add_dict = {
                'object_name': f'wall_{wall_idx}_{self.scene_num}',
                'mesh_names': [wall_mesh_file],
                'pos': middle,
                'size': [1,1,1],
                'color': [0,0,0],
                'quat': quat,
            }
            add_objects(
                xml_fname,
                mujoco_add_dict,
            )
    
    def create_convex_decomposed_scene(self):
        _ = self.create_walls_on_table(self.convex_decomp_xml_file) 
        self.add_lights_to_scene(self.convex_decomp_xml_file)
        add_objects(self.convex_decomp_xml_file, self.table_info.get_mujoco_add_dict(), run_id=None, material_name=None)
    
        for object_idx in range(self.num_objects):
            object_info = self.object_info_dict[object_idx]
            convex_decomp_mesh_fnames = object_info.load_decomposed_mesh()
            object_add_dict = object_info.get_mujoco_add_dict()
            
            start_position = self.generate_default_add_position(object_idx)
            object_add_dict.update({
                'mesh_names' : convex_decomp_mesh_fnames,
                'pos' : start_position,
            })
            add_objects(
                self.convex_decomp_xml_file,
                object_add_dict,
            )
        
        convex_decomp_mujoco_env = MujocoEnv(self.convex_decomp_xml_file, 1, has_robot=False)
        error_object_idx = None
        try:
            for object_idx in range(self.num_objects):
                object_info = self.object_info_dict[object_idx]
                convex_decomp_mesh_height = -object_info.convex_decomp_mesh.bounds[0,2]
                pos_x, pos_y = object_info.pos_x, object_info.pos_y
                
                moved_location = [pos_x, pos_y, self.table_info.height+convex_decomp_mesh_height+0.01]
                move_object(convex_decomp_mujoco_env, object_idx, moved_location, quat_xyzw_to_wxyz(object_info.rot.as_quat()), num_ind_prev=self.num_meshes_before_object)
                error_object_idx = object_idx
                for _ in range(4000):
                    convex_decomp_mujoco_env.model.step()
        except:
            print('\nERROR try to move object: ', self.scene_num, " at index: ", error_object_idx)
            error_all_poses = convex_decomp_mujoco_env.data.qpos.ravel().copy().reshape(-1,7)
            for object_idx in range(self.num_objects):
                if object_idx > error_object_idx:
                    break
                object_info = self.object_info_dict[object_idx]
                print(object_idx, [object_info.pos_x, object_info.pos_y, self.table_info.height+convex_decomp_mesh_height+0.05], object_info.rot.as_quat(), error_all_poses[object_idx+1])
            print("\n")
            raise
        
        self.convex_decomp_mujoco_env_state = convex_decomp_mujoco_env.get_env_state().copy()
        all_current_poses = convex_decomp_mujoco_env.data.qpos.ravel().copy().reshape(-1,7) 
        
        objects_current_positions = all_current_poses[self.num_meshes_before_object:][:,:3]
        for _,_,cur_z in objects_current_positions:
            assert cur_z > self.table_info.height
        table_current_position = all_current_poses[self.num_meshes_before_object-1][:3].reshape(-1,)
        table_current_position[2] = self.table_info.height
        # object_closest_to_table_center = np.argmin(np.linalg.norm(objects_current_positions - table_current_position, axis=1)) 
        # new_cam_center_x, new_cam_center_y,_ = objects_current_positions[object_closest_to_table_center]
        # new_cam_center_x, new_cam_center_y,_ = np.mean(objects_current_positions, axis=0)
        radius = np.max(np.linalg.norm(self.table_info.table_top_corners - table_current_position, axis=1))
        new_cam_center_x, new_cam_center_y = table_current_position[:2]
        self.add_cameras(
            sphere_sampling=True,
            center_x=new_cam_center_x, 
            center_y=new_cam_center_y, 
            camera_z_above_table=self.object_info_dict[0].canonical_size, 
            num_angles=20, 
            radius=self.object_info_dict[0].canonical_size*2,
        )
        self.add_cameras(
            sphere_sampling=True,
            center_x=new_cam_center_x, 
            center_y=new_cam_center_y, 
            camera_z_above_table=0.1, 
            num_angles=100, 
            radius=self.object_info_dict[0].canonical_size*2,
            upper_limit = 0.1,
        )
        ### DEBUG PERPOSE CAMERAs, bird-eye view to better see what is going on
        # level_height = 2
        # locations = [
        #     [0,0,level_height+self.table_info.height],
        #     [1,0,level_height+self.table_info.height],
        #     [0,1,level_height+self.table_info.height],
        #     [level_height,level_height,level_height+self.table_info.height],
        # ]
        # targets = [[0,0,0]] * len(locations)
        # for location, target in zip(locations, targets):
        #     new_camera = SceneCamera(location, target, self.total_camera_num)
        #     new_camera.rgb_save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'rgb_{(self.total_camera_num):05}.png')
        #     new_camera.depth_save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'depth_{(self.total_camera_num):05}.png')
        #     self.camera_info_dict[self.total_camera_num] = new_camera
        #     self.total_camera_num += 1
    
    def create_camera_scene(self):
        _ = self.create_walls_on_table(self.cam_temp_scene_xml_file)
        self.add_lights_to_scene(self.cam_temp_scene_xml_file)
        add_objects(self.cam_temp_scene_xml_file, self.table_info.get_mujoco_add_dict(), run_id=None, material_name=None)

        for object_idx in range(self.num_objects):
            object_info = self.object_info_dict[object_idx]

            object_add_dict = object_info.get_mujoco_add_dict()
            start_position = self.generate_default_add_position(object_idx)
            object_add_dict.update({
                'pos' : start_position,
            })
            add_objects(
                self.cam_temp_scene_xml_file,
                object_add_dict,
            )
        
        # mujoco_env = MujocoEnv(self.cam_temp_scene_xml_file, 1, has_robot=False)
        # mujoco_env.set_env_state(self.convex_decomp_mujoco_env_state)
        # mujoco_env.sim.physics.forward()
        # all_current_poses = mujoco_env.data.qpos.ravel().copy().reshape(-1,7)  

        # # all_bboxs = []
        # for object_idx in self.object_info_dict.keys(): 
        #     model_name = f'{self.train_or_test}_scene_{self.scene_num}_object_{object_idx}'           
        #     object_info = self.object_info_dict[object_idx]
        #     object_mesh = object_info.save_correct_size_model(os.path.join(args.scene_save_dir, 'models'), model_name)

        #     mesh_bounds = object_mesh.bounds
        #     current_pose = all_current_poses[object_idx+1]
            
        #     final_position = current_pose[:3]
        #     final_quat = quat_wxyz_to_xyzw(current_pose[3:])
        #     final_rot_obj = R.from_quat(final_quat)
        #     mesh_bounds, mesh_bbox, _ = get_corners(mesh_bounds, current_pose[:3], final_rot_obj, f'object_{object_idx}')
        #     assert np.all(np.abs(np.linalg.norm(mesh_bbox - final_position, axis=1) - np.linalg.norm(mesh_bounds, axis=1)) < 1e-5)

        #     locations, targets = self.generate_cameras_around(
        #         center_x=final_position[0], 
        #         center_y=final_position[1],
        #         num_angles=8, 
        #         radius=1,
        #     )
        #     for location, target in zip(locations, targets):
        #         new_camera = SceneCamera(location, target, self.total_camera_num)
        #         new_camera.rgb_save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'rgb_{(self.total_camera_num):05}.png')
        #         new_camera.depth_save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'depth_{(self.total_camera_num):05}.png')
        #         self.camera_info_dict[self.total_camera_num] = new_camera
        #         self.total_camera_num += 1
            
        #     self.object_info_dict[object_idx].bbox = mesh_bbox
        #     self.object_info_dict[object_idx].final_position = final_position
        #     self.object_info_dict[object_idx].final_quat = final_quat

        
        # # Generate around the center of all the bounding boxes
        # all_bboxs = np.hstack(all_bboxs)
        # avg_pt = np.mean(all_bboxs, axis=0)
        # locations, targets = self.generate_cameras_around(center_x=avg_pt[0], center_y=avg_pt[1])
        # for cam_num, (location, target) in enumerate(zip(locations, targets)):
        #     self.camera_info_dict[cam_num].pos = location
        #     self.camera_info_dict[cam_num].target = target

        for cam_num in self.camera_info_dict.keys():
            new_camera = self.camera_info_dict[cam_num]
            new_camera.add_camera_to_file(self.cam_temp_scene_xml_file)
        
        mujoco_env = MujocoEnv(self.cam_temp_scene_xml_file, 1, has_robot=False)
        mujoco_env.set_env_state(self.convex_decomp_mujoco_env_state)
        mujoco_env.sim.physics.forward()

        ## Calculate object bounding box in the scene and save in the mujoco object 
        all_current_poses = mujoco_env.data.qpos.ravel().copy().reshape(-1,7)  
        # if self.num_meshes_before_object > 1:
        #     for wall_idx in range(self.num_meshes_before_object-1):
        #         wall_location = self.generate_default_add_position(wall_idx, distance_away=80)
        #         move_object(mujoco_env, wall_idx, wall_location, [0,0,0,0], num_ind_prev=0)
                
        ## Save the scaled object, which is used for perch in the models directory
        for object_idx in self.object_info_dict.keys():
            model_name = f'{self.train_or_test}_scene_{self.scene_num}_object_{object_idx}'
            
            object_info = self.object_info_dict[object_idx]
            object_mesh = object_info.save_correct_size_model(os.path.join(args.scene_save_dir, 'models'), model_name)

            mesh_bounds = object_mesh.bounds
            current_pose = all_current_poses[object_idx + self.num_meshes_before_object]
            
            final_position = current_pose[:3]
            final_quat = quat_wxyz_to_xyzw(current_pose[3:])
            final_rot_obj = R.from_quat(final_quat)
            mesh_bounds, mesh_bbox, _ = get_corners(mesh_bounds, current_pose[:3], final_rot_obj, f'object_{object_idx}')
            assert np.all(np.abs(np.linalg.norm(mesh_bbox - final_position, axis=1) - np.linalg.norm(mesh_bounds, axis=1)) < 1e-5)
            self.object_info_dict[object_idx].bbox = mesh_bbox
            self.object_info_dict[object_idx].final_position = final_position
            self.object_info_dict[object_idx].final_quat = final_quat
        
        ## Save camera information, like intrinsics 
        for cam_num in self.camera_info_dict.keys():
            self.camera_info_dict[cam_num].set_camera_info_with_mujoco_env(mujoco_env, self.height, self.width)

        for cam_num in self.camera_info_dict.keys():
            fname = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'rgb_{(cam_num):05}.png')
            self.render_rgb(mujoco_env, self.height, self.width, cam_num, save_path=fname)
            self.render_depth(mujoco_env, self.height, self.width, cam_num)
        
        for cam_num in self.camera_info_dict.keys():
            self.render_object_segmentation_and_create_rgb_annotation(mujoco_env, self.height, self.width, cam_num)
        
        self.output_camera_annotations()

    
    def render_rgb(self, mujoco_env, cam_height, cam_width, cam_num, save=True, save_path=None):
        # self.camera_info_dict[cam_num]
        rgb = mujoco_env.model.render(
            height=cam_height, 
            width=cam_width, 
            camera_id=cam_num, 
            depth=False, 
            segmentation=False
        )
        if save:
            if save_path is None:
                save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'rgb_{(cam_num):05}.png')
            cv2.imwrite(os.path.join(self.scene_save_dir, save_path), rgb)
            self.camera_info_dict[cam_num].rgb_save_path = save_path
        return rgb
    
    def render_depth(self, mujoco_env, cam_height, cam_width, cam_num, save=True, save_path=None):
        depth = mujoco_env.model.render(
            height=cam_height, 
            width=cam_width, 
            camera_id=cam_num, 
            depth=True, 
            segmentation=False
        )        
        # depth value image overflow
        depth_scaled = depth * self.depth_factor
        depth_scaled[depth_scaled > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
        if save:     
            if save_path is None:
                save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'depth_{(cam_num):05}.png')
            # depth_img = Image.fromarray(depth_scaled.astype(np.int32))
            # depth_img.save(os.path.join(self.scene_save_dir, save_path))
            cv2.imwrite(os.path.join(self.scene_save_dir, save_path), depth_scaled.astype(np.uint16))
            self.camera_info_dict[cam_num].depth_save_path = save_path
        return depth
    
    def render_object_segmentation_and_create_rgb_annotation(self, mujoco_env, cam_height, cam_width, cam_num):
        '''
        Process segmentation mask for each object in front the given camera, since the segmentation is 
        needed for PERCH.
        Also save the annotation for each object, which involves 
        - object center in the image
        - object bounding box in the image (?)
        - segmentation mask file path
        - object model file path
        '''
        camera_info = self.camera_info_dict[cam_num]
        camera = Camera(physics=mujoco_env.model, height=cam_height, width=cam_width, camera_id=cam_num)
        original_state = mujoco_env.get_env_state().copy()
        whole_scene_segmentation = camera.render(segmentation=True)[:, :, 0]
        camera_scene_geoms = camera.scene.geoms
        occluded_geom_id_to_seg_id = {
            camera_scene_geoms[geom_ind][3] : camera_scene_geoms[geom_ind][8] for geom_ind in range(camera_scene_geoms.shape[0])
        }

        existed_object_ids, object_pixel_count = np.unique(whole_scene_segmentation, return_counts=True)
        existed_object_idxs = []
        
        for object_idx in self.object_info_dict.keys():
            geom_id = mujoco_env.model.model.name2id(f'gen_geom_object_{object_idx}_{self.scene_num}_0', "geom")
            seg_id = occluded_geom_id_to_seg_id[geom_id]
            # 
            if not seg_id in existed_object_ids:
                continue
            # This object is in this rgb image, so it can have an annotation
            existed_object_idxs.append(object_idx)
            object_segmentation = whole_scene_segmentation == seg_id

            # Move all other objects far away, except the table, so that we can capture
            # only one object in a scene.
            for move_obj_ind in self.object_info_dict.keys():
                if move_obj_ind != object_idx:
                    start_position = self.generate_default_add_position(move_obj_ind)
                    move_object(mujoco_env, move_obj_ind, start_position, [0, 0, 0, 0], num_ind_prev=self.num_meshes_before_object)
            mujoco_env.sim.physics.forward()

            
            save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'segmentation_{(cam_num):05}_{object_idx}.png')

            unocc_target_id = mujoco_env.model.model.name2id(f'gen_geom_object_{object_idx}_{self.scene_num}_0', "geom")
            unoccluded_camera = Camera(physics=mujoco_env.model, height=cam_height, width=cam_width, camera_id=cam_num)
            unoccluded_segs = unoccluded_camera.render(segmentation=True)

            # Move other objects back onto table
            mujoco_env.set_env_state(original_state)
            mujoco_env.sim.physics.forward()

            unoccluded_geom_id_to_seg_id = {
                unoccluded_camera.scene.geoms[geom_ind][3]: unoccluded_camera.scene.geoms[geom_ind][8] for geom_ind in range(unoccluded_camera.scene.geoms.shape[0])
            }
            unoccluded_segmentation = unoccluded_segs[:, :,0] == unoccluded_geom_id_to_seg_id[unocc_target_id]
            unoccluded_pixel_num = np.argwhere(unoccluded_segmentation).shape[0]
            
            if unoccluded_pixel_num == 0:
                # return -1, 0, None
                continue 

            segmentation = np.logical_and(object_segmentation, unoccluded_segmentation)
            number_pixels = np.argwhere(segmentation).shape[0]
            
            if number_pixels == 0:
                continue
            pix_left_ratio = number_pixels / unoccluded_pixel_num
            cv2.imwrite(os.path.join(self.scene_save_dir, save_path), segmentation.astype(np.uint8))
            
            bbox_2d = self.camera_info_dict[cam_num].project_2d(self.object_info_dict[object_idx].bbox)
            bbox_2d[:,0] = self.width - bbox_2d[:,0]
            # xmin, ymin, width, height
            xmin, ymin = np.min(bbox_2d, axis=0)
            xmax, ymax = np.max(bbox_2d, axis=0)
            
            if not (0 <= xmin and xmin <= self.width) or not(0 <= xmax and xmax <= self.width):
                continue 
            
            if not (0 <= ymin and ymin <= self.height) or not(0 <= ymax and ymax <= self.height):
                continue
            bbox_ann = [xmin, ymin, xmax-xmin, ymax-ymin]
            model_annotation = self.object_info_dict[object_idx].get_model_annotation()
            center_3d = self.object_info_dict[object_idx].final_position
            center_2d = self.camera_info_dict[cam_num].project_2d(center_3d.reshape(-1,3)).reshape(-1,)
            center_2d[0] = self.width - center_2d[0]
            
            # # DEBUG purpose
            cv2_image = cv2.imread(os.path.join(self.scene_save_dir, self.camera_info_dict[cam_num].rgb_save_path))
            # cv2_image = cv2.rectangle(cv2_image, (int(xmin), int(ymax)), (int(xmax),int(ymin)), (0,255,0), 3)
            rmin, rmax, cmin, cmax = bbox_ann[1], bbox_ann[1] + bbox_ann[3], bbox_ann[0], bbox_ann[0] + bbox_ann[2]
            perch_box = [cmin, rmin, cmax, rmax]
            cv2_image = cv2.rectangle(cv2_image, (int(cmin), int(rmin)), (int(cmax),int(rmax)), (0,255,0), 3)
            centroid_x, centroid_y = [(cmin+cmax)/2, (rmin+rmax)/2]
            cv2_image = cv2.circle(cv2_image, (int(centroid_x), int(centroid_y)), radius=0, color=(255, 0, 0), thickness=5)
            # # Draw the 8 bounding box corners in image
            for x,y in bbox_2d:
                cv2_image = cv2.circle(cv2_image, (x,y), radius=0, color=(0, 0, 255), thickness=3)
            cv2_image = cv2.circle(cv2_image, (center_2d[0],center_2d[1]), radius=0, color=(255, 255, 255), thickness=5)
            cv2.imwrite(os.path.join(self.scene_folder_path, f'rgb_with_bbox_{(cam_num):05}_{object_idx}.png') ,cv2_image)
            # #

            # object_location_in_camera_frame = 
            object_annotation = ImageAnnotation(
                center = list(center_2d),
                bbox = bbox_ann,
                rgb_file_path = self.camera_info_dict[cam_num].rgb_save_path,
                mask_file_path = save_path,
                model_name = self.object_info_dict[object_idx].model_name,
                percentage_not_occluded = pix_left_ratio,
                number_pixels = number_pixels,
                cam_intrinsics = camera_info.intrinsics, 
                object_location = model_annotation['position'], 
                object_quat = model_annotation['quat'], 
                model_annotation = model_annotation,
                camera_annotation= self.camera_info_dict[cam_num].get_camera_annotation(),
                width=self.width,
                height=self.height,
            )
            self.camera_info_dict[cam_num].rgb_image_annotations.append(object_annotation)
    
    def output_camera_annotations(self):
        all_annotations = []
        for cam_num in self.camera_info_dict.keys():
            camera_info = self.camera_info_dict[cam_num]
            anno_dict_list = camera_info.get_annotations_dict()
            all_annotations += anno_dict_list
        
        # json_string = json.dumps(all_annotations)
        # annotation_fname = os.path.join(self.scene_folder_path, 'annotations.json')
        # json_file = open(annotation_fname, "w+")
        # json_file.write(json_string)
        # json_file.close()

        images = []
        image_id_acc = -1
        image_id_dict = dict()

        categories = []
        category_id_acc = -1
        category_id_dict = {}

        annotation_id_acc = 0
        annotations = []

        for annotation in all_annotations:
            image_id = image_id_dict.get(annotation['rgb_file_path'], -1)
            if image_id < 0:
                image_id_acc += 1
                image_id_dict[annotation['rgb_file_path']] = image_id_acc
                image_anno = {
                    "id" : image_id_acc,
                    "file_name" : annotation['rgb_file_path'],
                    "width" : annotation['width'],
                    "height" : annotation['height'],
                    "date_captured" : "2021",
                    "license" : 1,
                    "coco_url": "", 
                    "flickr_url": "",
                }
                images.append(image_anno)
                image_id = image_id_acc

            model_name = annotation['model_name']
            category_id = category_id_dict.get(model_name, -1)

            if category_id < 0:
                category_id_acc += 1
                category_id_dict[model_name] = category_id_acc
                model_category = {
                    "id": category_id_acc,
                    "name" : model_name,
                    "supercategory": "shape",
                }
                categories.append(model_category)
                category_id = category_id_acc

            annotation.update({
                "id" : annotation_id_acc,
                "image_id" : image_id,
                "category_id" : category_id,
            })
            annotation_id_acc += 1
            annotations.append(annotation)
        
        camera_intrinsic_settings = {
            'fx' : self.camera_info_dict[0].fx,
            'fy' : self.camera_info_dict[0].fy,
            'cx' : self.camera_info_dict[0].width / 2,
            'cy' : self.camera_info_dict[0].height / 2,
        }

        camera_intrinsic_matrix = get_json_cleaned_matrix(self.camera_info_dict[0].intrinsics, type='float')
        
        coco_json_dict = {
            "info": {
                "description": "Example Dataset", 
                "url": "https://github.com/waspinator/pycococreator", 
                "version": "0.1.0", 
                "year": 2018, 
                "contributor": "waspinator", 
                "date_created": "2020-03-17 15:05:18.264435"
                }, 
            "licenses": [
                {"id": 1, "name": 
                "Attribution-NonCommercial-ShareAlike License", 
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }
            ],
            "categories" : categories,
            "camera_intrinsic_settings": camera_intrinsic_settings, 
            "camera_intrinsic_matrix" : camera_intrinsic_matrix,
            # "fixed_transforms": None,
            "images" : images,
            "annotations" : annotations,
        }

        json_string = json.dumps(coco_json_dict)
        annotation_fname = os.path.join(self.scene_folder_path, 'annotations.json')
        json_file = open(annotation_fname, "w+")
        json_file.write(json_string)
        json_file.close()

    def render_whole_scene_segmentation(self, mujoco_env, cam_height, cam_width, cam_num, save=True, save_path=None):
        camera = Camera(physics=mujoco_env.model, height=cam_height, width=cam_width, camera_id=cam_num)
        segmentation = camera.render(segmentation=True)[:,:,0] #(480, 640, 2)

        if save:
            if save_path is None:
                save_path = os.path.join(f'{self.train_or_test}/scene_{self.scene_num:06}', f'segmentation_{(cam_num):05}.png')
            cv2.imwrite(os.path.join(self.scene_save_dir, save_path), segmentation)
        return segmentation
    
    def render_all_object_segmentation(self, mujoco_env, cam_height, cam_width, cam_num, object_idx, save=True, save_path=None):
        raise 
    
    def render_point_cloud(self, depth, cam_num, mask=None, save=True, save_path=None):
        o3d_intrinsics = self.camera_info_dict[cam_num].get_o3d_intrinsics()
        o3d_depth = o3d.geometry.Image(depth)
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsics)
        return None 
    
    # def output_json_information(self):
    #     date_captured = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
    # def generate_cameras_old_style(self):
    #     num_rotates = 1
    #     step_deg = 10
    #     # for object_i in object_idx_to_obj_info.keys(): 
    #     for object_i, object_info in self.object_info_dict.items():
    #         xyz1 = object_info.final_position
    #         # height1 = object_idx_to_obj_info[object_i]['object_height']
            
    #         pairwise_diff = xys - xyz1[:2].reshape((1,2))
    #         dist = np.linalg.norm(pairwise_diff, axis=1)
    #         max_dist = np.max(dist)
            
    #         # for object_j in object_idx_to_obj_info.keys():  
    #         for object_j in new_obj_keys:
    #             if object_i == object_j:
    #                 continue 
    #             xyz2 = object_idx_to_obj_info[object_j]['xyz']
    #             height2 = object_idx_to_obj_info[object_j]['object_height']
                
    #             for sign in [1,-1]:
    #                 keep_rotating = True
    #                 for deg_i in range(num_rotates):
    #                     if not keep_rotating:
    #                         break
    #                     low_deg = deg_i*step_deg*sign
    #                     high_deg = (deg_i+1)*step_deg*sign
    #                     cam_xyz1,cam_target,cam_xyz2 = get_camera_position_occluded_one_cam(table_height, xyz1, xyz2,height1,height2,max_dist,[low_deg,high_deg])
                        
    #                     for cam_i,cam_xyz in enumerate([cam_xyz1, cam_xyz2]):
    #                         if cam_i == 2:
    #                             if not np.random.binomial(n=1, p=(1/5), size=1)[0]:
    #                                 continue

    #                         add_camera(cam_temp_scene_xml_file, f'gen_cam_{cam_num}', cam_xyz, cam_target, cam_num)

    #                         cam_xyzs[cam_num] = cam_xyz
    #                         cam_targets[cam_num] = cam_target
    #                         cam_num_to_occlusion_target[cam_num] = object_i
    #                         cam_num += 1
