import numpy as np
import os
import copy
import math
import cv2
import pickle
import shutil
import trimesh
from scipy.spatial.transform import Rotation as R, rotation

from mujoco_env import MujocoEnv


from data_gen_args import *
from simple_clutter_utils import *

class SceneCamera(object):
    def __init__(self, scene_name, location, target, cam_num):
        self.scene_name = scene_name
        self.location  = location
        self.target = target 
        self.cam_num = cam_num
        self.cam_name = f'gen_cam_{cam_num}'
    
    def add_camera_to_file(self):
        add_camera(self.scene_name, self.cam_name, self.location, self.target, self.cam_num)

    # def get_cemera_info(self, mujoco_env, cam_height, cam_width):
    #     camera = Camera(physics=mujoco_env.model, height=cam_height, width=cam_width, camera_id=self.cam_num)
    #     camera_id, camera_tf,camera_res = get_camera_matrix(camera)
    #     rgb = mujoco_env.model.render(height=cam_height, width=cam_width, camera_id=self.cam_num, depth=False, segmentation=False)
        

class MujocoObject(object):
    def __init__(
        self,
        scene_name,
        object_name,
        shapenet_file_name,
        transformed_table_mesh_fname,
        color,
        num_objects_in_scene,
    ):
        self.scene_name = scene_name
        self.object_name = object_name
        self.shapenet_file_name = shapenet_file_name
        self.transformed_table_mesh_fname = transformed_table_mesh_fname
        self.color = color
        self.num_objects_in_scene = num_objects_in_scene

        object_mesh = trimesh.load(shapenet_file_name, force='mesh')
        r = R.from_euler('xyz', [(1/2)*np.pi, 0, 0], degrees=False)
        upright_mat = np.eye(4)
        upright_mat[0:3, 0:3] = r.as_matrix()
        object_mesh.apply_transform(upright_mat) 
        f = open(transformed_table_mesh_fname, "w+")
        f.close()
        object_mesh.export(transformed_table_mesh_fname)
        self.object_mesh = object_mesh

        # Chaning rotation changes the object mesh
        self.rot = np.zeros(3)
        self.pos = np.zeros(3)
        # Chaning size scale changes the object mesh
        self.size = np.ones(3)


    def set_object_scale(self, scale = None):
        pass
    
    def set_object_rot(self, rot):
        r = R.from_euler('xyz', rot, degrees=False)
        rotation_mat = np.eye(4)
        rotation_mat[0:3, 0:3] = r.as_matrix()
        self.object_mesh.apply(rotation_mat)
        self.rot = rot 

    def get_mujoco_add_dict(self):
        return {
            'scene_name' : self.scene_name,
            'object_name': self.object_name,
            'mesh_names': [self.transformed_table_mesh_fname],
            'pos': self.pos,
            'size': self.size,
            'color': self.color,
            'rot': self.rot,
        }

class MujocoTable(MujocoObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_object_scale()

    def set_object_rot(self, rot):
        # If change rotation, then need to change table_frame_to_world and other stuff
        # Not for now
        raise

    def set_object_scale(self, scale = None):
        table_bounds = self.object_mesh.bounds
        table_xyz_range = np.min(table_bounds[1, :2] - table_bounds[0, :2])
        table_size = 2*(self.num_objects_in_scene + 2)/table_xyz_range
        scale_vec = np.array([table_size]*3)
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale_vec
        self.object_mesh.apply_transform(scale_matrix)
        table_bounds = self.object_mesh.bounds

        self.size = scale_vec
        self.pos = np.array([0.0, 0.0, -self.object_mesh.bounds[0][2]])

        self.height = self.object_mesh.bounds[1][2] - self.object_mesh.bounds[0][2]
        table_r = R.from_euler('xyz', self.rot, degrees=False)
        self.table_frame_to_world = autolab_core.RigidTransform(
            rotation=table_r.as_matrix(),
            translation=self.pos,
            from_frame='table',
            to_frame='world',
        )
        table_corners = bounds_xyz_to_corners(table_bounds)
        table_top_corners = table_corners[table_corners[:,2] == table_bounds[1,2]]
        self.table_top_corners = transform_3d_frame(self.table_frame_to_world.matrix, table_top_corners)
        # table_min_x, table_min_y,_ = np.min(table_top_corners, axis=0)
        # table_max_x, table_max_y,_ = np.max(table_top_corners, axis=0)


class MujocoNonTable(MujocoObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_object_scale()
    
    def reset_size(self):
        # Normalize the size by scaling the longest edge
        self.object_mesh = trimesh.load(self.transformed_table_mesh_fname, force='mesh')
        object_bounds = self.object_mesh.bounds
        range_max = np.max(object_bounds[1] - object_bounds[0])
        object_size = 1 / range_max
        normalize_vec = [object_size, object_size, object_size]
        normalize_matrix = np.eye(4)
        normalize_matrix[:3, :3] *= normalize_vec
        self.object_mesh.apply_transform(normalize_matrix)
        self.size = np.asarray(normalize_vec)
    
    def set_object_scale(self, scale = None):
        '''
        scale : np.array (3,)
        '''
        self.reset_size()
        if scale is None:
            scale = [np.random.choice([0.5, 0.75, 1.0])] * 3
            
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale
        self.object_mesh.apply_transform(scale_matrix)
        self.size *= np.array(scale)


class PerchScene(object):
    
    def __init__(self, scene_num, selected_objects, args):
        self.scene_num = scene_num
        self.selected_objects = selected_objects
        self.shapenet_filepath, top_dir, self.train_or_test = args.shapenet_filepath, args.top_dir, args.train_or_test
        self.num_lights = args.num_lights
        self.num_objects = len(selected_objects) 
        self.selected_colors = [ALL_COLORS[i] for i in np.random.choice(len(ALL_COLORS), self.num_objects+1, replace=False)]
        self.depth_factor = args.depth_factor

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
        
        self.cam_temp_scene_xml_file = os.path.join(xml_folder, f'cam_temp_data_gen_scene_{scene_num}.xml')
        shutil.copyfile(scene_xml_file, self.cam_temp_scene_xml_file)

        self.add_table_to_scene()

        self.object_info_dict = dict()
        for object_idx in range(self.num_objects):
            synset_category, shapenet_model_id = selected_objects[object_idx][0], selected_objects[object_idx][2]
            mesh_fname = os.path.join(
                self.shapenet_filepath,
                '0{}/{}/models/model_normalized.obj'.format(synset_category, shapenet_model_id),
            )
            transformed_fname = os.path.join(
                scene_folder_path,
                f'assets/model_normalized_{scene_num}_{object_idx}.stl'
            )
            
            object_info = MujocoNonTable(
                scene_name=self.cam_temp_scene_xml_file,
                object_name=f'object_{object_idx}_{scene_num}',
                shapenet_file_name=mesh_fname,
                transformed_table_mesh_fname=transformed_fname,
                color=self.selected_colors[object_idx+1],
                num_objects_in_scene=self.num_objects,
            )
            self.object_info_dict[object_idx] = object_info

        
        self.table_min_x, self.table_min_y,_ = np.min(self.table_info.table_top_corners, axis=0)
        self.table_max_x, self.table_max_y,_ = np.max(self.table_info.table_top_corners, axis=0)
        self.add_lights_to_scene()

        self.total_camera_num = 0
        self.camera_info_dict = dict()

    def add_table_to_scene(self):
        table_id = '97b3dfb3af4487b2b7d2794d2db4b0e7'
        table_mesh_fname = os.path.join(self.shapenet_filepath, f'04379243/{table_id}/models/model_normalized.obj')
        transformed_table_mesh_fname = os.path.join(self.scene_folder_path, f'assets/table_{self.scene_num}.stl')
        self.table_info = MujocoTable(
            scene_name=self.cam_temp_scene_xml_file,
            object_name='table',
            shapenet_file_name=table_mesh_fname,
            transformed_table_mesh_fname=transformed_table_mesh_fname,
            color=self.selected_colors[0],
            num_objects_in_scene=self.num_objects,
        )
        add_objects(self.table_info.get_mujoco_add_dict(), run_id=None, material_name=None)

    def add_objects_to_scene(self):
        pass 
    
    def add_lights_to_scene(self):
        light_position, light_direction = get_light_pos_and_dir(self.num_lights)
        ambients = np.random.uniform(0,0.05,self.num_lights*3).reshape(-1,3)
        diffuses = np.random.uniform(0.25,0.35,self.num_lights*3).reshape(-1,3)
        speculars = np.random.uniform(0.25,0.35,self.num_lights*3).reshape(-1,3)
       
        for light_id in range(self.num_lights):
            add_light(
                self.cam_temp_scene_xml_file,
                directional=True,
                ambient=ambients[light_id],
                diffuse=diffuses[light_id],
                specular=speculars[light_id],
                castshadow=False,
                pos=light_position[light_id],
                dir=light_direction[light_id],
                name=f'light{light_id}'
            )
    
    def add_camera_to_scene(self, location, target):
        new_camera = SceneCamera(self.cam_temp_scene_xml_file, location, target, self.total_camera_num)
        new_camera.add_camera_to_file()
        self.camera_info_dict[self.total_camera_num] = new_camera
        self.total_camera_num += 1
    
    def render_rgb(self, cam_height, cam_width, cam_num, save=True):
        # self.camera_info_dict[cam_num]
        rgb = self.mujoco_env.model.render(
            height=cam_height, 
            width=cam_width, 
            camera_id=cam_num, 
            depth=False, 
            segmentation=False
        )
        if save:
            cv2.imwrite(os.path.join(self.scene_folder_path, f'rgb_{(cam_num):05}.png'), rgb)
        return rgb
    
    def render_depth(self, cam_height, cam_width, cam_num, save=True):
        depth = self.mujoco_env.model.render(
            height=cam_height, 
            width=cam_width, 
            camera_id=cam_num, 
            depth=True, 
            segmentation=False
        )
        depth_scaled = (depth*self.depth_factor).astype(np.uint16) #(height, width)   
        if save:     
            cv2.imwrite(os.path.join(self.scene_folder_path, f'depth_{(cam_num):05}.png'), depth_scaled)
        return depth
    
    def render_whole_scene_segmentation(self, cam_height, cam_width, cam_num, save=True):
        camera = Camera(physics=self.mujoco_env.model, height=cam_height, width=cam_width, camera_id=cam_num)
        segmentation = camera.render(segmentation=True)[:,:,0] #(480, 640, 2)
        if save:
            cv2.imwrite(os.path.join(self.scene_folder_path, f'segmentation_{(cam_num):05}.png'), segmentation)
        return segmentation
    
    def create_env(self):
        self.mujoco_env = MujocoEnv(self.cam_temp_scene_xml_file, 1, has_robot=False)
        self.mujoco_env.sim.physics.forward()
        
        for _ in range(self.num_objects):
            for _ in range(4000):
                self.mujoco_env.model.step()
        all_poses = self.mujoco_env.data.qpos.ravel().copy().reshape(-1,7)
        return all_poses

class PerchScene1(PerchScene):
    def __init__(self, scene_num, args):
        # 3152c7a0e8ee4356314eed4e88b74a21
        selected_objects = [
            ('2880940',2,'95ac294f47fd7d87e0b49f27ced29e3',7),
            ('2946921',3,'d44cec47dbdead7ca46192d8b30882',8),
        ]
        super().__init__(scene_num, selected_objects, args)
    
    def add_objects_to_scene(self):
        bowl_xyz = np.zeros(3)
        bowl_width_limit = None
        for object_idx in range(self.num_objects):
            mujoco_object = self.object_info_dict[object_idx]
            if object_idx == 0:
                obj_bound = self.object_info_dict[object_idx].object_mesh.bounds
                
                object_bottom = -obj_bound[0][2]
                bowl_xyz[2] = self.table_info.height + object_bottom + 0.002
                bowl_xyz[0] = np.random.uniform(self.table_min_x+2, self.table_max_x-2)
                bowl_xyz[1] = np.random.uniform(self.table_min_y+2, self.table_max_y-2)
                mujoco_object.pos = bowl_xyz
                mujoco_object.rot = np.zeros(3)
                scale = [np.random.choice([0.75, 0.85, 1.0])] * 3
                mujoco_object.set_object_scale(scale=scale)
                bowl_bounds = mujoco_object.object_mesh.bounds
                bowl_width_limit = np.min(bowl_bounds[1,:2]-bowl_bounds[0,:2])
                
            if object_idx == 1:
                position = copy.deepcopy(bowl_xyz)
                position[2] += 0.003
                mujoco_object.pos = [10, 10, 1]
                mujoco_object.rot = np.random.uniform(0, 2*np.pi, 3)

                mujoco_object.reset_size()
                bounds = mujoco_object.object_mesh.bounds
                x_range, y_range, z_range = bounds[1] - bounds[0]
                scale = [bowl_width_limit / x_range, bowl_width_limit / y_range, bowl_width_limit / z_range]
                mujoco_object.set_object_scale(scale=scale)
                
            
            self.object_info_dict[object_idx] = mujoco_object
        
        for object_idx in range(self.num_objects):
            add_objects(self.object_info_dict[object_idx].get_mujoco_add_dict(), run_id=None, material_name=None)

        # Add camera around the scene
        num_angles = 8
        quad = (2.0*math.pi) / num_angles
        normal_thetas = [i*quad for i in range(num_angles)]
        # [np.random.uniform(i*quad, (i+1.0)*quad, 1)[0] for i in range(num_angles)]
        for theta in normal_thetas:
            cam_x = np.cos(theta) * 4 + self.object_info_dict[0].pos[0]
            cam_y = np.sin(theta) * 4 + self.object_info_dict[0].pos[1]
            location = [cam_x, cam_y, self.table_info.height+1.5]
            target = self.object_info_dict[0].pos
            self.add_camera_to_scene(location, target)
        
        all_poses = self.create_env()
        bowl_position = all_poses[1][:3]
        can_position = copy.deepcopy(bowl_position)
        can_position[2] += 0.008
        import pdb; pdb.set_trace()
        rot_x,rot_y,rot_z,rot_w = R.from_rotvec(self.object_info_dict[1].rot).as_quat()
        new_rot = [rot_w, rot_x, rot_y, rot_z]
        move_object(self.mujoco_env, 1, can_position, new_rot)
        for cam_num in self.camera_info_dict.keys():
            self.render_rgb(480, 640, cam_num)

class PerchSceneCompleteRandom(PerchScene):
    def __init__(self, scene_num, selected_objects, args):
        super().__init__(scene_num, selected_objects, args)
    
    def add_objects_to_scene(self):
        for object_idx in range(self.num_objects):
            mujoco_object = self.object_info_dict[object_idx]
            object_bottom = -1.0 * self.object_info_dict[object_idx].object_mesh.bounds[0,2]
            position = [
                np.random.uniform(self.table_min_x+1, self.table_max_x-1),
                np.random.uniform(self.table_min_y+1, self.table_max_y-1),
                self.table_info.height + object_bottom + 0.002
            ]
            mujoco_object.pos = np.asrray(position)
            mujoco_object.rot = np.random.uniform(0, 2*np.pi, 3)
            self.object_info_dict[object_idx] = mujoco_object
        
        for object_idx in range(self.num_objects):
            add_objects(self.object_info_dict[object_idx].get_mujoco_add_dict(), run_id=None, material_name=None)
