import numpy as np
import os
import copy
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
from mujoco_object_utils import MujocoObject, MujocoNonTable, MujocoTable

class SceneCamera(object):
    def __init__(self, location, target, cam_num):
        self.pos  = location
        self.target = target 
        self.cam_num = cam_num
        self.cam_name = f'gen_cam_{cam_num}'
    
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
        # cam_rot_matrix = self.rot_quat.as_matrix()
        # camera_tf = autolab_core.RigidTransform(rotation=cam_rot_matrix, translation=np.asarray(
        #     self.pos), from_frame='camera_{}'.format(self.cam_num), to_frame='world')
        # world_to_camera_tf_mat = camera_tf.inverse().matrix
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
    
    def get_camera_info_dict(self):
        return {
            'intrinsics': self.intrinsics,
            'pos' : self.pos, 
            'rot' : self.rot_quat,
        }

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

        # First create the scene with convex decomposed parts
        self.create_convex_decomposed_scene()


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
        )
        self.object_info_dict[object_idx] = object_info
    
    
    def create_convex_decomposed_scene(self):
        self.add_lights_to_scene(self.convex_decomp_xml_file)
        add_objects(self.convex_decomp_xml_file, self.table_info.get_mujoco_add_dict(), run_id=None, material_name=None)
        for object_idx in range(self.num_objects):
            object_info = self.object_info_dict[object_idx]
            convex_decomp_mesh_fnames = object_info.load_decomposed_mesh()
            object_add_dict = object_info.get_mujoco_add_dict()
            object_add_dict.update({
                'mesh_names' : convex_decomp_mesh_fnames,
                'pos' : [50,50,object_idx],
            })
            add_objects(
                self.convex_decomp_xml_file,
                object_add_dict,
            )
        
        num_angles = 8
        quad = (2.0*math.pi) / num_angles
        normal_thetas = [i*quad for i in range(num_angles)]
        table_width = np.max(self.table_info.object_mesh.bounds[1,:2] - self.table_info.object_mesh.bounds[0,:2])
        
        for theta in normal_thetas:
            cam_x = np.cos(theta) * table_width #+ self.object_info_dict[0].pos[0]
            cam_y = np.sin(theta) * table_width #+ self.object_info_dict[0].pos[1]
            location = [cam_x, cam_y, self.table_info.height + 1.5]
            target = [0,0,self.table_info.height]
            new_camera = SceneCamera(location, target, self.total_camera_num)
            new_camera.add_camera_to_file(self.convex_decomp_xml_file)
            self.camera_info_dict[self.total_camera_num] = new_camera
            self.total_camera_num += 1
        
        convex_decomp_mujoco_env = MujocoEnv(self.convex_decomp_xml_file, 1, has_robot=False)
        for object_idx in range(self.num_objects):
            object_info = self.object_info_dict[object_idx]
            
            convex_decomp_mesh_height = -object_info.convex_decomp_mesh.bounds[0,2]
            move_object(convex_decomp_mujoco_env, object_idx, [object_info.pos_x, object_info.pos_y, self.table_info.height+convex_decomp_mesh_height+0.1], rotvec_to_mujoco_quat(object_info.rot))
            for _ in range(2000):
                convex_decomp_mujoco_env.model.step()
        
        for cam_num in self.camera_info_dict.keys():
            self.camera_info_dict[cam_num].set_camera_info_with_mujoco_env(convex_decomp_mujoco_env, self.height, self.width)

        for cam_num in self.camera_info_dict.keys():
            fname = os.path.join(self.scene_folder_path, f'rgb_beforehand_{(cam_num):05}.png')
            self.render_rgb(convex_decomp_mujoco_env, self.height, self.width, cam_num, save_path=fname)

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
    
    def add_camera_to_scene(self, xml_fname, location, target):
        new_camera = SceneCamera(location, target, self.total_camera_num)
        new_camera.add_camera_to_file(xml_fname)
        self.camera_info_dict[self.total_camera_num] = new_camera
        self.total_camera_num += 1
    
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
                cv2.imwrite(os.path.join(self.scene_folder_path, f'rgb_{(cam_num):05}.png'), rgb)
            else:
                cv2.imwrite(save_path, rgb)
        return rgb
    
    def render_depth(self, mujoco_env, cam_height, cam_width, cam_num, save=True, save_path=None):
        depth = mujoco_env.model.render(
            height=cam_height, 
            width=cam_width, 
            camera_id=cam_num, 
            depth=True, 
            segmentation=False
        )
        depth_scaled = (depth*self.depth_factor).astype(np.int32) #(height, width)   
        
        if save:     
            # if save_path is None:
            #     cv2.imwrite(os.path.join(self.scene_folder_path, f'depth_{(cam_num):05}.png'), depth_scaled)
            # else:
            #     cv2.imwrite(save_path, depth_scaled)
            if save_path is None:
                save_path = os.path.join(self.scene_folder_path, f'depth_{(cam_num):05}.png')
            depth_img = Image.fromarray(depth_scaled)
            depth_img.save(save_path)
        return depth
    
    def render_whole_scene_segmentation(self, mujoco_env, cam_height, cam_width, cam_num, save=True, save_path=None):
        camera = Camera(physics=mujoco_env.model, height=cam_height, width=cam_width, camera_id=cam_num)
        segmentation = camera.render(segmentation=True)[:,:,0] #(480, 640, 2)
        if save:
            cv2.imwrite(os.path.join(self.scene_folder_path, f'segmentation_{(cam_num):05}.png'), segmentation)
        return segmentation
    
    def render_object_segmentation(self, mujoco_env, cam_height, cam_width, cam_num, object_idx, save=True, save_path=None):
        camera = Camera(physics=mujoco_env.model, height=cam_height, width=cam_width, camera_id=cam_num)
        original_state = self.mujoco_env.get_env_state().copy()
        whole_scene_segmentation = camera.render(segmentation=True)[:, :, 0]
        camera_scene_geoms = camera.scene.geoms
        occluded_geom_id_to_seg_id = {
            camera_scene_geoms[geom_ind][3] : camera_scene_geoms[geom_ind][8] for geom_ind in range(camera_scene_geoms.shape[0])
        }

        target_id = self.mujoco_env.model.model.name2id(f'gen_geom_object_{object_idx}_{self.scene_num}_0', "geom")
        object_segmentation = whole_scene_segmentation == occluded_geom_id_to_seg_id[target_id]

        # Move all other objects far away, except the table, so that we can capture
        # only one object in a scene.
        for move_obj_ind in self.object_info_dict.keys():
            if move_obj_ind != object_idx:
                move_object(self.mujoco_env, move_obj_ind, [40, 40, move_obj_ind], [0, 0, 0, 0])

        self.mujoco_env.sim.physics.forward()

        unocc_target_id = self.mujoco_env.model.model.name2id(f'gen_geom_object_{object_idx}_{self.scene_num}_0', "geom")
        unoccluded_camera = Camera(physics=self.mujoco_env.model, height=cam_height, width=cam_width, camera_id=cam_num)
        unoccluded_segs = unoccluded_camera.render(segmentation=True)

        # Move other objects back onto table
        self.mujoco_env.set_env_state(original_state)
        self.mujoco_env.sim.physics.forward()

        unoccluded_geom_id_to_seg_id = {
            unoccluded_camera.scene.geoms[geom_ind][3]: unoccluded_camera.scene.geoms[geom_ind][8] for geom_ind in range(unoccluded_camera.scene.geoms.shape[0])
        }
        unoccluded_segmentation = unoccluded_segs[:, :,0] == unoccluded_geom_id_to_seg_id[unocc_target_id]

        unoccluded_pixel_num = np.argwhere(unoccluded_segmentation).shape[0]
        if unoccluded_pixel_num == 0:
            return -1, 0, None 

        segmentation = np.logical_and(object_segmentation, unoccluded_segmentation)
        pix_left_ratio = np.argwhere(segmentation).shape[0] / unoccluded_pixel_num

        if save:
            if save_path is None:
                save_path = os.path.join(self.scene_folder_path, f'segmentation_{(cam_num):05}_{object_idx}.png')
                
            cv2.imwrite(save_path, segmentation)
        
        return pix_left_ratio, unoccluded_pixel_num, segmentation

    def render_all_object_segmentation(self, mujoco_env, cam_height, cam_width, cam_num, object_idx, save=True, save_path=None):
        return None 
    
    def render_point_cloud(self, depth, cam_num, mask=None, save=True, save_path=None):
        o3d_intrinsics = self.camera_info_dict[cam_num].get_o3d_intrinsics()
        o3d_depth = o3d.geometry.Image(depth)
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsics)
        import pdb; pdb.set_trace()
        return None 
    
    # def output_json_information(self):
    #     date_captured = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    def create_env(self):
        self.mujoco_env = MujocoEnv(self.cam_temp_scene_xml_file, 1, has_robot=False)
        self.mujoco_env.sim.physics.forward()
        
        for _ in range(self.num_objects):
            for _ in range(4000):
                self.mujoco_env.model.step()
        all_poses = self.mujoco_env.data.qpos.ravel().copy().reshape(-1,7)
        return all_poses

# class PerchSceneBowlAndCan(PerchScene):
#     def __init__(self, scene_num, args):
#         # 3152c7a0e8ee4356314eed4e88b74a21
#         selected_objects = [
#             ('2880940',2,'95ac294f47fd7d87e0b49f27ced29e3',7),
#             ('2946921',3,'d44cec47dbdead7ca46192d8b30882',8),
#         ]
#         super().__init__(scene_num, selected_objects, args)
    
#     def add_objects_to_scene(self):
#         bowl_xyz = np.zeros(3)
#         bowl_width_limit = None
#         for object_idx in range(self.num_objects):
#             mujoco_object = self.object_info_dict[object_idx]
#             if object_idx == 0:
#                 obj_bound = self.object_info_dict[object_idx].object_mesh.bounds
                
#                 object_bottom = -obj_bound[0][2]
#                 bowl_xyz[2] = self.table_info.height + object_bottom + 0.002
#                 # self.
#                 bowl_xyz[0] = np.random.uniform(self.table_min_x+2, self.table_max_x-2)
#                 bowl_xyz[1] = np.random.uniform(self.table_min_y+2, self.table_max_y-2)
#                 mujoco_object.pos = bowl_xyz
#                 mujoco_object.set_object_rot(np.zeros(3))
#                 scale = [np.random.choice([0.75, 0.85, 1.0])] * 3
#                 mujoco_object.set_object_scale(scale=scale)
#                 bowl_bounds = mujoco_object.object_mesh.bounds
#                 bowl_width_limit = np.min(bowl_bounds[1,:2]-bowl_bounds[0,:2])
                
#             if object_idx == 1:
#                 mujoco_object.pos = [10, 10, 1]
#                 can_rot = [
#                     np.random.uniform(-45.0, 45),
#                     np.random.uniform(-45.0, 45),
#                     np.random.uniform(0, 360),
#                 ]
#                 can_rot = [
#                     np.random.uniform(-75.0, -45),
#                     np.random.uniform(-75.0, -45),
#                     np.random.uniform(0, 360),
#                 ]
#                 can_rot_r = R.from_euler('xyz', can_rot, degrees=True)
#                 mujoco_object.set_object_rot(can_rot_r.as_rotvec())
                
#                 mujoco_object.reset_size()
#                 bounds = mujoco_object.object_mesh.bounds
#                 x_range, y_range, z_range = bounds[1] - bounds[0]
#                 max_range = np.max(bounds[1,:2] - bounds[0,:2])
#                 max_range = np.max(bounds[1] - bounds[0])
#                 scale = [(bowl_width_limit * 0.5) / max_range] * 3
#                 mujoco_object.set_object_scale(scale=scale)                
            
#             self.object_info_dict[object_idx] = mujoco_object
        
#         for object_idx in range(self.num_objects):
#             add_objects(self.object_info_dict[object_idx].get_mujoco_add_dict(), run_id=None, material_name=None)

#         # Add camera around the scene
#         num_angles = 8
#         quad = (2.0*math.pi) / num_angles
#         normal_thetas = [i*quad for i in range(num_angles)]
#         # [np.random.uniform(i*quad, (i+1.0)*quad, 1)[0] for i in range(num_angles)]
#         bowl_height = self.object_info_dict[0].object_mesh.bounds[1,2] - self.object_info_dict[0].object_mesh.bounds[0,2]
#         bowl_width = self.object_info_dict[0].object_mesh.bounds[1,1] - self.object_info_dict[0].object_mesh.bounds[0,1]
#         for theta in normal_thetas:
#             cam_x = np.cos(theta) * (bowl_width * 3) + self.object_info_dict[0].pos[0]
#             cam_y = np.sin(theta) * (bowl_width * 3) + self.object_info_dict[0].pos[1]
#             location = [cam_x, cam_y, self.table_info.height + bowl_height + 1]
#             target = self.object_info_dict[0].pos
#             self.add_camera_to_scene(location, target)
        
#         all_poses = self.create_env()
#         
        
#         bowl_position = all_poses[1][:3]
#         bowl_rotvec = mujoco_quat_to_rotation_object(all_poses[1][3:]).as_rotvec()
#         bowl_bbox = self.object_info_dict[0].get_object_bbox(bowl_position, bowl_rotvec)

#         can_length = self.object_info_dict[1].object_mesh.bounds[1] - self.object_info_dict[1].object_mesh.bounds[0]
#         can_length = np.linalg.norm(can_length)
        
#         valid = False 
#         while not valid:
#             corner_from = np.random.choice(4)
#             x,y,z = bowl_bbox[(corner_from+1)*2]
#             x2,y2 = x-can_length,y
            

        
#         # can_position = copy.deepcopy(bowl_position)
#         # can_height = self.object_info_dict[1].object_mesh.bounds[1,2] - self.object_info_dict[1].object_mesh.bounds[0,2]
#         # can_bottom = -self.object_info_dict[1].object_mesh.bounds[0,2]
#         # # can_position[0] -= 0.001
#         # # can_position[1] 
#         # can_position[2] = self.table_info.height + can_bottom  #(0.0005 + can_bottom)
        
#         rot_x,rot_y,rot_z,rot_w = R.from_rotvec(self.object_info_dict[1].rot).as_quat()
#         new_rot = [rot_w, rot_x, rot_y, rot_z]
#         all_new_poses = move_object(self.mujoco_env, 1, can_position, new_rot)
#         # 

#         for cam_num in self.camera_info_dict.keys():
#             self.camera_info_dict[cam_num].set_camera_info_with_mujoco_env(self.mujoco_env, self.height, self.width)

#         for cam_num in self.camera_info_dict.keys():
#             fname = os.path.join(self.scene_folder_path, f'rgb_beforehand_{(cam_num):05}.png')
#             self.render_rgb(self.height, self.width, cam_num, save_path=fname)
        
#         import pdb; pdb.set_trace()
#         # for _ in range(self.num_objects):
#         for _ in range(1000):
#             self.mujoco_env.model.step()

#         for cam_num in self.camera_info_dict.keys():
            
#             self.render_rgb(self.height, self.width, cam_num)
#             self.render_depth(self.height, self.width, cam_num)

#             for object_idx in self.object_info_dict.keys():
#                 self.render_object_segmentation(self.height, self.width, cam_num, object_idx)


def generate_coco_file(root_dir):
    json_dict = {
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
        "categories" : [],
        "camera_intrinsic_settings": None, 
        "fixed_transforms": None,
        "images" : [],
        "annotations" : [],
    }
    scene_dirs = os.listdir(root_dir)
    scene_dirs = [scene_dir.startswith("scene_") for scene_dir in scene_dirs]