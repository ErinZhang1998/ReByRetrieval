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
    
    def get_camera_info_dict(self):
        return {
            'intrinsics': self.intrinsics,
            'pos' : self.pos, 
            'rot' : self.rot_quat,
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
        self.add_cameras()
        self.create_convex_decomposed_scene()
        
        self.create_camera_scene()

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
            scale=self.selected_objects[object_idx][4],
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
    
    def generate_cameras_around(self, center_x=0, center_y=0):
        num_angles = 20
        quad = (2.0*math.pi) / num_angles
        normal_thetas = [i*quad for i in range(num_angles)]
        table_width = np.max(self.table_info.object_mesh.bounds[1,:2] - self.table_info.object_mesh.bounds[0,:2])
        
        locations = []
        targets = []
        for theta in normal_thetas:
            cam_x = np.cos(theta) * (table_width/2) + center_x
            cam_y = np.sin(theta) * (table_width/2) + center_y
            location = [cam_x, cam_y, self.table_info.height + 1.5]
            target = [center_x,center_y,self.table_info.height]

            locations.append(location)
            targets.append(target)

        return locations, targets
    
    def add_cameras(self):
        locations, targets = self.generate_cameras_around()
        for location, target in zip(locations, targets):
            new_camera = SceneCamera(location, target, self.total_camera_num)
            new_camera.rgb_save_path = os.path.join(self.scene_folder_path, f'rgb_{(self.total_camera_num):05}.png')
            new_camera.depth_save_path = os.path.join(self.scene_folder_path, f'depth_{(self.total_camera_num):05}.png')
            self.camera_info_dict[self.total_camera_num] = new_camera
            self.total_camera_num += 1
    
    def generate_default_add_position(self, object_idx):
        theta = ((2.0*math.pi) / self.num_objects) * object_idx
        start_x = np.cos(theta) * 10
        start_y = np.sin(theta) * 10
        start_z = object_idx + 1
        return [start_x, start_y, start_z]

    def create_convex_decomposed_scene(self):
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
        
        # #  DEBUG 
        # for cam_num in self.camera_info_dict.keys():
        #     new_camera = self.camera_info_dict[cam_num]
        #     new_camera.add_camera_to_file(self.convex_decomp_xml_file)

        convex_decomp_mujoco_env = MujocoEnv(self.convex_decomp_xml_file, 1, has_robot=False)
        for object_idx in range(self.num_objects):
            object_info = self.object_info_dict[object_idx]
            convex_decomp_mesh_height = -object_info.convex_decomp_mesh.bounds[0,2]
            if False:
                prev_object_info = self.object_info_dict[object_idx-1]
                prev_object_info_bounds = prev_object_info.convex_decomp_mesh.bounds
                prev_pose = convex_decomp_mujoco_env.data.qpos.ravel().copy().reshape(-1,7)[object_idx]
                prev_x, prev_y, prev_z = prev_pose[:3]
                if prev_z < self.table_info.height:
                    
                    prev_x, prev_y = 0,0
                pos_std = np.linalg.norm(prev_object_info_bounds[1] - prev_object_info_bounds[0])
                # 
                # _, prev_object_corners,_ = get_corners(prev_object_info_bounds, prev_pose[:3], mujoco_quat_to_rotation_object(prev_pose[3:]).as_rotvec(), f'prev_object_{object_idx-1}')
                # pos_x, pos_y = np.random.normal(loc=[prev_x,prev_y], scale=np.array([pos_std/2]*2)) 
            else:
                pos_x, pos_y = object_info.pos_x, object_info.pos_y
            
            moved_location = [pos_x, pos_y, self.table_info.height+convex_decomp_mesh_height+0.05]
            move_object(convex_decomp_mujoco_env, object_idx, moved_location, euler_xyz_to_mujoco_quat(object_info.rot))
            
            for _ in range(4000):
                convex_decomp_mujoco_env.model.step()

        self.convex_decomp_mujoco_env_state = convex_decomp_mujoco_env.get_env_state().copy()
        all_current_poses = convex_decomp_mujoco_env.data.qpos.ravel().copy().reshape(-1,7) 
        objects_current_positions = all_current_poses[1:][:,:3]
        
        table_current_position = all_current_poses[0][:3].reshape(-1,)
        table_current_position[2] = self.table_info.height
        object_closest_to_table_center = np.argmin(np.linalg.norm(objects_current_positions - table_current_position, axis=1))
        
        new_cam_center_x, new_cam_center_y,_ = objects_current_positions[object_closest_to_table_center]
        locations, targets = self.generate_cameras_around(center_x=new_cam_center_x, center_y=new_cam_center_y)
        for cam_num, (location, target) in enumerate(zip(locations, targets)):
            self.camera_info_dict[cam_num].pos = location
            self.camera_info_dict[cam_num].target = target
        
        # #  DEBUG
        # for cam_num in self.camera_info_dict.keys():
        #     fname = os.path.join(self.scene_folder_path, f'rgb_decomp_{(cam_num):05}.png')
        #     self.render_rgb(convex_decomp_mujoco_env, self.height, self.width, cam_num, save_path=fname)
    
    
    def create_camera_scene(self):
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
        for cam_num in self.camera_info_dict.keys():
            new_camera = self.camera_info_dict[cam_num]
            new_camera.add_camera_to_file(self.cam_temp_scene_xml_file)
        
        mujoco_env = MujocoEnv(self.cam_temp_scene_xml_file, 1, has_robot=False)
        mujoco_env.set_env_state(self.convex_decomp_mujoco_env_state)
        mujoco_env.sim.physics.forward()

        # Calculate object bounding box in the scene and save in the mujoco object 
        all_current_poses = mujoco_env.data.qpos.ravel().copy().reshape(-1,7)  
        
        ###################################### Save the scaled object, which is used for perch in the models directory
        for object_idx in self.object_info_dict.keys():
            model_save_dir = os.path.join(args.scene_save_dir, 'models', f'{self.train_or_test}_scene_{self.scene_num}_object_{object_idx}')
            if os.path.exists(model_save_dir):
                shutil.rmtree(model_save_dir)
            os.mkdir(model_save_dir)
            model_fname = os.path.join(model_save_dir, 'textured.obj')

            object_info = self.object_info_dict[object_idx]
            object_mesh = object_info.save_correct_size_model(model_fname)

            mesh_bounds = object_mesh.bounds
            current_pose = all_current_poses[object_idx+1]
            
            mesh_bounds, mesh_bbox, _ = get_corners(mesh_bounds, current_pose[:3], mujoco_quat_to_rotation_object(current_pose[3:]).as_euler('xyz'), f'object_{object_idx}')
            assert np.all(np.abs(np.linalg.norm(mesh_bbox - current_pose[:3], axis=1) - np.linalg.norm(mesh_bounds, axis=1)) < 1e-5)
            self.object_info_dict[object_idx].bbox = mesh_bbox
            self.object_info_dict[object_idx].object_center = current_pose[:3]
        
        ###################################### Save camera information, like intrinsics 
        for cam_num in self.camera_info_dict.keys():
            self.camera_info_dict[cam_num].set_camera_info_with_mujoco_env(mujoco_env, self.height, self.width)

        for cam_num in self.camera_info_dict.keys():
            fname = os.path.join(self.scene_folder_path, f'rgb_{(cam_num):05}.png')
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
                save_path = os.path.join(self.scene_folder_path, f'rgb_{(cam_num):05}.png')
            cv2.imwrite(save_path, rgb)
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
        if save:     
            if save_path is None:
                save_path = os.path.join(self.scene_folder_path, f'depth_{(cam_num):05}.png')
            # depth_img = Image.fromarray((depth*self.depth_factor).astype(np.int32))
            # depth_img.save(save_path)
            cv2.imwrite(save_path, (depth*self.depth_factor).astype(np.uint16))
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
                    move_object(mujoco_env, move_obj_ind, start_position, [0, 0, 0, 0])
            mujoco_env.sim.physics.forward()
            
            save_path = os.path.join(self.scene_folder_path, f'segmentation_{(cam_num):05}_{object_idx}.png')

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
            pix_left_ratio = np.argwhere(segmentation).shape[0] / unoccluded_pixel_num
            cv2.imwrite(save_path, segmentation.astype(np.uint8))
            
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
            center_3d = self.object_info_dict[object_idx].object_center
            center_2d = self.camera_info_dict[cam_num].project_2d(center_3d.reshape(-1,3)).reshape(-1,)
            center_2d[0] = self.width - center_2d[0]
            
            # # DEBUG purpose
            cv2_image = cv2.imread(self.camera_info_dict[cam_num].rgb_save_path)
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

            object_annotation = ImageAnnotation(
                center_3d = list(center_3d.reshape(-1,)),
                center = list(center_2d),
                bbox = bbox_ann,
                rgb_file_path = self.camera_info_dict[cam_num].rgb_save_path,
                mask_file_path = save_path,
                model_file_path = self.object_info_dict[object_idx].textured_obj_fname,
                percentage_not_occluded = pix_left_ratio,
                cam_intrinsics = camera_info.intrinsics, 
                cam_location = camera_info.pos, 
                cam_quat = camera_info.rot_quat, 
            )
            self.camera_info_dict[cam_num].rgb_image_annotations.append(object_annotation)
    
    def output_camera_annotations(self):
        all_annotations = []
        for cam_num in self.camera_info_dict.keys():
            camera_info = self.camera_info_dict[cam_num]
            anno_dict_list = camera_info.get_annotations_dict()
            all_annotations += anno_dict_list
        
        json_string = json.dumps(all_annotations)
        annotation_fname = os.path.join(self.scene_folder_path, 'annotations.json')
        json_file = open(annotation_fname, "w+")
        json_file.write(json_string)
        json_file.close()

    def render_whole_scene_segmentation(self, mujoco_env, cam_height, cam_width, cam_num, save=True, save_path=None):
        camera = Camera(physics=mujoco_env.model, height=cam_height, width=cam_width, camera_id=cam_num)
        segmentation = camera.render(segmentation=True)[:,:,0] #(480, 640, 2)
        if save:
            cv2.imwrite(os.path.join(self.scene_folder_path, f'segmentation_{(cam_num):05}.png'), segmentation)
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