import math
import autolab_core
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from dm_control.mujoco.engine import Camera 

import datagen_utils

class SceneCamera(object):
    def __init__(self, location, target, cam_num):
        self.pos  = location
        self.target = target 
        self.cam_num = cam_num
        self.cam_name = f'gen_cam_{cam_num}'

        self.rgb_image_annotations = []

    def add_camera_to_file(self, xml_fname):
        return datagen_utils.add_camera(xml_fname, self.cam_name, self.pos, self.target, self.cam_num)
        

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
            'all_object_with_table_segmentation_path' : self.all_object_with_table_segmentation_path,
            'all_object_segmentation_path' : self.all_object_segmentation_path,
            'intrinsics_matrix': datagen_utils.get_json_cleaned_matrix(self.intrinsics, type='float'),
            'pos' : datagen_utils.get_json_cleaned_matrix(self.pos, type='float'), 
            'rot_quat' : datagen_utils.get_json_cleaned_matrix(self.rot_quat, type='float'),
            'camera_frame_to_world_frame_mat' : datagen_utils.get_json_cleaned_matrix(self.camera_frame_to_world_frame_mat, type='float'),
            'world_frame_to_camera_frame_mat' : datagen_utils.get_json_cleaned_matrix(self.world_frame_to_camera_frame_mat, type='float'),
            'all_object_bbox' : self.all_object_bbox,
        }
    
    def get_annotations_dict(self):
        coco_annos = []
        for coco_anno in self.rgb_image_annotations:
            coco_annos.append(coco_anno.output_dict())
        return coco_annos
