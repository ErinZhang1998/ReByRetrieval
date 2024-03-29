# from math import comb
import os
import shutil 
import numpy as np
import trimesh 
import open3d as o3d
import autolab_core
from scipy.spatial.transform import Rotation as R, rotation        
import utils.datagen_utils as datagen_utils

# # bag, bottle, bowl, can, clock, jar, laptop, camera, mug
# ACTUAL_LIMIT_DICT = {
#     '02773838' : 0.2,
#     '02876657' : 0.1,
#     '02880940' : 0.15,
#     '02946921' : 0.1,
#     '03593526' : 0.25,
#     '03046257' : 0.15,
#     '03642806' : 0.3,
#     '02942699' : 0.15,
#     '03797390' : 0.1,
# }

class MujocoObject(object):
    def __init__(
        self,
        object_name,
        shapenet_file_name,
        transformed_mesh_fname,
        color,
        num_objects_in_scene,
    ):
        '''
        Transform the object mesh so that by default, it will stand upright on the table when put into the scene
        By default:
            position : [0,0,0]
            rotation : [0,0,0]
            size : [1,1,1]
        
        self.pos : (3,)
        self.rot : Rotation scipy transform object, from euler [0,0,0]
        self.size : (3,)
        '''
        self.object_name = object_name
        self.shapenet_file_name = shapenet_file_name
        self.from_mesh_fname_to_ids()
        self.transformed_mesh_fname = transformed_mesh_fname
        self.color = color
        self.num_objects_in_scene = num_objects_in_scene

        object_mesh = trimesh.load(shapenet_file_name, force='mesh')
        r = R.from_euler('xyz', [(1/2)*np.pi, 0, 0], degrees=False)
        upright_mat = np.eye(4)
        upright_mat[0:3, 0:3] = r.as_matrix()
        self.upright_mat = upright_mat
        object_mesh.apply_transform(upright_mat) 
        f = open(transformed_mesh_fname, "w+")
        f.close()
        object_mesh.export(transformed_mesh_fname)
        self.object_mesh = object_mesh

        # Chaning rotation changes the object mesh
        self.rot = R.from_euler('xyz', np.zeros(3), degrees=False) 
        self.pos = np.zeros(3)
        # Chaning size scale changes the object mesh
        self.size = None

    def from_mesh_fname_to_ids(self):
        fname_list = self.shapenet_file_name.split('/')
        self.synset_id = fname_list[-4]
        self.model_id = fname_list[-3] 
    
    def set_object_scale(self, scale = None):
        pass
    
    def get_mujoco_add_dict(self):
        return {
                'object_name': self.object_name,
                'mesh_names': [self.transformed_mesh_fname],
                'pos': self.pos,
                'size': self.size,
                'color': self.color,
                'quat': datagen_utils.quat_xyzw_to_wxyz(self.rot.as_quat()),
            }


class MujocoNonTable(MujocoObject):
    def __init__(self, **kwargs):
        super().__init__(
            object_name = kwargs['object_name'],
            shapenet_file_name = kwargs['shapenet_file_name'],
            transformed_mesh_fname = kwargs['transformed_mesh_fname'],
            color = kwargs['color'],
            num_objects_in_scene = kwargs['num_objects_in_scene'],
        )
        '''
        When initialized, generate random rotation and position
        '''
        self.shapenet_convex_decomp_dir = kwargs['shapenet_convex_decomp_dir']
        self.object_idx = kwargs['object_idx']
        self.half_or_whole = kwargs['selected_object_info']['half_or_whole']
        self.perch_rot_angle = kwargs['selected_object_info']['perch_rot_angle']
        self.upright_ratio = kwargs['upright_ratio']

        if 'size_xyz' in kwargs['selected_object_info']:
            self.actual_size = kwargs['selected_object_info']['size']  
            mesh_scale = kwargs['selected_object_info']['size'] / (self.object_mesh.bounds[1] - self.object_mesh.bounds[0])
            self.size = list(mesh_scale)
        else:
            xy_range_max = max((self.object_mesh.bounds[1] - self.object_mesh.bounds[0])[:2])        
            self.actual_size = (self.object_mesh.bounds[1] - self.object_mesh.bounds[0]) * (kwargs['selected_object_info']['size'] / xy_range_max)
            mesh_scale = self.actual_size / (self.object_mesh.bounds[1] - self.object_mesh.bounds[0])
            self.size = list(mesh_scale)

        if 'quaternion_xyzw' in kwargs['selected_object_info']:
            self.rot = R.from_quat(kwargs['selected_object_info']['quaternion_xyzw'])
            self.upright = True
        else:
            if np.random.uniform(0,1) > self.upright_ratio:
                random_rotation = [
                    np.random.uniform(-90.0, 90),
                    np.random.uniform(-90.0, 90),
                    np.random.uniform(0, 360),
                ]
                self.upright = False
            else:
                random_rotation = [
                    0,
                    0,
                    np.random.uniform(0, 360),
                ]
                self.upright = True
            self.rot = R.from_euler('xyz', random_rotation, degrees=True)

        if 'position' in kwargs['selected_object_info']:
            pre_selected_position = kwargs['selected_object_info']['position']
            if len(pre_selected_position) == 2:
                self.pos_x, self.pos_y = pre_selected_position
                self.pos_z = None
            else:
                self.pos_x, self.pos_y, self.pos_z = pre_selected_position
        else:
            self.pos_x, self.pos_y = np.random.normal(loc=[0,0], scale=np.array([0.15, 0.15]))
            self.pos_z = None

        self.bbox = None

    def load_decomposed_mesh(self):
        comb_mesh, obj_convex_decomp_dir = datagen_utils.get_convex_decomp_mesh(
            self.shapenet_file_name, 
            self.shapenet_convex_decomp_dir, 
            self.synset_id, 
            model_id=self.model_id,
        )
        comb_mesh.apply_transform(self.upright_mat) 
        comb_mesh.export(os.path.join(obj_convex_decomp_dir, 'convex_decomp.stl'))
        comb_mesh_scale = self.actual_size / (comb_mesh.bounds[1] - comb_mesh.bounds[0])
        comb_mesh = datagen_utils.apply_scale_to_mesh(comb_mesh, comb_mesh_scale)
        
        mesh_names = [os.path.join(obj_convex_decomp_dir, 'convex_decomp.stl')]
        self.convex_decomp_mesh_fnames = mesh_names
        self.convex_decomp_mesh = datagen_utils.apply_rot_to_mesh(comb_mesh, self.rot)
        return mesh_names, list(comb_mesh_scale)
    
    def save_correct_size_model(self, model_save_root_dir, model_name):
        import shutil
        model_save_dir = os.path.join(model_save_root_dir, model_name)
        if os.path.exists(model_save_dir):
            shutil.rmtree(model_save_dir)
        os.mkdir(model_save_dir)
        model_fname = os.path.join(model_save_dir, 'textured.obj')

        object_mesh = trimesh.load(self.shapenet_file_name, force='mesh')
        object_mesh.apply_transform(self.upright_mat)
        object_mesh = datagen_utils.apply_scale_to_mesh(object_mesh, self.size)
        object_mesh.export(model_fname)
        self.model_name = model_name

        # Save as textured.ply
        copy_textured_mesh = o3d.io.read_triangle_mesh(model_fname)
        o3d.io.write_triangle_mesh(os.path.join(model_save_dir, 'textured.ply'), copy_textured_mesh)

        return object_mesh
    
    def get_model_annotation(self):
        # final_quat is xyzw
        return {
            "synset_id" : self.synset_id,
            "model_id" : self.model_id,
            "size" : [float(item) for item in self.size],
            "actual_size" : [float(item) for item in self.actual_size],
            "position" : [float(item) for item in self.final_position],
            "quat" : [float(item) for item in self.final_quat],
            "half_or_whole" : int(self.half_or_whole),
            "perch_rot_angle" : int(self.perch_rot_angle),
        }


class MujocoTable(MujocoObject):
    def __init__(self, **kwargs):
        super().__init__(
            object_name = kwargs['object_name'],
            shapenet_file_name = kwargs['shapenet_file_name'],
            transformed_mesh_fname = kwargs['transformed_mesh_fname'],
            color = kwargs['color'],
            num_objects_in_scene = kwargs['num_objects_in_scene'],
        )
        self.table_size = kwargs['table_size']
        self.set_object_scale()

        self.pos = np.array([0.0, 0.0, -self.object_mesh.bounds[0][2]])

        self.height = self.object_mesh.bounds[1][2] - self.object_mesh.bounds[0][2]

        self.table_frame_to_world = autolab_core.RigidTransform(
            rotation=self.rot.as_matrix(),
            translation=self.pos,
            from_frame='table',
            to_frame='world',
        )
        table_bounds = self.object_mesh.bounds
        table_corners = datagen_utils.bounds_xyz_to_corners(table_bounds)
        table_top_corners = table_corners[table_corners[:,2] == table_bounds[1,2]]
        self.table_top_corners = datagen_utils.transform_3d_frame(self.table_frame_to_world.matrix, table_top_corners)
        
        table_corners, _, obj_to_world_mat = datagen_utils.get_corners(table_bounds, self.pos, self.rot, 'table')
        self.table_frame_to_world_mat = obj_to_world_mat
        table_top_corners = table_corners[table_corners[:,2] == table_bounds[1,2]]
        self.table_top_corners = datagen_utils.transform_3d_frame(self.table_frame_to_world_mat, table_top_corners)

    def set_object_scale(self, scale = None):
        '''
        Scale the table so that it can hold many objects
        '''
        table_bounds = self.object_mesh.bounds
        table_xyz_range = np.min(table_bounds[1, :2] - table_bounds[0, :2])
        table_scale = self.table_size/table_xyz_range
        scale_vec = np.array([table_scale]*3)
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale_vec
        self.object_mesh.apply_transform(scale_matrix)
        table_bounds = self.object_mesh.bounds
        self.size = scale_vec