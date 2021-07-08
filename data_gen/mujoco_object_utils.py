import os 
import numpy as np
import trimesh 
import autolab_core
from scipy.spatial.transform import Rotation as R, rotation        
import simple_clutter_utils as utils

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
        self.rot : (3,) Rotation vector
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
        self.rot = np.zeros(3)
        self.pos = np.zeros(3)
        # Chaning size scale changes the object mesh
        self.size = np.ones(3)

    def from_mesh_fname_to_ids(self):
        fname_list = self.shapenet_file_name.split('/')
        self.synset_id = fname_list[-4]
        self.model_id = fname_list[-3] 

    def set_object_scale(self, scale = None):
        pass
    
    def set_object_rot(self, rot):
        rot_obj = R.from_euler('xyz', rot, degrees=False)
        # rotation_mat = np.eye(4)
        # rotation_mat[0:3, 0:3] = r.as_matrix()
        # self.object_mesh.apply_transform(rotation_mat)
        self.object_mesh = utils.apply_rot_to_mesh(self.object_mesh, rot_obj)
        self.rot = rot 
    
    def get_mujoco_add_dict(self):
        return {
                'object_name': self.object_name,
                'mesh_names': [self.transformed_mesh_fname],
                'pos': self.pos,
                'size': self.size,
                'color': self.color,
                'rot': self.rot,
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
        self.canonical_size = 2

        self.actual_size = np.random.choice([0.75, 0.85, 1.0]) * self.canonical_size
        
        if np.random.uniform(0,1) < 0.8:
            random_rotation = [
                np.random.uniform(-90.0, 90),
                np.random.uniform(-90.0, 90),
                np.random.uniform(0, 360),
            ]
        else:
            random_rotation = [
                0,
                0,
                np.random.uniform(0, 360),
            ]
        
        random_rotation_r = R.from_euler('xyz', random_rotation, degrees=True)
        self.rot = random_rotation_r.as_euler('xyz')

        self.pos_x, self.pos_y = np.random.normal(loc=[0,0], scale=np.array([2,2]))
        self.bbox = None

    def load_decomposed_mesh(self):
        obj_convex_decomp_dir = os.path.join(self.shapenet_convex_decomp_dir, f'{self.synset_id}/{self.model_id}')
        comb_mesh = trimesh.load(os.path.join(obj_convex_decomp_dir, 'convex_decomp.obj'), force='mesh')
        # make mesh stand upright
        comb_mesh.apply_transform(self.upright_mat) 
        comb_mesh.export(os.path.join(obj_convex_decomp_dir, 'convex_decomp.stl'))

        range_max = np.linalg.norm(comb_mesh.bounds[1] - comb_mesh.bounds[0])
        comb_mesh_scale = self.actual_size / range_max
        comb_mesh_scale = [comb_mesh_scale] * 3
        # scale the mesh
        comb_mesh = utils.apply_scale_to_mesh(comb_mesh, comb_mesh_scale)
        self.size = comb_mesh_scale
        # comb_mesh.export(os.path.join(obj_convex_decomp_dir, 'convex_decomp.stl'))
        mesh_names = [os.path.join(obj_convex_decomp_dir, 'convex_decomp.stl')]
        self.convex_decomp_mesh_fnames = mesh_names
        # Apply rotation, generated during initialization
        rot_obj = R.from_euler('xyz', self.rot, degrees=False)
        self.convex_decomp_mesh = utils.apply_rot_to_mesh(comb_mesh, rot_obj)
        return mesh_names
    
    def save_correct_size_model(self, model_fname):
        object_mesh = trimesh.load(self.shapenet_file_name, force='mesh')
        object_mesh.apply_transform(self.upright_mat)
        object_mesh = utils.apply_scale_to_mesh(object_mesh, self.size)
        object_mesh.export(model_fname)
        self.textured_obj_fname = model_fname

        return object_mesh


class MujocoTable(MujocoObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_object_scale()

        self.pos = np.array([0.0, 0.0, -self.object_mesh.bounds[0][2]])

        self.height = self.object_mesh.bounds[1][2] - self.object_mesh.bounds[0][2]
        table_r = R.from_euler('xyz', self.rot, degrees=False)
        self.table_frame_to_world = autolab_core.RigidTransform(
            rotation=table_r.as_matrix(),
            translation=self.pos,
            from_frame='table',
            to_frame='world',
        )
        table_bounds = self.object_mesh.bounds
        table_corners = utils.bounds_xyz_to_corners(table_bounds)
        table_top_corners = table_corners[table_corners[:,2] == table_bounds[1,2]]
        self.table_top_corners = utils.transform_3d_frame(self.table_frame_to_world.matrix, table_top_corners)
        
        table_corners, _, obj_to_world_mat = utils.get_corners(table_bounds, self.pos, self.rot, 'table')
        self.table_frame_to_world_mat = obj_to_world_mat
        table_top_corners = table_corners[table_corners[:,2] == table_bounds[1,2]]
        self.table_top_corners = utils.transform_3d_frame(self.table_frame_to_world_mat, table_top_corners)

    def set_object_rot(self, rot):
        # If change rotation, then need to change table_frame_to_world and other stuff
        # Not for now
        raise

    def set_object_scale(self, scale = None):
        '''
        Scale the table so that it can hold many objects
        '''
        table_bounds = self.object_mesh.bounds
        table_xyz_range = np.min(table_bounds[1, :2] - table_bounds[0, :2])
        table_size = 2*(self.num_objects_in_scene + 2)/table_xyz_range
        scale_vec = np.array([table_size]*3)
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale_vec
        self.object_mesh.apply_transform(scale_matrix)
        table_bounds = self.object_mesh.bounds
        self.size = scale_vec
