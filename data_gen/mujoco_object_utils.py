import os 
import numpy as np
import trimesh 
import autolab_core
from scipy.spatial.transform import Rotation as R, rotation        
import simple_clutter_utils as utils

class MujocoObject(object):
    def __init__(
        self,
        scene_name,
        object_name,
        shapenet_file_name,
        transformed_mesh_fname,
        color,
        num_objects_in_scene,
    ):
        self.scene_name = scene_name
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

    def apply_object_scale(self, mesh):
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= self.size
        mesh.apply_transform(scale_matrix)
        return mesh 
    
    def set_object_rot(self, rot):
        r = R.from_euler('xyz', rot, degrees=False)
        rotation_mat = np.eye(4)
        rotation_mat[0:3, 0:3] = r.as_matrix()
        self.object_mesh.apply_transform(rotation_mat)
        self.rot = rot 

    def get_corners(self, bounds, pos, rot, from_frame_name):
        '''
        bounds: (2,3)
        pos: (3,)
        rot: (3,) radian
        from_frame_name: str
        '''
        rot_obj = R.from_euler('xyz', rot, degrees=False)
        obj_frame_to_world = autolab_core.RigidTransform(
            rotation=rot_obj.as_matrix(),
            translation=pos,
            from_frame=from_frame_name,
            to_frame='world',
        )
        #                  (6)_________(8)
        # (2)_________(4)   |           |
        # |           |     |           |
        # |           |    (5)_________(7)   
        # (1)_________(3)
        corners_obj = utils.bounds_xyz_to_corners(bounds)
        corners_world = utils.transform_3d_frame(obj_frame_to_world.matrix, corners_obj)
        return corners_obj, corners_world, obj_frame_to_world.matrix
    
    def get_mujoco_add_dict(self):
        return {
            'scene_name' : self.scene_name,
            'object_name': self.object_name,
            'mesh_names': [self.transformed_mesh_fname],
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
        table_corners = utils.bounds_xyz_to_corners(table_bounds)
        table_top_corners = table_corners[table_corners[:,2] == table_bounds[1,2]]
        self.table_top_corners = utils.transform_3d_frame(self.table_frame_to_world.matrix, table_top_corners)
        
        table_corners, _, obj_to_world_mat = self.get_corners(table_bounds, self.pos, self.rot, 'table')
        self.table_frame_to_world_mat = obj_to_world_mat
        table_top_corners = table_corners[table_corners[:,2] == table_bounds[1,2]]
        self.table_top_corners = utils.transform_3d_frame(self.table_frame_to_world_mat, table_top_corners)


class MujocoNonTable(MujocoObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.shapenet_convex_decomp_dir = kwargs['shapenet_convex_decomp_dir']
        
        self.canonical_size = 2
        self.set_object_scale()
        self.object_idx = kwargs['object_idx']
    
    def load_decomposed_mesh(self):
        obj_convex_decomp_dir = os.path.join(self.shapenet_convex_decomp_dir, f'{self.synset_id}/{self.model_id}')

        comb_mesh = None
        for mesh_file in os.listdir(obj_convex_decomp_dir):
            object_mesh_part = trimesh.load(os.path.join(obj_convex_decomp_dir, mesh_file))
            if comb_mesh == None:
                comb_mesh = object_mesh_part
            else:
                comb_mesh += object_mesh_part
        self.apply_object_scale(comb_mesh)
        trimesh.repair.fix_inversion(comb_mesh)
        meshes = comb_mesh.split()
        
        mesh_names = []
        mesh_masses = []
        
        combined_mesh=None
        mesh_file_ind=0
        for mesh_file in os.listdir(obj_convex_decomp_dir):
            object_mesh_part = meshes[mesh_file_ind]
            if object_mesh_part.faces.shape[0] > 10 and object_mesh_part.mass>10e-7:
                object_mesh_part_fname = os.path.join(obj_convex_decomp_dir, mesh_file[:-3]+'stl')
                object_mesh_part.export(object_mesh_part_fname)
                mesh_names.append(object_mesh_part_fname)
                mesh_masses.append(object_mesh_part_fname.mass)
                
                if combined_mesh == None:
                    combined_mesh = object_mesh_part_fname
                else:
                    combined_mesh += object_mesh_part_fname
            mesh_file_ind += 1
            if mesh_file_ind >= len(meshes):
                break
        
        if len(mesh_names) > 100:
            heavy_inds = np.argsort(np.array(mesh_masses))
            new_mesh_names=[]
            for ind in range(100):
                new_mesh_names.append(mesh_names[heavy_inds[-ind]])
            mesh_names = new_mesh_names

    
    def reload_mesh(self):
        object_mesh = trimesh.load(self.transformed_mesh_fname, force='mesh')
        object_bounds = self.object_mesh.bounds
        range_max = np.max(object_bounds[1] - object_bounds[0])
        object_size = self.canonical_size / range_max
        normalize_vec = [object_size] * 3
        normalize_matrix = np.eye(4)
        normalize_matrix[:3, :3] *= normalize_vec
        object_mesh.apply_transform(normalize_matrix)
        return object_mesh, normalize_vec

    def reset_size(self):
        # Normalize the size by scaling the longest edge
        self.object_mesh, normalize_vec = self.reload_mesh()
        self.size = np.asarray(normalize_vec)
        self.set_object_rot(self.rot)
    
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
    
    def get_object_bbox(self, pos, rot):
        object_mesh, _ = self.reload_mesh()
        bounds = object_mesh.bounds
        corners, corners_world, obj_to_world_mat = self.get_corners(bounds, pos, rot, 'object_{}'.format(self.object_idx))
        return corners_world
