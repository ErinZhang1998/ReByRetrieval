import re 
import os 
import numpy as np 
import trimesh
import h5py

from scipy.spatial.transform import Rotation as R, rotation        

class BlenderProcObject(object):
    def __init__(
        self,
        model_name,
        synset_id,
        model_id,
        shapenet_file_name,
        num_objects_in_scene,
    ):
        self.synset_id = synset_id
        self.model_id = model_id
        self.model_name = model_name
        self.shapenet_file_name = shapenet_file_name
        self.num_objects_in_scene = num_objects_in_scene

        object_mesh = trimesh.load(shapenet_file_name, force='mesh')
        r = R.from_euler('xyz', [(1/2)*np.pi, 0, 0], degrees=False)
        upright_mat = np.eye(4)
        upright_mat[0:3, 0:3] = r.as_matrix()
        self.upright_mat = upright_mat
        object_mesh.apply_transform(upright_mat) 
        self.object_mesh = object_mesh
        
        self.rot = R.from_euler('xyz', np.zeros(3), degrees=False) 
        self.pos = np.zeros(3)
        self.actual_size = np.ones(3)
    
    def get_blender_proc_dict(self):
        upright_rot = self.upright_mat[0:3, 0:3]
        final_rot_matrix = self.rot.as_matrix() @ upright_rot
        final_rot = R.from_matrix(final_rot_matrix)
        return {
                'model_name': self.model_name,
                'synset_id': self.synset_id,
                'model_id': self.model_id,
                'actual_size': self.actual_size,
                'position': self.pos,
                'euler' : final_rot.as_euler('xyz'),
            }


class BlenderProcTable(BlenderProcObject):
    def __init__(self, **kwargs):
        super().__init__(
            model_name = kwargs['model_name'],
            synset_id = kwargs['synset_id'],
            model_id = kwargs['model_id'],
            shapenet_file_name = kwargs['shapenet_file_name'],
            num_objects_in_scene = kwargs['num_objects_in_scene'],
        )
        self.table_size = kwargs['table_size']
        self.pos = np.array([0.0, 0.0, -self.object_mesh.bounds[0][2]])
        self.rot = R.from_euler('xyz', np.zeros(3), degrees=False) 
        

        table_bounds = self.object_mesh.bounds
        table_xyz_range = np.min(table_bounds[1, :2] - table_bounds[0, :2])
        table_scale = self.table_size/table_xyz_range
        scale_vec = np.array([table_scale]*3)
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale_vec
        self.object_mesh.apply_transform(scale_matrix)
        table_bounds = self.object_mesh.bounds
        self.actual_size = table_bounds[1] - table_bounds[0]
        self.height = self.object_mesh.bounds[1][2] - self.object_mesh.bounds[0][2]
        

    
class BlenderProcNonTable(BlenderProcObject):
    def __init__(self, **kwargs):
        super().__init__(
            model_name = kwargs['model_name'],
            synset_id = kwargs['synset_id'],
            model_id = kwargs['model_id'],
            shapenet_file_name = kwargs['shapenet_file_name'],
            num_objects_in_scene = kwargs['num_objects_in_scene'],
        )

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
                self.pos_z = kwargs['table_height'] + (-self.object_mesh.bounds[0,2]) + 0.005
            else:
                self.pos_x, self.pos_y, self.pos_z = pre_selected_position
        else:
            self.pos_x, self.pos_y = np.random.normal(loc=[0,0], scale=np.array([0.15, 0.15]))
            self.pos_z = kwargs['table_height'] + (-self.object_mesh.bounds[0,2]) + 0.005
        self.pos = np.array([self.pos_x, self.pos_y, self.pos_z])
        


def read_obj(model_path, flags = ('v')):
    fid = open(model_path, 'r', encoding="utf-8")

    data = {}

    for head in flags:
        data[head] = []

    for line in fid:
        line = line.strip()
        if not line:
            continue
        line = re.split('\s+', line)
        if line[0] in flags:
            data[line[0]].append(line[1:])

    fid.close()

    if 'v' in data.keys():
        data['v'] = np.array(data['v']).astype(np.float)

    if 'vt' in data.keys():
        data['vt'] = np.array(data['vt']).astype(np.float)

    if 'vn' in data.keys():
        data['vn'] = np.array(data['vn']).astype(np.float)

    return data


def normalize_points(in_points, padding = 0.1):
    '''
    normalize points into [-0.5, 0.5]^3, and output point set and scale info.
    :param in_points: input point set, Nx3.
    :return:
    '''
    vertices = in_points.copy()

    bb_min = vertices.min(0)
    bb_max = vertices.max(0)
    total_size = (bb_max - bb_min).max() / (1 - padding)

    # centroid
    centroid = (bb_min + bb_max)/2.
    vertices = vertices - centroid

    vertices = vertices/total_size

    return vertices, total_size, centroid


def normalize_obj_file(input_obj_file, output_obj_file, padding = 0.1):
    """
    normalize vertices into [-0.5, 0.5]^3, and write the result into another .obj file.
    :param input_obj_file: input .obj file
    :param output_obj_file: output .obj file
    :return:
    """
    vertices = read_obj(input_obj_file)['v']

    vertices, total_size, centroid = normalize_points(vertices, padding=padding)

    input_fid = open(input_obj_file, 'r', encoding='utf-8')
    output_fid = open(output_obj_file, 'w', encoding='utf-8')

    v_id = 0
    for line in input_fid:
        if line.strip().split(' ')[0] != 'v':
            output_fid.write(line)
        else:
            output_fid.write(('v' + ' %f' * len(vertices[v_id]) + '\n') % tuple(vertices[v_id]))
            v_id = v_id + 1

    output_fid.close()
    input_fid.close()

    return total_size, centroid


def scale_points(in_points, actual_size):
    actual_size = np.asarray(actual_size)
    vertices = in_points.copy()
    bb_min = vertices.min(0)
    bb_max = vertices.max(0)
    total_size = (bb_max - bb_min) / actual_size

    # centroid
    centroid = (bb_min + bb_max)/2.
    vertices = vertices - centroid

    vertices = vertices/total_size

    return vertices, total_size, centroid


def scale_obj_file(input_obj_file, output_obj_file, actual_size, add_name = None):
    vertices = read_obj(input_obj_file)['v']

    vertices, total_size, centroid = scale_points(vertices, actual_size)

    input_fid = open(input_obj_file, 'r', encoding='utf-8')
    output_fid = open(output_obj_file, 'w', encoding='utf-8')

    name_added = False

    v_id = 0
    for line in input_fid:
        first_char = line.strip().split(' ')[0]
        if first_char != 'v':
            output_fid.write(line)
            if first_char == 'mtllib':
                if add_name is not None:
                    name_line = 'o {}\n'.format(add_name)
                    output_fid.write(name_line)
                    name_added = True
        else:
            output_fid.write(('v' + ' %f' * len(vertices[v_id]) + '\n') % tuple(vertices[v_id]))
            v_id = v_id + 1
    if add_name is not None:
        assert name_added
    output_fid.close()
    input_fid.close()

    return total_size, centroid


def load_h5py_result(path):
    output_dict = {}
    if os.path.exists(path):
        if os.path.isfile(path):
            fh = h5py.File(path, 'r')
            print(path + " contains the following keys: " + str(data.keys()))
            keys = [key for key in data.keys()]

            # Visualize every key
            for key in keys:
                value = np.array(data[key])
                output_dict[key] = data[key]
                
            fh.close()
        else:
            print("The path is not a file")
    else:
        print("The file does not exist")
    return output_dict