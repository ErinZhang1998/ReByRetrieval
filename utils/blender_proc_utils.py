import re 
import os 
import json
import ast 
import numpy as np 
import trimesh
import h5py
import shutil
import pickle

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
        self.scale = np.ones(3)
    
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
                'scale' : self.scale,
            }


class BlenderProcTable(BlenderProcObject):
    def __init__(self, **kwargs):
        
        table_id = '97b3dfb3af4487b2b7d2794d2db4b0e7'#np.random.choice(kwargs['model_id_available'])
        synset_id = kwargs['synset_id']
        table_mesh_fname = os.path.join(kwargs['shapenet_filepath'], f'{synset_id}/{table_id}/models/model_normalized.obj')
        super().__init__(
            model_name = kwargs['model_name'],
            synset_id = kwargs['synset_id'],
            model_id = table_id,
            shapenet_file_name = table_mesh_fname,
            num_objects_in_scene = kwargs['num_objects_in_scene'],
        )  
        
        self.table_size = kwargs['table_size']
        self.table_size_xyz = kwargs['table_size_xyz']
        self.rot = R.from_euler('xyz', np.zeros(3), degrees=False) 

        table_bounds = self.object_mesh.bounds
        # table_xyz_range = np.min(table_bounds[1, :2] - table_bounds[0, :2])
        # table_scale = self.table_size/table_xyz_range
        # scale_vec = np.array([table_scale]*3)
        scale_vec = np.asarray(self.table_size_xyz) / (table_bounds[1] - table_bounds[0])
       
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale_vec
        self.object_mesh.apply_transform(scale_matrix)
        
        table_bounds = self.object_mesh.bounds
        
        self.actual_size = table_bounds[1] - table_bounds[0]
        
        self.height = self.object_mesh.bounds[1][2] - self.object_mesh.bounds[0][2]
        self.pos = np.array([0.0, 0.0, -self.object_mesh.bounds[0][2]])
        self.scale = scale_vec

              

    
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


def normalize_obj_file(input_obj_file, output_obj_file, padding = 0, add_name = None):
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

    # v_id = 0
    # for line in input_fid:
    #     if line.strip().split(' ')[0] != 'v':
    #         output_fid.write(line)
    #     else:
    #         output_fid.write(('v' + ' %f' * len(vertices[v_id]) + '\n') % tuple(vertices[v_id]))
    #         v_id = v_id + 1
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

    bb_min = vertices.min(0)
    bb_max = vertices.max(0)

    return bb_max, bb_min, total_size, centroid


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

    bb_min = vertices.min(0)
    bb_max = vertices.max(0)

    return bb_max, bb_min, total_size, centroid

def save_normalized_object_to_file(shapenet_filepath, normalized_model_save_dir, ann, actual_size_used=False):
    model_name = '{}_{}'.format(ann['synsetId'], ann['ShapeNetModelId'])
    shapenet_dir = os.path.join(
        shapenet_filepath,
        '{}/{}'.format(ann['synsetId'], ann['ShapeNetModelId']),
    )
    input_obj_file = os.path.join(shapenet_dir, 'models', 'model_normalized.obj')
    mtl_path = os.path.join(shapenet_dir, 'models', 'model_normalized.mtl')
    image_material_dir = os.path.join(shapenet_dir, 'images')

    synset_dir = os.path.join(normalized_model_save_dir, ann['synsetId'])
    if not os.path.exists(synset_dir):
        os.mkdir(synset_dir)
    model_dir = os.path.join(synset_dir, ann['ShapeNetModelId'])
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir_models = os.path.join(model_dir, 'models')
    if not os.path.exists(model_dir_models):
        os.mkdir(model_dir_models)
    
    output_obj_file = os.path.join(model_dir_models, 'model_normalized.obj')
    new_mtl_path = os.path.join(model_dir_models, 'model_normalized.mtl')
    shutil.copyfile(mtl_path, new_mtl_path)

    if os.path.exists(image_material_dir):
        new_image_dir = os.path.join(model_dir, 'images')
        if os.path.exists(new_image_dir):
            shutil.rmtree(new_image_dir)
        shutil.copytree(image_material_dir, new_image_dir) 

    normalize_obj_file(input_obj_file, output_obj_file, add_name=model_name)
    if actual_size_used:
        x,y,z = ann['actual_size']
        bb_max, bb_min, _, _ = scale_obj_file(input_obj_file, output_obj_file, np.array([x,z,y]), add_name=model_name)
    else:
        bb_max, bb_min, _, _ = normalize_obj_file(input_obj_file, output_obj_file, add_name=model_name)

    max_min_info_fname = os.path.join(model_dir, 'info.pkl')
    with open(max_min_info_fname, 'wb+') as fh:
        pickle.dump([bb_max, bb_min], max_min_info_fname)

    return output_obj_file, bb_max, bb_min

def load_max_min_info(model_dir):
    bb_max, bb_min = None, None 
    max_min_info_fname = os.path.join(model_dir, 'info.pkl')
    with open(max_min_info_fname, 'rb') as fh:
        bb_max, bb_min = pickle.load(max_min_info_fname)

    return bb_max, bb_min

def load_h5py_result(scene_dir, image_id):
    # coco_annos = json.load(open(os.path.join(scene_dir, 'coco_data', 'coco_annotations.json')))
    # for ann in coco_annos['images']:
        # image_id = ann['id']
    fh = h5py.File(os.path.join(scene_dir, '{}.hdf5'.format(image_id)), 'r')
    # object_mask_path = ann['mask_file_path']
    segcolormap = ast.literal_eval(np.array(fh.get('segcolormap')).tolist().decode('UTF-8'))