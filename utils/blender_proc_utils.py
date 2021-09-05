import re 
import os 
import json
import ast 
import numpy as np 
import trimesh
import h5py
import cv2
import yaml
import shutil
import pickle

import pycocotools.mask as coco_mask
from pycocotools.coco import COCO

from scipy.spatial.transform import Rotation as R, rotation    
import utils.datagen_utils as datagen_utils

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

    # normalize_obj_file(input_obj_file, output_obj_file, add_name=model_name)
    if actual_size_used:
        x,y,z = ann['actual_size']
        bb_max, bb_min, _, _ = scale_obj_file(input_obj_file, output_obj_file, np.array([x,z,y]), add_name=model_name)
    else:
        bb_max, bb_min, _, _ = normalize_obj_file(input_obj_file, output_obj_file, add_name=model_name)

    max_min_info_fname = os.path.join(model_dir, 'info.pkl')
    with open(max_min_info_fname, 'wb+') as fh:
        pickle.dump([bb_max, bb_min], fh)

    return output_obj_file, bb_max, bb_min



def load_max_min_info(model_dir):
    bb_max, bb_min = None, None 
    max_min_info_fname = os.path.join(model_dir, 'info.pkl')
    with open(max_min_info_fname, 'rb') as fh:
        bb_max, bb_min = pickle.load(fh)

    return bb_max, bb_min

def rleToMask(rleNumbers, height, width):
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(height*width,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(width,height)
    img = img.T
    return img

def from_yaml_to_object_information(yami_file_obj, df):
    object_from_yaml = {}
    for module in yami_file_obj['modules']:
        if module['module'] == 'loader.ObjectLoader':
            if 'cp_shape_net_table' in module['config']['add_properties']:
                continue
            
            model_name = module['config']['add_properties']['cp_model_name']
            model_info = {}
            if 'path' in module['config']:
                path = module['config']['path']
            else:
                path = module['config']['paths'][0]
            
            model_info = {
                'category_id' : module['config']['add_properties']['cp_category_id'],
                'path' : path,
            }
            
            if model_name.split('_')[-2] == 'object':
                synset_id = path.split('/')[-4]
                model_id = path.split('/')[-3]
                obj_cat = df[df['ShapeNetModelId'] == model_id].catId.values[0]
                obj_id = df[df['ShapeNetModelId'] == model_id].objId.values[0]
                
                model_info.update({
                    'synset_id' : synset_id,
                    'model_id' : model_id,
                    'obj_cat' : obj_cat,
                    'obj_id' : obj_id,
                })
                
            object_from_yaml[model_name] = model_info

        if module['module'] == 'manipulators.EntityManipulator':
            if 'scale' not in module['config']:
                continue
            model_name = module['config']['selector']['conditions']['cp_model_name']
            model_info = object_from_yaml.get(model_name, {})
            # WARNING
            model_info.update({
                'scale' : module['config']['scale'][0]
            })
    
    model_info_category_id = {}
    for model_name, model_info in object_from_yaml.items():
        model_info_category_id[int(model_info['category_id'])] = model_info
    return model_info_category_id


def bbox_to_bbox_2d_and_center(bbox):
    xmin, ymin, xleng, yleng = bbox
    ymax = ymin + yleng
    xmax = xmin + xleng
    bbox_2d = [[xmin, ymin],[xmax, ymax]]
    center = [int((xmin + xmax) * 0.5), int((ymin + ymax) * 0.5)]
    return bbox_2d, center


def to_old_annotaiton_format(
    perch_model_dir, 
    yaml_file_root_dir, 
    df, 
    one_scene_dir,
    storage_root = None,
    train_or_test = None,
    scene_num = None,
):
    '''
    Args:
        blender_proc_model_dir: use for save_correct_size_model
            where Perch will be reading the model from  
        yaml_file_root_dir: where to find the yaml_files that are used to produce the blender proc data
        df: mainly used in from_yaml_to_object_information
            Use to label obj_cat and obj_id
        one_scene_dir: 
    '''
    
    # model_dir = os.path.join(blender_proc_model_dir, 'perch_output_models')
    # if not os.path.exists(model_dir):
    #     os.mkdir(model_dir)
    scene_dir_split_list = one_scene_dir.split('/')

    root_dir = '/'.join(scene_dir_split_list[:-2])

    if train_or_test is None:
        train_or_test = scene_dir_split_list[-2]
    if scene_num is None:
        scene_num = int(scene_dir_split_list[-1].split('_')[-1])
    
    scene_name = f'{train_or_test}_scene_{scene_num:06}'
    
    # yaml_file_prefix = '_'.join(one_scene_dir.split('/')[-2:])
    yaml_file = os.path.join(yaml_file_root_dir, '{}.yaml'.format(scene_name))
    yaml_file_obj = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
    datagen_yaml = from_yaml_to_object_information(yaml_file_obj, df)

    if not os.path.exists(os.path.join(one_scene_dir, 'coco_data')):
        return 

    coco_fname = os.path.join(one_scene_dir, 'coco_data', 'coco_annotations.json')
    coco_anno = json.load(open(coco_fname))
    coco = COCO(coco_fname)
    
    image_id_to_h5py_fh = {}
    image_id_to_fname = {}
    images_ann_new = []
    for image_ann in coco_anno['images']:
        image_id = image_ann['id']
        h5py_fh = image_id_to_h5py_fh.get(image_id, None)
        if h5py_fh is None:
            h5py_file_name = os.path.join(one_scene_dir, '{}.hdf5'.format(image_id))
            if not os.path.exists(h5py_file_name):
                continue
            h5py_fh = h5py.File(h5py_file_name, 'r')
            image_id_to_h5py_fh[image_id] = h5py_fh
        cam_pose_dict = ast.literal_eval(np.array(h5py_fh.get('campose')).tolist().decode('UTF-8'))[0]
        
        image_ann_new = {}
        image_ann_new.update(image_ann)

        fname = os.path.join(train_or_test, f'scene_{scene_num:06}', image_ann['file_name']) 
        #os.path.join(*one_scene_dir.split('/')[-2:], image_ann['file_name'])
        
        image_id_to_fname[image_id] = fname
        intrinsics_matrix = cam_pose_dict['cam_K']
        camera_frame_to_world_frame_mat = np.asarray(cam_pose_dict['cam2world_matrix'])
        rot_quat = R.from_matrix(camera_frame_to_world_frame_mat[:3,:3]).as_quat()
        rot_quat = datagen_utils.get_json_cleaned_matrix(rot_quat)
        world_frame_to_camera_frame_mat = np.linalg.inv(camera_frame_to_world_frame_mat)
        
        #missing: all_object_bbox, all_object_segmentation_path, all_object_with_table_segmentation_path
        image_ann_new.update({
            'file_name' : fname,
            'intrinsics_matrix' : intrinsics_matrix,
            'pos' : cam_pose_dict['location'],
            'rot_quat' : rot_quat,
            'camera_frame_to_world_frame_mat' : cam_pose_dict['cam2world_matrix'],
            'world_frame_to_camera_frame_mat' : datagen_utils.get_json_cleaned_matrix(world_frame_to_camera_frame_mat),
        })
        images_ann_new.append(image_ann_new)
    
    '''
    object_state, records object position and orientation information: 
    {
        'customprop_model_name': 'testing_set_scene_0_object_0',
        'customprop_category_id': 1,
        'name': '02946921_f6316c6702c49126193d9e76bb15876',
        'location': [-0.08852141350507736, 0.2833922207355499, 1.1032110452651978],
        'rotation_euler': [1.5707963705062866, 0.0, 6.015883445739746],
        'matrix_world': [
            [0.19909153878688812,-2.383245822912272e-09, -0.0545223131775856,-0.08852141350507736],
            [-0.0545223131775856,-8.702567555474161e-09,-0.19909153878688812,0.2833922207355499],
            [0.0, 0.20642219483852386, -9.023001013019893e-09, 1.1032110452651978],
            [0.0, 0.0, 0.0, 1.0]]
    }
    '''
    category_id_to_object_state = {}
    for image_id, h5py_fh in image_id_to_h5py_fh.items():
        object_states = ast.literal_eval(np.array(h5py_fh.get('object_states')).tolist().decode('UTF-8'))
        for ann in object_states:
            if ann['customprop_model_name'].split('_')[-1] == 'table':
                continue
            category_id = int(ann['customprop_category_id'])
            object_state = category_id_to_object_state.get(category_id, None)
            
            if object_state is None:
                category_id_to_object_state[category_id] = ann
        
        # save depth image
        depth = np.array(h5py_fh.get('depth'))
        depth_scaled = depth * 1000
        depth_scaled[depth_scaled > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
        depth_save_dir = image_id_to_fname[image_id].replace('rgb', 'depth')
        depth_save_dir = os.path.join(root_dir, depth_save_dir)
        # print("Save depth: ", depth_save_dir)
        cv2.imwrite(depth_save_dir, depth_scaled.astype(np.uint16))
    
    categories_ann_new = []
    category_id_to_model_name = {}
    
    failed_category = []
    failed_model_name = []
    
    for category_ann in coco_anno['categories']:
        category_id = category_ann['id'] # this is used during blender proc
        
        if category_id not in datagen_yaml:
            continue 
        if 'scale' not in datagen_yaml[category_id]:
            continue
        datagen_yaml_info = datagen_yaml[category_id]
        object_state = category_id_to_object_state[category_id]
        
        mesh_file_name = datagen_yaml_info['path']
        file_storage_root = mesh_file_name[:mesh_file_name.index('/xiaoyuz1')]
        mesh_file_name = mesh_file_name.replace(file_storage_root, storage_root)
        
        category_ann_new = {}
        category_ann_new.update(category_ann)

        bb_max, bb_min = load_max_min_info('/'.join(mesh_file_name.split('/')[:-2]))
        model_name = '_'.join([train_or_test, f'scene_{scene_num}', f'object_{category_id}'])
        category_id_to_model_name[category_id] = model_name
        scale = datagen_yaml_info['scale']

        actual_size = (bb_max - bb_min) * np.array([scale] * 3)
        returned_mesh, _ = datagen_utils.save_correct_size_model(
            perch_model_dir, 
            model_name, 
            actual_size, 
            mesh_file_name, 
            turn_upright_before_scale = False,
            turn_upright_after_scale = True,
        )

        if returned_mesh is None:
            failed_category.append(category_id)
            failed_model_name += [model_name]

        #object_frame_to_world_frame_mat = np.asarray(object_state['matrix_world'])
        #object_frame_to_world_frame_mat /= scale
        #rot_quat = R.from_matrix(object_frame_to_world_frame_mat[:3,:3]).as_quat()
        
        rot_quat = R.from_euler('xyz', np.asarray(object_state['rotation_euler'])).as_quat()
        rot_quat = datagen_utils.get_json_cleaned_matrix(rot_quat)
        
        # actual_size is a must have 
        category_ann_new.update({
            'name' : model_name,
            'shapenet_category_id' : int(datagen_yaml_info['obj_cat']),
            'shapenet_object_id' : int(datagen_yaml_info['obj_id']),
            'synset_id' : datagen_yaml_info['synset_id'],
            'model_id' : datagen_yaml_info['model_id'],
            'size' : [float(scale)] * 3,
            'actual_size' : [float(item) for item in actual_size],
            'position' : [float(item) for item in object_state['location']],
            'quat' : [float(item) for item in rot_quat],
            'half_or_whole' : 0,
            'perch_rot_angle' : 0,
        })
        categories_ann_new.append(category_ann_new)
    
    annotations_new = []
    for anno in coco_anno['annotations']:
        if anno['category_id'] not in category_id_to_model_name:
            continue
        anno_new = {}
        anno_new.update(anno)

        category_id = anno['category_id']
        image_id = anno['image_id']

        if category_id in failed_category:
            continue

        _, center = bbox_to_bbox_2d_and_center(anno['bbox'])
        anno_new.update({
            'center' : [float(item) for item in center],
            'object_idx' : int(anno['category_id']-1),
            'model_name' : category_id_to_model_name[anno['category_id']],
            'percentage_not_occluded' : None,
            'number_pixels' : anno['area'],
            'mask_file_path' : None,
        }) 

        # Update mask to get rid of outliers 
        h5py_fh = image_id_to_h5py_fh[anno_new['image_id']]
        depth = np.array(h5py_fh.get('depth'))
        object_mask = coco.annToMask(anno)
        masked_depth = depth * object_mask # depth of the objects
        masked_depth_above_zero = masked_depth[masked_depth > 0]
        median_depth = np.median(masked_depth_above_zero)
        min_depth = max(0, median_depth-0.5)
        max_depth = median_depth+0.5
        indices = np.vstack(np.where(masked_depth < min_depth))
        object_mask[tuple(indices)] = 0

        indices = np.vstack(np.where(masked_depth > max_depth))
        object_mask[tuple(indices)] = 0

        seg_save_path = f'coco_data/segmentation_{image_id}_{category_id}.png'
        
        seg_save_path_full = os.path.join(one_scene_dir, seg_save_path)
        # print("Saving segmentation: ", seg_save_path_full)
        cv2.imwrite(seg_save_path_full, object_mask.astype(np.uint8))

        anno_new.update({
            'area': int(np.sum(object_mask)),
            'segmentation' : None,
            'mask_file_path' : os.path.join(train_or_test, f'scene_{scene_num:06}', seg_save_path),
        })
        annotations_new.append(anno_new)
    
    import copy
    json_dict = {
        'info' : copy.deepcopy(coco_anno['info']),
        'licenses' : copy.deepcopy(coco_anno['licenses']),
        'categories' : categories_ann_new,
        'images' : images_ann_new,
        'annotations' : annotations_new,
    }
    path = os.path.join(one_scene_dir, 'annotations.json')
    datagen_utils.output_json(json_dict, path)

    return failed_model_name


