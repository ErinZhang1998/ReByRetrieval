# import pybullet as p
import numpy as np
import copy
import shutil
import json
import pandas as pd
from scipy.spatial.transform import Rotation as R, rotation
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors

import autolab_core
from shapely.geometry import Polygon

import matplotlib
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from xml.dom import minidom
import os
import trimesh
# import dm_control.mujoco as mujoco
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image as PIL_Image

r = R.from_euler('xyz', [(1/2)*np.pi, 0, 0], degrees=False)
UPRIGHT_MAT = np.eye(4)
UPRIGHT_MAT[0:3, 0:3] = r.as_matrix()

def format_ply_file(input_ply_file):
    back_up = input_ply_file.split('.')[0] + '_backup' + '.ply'
    shutil.copyfile(input_ply_file, back_up)

    input_fid = open(back_up, 'rb')
    output_fid = open(input_ply_file, 'wb')

    for line in input_fid:
        try:
            line_decoded = line.decode('UTF-8')
            output_fid.write(line)
            if line_decoded.strip().startswith('element vertex'):
                new_line = 'element face 0\n'.encode('UTF-8')
                output_fid.write(new_line)
        except:
            output_fid.write(line)
    output_fid.close()
    input_fid.close()

def save_correct_size_model(
    model_save_root_dir, 
    model_name, 
    actual_size, 
    mesh_file_name, 
    turn_upright_before_scale = False,
    turn_upright_after_scale = False,
):
    '''
    Turn upright before: if the actual size is defined in terms of x,y,z
        otherwise, defined x,z,y
    Turn upright after is used for PERCH.
    Args:
        model_save_root_dir: 
            the "model_dir" directory used in Perch 
        model_name: 
            name of model used by Perch 
        actual_size: 
            the size (x,y,z) that the model will be scaled to 
        mesh_file_name: 
            path to model_normalized.obj file 
    
    Return:
        object_mesh:
            scaled object mesh, according to actual_size
        mesh_scale:
            list of 3 numbers representing scale
    '''
    model_save_dir = os.path.join(model_save_root_dir, model_name)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_fname = os.path.join(model_save_dir, 'textured.obj')
    model_ply_fname = os.path.join(model_save_dir, 'textured.ply')

    model_fname_backup = os.path.join(model_save_dir, 'textured_backup.obj')
    model_ply_fname_backup = os.path.join(model_save_dir, 'textured_backup.ply')

    if os.path.exists(model_fname) and not os.path.exists(model_fname_backup):
        shutil.copyfile(model_fname, model_fname_backup)
    
    if os.path.exists(model_ply_fname) and not os.path.exists(model_ply_fname_backup):
        shutil.copyfile(model_ply_fname, model_ply_fname_backup)

    print("Loading: ", mesh_file_name)
    object_mesh = trimesh.load(mesh_file_name, force='mesh')
    if turn_upright_before_scale:
        object_mesh.apply_transform(UPRIGHT_MAT)
    
    # scale the object_mesh to have the actual_size
    actual_size = np.asarray(actual_size).reshape(-1,)
    # print("object_mesh.bounds[1] - object_mesh.bounds[0]: ", object_mesh.bounds[1] - object_mesh.bounds[0])
    mesh_scale = actual_size / (object_mesh.bounds[1] - object_mesh.bounds[0])
    mesh_scale = list(mesh_scale)
    object_mesh = apply_scale_to_mesh(object_mesh, mesh_scale)
    if turn_upright_after_scale:
        object_mesh.apply_transform(UPRIGHT_MAT)
    
    if turn_upright_before_scale and turn_upright_before_scale:
        print("WARNING WARNING: turning ShapeNet meshes upright before and after scaling!")
    
    print("Exporting: ", model_fname)
    object_mesh.export(model_fname)
    copy_textured_mesh = o3d.io.read_triangle_mesh(model_fname)
    print("Exporting pointcloud: ", model_ply_fname)
    o3d.io.write_triangle_mesh(model_ply_fname, copy_textured_mesh)
    pcd = copy_textured_mesh.sample_points_uniformly(number_of_points=5000)
    o3d.io.write_point_cloud(os.path.join(model_save_dir, 'sampled.ply'), pcd)

    # # ## DEBUG
    # # from plyfile import PlyData, PlyElement
    # # cloud = PlyData.read(model_ply_fname).elements[0].data
    # # cloud = np.transpose(np.vstack((cloud['x'], cloud['y'], cloud['z'])))
    # cloud = o3d.io.read_point_cloud(model_ply_fname)
    # if np.asarray(cloud.points).shape[0] > 100000:
    #     # import pdb; pdb.set_trace()
    #     pcd = copy_textured_mesh.sample_points_uniformly(number_of_points=20000)
    #     # pcd = copy_textured_mesh.sample_points_poisson_disk(number_of_points=10000, pcl=pcd)
    #     os.remove(model_ply_fname)
    #     o3d.io.write_point_cloud(model_ply_fname, pcd)

    #     return None, mesh_scale

    return object_mesh, mesh_scale

def bounds_xyz_to_corners(bounds):
    '''
    bounds : (2,3)

    bbox: (8,3)
    '''
    lower, upper = bounds
    x0, y0, z0 = lower
    x1, y1, z1 = upper

    bbox = np.array(np.meshgrid([x0, x1], [y0, y1], [z0, z1])).reshape(3, -1).T
    return bbox


def determine_object_scale(obj_cat, mesh):
    object_bounds = mesh.bounds
    a, b, c = object_bounds[1] - object_bounds[0]
    len_x, len_y, len_z = None, None, None
    # import pdb; pdb.set_trace()
    if obj_cat == 2773838:
        # bag
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = np.random.uniform(1, 4, 1)[0]
        len_z = np.random.uniform(1, 2, 1)[0]
    elif obj_cat == 2876657:
        # bottle
        # maintain the neck part for wine-bottles
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(1.5, 5, 1)[0]
        len_z = len_x * np.random.uniform(0.8, 1.7, 1)[0]
    elif obj_cat == 2880940:
        # bowl
        # want to distinguish it from a cup
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(0.1, 0.8, 1)[0]
        len_z = len_x * np.random.uniform(0.8, 1.7, 1)[0]
    elif obj_cat == 2946921:
        # can
        # maintain the neck part for wine-bottles
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(1.5, 5, 1)[0]
        len_z = len_x * np.random.uniform(0.8, 1.7, 1)[0]
    elif obj_cat == 3046257:
        # clock
        # x,y cannot differ by too much
        # z gt is half
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(0.8, 1.5, 1)[0]
        len_z = len_x * np.random.uniform(0.4, 1.5, 1)[0]
    elif obj_cat == 3593526:
        # vase
        # similar to bottle
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(1.2, 4, 1)[0]
        len_z = len_x * np.random.uniform(0.8, 1.7, 1)[0]
    elif obj_cat == 3642806:
        # laptop
        # y and z should not be too different from x
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(0.8, 1.5, 1)[0]
        len_z = len_x * np.random.uniform(0.8, 2, 1)[0]
    elif obj_cat == 2942699:
        # cameras
        # x,y should be rectangular
        # z controls the length of the lens
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(0.8, 2, 1)[0]
        len_z = len_x * np.random.uniform(0.5, 2, 1)[0]
    elif obj_cat == 3797390:
        # mug
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(1, 2.5, 1)[0]
        len_z = len_x * np.random.uniform(0.8, 1.7, 1)[0]
    elif obj_cat == 2801938:
        len_x = np.random.uniform(1, 4, 1)[0]
        len_y = len_x * np.random.uniform(0.5, 2.5, 1)[0]
        len_z = len_x * np.random.uniform(0.8, 1.7, 1)[0]
    # print(obj_cat, len_x, len_y, len_z)
    scale_x = len_x / a
    scale_y = len_y / b
    scale_z = len_z / c
    scale_vec = [scale_x, scale_y, scale_z]
    matrix = np.eye(4)
    matrix[:3, :3] *= scale_vec
    return scale_vec, matrix


def move_object(mujoco_env, ind, pos, rot):
    all_poses = mujoco_env.data.qpos.ravel().copy()
    all_vels = mujoco_env.data.qvel.ravel().copy()

    all_poses[7*ind : 7*ind+3] = pos
    all_poses[7*ind+3 : 7*ind+7] = rot

    all_vels[6+6*ind: 6+6*ind+6] = 0
    mujoco_env.set_state(all_poses, all_vels)

    num_objects = all_poses.reshape(-1, 7).shape[0]
    # for _ in range(num_objects):
    #     for _ in range(10000):
    #         mujoco_env.model.step()
    return mujoco_env.data.qpos.ravel().copy().reshape(-1, 7)


def transform_3d_frame(mat, pts):
    '''
    pts : (N,3)
    mat : (4,4)
    '''
    pts_pad = np.append(pts.T, np.ones(len(pts)).astype(
        'int').reshape(1, -1), axis=0)  # (4,N)
    pts_other_frame = mat @ pts_pad
    pts_other_frame = pts_other_frame / pts_other_frame[-1, :]
    pts_other_frame = pts_other_frame[:-1, :]  # (3,N)
    pts_other_frame = pts_other_frame.T  # (N,3)
    return pts_other_frame


def quat_xyzw_to_wxyz(quat_xyzw):
    x,y,z,w = quat_xyzw
    return np.array([w,x,y,z])


def quat_wxyz_to_xyzw(quat_wxyz):
    w,x,y,z = quat_wxyz
    return np.array([x,y,z,w])


def euler_xyz_to_quat_wxyz(euler_xyz):
    rot_obj = R.from_euler('xyz', euler_xyz, degrees=False)
    x,y,z,w = rot_obj.as_quat()
    return [w,x,y,z]

def quat_xyzw_to_euler(quat):
    x,y,z,w = quat
    rot = R.from_quat(np.array([x,y,z,w]))
    return rot.as_euler('xyz', degrees=True)

def apply_scale_to_mesh(mesh, scale):
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale
    mesh.apply_transform(scale_matrix)
    return mesh 


def apply_rot_to_mesh(mesh, rot_obj):
    rotation_mat = np.eye(4)
    rotation_mat[0:3, 0:3] = rot_obj.as_matrix()
    mesh.apply_transform(rotation_mat)
    return mesh

def get_corners(bounds, pos, rot_obj, from_frame_name):
    '''
    Get the 8 3D object bounding box corners, in world-frame

    bounds: (2,3), min and max pts of the bounding box
    pos: (3,), object position in world frame
    rot_obj: scipy.spatial.transform object, object orientation in world frame
    from_frame_name: str
    '''
    # rot_obj = R.from_euler('xyz', euler_xyz, degrees=False)
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
    corners_obj = bounds_xyz_to_corners(bounds)
    corners_world = transform_3d_frame(obj_frame_to_world.matrix, corners_obj)
    return corners_obj, corners_world, obj_frame_to_world.matrix


def get_json_cleaned_matrix(mat, type='float'):
    '''
    Return json serializable matrix 
    ''' 
    try:
        mat[0]
        if len(mat.shape) == 1:
            if type == 'float':
                return [float(item) for item in mat]
            else:
                return [int(item) for item in mat]
        cleaned_mat = []
        for sub_mat in mat:
            cleaned_mat.append(get_json_cleaned_matrix(sub_mat))
        return cleaned_mat
    except:
        return mat

##############################################################################################################
class XmlObjectNameTracker(object):
    def __init__(self, scene_name):
        self.scene_name = scene_name
        self.light_names = {} # light_id --> name
        self.camera_names = {} # cam_num --> name
        self.camera_target_names = {} # cam_num --> (target_body_name, target_body_geom_name)
        self.object_mesh_names = {} # object_idx --> [mesh_names]
        self.object_body_names = {} # object_idx --> object_body_name
        self.object_geom_names = {} # object_idx --> [geom_names]
        self.object_joint_names = {} # object_idx --> joint_name
    
    def set_object_dicts(self, object_idx, body_name, added_mesh_names, geom_names, joint_name):
        self.object_mesh_names[object_idx] = added_mesh_names
        self.object_body_names[object_idx] = body_name
        self.object_geom_names[object_idx] = geom_names
        self.object_joint_names[object_idx] = joint_name

    def get_object_geom_name(self, object_idx):
        '''
        -1 --> table
        '''
        geom_names = self.object_geom_names[object_idx]
        if len(geom_names) == 1:
            return geom_names[0]
        else:
            return geom_names


def add_light(scene_name, directional, ambient, diffuse, specular, castshadow, pos, dir, name):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]

    new_body = xmldoc.createElement('light')
    new_body.setAttribute('name', name)
    new_body.setAttribute('pos', f'{pos[0]} {pos[1]} {pos[2]}')
    new_body.setAttribute('dir', f'{dir[0]} {dir[1]} {dir[2]}')
    directional_s = "true" if directional else "false"
    new_body.setAttribute('directional', directional_s)
    new_body.setAttribute('ambient', f'{ambient[0]} {ambient[1]} {ambient[2]}')
    new_body.setAttribute('diffuse', f'{diffuse[0]} {diffuse[1]} {diffuse[2]}')
    new_body.setAttribute('specular', f'{specular[0]} {specular[1]} {specular[2]}')
    castshadow_s = "true" if castshadow else "false"
    new_body.setAttribute('castshadow', castshadow_s)
    world_body.appendChild(new_body)

    with open(scene_name, "w") as f:
        xmldoc.writexml(f)


def get_light_pos_and_dir(num_lights):
    pos = []
    direction = []
    quad = (2.0*math.pi) / num_lights
    light_angles = np.arange(num_lights) * quad
    distance = np.random.uniform(1, 1.2, 1)[0]
    for light_id in range(num_lights):
        base_angle = light_angles[light_id]
        theta = np.random.uniform(base_angle - math.pi/9, base_angle + math.pi/9, 1)[0]
        x = np.cos(theta) * distance
        y = np.sin(theta) * distance
        z = np.random.uniform(2, 4, 1)[0]
        pos.append([x, y, z])
        direction_z = np.random.uniform(-0.5, -1, 1)[0]
        direction.append([0, 0, direction_z])
    return pos, direction


def add_camera(scene_name, cam_name, cam_pos, cam_target, cam_id):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]

    cam_target_body_name = f'added_cam_target_{cam_id}'
    geom_name = f'added_cam_target_geom_{cam_id}'

    new_body = xmldoc.createElement('camera')
    new_body.setAttribute('name', cam_name)
    new_body.setAttribute('mode', 'targetbody')
    new_body.setAttribute('fovy', '65')
    new_body.setAttribute('pos', f'{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}')
    new_body.setAttribute('target', cam_target_body_name)
    world_body.appendChild(new_body)

    new_body = xmldoc.createElement('body')
    new_body.setAttribute('name', cam_target_body_name)
    new_body.setAttribute('pos', f'{cam_target[0]} {cam_target[1]} {cam_target[2]}')
    
    new_geom = xmldoc.createElement('geom')
    new_geom.setAttribute('name', geom_name)
    new_geom.setAttribute('class', '/')
    new_geom.setAttribute('type', 'box')
    new_geom.setAttribute('contype', '0')
    new_geom.setAttribute('conaffinity', '0')
    new_geom.setAttribute('group', '1')
    new_geom.setAttribute('size', "1 1 1")
    new_geom.setAttribute('rgba', f'0 0 0 0')
    new_body.appendChild(new_geom)
    world_body.appendChild(new_body)

    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

    return cam_name, (cam_target_body_name, geom_name)


def add_objects(scene_name, object_info, material_name=None):
    # scene_name = object_info['scene_name']
    object_name = object_info['object_name']
    mesh_names = object_info['mesh_names']
    pos = object_info['pos']
    size = object_info['size']
    color = object_info['color']
    quat = object_info['quat']

    xmldoc = minidom.parse(scene_name)

    assets = xmldoc.getElementsByTagName('asset')[0]
    added_mesh_names = []
    for mesh_ind in range(len(mesh_names)):
        new_mesh = xmldoc.createElement('mesh')
        mesh_name = f'gen_mesh_{object_name}_{mesh_ind}'
        added_mesh_names.append(mesh_name)
        new_mesh.setAttribute('name', mesh_name)
        new_mesh.setAttribute('class', 'geom0')
        new_mesh.setAttribute('scale', f'{size[0]} {size[1]} {size[2]}')
        new_mesh.setAttribute('file', mesh_names[mesh_ind])
        assets.appendChild(new_mesh)

    world_body = xmldoc.getElementsByTagName('worldbody')[0]

    new_body = xmldoc.createElement('body')
    body_name = f'gen_body_{object_name}'
    new_body.setAttribute('name', body_name)
    new_body.setAttribute('pos', f'{pos[0]} {pos[1]} {pos[2]}')
    new_body.setAttribute('quat', f'{quat[0]} {quat[1]} {quat[2]} {quat[3]}')
    if object_info.get('mocap', False):
        new_body.setAttribute('mocap', 'true')

    geom_names = []
    for geom_ind in range(len(mesh_names)):
        geom_name = f'gen_geom_{object_name}_{geom_ind}'
        if object_info.get('site', False):
            new_geom = xmldoc.createElement('site')

            new_geom.setAttribute('type', object_info.get('type', 'box'))
            site_sizes = object_info.get('site_sizes', [1,1,1])
            new_geom.setAttribute('size', f'{site_sizes[0]} {site_sizes[1]} {site_sizes[2]}')
        else:
            new_geom = xmldoc.createElement('geom')
            new_geom.setAttribute('type', 'mesh')
        
        geom_names.append(geom_name)
        new_geom.setAttribute('name', geom_name)
        # new_geom.setAttribute('mass', '1')
        new_geom.setAttribute('class', '/')
        
        if not material_name is None:
            new_geom.setAttribute('material', material_name)
        if material_name is None:
            if len(color) == 3:
                new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} 1')
            else:
                new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} {color[3]}')
        
        if not object_info.get('site', False):
            new_geom.setAttribute('mesh', f'gen_mesh_{object_name}_{geom_ind}')

        new_body.appendChild(new_geom)

    joint_name = f'gen_joint_{object_name}'
    if not object_info.get('mocap', False) and not object_info.get('site', False):
        new_joint = xmldoc.createElement('joint')
        new_joint.setAttribute('name', joint_name)
        new_joint.setAttribute('class', '/')
        new_joint.setAttribute('type', 'free')
        #new_joint.setAttribute('damping', '0.001')
        new_body.appendChild(new_joint)
    world_body.appendChild(new_body)

    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

    return body_name, added_mesh_names, geom_names, joint_name


def add_texture(scene_name, texture_name, filename, texture_type="cube"):
    '''
    <texture name="concretecube" type="cube" file="concrete-7baa937dd9eb091794feb091c728eb4f234150ae.png"/>
    <material name="concrete_2d" class="/" texture="concrete_2d" reflectance="0.3" rgba="0.8 0.8 0.8 1"/>
    '''
    xmldoc = minidom.parse(scene_name)

    assets = xmldoc.getElementsByTagName('asset')[0]

    new_texture = xmldoc.createElement('texture')
    new_texture.setAttribute('name', texture_name)
    new_texture.setAttribute('type', texture_type)
    new_texture.setAttribute('file', filename)
    assets.appendChild(new_texture)

    new_material = xmldoc.createElement('material')
    new_material.setAttribute('name', texture_name)
    new_material.setAttribute('class', "/")
    new_material.setAttribute('texture', texture_name)
    # new_material.setAttribute('reflectance', "0.5")
    # new_material.setAttribute('rgba', "1 1 1 1")
    assets.appendChild(new_material)

    with open(scene_name, "w") as f:
        xmldoc.writexml(f)


############################################################

all_colors_dict = mcolors.CSS4_COLORS  # TABLEAU_COLORS #
ALL_COLORS = []
excluded_color_names = ['black', 'midnightblue',
                        'darkslategray', 'darkslategrey', 'dimgray', 'dimgrey']
for name, color in all_colors_dict.items():
    c = mcolors.to_rgb(color)
    if c[0] > 0.8 and c[1] > 0.8 and c[2] > 0.8:
        continue
    if name in excluded_color_names:
        continue
    ALL_COLORS.append(np.asarray(mcolors.to_rgb(color)))

def get_convex_decomp_mesh(mesh_file_name, convex_decomp_dir, synset_id, model_id=None):
    import pybullet as p

    convex_decomp_synset_dir = os.path.join(convex_decomp_dir, synset_id)
    if not os.path.exists(convex_decomp_synset_dir):
        os.mkdir(convex_decomp_synset_dir)
    if model_id is not None:
        obj_convex_decomp_dir = os.path.join(convex_decomp_dir, f'{synset_id}/{model_id}')
        if not os.path.exists(obj_convex_decomp_dir):
            os.mkdir(obj_convex_decomp_dir)
    else:
        obj_convex_decomp_dir = convex_decomp_synset_dir
    
    obj_convex_decomp_fname = os.path.join(obj_convex_decomp_dir, 'convex_decomp.obj')
    if not os.path.exists(obj_convex_decomp_fname):
        name_log = os.path.join(obj_convex_decomp_dir, 'convex_decomp_log.txt')
        p.vhacd(mesh_file_name, obj_convex_decomp_fname, name_log, alpha=0.04,resolution=50000)
        
    assert os.path.exists(obj_convex_decomp_fname)
    return trimesh.load(obj_convex_decomp_fname, force='mesh'), obj_convex_decomp_dir


def get_convex_decomp_mesh_csv(csv_fname, shapenet_filepath, shapenet_convex_decomp_dir):
    df = pd.read_csv(csv_fname)
    for i in range(len(df)):
        row = df.iloc[i]
        synset_id = row['synsetId']
        model_id = row['ShapeNetModelId']
        mesh_fname = os.path.join(
            shapenet_filepath,
            '0{}/{}/models/model_normalized.obj'.format(synset_id, model_id),
        )
        comb_mesh, obj_convex_decomp_dir = get_convex_decomp_mesh(
            mesh_fname, 
            shapenet_convex_decomp_dir, 
            f'0{synset_id}', 
            model_id=model_id,
        )


def create_walls(inner_pts, outer_pts, bottom_height=0):
    '''
    Create 4 walls such that the space between inner_pts and outer_pts are filled.

    inner_pts: (4,3)
    outer_pts: (4,3)
    '''
    E,F,G,H = inner_pts
    A,B,C,D = outer_pts 
    wall_height = 1
    lx_room = 0
    ly_room = 0

    wall_infos = dict()
    
    x0 = np.mean([A,B], axis=0)[0]
    assert np.abs(np.mean([A,E], axis=0)[1] - np.mean([B,F], axis=0)[1]) < 1e-3 
    y0 = np.mean([A,E], axis=0)[1]
    lx = np.abs(E[0] - F[0]) + 0.1
    ly = np.abs(E[1] - A[1]) - ly_room
    lz = wall_height
    pos0 = [x0,y0,bottom_height+lz*0.5]
    wall_infos[0] = [pos0, [0,0,0,0], [lx,ly,lz]]

    x1 = np.mean([C,D], axis=0)[0]
    assert np.abs(np.mean([C,G], axis=0)[1] - np.mean([H,D], axis=0)[1]) < 1e-3 
    y1 = np.mean([C,G], axis=0)[1]
    lx = np.abs(G[0] - H[0]) + 0.1
    ly = np.abs(C[1] - G[1]) - ly_room
    lz = wall_height
    pos1 = [x1,y1,bottom_height+lz*0.5]
    wall_infos[1] = [pos1, [0,0,0,0], [lx,ly,lz]]
    
    y2 = np.mean([A,C], axis=0)[1]
    assert np.abs(np.mean([A,E], axis=0)[0] - np.mean([C,G], axis=0)[0]) < 1e-3 
    x2 = np.mean([C,G], axis=0)[0]
    lx = np.abs(A[0] - E[0]) - lx_room
    ly = np.abs(A[1] - C[1]) - ly_room
    lz = wall_height
    pos2 = [x2,y2,bottom_height+lz*0.5]
    wall_infos[2] = [pos2, [0,0,0,0], [lx,ly,lz]]
    
    y3 = np.mean([B,D], axis=0)[1]
    assert np.abs(np.mean([H,D], axis=0)[0] - np.mean([B,F], axis=0)[0]) < 1e-3 
    x3 = np.mean([H,D], axis=0)[0]
    lx = np.abs(B[0] - F[0]) - lx_room
    ly = np.abs(B[1] - D[1]) - ly_room
    lz = wall_height
    pos3 = [x3,y3,bottom_height+wall_height*0.5]
    wall_infos[3] = [pos3, [0,0,0,0], [lx,ly,lz]]

    return wall_infos


def generate_default_add_position(object_idx, num_objects, distance_away=50):
    theta = ((2.0*math.pi) / num_objects) * object_idx
    start_x = np.cos(theta) * distance_away
    start_y = np.sin(theta) * distance_away
    start_z = object_idx + 1
    return [start_x, start_y, start_z]

def output_json(json_dict, path):
    json_string = json.dumps(json_dict)
    json_file = open(path, "w+")
    json_file.write(json_string)
    json_file.close()


def from_perch_cam_annotations_to_world_frame(position, quaternion_xywz, image_ann):
    gt_to_perch_cam = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    perch_gt_to_cam = np.linalg.inv(gt_to_perch_cam)
    
    cam_to_world = image_ann['camera_frame_to_world_frame_mat']
    cam_to_world = np.asarray(cam_to_world)
    object_position_perch = np.asarray(position)
    object_position_world = transform_3d_frame(perch_gt_to_cam, object_position_perch.reshape(-1,3))
    object_position_world = transform_3d_frame(cam_to_world, object_position_world)
    
    
    object_quat_perch = np.array(quaternion_xywz)
    perch_gt_mat = np.zeros((4,4))
    perch_gt_mat[:3,:3] = R.from_quat(object_quat_perch).as_matrix()
    perch_gt_mat[:,3] = list(object_position_perch) + [1]

    cam_mat = perch_gt_to_cam @ perch_gt_mat 
    perch_gt_to_world = cam_to_world @ cam_mat
    quat_world = R.from_matrix(perch_gt_to_world[:3,:3]).as_quat().reshape(4,)
    
    return object_position_world.reshape(-1,), quat_world.reshape(-1,)

def from_world_frame_annotations_to_perch_cam(position, quaternion_xywz, image_ann):
    gt_to_perch_cam = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    
    world_to_cam = image_ann['world_frame_to_camera_frame_mat']
    world_to_cam = np.asarray(world_to_cam)

    object_position = np.asarray(position)
    object_position_cam = transform_3d_frame(world_to_cam, object_position.reshape(-1,3))
    object_position_cam = transform_3d_frame(gt_to_perch_cam, object_position_cam)
    
    object_quat_world = np.array(quaternion_xywz)

    object_to_world_mat = np.zeros((4,4))
    object_to_world_mat[:3,:3] = R.from_quat(object_quat_world).as_matrix()
    object_to_world_mat[:,3] = list(object_position) + [1]

    object_to_cam_mat = world_to_cam @ object_to_world_mat 
    object_to_cam_mat = gt_to_perch_cam @ object_to_cam_mat
    new_quat_cam = R.from_matrix(object_to_cam_mat[:3,:3]).as_quat().reshape(4,)
    
    return object_position_cam.reshape(-1,), new_quat_cam.reshape(-1,)
        
### copying from the perch object to do rotation sampling 
def get_rotation_samples(num_samples, perch_rot_angle=None):
    from dipy.core.geometry import cart2sphere, sphere2cart
    num_pts = num_samples * 2
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    locations = np.asarray([x, y, z]).T
    locations = locations[locations[:,2] >= 0]
    
    all_rots = []
    # for viewpoint in locations:
    #     r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
    #     theta = math.pi/2 - theta
    #     xyz_rotation_angles = [-phi, theta, 0] # euler xyz
    #     all_rots.append(xyz_rotation_angles)
    for viewpoint in locations:
        r, theta, phi = cart2sphere(viewpoint[0], viewpoint[1], viewpoint[2])
        ## sphere2euler : convert_fat_coco 
        theta = math.pi/2 - theta

        if perch_rot_angle == 0:
            xyz_rotation_angles = [-phi, theta, 0] # euler xyz
            all_rots.append(xyz_rotation_angles)
        elif perch_rot_angle == 1:
            step_size = math.pi/2
            for yaw_temp in np.arange(0,math.pi, step_size):
                xyz_rotation_angles = [-phi, yaw_temp, theta]
                # xyz_rotation_angles = [yaw_temp, -phi, theta]
                all_rots.append(xyz_rotation_angles)
        elif perch_rot_angle == 2:
            step_size = math.pi/4
            for yaw_temp in np.arange(0,math.pi, step_size):
                xyz_rotation_angles = [-phi, yaw_temp, theta]
                # xyz_rotation_angles = [yaw_temp, -phi, theta]
                all_rots.append(xyz_rotation_angles)
        elif perch_rot_angle == 3:
            xyz_rotation_angles = [-phi, 0, theta]
            all_rots.append(xyz_rotation_angles)
            # xyz_rotation_angles = [-phi, math.pi/2, theta]
            # all_rots.append(xyz_rotation_angles)
            xyz_rotation_angles = [-phi, 2*math.pi/3, theta]
            all_rots.append(xyz_rotation_angles)
        elif perch_rot_angle == 4:
            # For upright sugar box
            xyz_rotation_angles = [-phi, math.pi+theta, 0]
            all_rots.append(xyz_rotation_angles)
        elif perch_rot_angle == 5:
            xyz_rotation_angles = [phi, theta, math.pi]
            all_rots.append(xyz_rotation_angles)
        elif perch_rot_angle == 6:
            # This causes sampling of inplane along z
            xyz_rotation_angles = [-phi, 0, theta]
            all_rots.append(xyz_rotation_angles)
            xyz_rotation_angles = [-phi, math.pi/3, theta]
            all_rots.append(xyz_rotation_angles)
            xyz_rotation_angles = [-phi, 2*math.pi/3, theta]
            all_rots.append(xyz_rotation_angles)
        elif perch_rot_angle == 7:
            # This causes sampling of inplane along z
            # xyz_rotation_angles = [-phi, 0, theta]
            # all_rots.append(xyz_rotation_angles)
            # xyz_rotation_angles = [-phi, math.pi/3, theta]
            # all_rots.append(xyz_rotation_angles)
            # xyz_rotation_angles = [-phi, 2*math.pi/3, theta]
            # all_rots.append(xyz_rotation_angles)
            # xyz_rotation_angles = [-phi, math.pi, theta]
            # all_rots.append(xyz_rotation_angles)
            step_size = math.pi/2
            for yaw_temp in np.arange(0, 2*math.pi, step_size):
                xyz_rotation_angles = [-phi, yaw_temp, theta]
                # xyz_rotation_angles = [yaw_temp, -phi, theta]
                all_rots.append(xyz_rotation_angles)
        elif perch_rot_angle == 8:
            step_size = math.pi/3
            for yaw_temp in np.arange(0, math.pi, step_size):
                xyz_rotation_angles = [yaw_temp, -phi, theta]
                all_rots.append(xyz_rotation_angles)
    
    return all_rots
    
    
    
    
    from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points_with_sym_metric
    from dipy.core.geometry import cart2sphere, sphere2cart

    half_or_whole = half_or_whole if not half_or_whole is None else NAME_SYM_DICT[label][0]
    perch_rot_angle = perch_rot_angle if not perch_rot_angle is None else NAME_SYM_DICT[label][1]

    # rotation hypothesis in euler angles
    all_rots = []
    
    viewpoints_xyz = sphere_fibonacci_grid_points_with_sym_metric(num_samples, half_or_whole)
    
    return all_rots
