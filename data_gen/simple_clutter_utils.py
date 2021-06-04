import pybullet as p
import numpy as np 
import os 
from scipy.spatial.transform import Rotation as R
import math 
from dm_control.mujoco.engine import Camera
import matplotlib.pyplot as plt


import autolab_core
from shapely.geometry import Polygon

import matplotlib
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from xml.dom import minidom
import os
import trimesh
import dm_control.mujoco as mujoco
from pathlib import Path
import numpy as np
import shutil

def determine_object_rotation(object_mesh):
    # Rotate object so that it appears upright in Mujoco
    rot_vec = [(1/2)*np.pi, 0, 0]
    r = R.from_euler('xyz', rot_vec, degrees=False) 
    upright_mat = np.eye(4)
    upright_mat[0:3,0:3] = r.as_matrix()
    
    return rot_vec, upright_mat 

def determine_object_scale(obj_cat, mesh):
    object_bounds = mesh.bounds
    a,b,c = object_bounds[1] - object_bounds[0]
    len_x,len_y,len_z = None,None,None
    # import pdb; pdb.set_trace()
    if obj_cat == 2773838:
        # bag
        len_x = np.random.uniform(1,4,1)[0]
        len_y = np.random.uniform(1,4,1)[0]
        len_z = np.random.uniform(1,2,1)[0]
    elif obj_cat == 2876657:
        # bottle
        # maintain the neck part for wine-bottles
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(1.5,5,1)[0]
        len_z = len_x * np.random.uniform(0.8,1.7,1)[0]
    elif obj_cat == 2880940:
        # bowl
        # want to distinguish it from a cup 
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(0.1,0.8,1)[0]
        len_z = len_x *  np.random.uniform(0.8,1.7,1)[0]
    elif obj_cat == 2946921:
        # can
        # maintain the neck part for wine-bottles
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(1.5,5,1)[0]
        len_z = len_x * np.random.uniform(0.8,1.7,1)[0]
    elif obj_cat == 3046257:
        # clock
        # x,y cannot differ by too much
        # z gt is half
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(0.8,1.5,1)[0]
        len_z = len_x * np.random.uniform(0.4,1.5,1)[0]
    elif obj_cat == 3593526:
        # vase
        # similar to bottle
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(1.2,4,1)[0]
        len_z = len_x * np.random.uniform(0.8,1.7,1)[0]
    elif obj_cat == 3642806:
        # laptop
        # y and z should not be too different from x
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(0.8,1.5,1)[0]
        len_z = len_x * np.random.uniform(0.8,2,1)[0]
    elif obj_cat == 2942699:
        # cameras
        # x,y should be rectangular
        # z controls the length of the lens
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(0.8,2,1)[0]
        len_z = len_x * np.random.uniform(0.5,2,1)[0]
    elif obj_cat == 3797390:
        # mug 
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(1,2.5,1)[0]
        len_z = len_x * np.random.uniform(0.8,1.7,1)[0]
    elif obj_cat == 2801938:
        len_x = np.random.uniform(1,4,1)[0]
        len_y = len_x * np.random.uniform(0.5,2.5,1)[0]
        len_z = len_x * np.random.uniform(0.8,1.7,1)[0]
    # print(obj_cat, len_x, len_y, len_z)                
    scale_x = len_x / a
    scale_y = len_y / b
    scale_z = len_z / c
    scale_vec = [scale_x, scale_y, scale_z]
    matrix = np.eye(4)
    matrix[:3, :3] *= scale_vec
    return scale_vec, matrix
    


def draw_boundary_points_rect(corners4s, layout_filename=None):
    fig, ax = plt.subplots()
    for corners4 in corners4s:
        poly = patches.Polygon(corners4, linewidth=1, edgecolor='r', facecolor='g')
        ax.add_artist(poly)
        
    plt.xlim([-4, 4]) 
    plt.ylim([-4, 4]) 
    ax.set_aspect('equal')
    
    if not layout_filename:
        plt.show()
    else:
        plt.savefig(layout_filename) 
    plt.close()

def update_object_position_region(object_position_region, probs, selected_region):
    new_probs = [0] * len(probs)

    for i in range(len(probs)):
        if i == selected_region:
            new_probs[i] = probs[i] * (1/5)
        else:
            new_probs[i] = probs[i] + (probs[selected_region] * (4/5) * (1/7)) 

    return new_probs
    

def generate_object_xy(object_rot, object_z, object_bounds, prev_bbox, object_position_region, probs, scene_folder_path):
    ratio_lower = 0.4
    ratio_upper = 0.8
    avoid_all_squares = False 
    x,y = None, None
    selected_region = None 
    MAX_TRY = 50
    lower_x, upper_x = object_bounds[:,0]
    lower_y, upper_y = object_bounds[:,1]
    lower_z,_ = object_bounds[:,2]
    object_x_width, object_y_width, _ = object_bounds[1] - object_bounds[0]
    add_x = object_x_width * 0.1
    add_y = object_y_width * 0.1
    upper_x, upper_y = upper_x+add_x, upper_y+add_y
    lower_x, lower_y = lower_x-add_x, lower_y-add_y
    new_corners_3d = np.array([[lower_x, lower_y, lower_z], \
            [upper_x, lower_y, lower_z], \
            [upper_x, upper_y, lower_z], \
            [lower_x, upper_y, lower_z]]) #(4,3) --> (3,4)
    try_count = 0
    while not avoid_all_squares:
        if MAX_TRY < 0:
            raise ValueError
        
        prev_center_idx = np.random.choice(len(prev_bbox), 1)[0]
        center_corners = prev_bbox[prev_center_idx]
        x_bottom, y_bottom = np.min(center_corners, axis=0)
        x_top, y_top = np.max(center_corners, axis=0)
        object_position_region = {
            0: [[x_top, 1],[y_bottom, 1]],
            1: [[x_bottom, 1],[y_top,1]],
            2: [[x_bottom, -1],[y_bottom, 1]],
            3: [[x_bottom, 1],[y_bottom, -1]],
            4: [[x_top,1],[y_bottom, -1]],
            6: [[x_bottom, -1],[y_top,1]],
            5: [[x_top,1],[y_top,1]],
            7: [[x_bottom, -1],[y_bottom, -1]],
        }

        region = np.random.choice(8, 1, p=probs)[0]
        region_range = object_position_region[region]
        x_start,x_sign = region_range[0]
        y_start,y_sign = region_range[1]
        
        x_dist = np.random.uniform(object_x_width*ratio_lower, object_x_width*ratio_upper, 1)[0]
        y_dist = np.random.uniform(object_y_width*ratio_lower, object_y_width*ratio_upper, 1)[0]

        x = x_start + x_dist * x_sign
        y = y_start + y_dist * y_sign

        if len(prev_bbox) == 0:
            avoid_all_squares = True
            continue 
        selected_region = region
        
        object_xyz = [x, y, object_z]
        r2 = R.from_rotvec(object_rot)
        object_tf = autolab_core.RigidTransform(rotation = r2.as_matrix(), translation = np.asarray(object_xyz), from_frame='object_tmp', to_frame='world')
        pt_3d_homo = np.append(new_corners_3d.T, np.ones(4).astype('int').reshape(1,-1), axis=0) #(4,4)
        bounding_coord = object_tf.matrix @ pt_3d_homo #(4,4)
        bounding_coord = bounding_coord / bounding_coord[-1, :]
        bounding_coord = bounding_coord[:-1, :].T 
        new_corners = bounding_coord[:,:2]
        poly = Polygon(new_corners)

        # layout_filename = os.path.join(scene_folder_path, 'layout_idx-{}_try-{}.png'.format(len(prev_bbox),try_count))
        # draw_boundary_points_rect(prev_bbox + [new_corners], layout_filename)

        all_outside = True
        min_corner_dist = 1000
        dists = []
        for prev_idx, old_corners in enumerate(prev_bbox):
            old_poly = Polygon(old_corners)
            if old_poly.intersects(poly):
                # print("\n#1: try-{}".format(try_count), x, y)
                # print(x_start,x_sign, y_start,y_sign)
                # print("::", x_start + object_x_width*ratio_lower*x_sign, x_start + object_x_width*ratio_upper*x_sign)
                # print("::", y_start + object_y_width*ratio_lower*y_sign, y_start + object_y_width*ratio_upper*y_sign)
                all_outside = False
                break

            for corner in new_corners:
                dists.append(np.linalg.norm(old_corners - corner, axis=1))
        
        if all_outside:
            min_corner_dist = np.min(np.stack(dists))
            if min_corner_dist > 0.35:
                # print("\n#2: try-{}".format(try_count), x, y)
                # print(x_start,x_sign, y_start,y_sign)
                # print("::", x_start + object_x_width*ratio_lower*x_sign, x_start + object_x_width*ratio_upper*x_sign)
                # print("::", y_start + object_y_width*ratio_lower*y_sign, y_start + object_y_width*ratio_upper*y_sign)
                # print(np.min(np.stack(dists), axis=1), min_corner_dist)
                all_outside = False

        avoid_all_squares = all_outside
        MAX_TRY -= 1
        try_count += 1
    # new_probs = update_object_position_region(object_position_region, probs, selected_region)
    new_probs = probs
    return x,y,new_probs, object_tf, new_corners

def get_2d_diagonal_corners(obj_xyzs, all_obj_bounds):
    corners = []
    for xyz, bound in zip(obj_xyzs, all_obj_bounds):
        x,y,_ = xyz 
        a,b = bound[:,0], bound[:,1]
        diag_length = np.sqrt((a[1] - a[0]) ** 2 + (b[1] - b[0]) ** 2)
        diag_length = diag_length * (1/2)
        pts = np.array([[-diag_length, -diag_length],
                        [-diag_length, diag_length],
                        [diag_length, diag_length],
                        [diag_length, -diag_length]])
        
        pts = pts + np.array([x,y])
        corners.append(pts)
    
    return corners


def doOverlap(l1, r1, l2, r2, buf): 
      
    # If one rectangle is on left side of other 
    l1x, l1y = l1[0],l1[1]
    r1x, r1y = r1[0],r1[1]
    l2x, l2y = l2[0],l2[1]
    r2x, r2y = r2[0],r2[1]

    if(l1x >= r2x+buf or l2x >= r1x+buf): 
        return False
  
    # If one rectangle is above other 
    if(l1y >= r2y+buf or l2y >= r1y+buf): 
        return False
  
    return True

def move_object(e, ind, pos, rot):
    # ASSUME THERE IS TABLE so 7+ and 6+
    all_poses=e.data.qpos.ravel().copy()
    all_vels=e.data.qvel.ravel().copy()
    
    all_poses[7+7*ind : 7+7*ind+3]=pos
    all_poses[7+7*ind+3 : 7+7*ind+7]=rot
    
    all_vels[6+6*ind : 6+6*ind+6] = 0
    e.set_state(all_poses, all_vels)

def get_camera_position_occluded_one_cam(table_height, xyz1,xyz2,height1,height2,max_dist, deg_candidate):
    distance_away = 3
    
    x,y,z = xyz1
    a,b,c = xyz2
    xdiff = a-x
    ydiff = b-y
    rad = np.arctan(ydiff / xdiff) if xdiff > 0 else np.arctan(ydiff / xdiff)+np.pi
    shifted_degree = np.random.uniform(deg_candidate[0],deg_candidate[1],1)[0]
    #sign = np.random.choice([1,-1])
    rad += np.deg2rad(shifted_degree)
    
    cam_x = np.cos(rad) * (max_dist+distance_away) + x 
    cam_y = np.sin(rad) * (max_dist+distance_away) + y
    
    if height2 > height1:
        cam_z = table_height + height2
    else:
        cam_z = table_height + height2 / 2
    
    cam_xyz = [cam_x, cam_y, cam_z]
    jitter = [0,0,0]#np.random.uniform(0.2,0.5,2)
    cam_target = [x+jitter[0],y+jitter[1],z+np.random.uniform(0.1,0.2,1)[0]]

    return cam_xyz,cam_target

def get_pixel_left_ratio(scene_num, camera, cam_num, e, object_i, all_obj_indices, cam_width, cam_height):
    state = e.get_env_state().copy()
    segs = camera.render(segmentation=True)[:,:,0]
    occluded_geom_id_to_seg_id = {camera.scene.geoms[geom_ind][3]: camera.scene.geoms[geom_ind][8] for geom_ind in range(camera.scene.geoms.shape[0])}

    
    target_id = e.model.model.name2id(f'gen_geom_object_{object_i}_{scene_num}_0', "geom")
    segmentation = segs == occluded_geom_id_to_seg_id[target_id]
        
    # Move all other objects far away, except the table, so that we can capture
    # only one object in a scene.
    for move_obj_ind in all_obj_indices:
        if move_obj_ind != object_i:
            move_object(e, move_obj_ind, [20, 20, move_obj_ind], [0,0,0,0])

    e.sim.physics.forward()

    unocc_target_id = e.model.model.name2id(f'gen_geom_object_{object_i}_{scene_num}_0', "geom")
    unoccluded_camera = Camera(physics=e.model, height=cam_height, width=cam_width, camera_id=cam_num)
    unoccluded_segs = unoccluded_camera.render(segmentation=True)

    # Move other objects back onto table 
    e.set_env_state(state)
    e.sim.physics.forward()
        
    unoccluded_geom_id_to_seg_id = {unoccluded_camera.scene.geoms[geom_ind][3]: unoccluded_camera.scene.geoms[geom_ind][8] for geom_ind in range(unoccluded_camera.scene.geoms.shape[0])}
    unoccluded_segmentation = unoccluded_segs[:,:,0] == unoccluded_geom_id_to_seg_id[unocc_target_id]
    
    onoccluded_pixel_num = np.argwhere(unoccluded_segmentation).shape[0]
    # If the object is not in the scene of this object 
    if onoccluded_pixel_num == 0:
        return -1, 0, None
    
    segmentation = np.logical_and(segmentation, unoccluded_segmentation)
    pix_left_ratio = np.argwhere(segmentation).shape[0] / onoccluded_pixel_num

    return pix_left_ratio, onoccluded_pixel_num, segmentation

def get_camera_position_occluded(camera_distance, table_height, max_object_height, xyzs, heights):
    num_angles = 6
    shifted_degree = 10
    distance_away = 3

    N = len(xyzs)
    # xys = xyzs[:,:2]
    # pairwise_diff = np.repeat(xys[None,:,:], N, axis=0) - xys.reshape((-1,1,2)) #(N,N,2)
    # pairwise_dist = np.linalg.norm(pairwise_diff, axis=2) #(N,N)
    # pairwise_dist = np.max(pairwise_dist, axis=1) #(N,1)

    xys = []
    for object_idx,v in xyzs.items():
        xys.append([v[0],v[1]])
    xys = np.asarray(xys)

    cam_xyzs = dict()
    cam_targets = dict()
    cam_num = 0
    cam_num_to_occlusion_target = dict()
    for i,(x,y,z) in xyzs.items():
        
        pairwise_diff = xys - np.array([x,y]).reshape((1,2))
        dist = np.linalg.norm(pairwise_diff, axis=1)
        max_dist = np.max(dist)
        
        for j,(a,b,c) in xyzs.items():
            if i == j:
                continue 
            # dist = max_dist #np.linalg.norm([a-x,b-y]) 
                
            xdiff = a-x
            ydiff = b-y
            rad = np.arctan(ydiff / xdiff) if xdiff > 0 else np.arctan(ydiff / xdiff)+np.pi
            # shifted_degrees = np.random.choice([5,9,13,],10)
            
            deg_candidate = np.asarray([0,2])#np.asarray([3,5,10,15,20,25])
            shifted_degrees = []
            for degi,degj in zip(deg_candidate[:-1], deg_candidate[1:]):
                shifted_degrees.append(np.random.uniform(degi,degj,1)[0])
            
            for shifted_degree in shifted_degrees:#
                sign = np.random.choice([1,-1])
                rad += np.deg2rad(sign * shifted_degree)
                
                cam_x = np.cos(rad) * (max_dist+distance_away) + x 
                cam_y = np.sin(rad) * (max_dist+distance_away) + y
                
                if heights[j] > heights[i]:
                    cam_z = table_height + heights[j]
                else:
                    cam_z = table_height + heights[j] / 2
                
                cam_xyzs[cam_num] = [cam_x, cam_y, cam_z]
                jitter = np.random.uniform(0.2,0.5,2)
                cam_targets[cam_num] = [x+jitter[0],y+jitter[1],z+np.random.uniform(0.1,0.2,1)[0]]

                cam_num_to_occlusion_target[cam_num] = i
                cam_num += 1
    
    #bird eye view
    num_angles = 8
    quad = (2.0*math.pi) / num_angles
    normal_thetas = [np.random.uniform(i*quad, (i+1.0)*quad,1)[0] for i in range(num_angles)]
    
    center = np.mean(xys, axis=0)
    pairwise_diff = xys - center.reshape((1,2))
    dist = np.linalg.norm(pairwise_diff, axis=1)
    max_dist = np.max(dist)
    
    for theta in normal_thetas:
        cam_x = np.cos(theta) * (max_dist+distance_away) + center[0]
        cam_y = np.sin(theta) * (max_dist+distance_away) + center[1]
        cam_z = max_object_height * np.random.uniform(1,1.3,1)[0]
        cam_xyzs[cam_num] = [cam_x, cam_y, cam_z]
        cam_targets[cam_num] = [center[0],center[1],table_height]

        cam_num_to_occlusion_target[cam_num] = -1
        cam_num += 1
    
    return cam_xyzs, cam_targets, cam_num_to_occlusion_target

def get_camera_matrix(camera):
    camera_id = camera._render_camera.fixedcamid
    pos = camera._physics.data.cam_xpos[camera_id]
    rot = camera._physics.data.cam_xmat[camera_id].reshape(3, 3)
    fov = camera._physics.model.cam_fovy[camera_id]

    # # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * camera.height / 2.0
    import math
    f = 0.5 * camera.height / math.tan(fov * np.pi / 360)
    assert np.abs(f - focal_scaling) < 1e-3

    P = np.array(((focal_scaling, 0, camera.width / 2), (0, focal_scaling, camera.height / 2), (0, 0, 1)))
    camera_tf = autolab_core.RigidTransform(rotation = rot, translation = np.asarray(pos), from_frame='camera_{}'.format(camera_id), to_frame='world')

    assert np.all(np.abs(camera_tf.matrix @ np.array([0, 0, 0, 1]).reshape(4,-1) - np.array([[pos[0],pos[1],pos[2],1]]).reshape(4,-1)) < 1e-5)

    return P,camera_tf


def project_2d(P, camera_tf, pt_3d):
    '''
    pt_3d: (N,3)
    '''
    N = len(pt_3d)
    world_to_camera_tf_mat = camera_tf.inverse().matrix #(4,4)
    pt_3d_homo = np.append(pt_3d.T, np.ones(N).astype('int').reshape(1,-1), axis=0) #(4,N)
    pt_3d_camera = world_to_camera_tf_mat @ pt_3d_homo #(4,N)
    assert np.all(np.abs(pt_3d_camera[-1] - 1) < 1e-6)
    pixel_coord = P @ (pt_3d_camera[:-1, :])
    pixel_coord = pixel_coord / pixel_coord[-1, :]
    pixel_coord = pixel_coord[:2, :] #(2,N)
    return pixel_coord.astype('int').T


def add_light(scene_name, directional, ambient, diffuse, specular, castshadow, pos, dir, name):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('light')
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
    distance = np.random.uniform(1,1.2,1)[0]
    for light_id in range(num_lights):
        base_angle = light_angles[light_id]
        theta = np.random.uniform(base_angle - math.pi/9, base_angle + math.pi/9, 1)[0] 
        x = np.cos(theta) * distance 
        y = np.sin(theta) * distance 
        z = np.random.uniform(2,4,1)[0]
        pos.append([x,y,z])
        direction_z = np.random.uniform(-0.5,-1,1)[0]
        direction.append([0,0,direction_z])
    return pos, direction

def add_camera(scene_name, cam_name, cam_pos, cam_target, cam_id):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('camera')
    new_body.setAttribute('name', cam_name)
    new_body.setAttribute('mode', 'targetbody')
    new_body.setAttribute('pos', f'{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}')
    new_body.setAttribute('target', f'added_cam_target_{cam_id}')
    world_body.appendChild(new_body)
    
    new_body=xmldoc.createElement('body')
    new_body.setAttribute('name', f'added_cam_target_{cam_id}')
    new_body.setAttribute('pos', f'{cam_target[0]} {cam_target[1]} {cam_target[2]}')
    new_geom=xmldoc.createElement('geom')
    geom_name=f'added_cam_target_geom_{cam_id}'
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

def add_objects(scene_name, object_name, mesh_names, pos, size, color, rot, run_id):
    xmldoc = minidom.parse(scene_name)
    # import pdb; pdb.set_trace()
    
    assets = xmldoc.getElementsByTagName('asset')[0]
    for mesh_ind in range(len(mesh_names)):
        new_mesh=xmldoc.createElement('mesh')
        new_mesh.setAttribute('name', f'gen_mesh_{object_name}_{mesh_ind}')
        new_mesh.setAttribute('class', 'geom0')
        # new_mesh.setAttribute('class', 'geom')
        new_mesh.setAttribute('scale', f'{size[0]} {size[1]} {size[2]}')
        new_mesh.setAttribute('file', mesh_names[mesh_ind])
        assets.appendChild(new_mesh)
    
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('body')
    body_name=f'gen_body_{object_name}'
    new_body.setAttribute('name', body_name)
    new_body.setAttribute('pos', f'{pos[0]} {pos[1]} {pos[2]}')
    new_body.setAttribute('euler', f'{rot[0]} {rot[1]} {rot[2]}')
    
    geom_names=[]
    for geom_ind in range(len(mesh_names)):
        new_geom=xmldoc.createElement('geom')
        geom_name=f'gen_geom_{object_name}_{geom_ind}'
        geom_names.append(geom_name)
        new_geom.setAttribute('name', geom_name)
        new_geom.setAttribute('class', '/')
        new_geom.setAttribute('type', 'mesh')
        new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} 1')
        new_geom.setAttribute('mesh', f'gen_mesh_{object_name}_{geom_ind}')
        new_body.appendChild(new_geom)
    
    new_joint=xmldoc.createElement('joint')
    new_joint.setAttribute('name', f'gen_joint_{object_name}')
    new_joint.setAttribute('class', '/')
    new_joint.setAttribute('type', 'free')
    #new_joint.setAttribute('damping', '0.001')
    new_body.appendChild(new_joint)
    world_body.appendChild(new_body)
  
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)
    
    return body_name, geom_names
