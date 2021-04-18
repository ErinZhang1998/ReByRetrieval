import pybullet as p
import numpy as np 
import trimesh
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

 
def from_object_to_world(trans, pt):
    
    pt_p = autolab_core.Point(pt.reshape(-1,1), trans.from_frame)
    return trans.apply(pt_p).vector


def draw_boundary_points(obj_trans, obj_xyzs,  all_obj_bounds, layout_filename=None):
    obj_xyzs = np.asarray(obj_xyzs)
    all_obj_bounds = np.asarray(all_obj_bounds)

    
    polys = []
    fig, ax = plt.subplots()
    for i in range(len(all_obj_bounds)):
        x,y,z = obj_xyzs[i]

        bound = all_obj_bounds[i]

        
        corners4 = get_bound_2d_4corners(obj_trans[i], bound)
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
    

# def generate_object_xy(object_position_region,
#                     probs,
#                     prev_polys, 
#                     obj_trans, 
#                     obj_rotations, 
#                     object_rot, 
#                     bound,  
#                     obj_xyzs, 
#                     all_obj_bounds, 
#                     z):
#     avoid_all_squares = False 
#     x,y = None, None
#     acc = 0
#     selected_region = None 
#     while not avoid_all_squares:
#         region = np.random.choice(8, 1, p=probs)[0]
        
#         acc += 1
#         region_range = object_position_region[region]
#         x = np.random.uniform(region_range[0][0]/2, region_range[0][1]/2, 1)[0]
#         y = np.random.uniform(region_range[1][0]/2, region_range[1][1]/2, 1)[0]
        
#         if len(prev_polys) == 0:
#             avoid_all_squares = True
#             continue 

#         selected_region = region 
        
#         r = R.from_euler('xyz', object_rot, degrees=False) 
#         r = R.from_euler('xyz', [0,0,0], degrees=False)
#         r = R.from_euler('xyz', [0,0,object_rot[-1]], degrees=False)
#         trans = autolab_core.RigidTransform(rotation = r.as_matrix(), translation = np.asarray([x,y,z]), from_frame='guess')
#         corners4 = get_bound_2d_4corners(trans, bound)
#         poly = Polygon(corners4)

        
#         all_outside = True
#         for prev_poly in prev_polys:
#             # draw_boundary_points(obj_trans+[trans], obj_xyzs+[[x,y,z]],  all_obj_bounds+[bound])
#             # draw_boundary_points(obj_trans, obj_xyzs,  all_obj_bounds)
#             if prev_poly.intersects(poly):
#                 all_outside = False
#                 break
#         avoid_all_squares = all_outside
    
#     # import pdb; pdb.set_trace()
#     new_probs = update_object_position_region(object_position_region, probs, selected_region)
#     return x,y, new_probs

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
            raise
        
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
                print("\n#1: try-{}".format(try_count), x, y)
                print(x_start,x_sign, y_start,y_sign)
                print("::", x_start + object_x_width*ratio_lower*x_sign, x_start + object_x_width*ratio_upper*x_sign)
                print("::", y_start + object_y_width*ratio_lower*y_sign, y_start + object_y_width*ratio_upper*y_sign)
                all_outside = False
                break

            for corner in new_corners:
                dists.append(np.linalg.norm(old_corners - corner, axis=1))
        
        if all_outside:
            min_corner_dist = np.min(np.stack(dists))
            if min_corner_dist > 0.35:
                print("\n#2: try-{}".format(try_count), x, y)
                print(x_start,x_sign, y_start,y_sign)
                print("::", x_start + object_x_width*ratio_lower*x_sign, x_start + object_x_width*ratio_upper*x_sign)
                print("::", y_start + object_y_width*ratio_lower*y_sign, y_start + object_y_width*ratio_upper*y_sign)
                print(np.min(np.stack(dists), axis=1), min_corner_dist)
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
        

def generate_object_xy_rect(bound, prev_bbox, object_position_region, probs):
    avoid_all_squares = False 
    x,y = None, None
    selected_region = None 
    while not avoid_all_squares:
        region = np.random.choice(8, 1, p=probs)[0]
        region_range = object_position_region[region]
        region_width = region_range[0][1] - region_range[0][0]
        region_height = region_range[1][1] - region_range[1][0]
        # x = np.random.uniform(region_range[0][0] + region_width/5, region_range[0][0] + region_width/5 + region_width/2, 1)[0]
        # y = np.random.uniform(region_range[1][0] + region_height/5, region_range[1][0] + region_height/5 + region_height/2, 1)[0]
        x = np.random.uniform(region_range[0][0], region_range[0][1], 1)[0]
        y = np.random.uniform(region_range[1][0], region_range[1][1], 1)[0]

        if len(prev_bbox) == 0:
            avoid_all_squares = True
            continue 
        selected_region = region
        corner = get_2d_diagonal_corners([[x,y,0]], [bound])[0]
        ll = corner[0]
        ur = corner[2]

        all_outside = True
        for prev_corner in prev_bbox:
            prev_ll = prev_corner[0]
            prev_ur = prev_corner[2]
            if doOverlap(ll, ur, prev_ll, prev_ur, 0):
                all_outside = False 
                break

            new_lower_x, new_lower_y = ll[0],ll[1]
            new_upper_x, new_upper_y = ur[0],ur[1]
            old_lower_x, old_lower_y = prev_ll[0], prev_ll[1]
            old_upper_x, old_upper_y = prev_ur[0], prev_ur[1]
            dists = []
            new_corners = np.asarray([[new_lower_x, new_lower_y], \
                [new_upper_x, new_lower_y], \
                [new_upper_x, new_upper_y], \
                [new_lower_x, new_upper_y]])
            old_corners = np.asarray([[old_lower_x, old_lower_y], \
                [old_upper_x, old_lower_y], \
                [old_upper_x, old_upper_y], \
                [old_lower_x, old_upper_y]])
            
            for corner in new_corners:
                dists.append(np.linalg.norm(old_corners - corner, axis=1))
            min_corner_dist = np.min(np.stack(dists))
            if min_corner_dist > 0.4:
                all_outside = False
                break
            # print(np.stack(dists), min_corner_dist)
            # print("x,y: ", x,y)

        avoid_all_squares = all_outside
    new_probs = update_object_position_region(object_position_region, probs, selected_region)
    return x,y, new_probs


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
            
            deg_candidate = np.asarray([3,5,10,15,20,25])
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
    num_angles = 16
    quad = (2.0*math.pi) / num_angles
    normal_thetas = [np.random.uniform(i*quad, (i+1.0)*quad,1)[0] for i in range(num_angles)]
    
    center = np.mean(xys, axis=0)
    pairwise_diff = xys - center.reshape((1,2))
    dist = np.linalg.norm(pairwise_diff, axis=1)
    max_dist = np.max(dist)
    
    for theta in normal_thetas:
        cam_x = np.cos(theta) * (max_dist+distance_away) + center[0]
        cam_y = np.sin(theta) * (max_dist+distance_away) + center[1]
        cam_z = max_object_height * np.random.uniform(1,2,1)[0]
        cam_xyzs[cam_num] = [cam_x, cam_y, cam_z]
        cam_targets[cam_num] = [center[0],center[1],table_height]

        cam_num_to_occlusion_target[cam_num] = -1
        cam_num += 1
    
    return cam_xyzs, cam_targets, cam_num_to_occlusion_target

def get_camera_position(camera_distance, table_height, max_object_height, obj_xyzs):
    num_angles = 12
    quad = (2.0*math.pi) / num_angles
    normal_thetas = [np.random.uniform(i*quad, (i+1.0)*quad,1)[0] for i in range(num_angles)]
    normal_thetas = np.array(normal_thetas)
    normal_x=np.cos(normal_thetas).flatten() 
    normal_y=np.sin(normal_thetas).flatten() 
    
    cam_xyzs = []
    cam_targets = []

    camera_pos_z = [max_object_height, max_object_height*1.3]
    mean_object_pos = np.mean(obj_xyzs, axis=0)
    cam_targs = [mean_object_pos, mean_object_pos]
    cam_distances = [camera_distance*1.7, camera_distance*0.7]
    for z,targ, cam_d in zip(camera_pos_z, cam_targs, cam_distances):
        camera_pos_x = normal_x * cam_d
        camera_pos_y = normal_y * cam_d
        num_camera_x = len(camera_pos_x)
        zs = np.repeat(z, num_camera_x)
        
        cam_xyzs.append(np.vstack([camera_pos_x, camera_pos_y, zs]).T)

        cam_targets.append(np.repeat(targ.reshape(-1,3), num_camera_x, axis=0))

    return np.vstack(cam_xyzs), np.vstack(cam_targets)

def get_fixed_camera_position(camera_distance, max_object_height, table_xyz):
    num_angles = 8
    normal_thetas = [((2.0*math.pi)/num_angles)*i for i in range(num_angles)]

    normal_x=np.cos(normal_thetas).flatten() 
    normal_y=np.sin(normal_thetas).flatten() 
    
    camera_pos_x = normal_x * camera_distance *2.5
    camera_pos_x = np.append(camera_pos_x, 0)
    camera_pos_y = normal_y * camera_distance *2.5
    camera_pos_y = np.append(camera_pos_y, 0)
    num_camera_x = len(camera_pos_x)
    # Generate camera heights
    camera_pos_z = [max_object_height,  max_object_height*3]
    num_camera_z = len(camera_pos_z)
    # [z1,z1,...,z1,z2,z2....z2,....,zn,...,zn]
    camera_pos_z = np.repeat(camera_pos_z, num_camera_x)
    # camera_pos_z = np.append(camera_pos_z,)
    # [x1,x2,...,xn,x1,x2,...xn,....,x1,...,xn]
    # [y1,y2,...,yn,y1,y2,...yn,....,y1,...,yn]
    camera_pos_x = np.tile(camera_pos_x, num_camera_z)
    camera_pos_y = np.tile(camera_pos_y, num_camera_z)
    
    cam_targets = np.array([table_xyz, table_xyz])
    cam_targets = np.repeat(cam_targets.reshape(-1,3), num_camera_x, axis=0)
    # 
    return np.vstack([camera_pos_x, camera_pos_y,camera_pos_z]).T, cam_targets


def transform_to_camera_vector(vector, camera_pos, lookat_pos, camera_up_vector):
    
    view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
    view_matrix = np.array(view_matrix).reshape(4,4, order='F')
    vector = np.concatenate((vector, np.array([1])))
    transformed_vector = view_matrix.dot(vector)
    return transformed_vector[:3]


def get_camera_matrix(camera):
    camera_id = camera._render_camera.fixedcamid
    pos = camera._physics.data.cam_xpos[camera_id]
    rot = camera._physics.data.cam_xmat[camera_id].reshape(3, 3)
    fov = camera._physics.model.cam_fovy[camera_id]

    
    # # Translation matrix (4x4).
    # translation = np.eye(4)
    # translation[0:3, 3] = -pos
    
    # # Rotation matrix (4x4).
    # rotation = np.eye(4)
    # rotation[0:3, 0:3] = rot
    
    # # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * camera.height / 2.0
    # focal = np.diag([focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    # # Image matrix (3x3).
    # image = np.eye(3)
    # image[0, 2] = (camera.width - 1) / 2.0
    # image[1, 2] = (camera.height - 1) / 2.0
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



def get_camera_matrix_1(camera):
    camera_id = camera._render_camera.fixedcamid
    pos = camera._physics.data.cam_xpos[camera_id]
    rot = camera._physics.data.cam_xmat[camera_id].reshape(3, 3).T
    fov = camera._physics.model.cam_fovy[camera_id]

    
    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos
    
    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot
    
    # # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * camera.height / 2.0
    focal = np.diag([focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (camera.width - 1) / 2.0
    image[1, 2] = (camera.height - 1) / 2.0

    return image @ focal @ rotation @ translation, None


def load_mesh_convex_parts(shapenet_decomp_filepath, obj_cat, obj_id, scale_mat):
    '''
    Load the convex-decomposed parts of an object
    Apply transformation according to scale_mat
    Export the parts back to the file
    '''
    comb_mesh=None
    decomp_shapenet_decomp_filepath = os.path.join(shapenet_decomp_filepath, f'{obj_cat}/{obj_id}')
    for mesh_file in os.listdir(decomp_shapenet_decomp_filepath):
        if len(mesh_file.split(".")) < 2:
            continue 
        decomp_object_mesh = trimesh.load(os.path.join(decomp_shapenet_decomp_filepath, mesh_file))
        if comb_mesh == None:
            comb_mesh = decomp_object_mesh
        else:
            comb_mesh += decomp_object_mesh
    comb_mesh.apply_transform(scale_mat)
    trimesh.repair.fix_inversion(comb_mesh)
    meshes = comb_mesh.split()
    
    mesh_names = []
    mesh_masses = []
    
    combined_mesh = None
    mesh_file_ind = 0
    for mesh_file in os.listdir(decomp_shapenet_decomp_filepath):
        if len(mesh_file.split(".")) < 2:
            continue 
        decomp_object_mesh = meshes[mesh_file_ind]
        if decomp_object_mesh.faces.shape[0]>10 and decomp_object_mesh.mass>10e-7:
            obj_mesh_filename = os.path.join(decomp_shapenet_decomp_filepath, mesh_file[:-3]+'stl')
            f = open(obj_mesh_filename, "w+")
            f.close()
            decomp_object_mesh.export(obj_mesh_filename)
            mesh_names.append(obj_mesh_filename)
            mesh_masses.append(decomp_object_mesh.mass)
            if combined_mesh == None:
                combined_mesh = decomp_object_mesh
            else:
                combined_mesh += decomp_object_mesh
        mesh_file_ind+=1
        if mesh_file_ind >= len(meshes):
            break
        
    if len(mesh_names)>100:
        heavy_inds=np.argsort(np.array(mesh_masses))
        new_mesh_names=[]
        for ind in range(100):
            new_mesh_names.append(mesh_names[heavy_inds[-ind]])
        mesh_names=new_mesh_names
    
    return mesh_names


def determine_object_rotation(object_mesh):
    # Rotate object so that it appears upright in Mujoco
    rot_vec = [(1/2)*np.pi, 0, 0]
    r = R.from_euler('xyz', rot_vec, degrees=False) 
    upright_mat = np.eye(4)
    upright_mat[0:3,0:3] = r.as_matrix()
    
    return rot_vec, upright_mat 

def return_centroid(obj_mesh_filename, upright_mat, object_xyz, object_size, object_rot):
    mesh = trimesh.load(obj_mesh_filename, force='mesh')
    mesh.apply_transform(upright_mat)
    size_mat = np.eye(4) * object_size
    size_mat[3,3] = 1.0
    mesh.apply_transform(size_mat)
    
    r = R.from_rotvec(object_rot)
    rot_mat = np.eye(4)
    rot_mat[0:3,0:3] = r.as_matrix()
    mesh.apply_transform(rot_mat)
    
    position_mat = np.eye(4)
    position_mat[0:3,3] = object_xyz
    mesh.apply_transform(position_mat)

    return mesh.centroid
