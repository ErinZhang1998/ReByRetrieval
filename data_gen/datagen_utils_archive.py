import pybullet as p
import numpy as np
import copy
import pickle
from scipy.spatial.transform import Rotation as R, rotation
import math
from dm_control.mujoco.engine import Camera
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
import dm_control.mujoco as mujoco
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image as PIL_Image

"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""


def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw),
                   2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw),
                   2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat


"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""


def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat


"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""


def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat


"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


#
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""


class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """

    def __init__(self, sim, min_bound=None, max_bound=None):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # I think these can be set to anything
        self.img_width = 640
        self.img_height = 480

        self.cam_names = self.sim.model.camera_names

        self.target_bounds = None
        if min_bound != None and max_bound != None:
            self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound, max_bound=max_bound)

        # List of camera intrinsic matrices
        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2),
                                (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def camera_information(self, cam_i):
        cam_body_id = self.sim.model.cam_bodyid[cam_i]
        cam_pos = self.sim.model.body_pos[cam_body_id]
        c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
        # In MuJoCo, we assume that a camera is specified in XML as a body
        #    with pose p, and that that body has a camera sub-element
        #    with pos and euler 0.
        #    Therefore, camera frame with body euler 0 must be rotated about
        #    x-axis by 180 degrees to align it with the world frame.
        b2w_r = quat2Mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        c2w = posRotMat2Mat(cam_pos, c2w_r)

        return {
            'P': self.cam_mats[cam_i],
            'camera_frame_to_world_frame_mat': c2w,
        }

    def generateCroppedPointCloud_onecam(self, cam_i):
        # Render and optionally save image from camera corresponding to cam_i
        depth_img = self.captureImage(cam_i)
        # If directory was provided, save color and depth images
        #    (overwriting previous)
        # convert camera matrix and depth image to Open3D format, then
        #    generate point cloud
        od_cammat = cammat2o3d(
            self.cam_mats[cam_i], self.img_width, self.img_height)
        od_depth = o3d.geometry.Image(depth_img)
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            od_depth, od_cammat)

        # Compute world to camera transformation matrix
        cam_body_id = self.sim.model.cam_bodyid[cam_i]
        cam_pos = self.sim.model.body_pos[cam_body_id]
        c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
        # In MuJoCo, we assume that a camera is specified in XML as a body
        #    with pose p, and that that body has a camera sub-element
        #    with pos and euler 0.
        #    Therefore, camera frame with body euler 0 must be rotated about
        #    x-axis by 180 degrees to align it with the world frame.
        b2w_r = quat2Mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        c2w = posRotMat2Mat(cam_pos, c2w_r)
        transformed_cloud = o3d_cloud.transform(c2w)

        return transformed_cloud

    def generateCroppedPointCloud(self, save_img_dir=None):
        o3d_clouds = []
        cam_poses = []
        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            depth_img = self.captureImage(cam_i)
            # If directory was provided, save color and depth images
            #    (overwriting previous)
            if save_img_dir != None:
                self.saveImg(depth_img, save_img_dir,
                             "depth_test_" + str(cam_i))
                color_img = self.captureImage(cam_i, False)
                self.saveImg(color_img, save_img_dir,
                             "color_test_" + str(cam_i))

            # convert camera matrix and depth image to Open3D format, then
            #    generate point cloud
            od_cammat = cammat2o3d(
                self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth_img)
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                od_depth, od_cammat)

            # Compute world to camera transformation matrix
            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            cam_pos = self.sim.model.body_pos[cam_body_id]
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            transformed_cloud = o3d_cloud.transform(c2w)

            # If both minimum and maximum bounds are provided, crop cloud to fit
            #    inside them.
            if self.target_bounds != None:
                transformed_cloud = transformed_cloud.crop(self.target_bounds)

            # Estimate normals of cropped cloud, then flip them based on camera
            #    position.
            transformed_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
            transformed_cloud.orient_normals_towards_camera_location(cam_pos)

            o3d_clouds.append(transformed_cloud)

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        return combined_cloud

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, cam_ind, capture_depth=True):
        rendered_images = self.sim.render(
            self.img_width, self.img_height, camera_name=self.cam_names[cam_ind], depth=capture_depth)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)
            real_depth = self.depthimg2Meters(depth)

            return real_depth
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".png")

##############################################################################################################


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


def determine_object_rotation(object_mesh):
    # Rotate object so that it appears upright in Mujoco
    rot_vec = [(1/2)*np.pi, 0, 0]
    r = R.from_euler('xyz', rot_vec, degrees=False)
    upright_mat = np.eye(4)
    upright_mat[0:3, 0:3] = r.as_matrix()

    return rot_vec, upright_mat


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


def scale_rotate_object(mesh, quat, scale):
    '''
    mesh: trimesh.Trimesh
    quat: [x, y, z, w] quaternion
    scale: [x_scale, y_scale, z_scale]
    '''
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale
    mesh.apply_transform(scale_matrix)

    r = R.from_euler('xyz', [(1/2)*np.pi, 0, 0], degrees=False)
    rot_matrix = np.eye(4)
    rot_matrix[0:3, 0:3] = r.as_matrix()
    mesh.apply_transform(rot_matrix)

    # r2 = R.from_quat(quat)
    # rot_matrix = np.eye(4)
    # rot_matrix[:3, :3] = r2.as_matrix()
    # mesh.apply_transform(rot_matrix)

    return mesh


def retrieve_mesh_from_description(object_description, shapenet_path):
    mesh_fname = os.path.join(
        shapenet_path, object_description['mesh_filename'])
    object_mesh = trimesh.load(mesh_fname, force='mesh')
    return scale_rotate_object(object_mesh, object_description['orientation_quat'], object_description['scale'])

##############################################################################################################


def draw_boundary_points_rect(corners4s, layout_filename=None):
    fig, ax = plt.subplots()
    for corners4 in corners4s:
        poly = patches.Polygon(corners4, linewidth=1,
                               edgecolor='r', facecolor='g')
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
    x, y = None, None
    selected_region = None
    MAX_TRY = 50
    lower_x, upper_x = object_bounds[:, 0]
    lower_y, upper_y = object_bounds[:, 1]
    lower_z, _ = object_bounds[:, 2]
    object_x_width, object_y_width, _ = object_bounds[1] - object_bounds[0]
    add_x = object_x_width * 0.1
    add_y = object_y_width * 0.1
    upper_x, upper_y = upper_x+add_x, upper_y+add_y
    lower_x, lower_y = lower_x-add_x, lower_y-add_y
    new_corners_3d = np.array([[lower_x, lower_y, lower_z],
                               [upper_x, lower_y, lower_z],
                               [upper_x, upper_y, lower_z],
                               [lower_x, upper_y, lower_z]])  # (4,3) --> (3,4)
    try_count = 0
    while not avoid_all_squares:
        if MAX_TRY < 0:
            raise ValueError

        prev_center_idx = np.random.choice(len(prev_bbox), 1)[0]
        center_corners = prev_bbox[prev_center_idx]
        x_bottom, y_bottom = np.min(center_corners, axis=0)
        x_top, y_top = np.max(center_corners, axis=0)
        object_position_region = {
            0: [[x_top, 1], [y_bottom, 1]],
            1: [[x_bottom, 1], [y_top, 1]],
            2: [[x_bottom, -1], [y_bottom, 1]],
            3: [[x_bottom, 1], [y_bottom, -1]],
            4: [[x_top, 1], [y_bottom, -1]],
            6: [[x_bottom, -1], [y_top, 1]],
            5: [[x_top, 1], [y_top, 1]],
            7: [[x_bottom, -1], [y_bottom, -1]],
        }

        region = np.random.choice(8, 1, p=probs)[0]
        region_range = object_position_region[region]
        x_start, x_sign = region_range[0]
        y_start, y_sign = region_range[1]

        x_dist = np.random.uniform(
            object_x_width*ratio_lower, object_x_width*ratio_upper, 1)[0]
        y_dist = np.random.uniform(
            object_y_width*ratio_lower, object_y_width*ratio_upper, 1)[0]

        x = x_start + x_dist * x_sign
        y = y_start + y_dist * y_sign

        if len(prev_bbox) == 0:
            avoid_all_squares = True
            continue
        selected_region = region

        object_xyz = [x, y, object_z]
        r2 = R.from_rotvec(object_rot)
        object_tf = autolab_core.RigidTransform(rotation=r2.as_matrix(
        ), translation=np.asarray(object_xyz), from_frame='object_tmp', to_frame='world')
        pt_3d_homo = np.append(new_corners_3d.T, np.ones(
            4).astype('int').reshape(1, -1), axis=0)  # (4,4)
        bounding_coord = object_tf.matrix @ pt_3d_homo  # (4,4)
        bounding_coord = bounding_coord / bounding_coord[-1, :]
        bounding_coord = bounding_coord[:-1, :].T
        new_corners = bounding_coord[:, :2]
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
    return x, y, new_probs, object_tf, new_corners


def get_2d_diagonal_corners(obj_xyzs, all_obj_bounds):
    corners = []
    for xyz, bound in zip(obj_xyzs, all_obj_bounds):
        x, y, _ = xyz
        a, b = bound[:, 0], bound[:, 1]
        diag_length = np.sqrt((a[1] - a[0]) ** 2 + (b[1] - b[0]) ** 2)
        diag_length = diag_length * (1/2)
        pts = np.array([[-diag_length, -diag_length],
                        [-diag_length, diag_length],
                        [diag_length, diag_length],
                        [diag_length, -diag_length]])

        pts = pts + np.array([x, y])
        corners.append(pts)

    return corners


def doOverlap(l1, r1, l2, r2, buf):

    # If one rectangle is on left side of other
    l1x, l1y = l1[0], l1[1]
    r1x, r1y = r1[0], r1[1]
    l2x, l2y = l2[0], l2[1]
    r2x, r2y = r2[0], r2[1]

    if(l1x >= r2x+buf or l2x >= r1x+buf):
        return False

    # If one rectangle is above other
    if(l1y >= r2y+buf or l2y >= r1y+buf):
        return False

    return True


def move_object(mujoco_env, ind, pos, rot, num_ind_prev=1):
    # ASSUME THERE IS TABLE so 7+ and 6+
    all_poses = mujoco_env.data.qpos.ravel().copy()
    all_vels = mujoco_env.data.qvel.ravel().copy()

    start_idx = 7*num_ind_prev
    all_poses[start_idx + 7*ind : start_idx + 7*ind+3] = pos
    all_poses[start_idx + 7*ind+3 : start_idx + 7*ind+7] = rot

    all_vels[6+6*ind: 6+6*ind+6] = 0
    mujoco_env.set_state(all_poses, all_vels)

    num_objects = all_poses.reshape(-1, 7).shape[0]
    # for _ in range(num_objects):
    #     for _ in range(10000):
    #         mujoco_env.model.step()
    return mujoco_env.data.qpos.ravel().copy().reshape(-1, 7)


def get_camera_position_occluded_one_cam(table_height, xyz1, xyz2, height1, height2, max_dist, deg_candidate):
    distance_away = 3

    x, y, z = xyz1
    a, b, c = xyz2
    xdiff = a-x
    ydiff = b-y
    rad = np.arctan(
        ydiff / xdiff) if xdiff > 0 else np.arctan(ydiff / xdiff)+np.pi
    shifted_degree = np.random.uniform(
        deg_candidate[0], deg_candidate[1], 1)[0]
    #sign = np.random.choice([1,-1])
    rad += np.deg2rad(shifted_degree)

    cam_x = np.cos(rad) * (max_dist+distance_away) + x
    cam_y = np.sin(rad) * (max_dist+distance_away) + y

    if height2 > height1:
        cam_z = table_height + height2
    else:
        cam_z = table_height + height2 / 2

    cam_xyz = [cam_x, cam_y, cam_z]
    jitter = [0, 0, 0]  # np.random.uniform(0.2,0.5,2)
    cam_target = [x+jitter[0], y+jitter[1],
                  z+np.random.uniform(0.1, 0.2, 1)[0]]

    cam_xyz2 = [cam_x, cam_y, np.random.uniform(
        table_height + height2*2, table_height + height2*3, 1)[0]]

    return cam_xyz, cam_target, cam_xyz2


def get_pixel_left_ratio(scene_num, camera, cam_num, e, object_i, all_obj_indices, cam_width, cam_height):
    state = e.get_env_state().copy()
    segs = camera.render(segmentation=True)[:, :, 0]
    occluded_geom_id_to_seg_id = {
        camera.scene.geoms[geom_ind][3]: camera.scene.geoms[geom_ind][8] for geom_ind in range(camera.scene.geoms.shape[0])}

    target_id = e.model.model.name2id(
        f'gen_geom_object_{object_i}_{scene_num}_0', "geom")
    segmentation = segs == occluded_geom_id_to_seg_id[target_id]

    # Move all other objects far away, except the table, so that we can capture
    # only one object in a scene.
    for move_obj_ind in all_obj_indices:
        if move_obj_ind != object_i:
            move_object(e, move_obj_ind, [20, 20, move_obj_ind], [0, 0, 0, 0])

    e.sim.physics.forward()

    unocc_target_id = e.model.model.name2id(
        f'gen_geom_object_{object_i}_{scene_num}_0', "geom")
    unoccluded_camera = Camera(
        physics=e.model, height=cam_height, width=cam_width, camera_id=cam_num)
    unoccluded_segs = unoccluded_camera.render(segmentation=True)

    # Move other objects back onto table
    e.set_env_state(state)
    e.sim.physics.forward()

    unoccluded_geom_id_to_seg_id = {unoccluded_camera.scene.geoms[geom_ind][3]: unoccluded_camera.scene.geoms[geom_ind][8] for geom_ind in range(
        unoccluded_camera.scene.geoms.shape[0])}
    unoccluded_segmentation = unoccluded_segs[:, :,
                                              0] == unoccluded_geom_id_to_seg_id[unocc_target_id]

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
    for object_idx, v in xyzs.items():
        xys.append([v[0], v[1]])
    xys = np.asarray(xys)

    cam_xyzs = dict()
    cam_targets = dict()
    cam_num = 0
    cam_num_to_occlusion_target = dict()
    for i, (x, y, z) in xyzs.items():

        pairwise_diff = xys - np.array([x, y]).reshape((1, 2))
        dist = np.linalg.norm(pairwise_diff, axis=1)
        max_dist = np.max(dist)

        for j, (a, b, c) in xyzs.items():
            if i == j:
                continue
            # dist = max_dist #np.linalg.norm([a-x,b-y])

            xdiff = a-x
            ydiff = b-y
            rad = np.arctan(
                ydiff / xdiff) if xdiff > 0 else np.arctan(ydiff / xdiff)+np.pi
            # shifted_degrees = np.random.choice([5,9,13,],10)

            deg_candidate = np.asarray([0, 2])  # np.asarray([3,5,10,15,20,25])
            shifted_degrees = []
            for degi, degj in zip(deg_candidate[:-1], deg_candidate[1:]):
                shifted_degrees.append(np.random.uniform(degi, degj, 1)[0])

            for shifted_degree in shifted_degrees:
                sign = np.random.choice([1, -1])
                rad += np.deg2rad(sign * shifted_degree)

                cam_x = np.cos(rad) * (max_dist+distance_away) + x
                cam_y = np.sin(rad) * (max_dist+distance_away) + y

                if heights[j] > heights[i]:
                    cam_z = table_height + heights[j]
                else:
                    cam_z = table_height + heights[j] / 2

                cam_xyzs[cam_num] = [cam_x, cam_y, cam_z]
                jitter = np.random.uniform(0.2, 0.5, 2)
                cam_targets[cam_num] = [x+jitter[0], y+jitter[1],
                                        z+np.random.uniform(0.1, 0.2, 1)[0]]

                cam_num_to_occlusion_target[cam_num] = i
                cam_num += 1

    # bird eye view
    num_angles = 8
    quad = (2.0*math.pi) / num_angles
    normal_thetas = [np.random.uniform(
        i*quad, (i+1.0)*quad, 1)[0] for i in range(num_angles)]

    center = np.mean(xys, axis=0)
    pairwise_diff = xys - center.reshape((1, 2))
    dist = np.linalg.norm(pairwise_diff, axis=1)
    max_dist = np.max(dist)

    for theta in normal_thetas:
        cam_x = np.cos(theta) * (max_dist+distance_away) + center[0]
        cam_y = np.sin(theta) * (max_dist+distance_away) + center[1]
        cam_z = max_object_height * np.random.uniform(1, 1.3, 1)[0]
        cam_xyzs[cam_num] = [cam_x, cam_y, cam_z]
        cam_targets[cam_num] = [center[0], center[1], table_height]

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

    P = np.array(((focal_scaling, 0, camera.width / 2),
                  (0, focal_scaling, camera.height / 2), (0, 0, 1)))
    camera_tf = autolab_core.RigidTransform(rotation=rot, translation=np.asarray(
        pos), from_frame='camera_{}'.format(camera_id), to_frame='world')

    assert np.all(np.abs(camera_tf.matrix @ np.array([0, 0, 0, 1]).reshape(
        4, -1) - np.array([[pos[0], pos[1], pos[2], 1]]).reshape(4, -1)) < 1e-5)

    res = {
        'P': P,
        'intrinsices': P,
        'pos': pos,
        'rot': rot,
        'fov': fov,
        'focal_scaling': focal_scaling,
        'camera_frame_to_world_frame_mat': camera_tf.matrix,
        'world_frame_to_camera_frame_mat': camera_tf.inverse().matrix,
    }
    return camera_id, camera_tf, res


def project_2d_mat(P, world_to_camera_tf_mat, pt_3d):
    '''
    pt_3d: (N,3)
    '''
    N = len(pt_3d)
    pt_3d_pad = np.append(pt_3d.T, np.ones(N).astype(
        'int').reshape(1, -1), axis=0)  # (4,N)
    pt_3d_camera = world_to_camera_tf_mat @ pt_3d_pad  # (4,N)
    assert np.all(np.abs(pt_3d_camera[-1] - 1) < 1e-6)
    pixel_coord = P @ (pt_3d_camera[:-1, :])
    mult = pixel_coord[-1, :]
    pixel_coord = pixel_coord / pixel_coord[-1, :]
    pixel_coord = pixel_coord[:2, :]  # (2,N)
    pixel_coord = pixel_coord.astype('int').T
    return pixel_coord


def project_2d(P, camera_tf, pt_3d):
    '''
    pt_3d: (N,3)
    '''
    world_to_camera_tf_mat = camera_tf.inverse().matrix  # (4,4)
    return project_2d_mat(P, world_to_camera_tf_mat, pt_3d)


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


def object_bounds_scaled_rotated(bounds, scale, rot=None):
    '''
    bounds : (2,3) directly loaded from ShapeNet, default bounds
    scale : (3,)

    '''
    _, basic_rot = determine_object_rotation(None)
    bounds = bounds * np.array([scale, scale])
    bounds_rotated = transform_3d_frame(basic_rot, bounds)
    if not rot is None:
        r = R.from_quat(rot)
        rotation_mat = np.eye(4)
        rotation_mat[0:3, 0:3] = r.as_matrix()
        bounds_rotated = transform_3d_frame(rotation_mat, bounds)
    return bounds_rotated


def quat_xyzw_to_wxyz(quat_xyzw):
    x,y,z,w = quat_xyzw
    return np.array([w,x,y,z])

def quat_wxyz_to_xyzw(quat_wxyz):
    w,x,y,z = quat_wxyz
    return np.array([x,y,z,w])

def mujoco_quat_to_rotation_object(quat_wxyz):
    w,x,y,z = quat_wxyz
    return R.from_quat([x,y,z,w])

def euler_xyz_to_mujoco_quat(euler_xyz):
    rot_obj = R.from_euler('xyz', euler_xyz, degrees=False)
    x,y,z,w = rot_obj.as_quat()
    return [w,x,y,z]

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
    bounds: (2,3)
    pos: (3,)
    rot_obj: scipy.spatial.transform object
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
    return json serializable matrix 
    ''' 
    if len(mat.shape) == 1:
        if type == 'float':
            return [float(item) for item in mat]
        else:
            return [int(item) for item in mat]
    cleaned_mat = []
    for sub_mat in mat:
        cleaned_mat.append(get_json_cleaned_matrix(sub_mat))
    return cleaned_mat

##############################################################################################################


def compile_mask(mask_path_lists):
    masks = []
    for mask_path in mask_path_lists:
        mask = mpimg.imread(mask_path)
        masks.append(mask)

    return np.sum(np.stack(masks), axis=0)


def from_depth_img_to_pc(depth_image, cam_cx, cam_cy, fx, fy, cam_scale=1.0, upsample=1):
    img_width = int(2*cam_cx)
    img_height = int(2*cam_cy)
    xmap = np.array([[j for i in range(int(upsample*img_width))]
                     for j in range(int(upsample*img_height))])
    ymap = np.array([[i for i in range(int(upsample*img_width))]
                     for j in range(int(upsample*img_height))])

    depth_masked = depth_image.flatten()[:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[:, np.newaxis].astype(np.float32)

    pt2 = depth_masked / cam_scale
    pt0 = (ymap_masked/upsample - cam_cx) * pt2 / (fx)
    pt1 = (xmap_masked/upsample - cam_cy) * pt2 / (fy)
    cloud = np.concatenate((pt0, -pt1, -pt2), axis=1)
    return cloud


def process_pointcloud(cloud, obj_points_inds, rot):
    obs_ptcld = cloud/1000.0
    obj_pointcloud = obs_ptcld[obj_points_inds]
    pc_mean = np.mean(obj_pointcloud, axis=0)

    obs_ptcld_min = np.amin(obj_pointcloud, axis=0)
    obs_ptcld_max = np.amax(obj_pointcloud, axis=0)
    scale = 4.0*float(np.max(obs_ptcld_max-obs_ptcld_min))

    obj_pointcloud = (obj_pointcloud - pc_mean) / scale
    obj_pointcloud = rot.dot(obj_pointcloud.T).T

    low = np.array([-0.5, -0.5, -0.5])
    hi = np.array([0.5, 0.5, 0.5])
    cloud_mask = np.argwhere(np.all(np.logical_and(
        obj_pointcloud >= low, obj_pointcloud <= hi), axis=1))  # (M, 1)
    obj_pointcloud = obj_pointcloud[cloud_mask][:, 0, :] + 0.5

    return obj_pointcloud, cloud_mask.flatten()


def get_pointcloud(mask, depth, all_ptcld, camera_rot, cam_height, cam_width):
    inds = np.where(mask, depth, 0.0).flatten().nonzero()[0]
    if inds.shape[0] == 0:
        return []
    # import pdb; pdb.set_trace()
    obj_points, obj_mask = process_pointcloud(all_ptcld, inds, camera_rot)
    if len(obj_mask) == 0:
        return []
    x_ind, y_ind = np.unravel_index(inds[obj_mask], (cam_height, cam_width))
    return [obj_points, x_ind, y_ind]


def create_o3d_pc(xyz, visualize=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def load_pc_from_pkl(pkl_filename):
    xyz, xind, yind = None, None, None
    with open(pkl_filename, 'rb') as f:
        xyz, xind, yind = pickle.load(f)
    return xyz


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def generateCroppedPointCloud(depth_img, intrinsic, camera_to_world_mat, img_width, img_height):
    od_cammat = cammat2o3d(intrinsic, img_width, img_height)
    od_depth = o3d.geometry.Image(depth_img)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        od_depth, od_cammat)
    transformed_cloud = o3d_cloud.transform(camera_to_world_mat)

    return transformed_cloud


##############################################################################################################
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
    new_body.setAttribute(
        'specular', f'{specular[0]} {specular[1]} {specular[2]}')
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
        theta = np.random.uniform(
            base_angle - math.pi/9, base_angle + math.pi/9, 1)[0]
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

    new_body = xmldoc.createElement('camera')
    new_body.setAttribute('name', cam_name)
    new_body.setAttribute('mode', 'targetbody')
    new_body.setAttribute('fovy', '65')
    new_body.setAttribute('pos', f'{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}')
    new_body.setAttribute('target', f'added_cam_target_{cam_id}')
    world_body.appendChild(new_body)

    new_body = xmldoc.createElement('body')
    new_body.setAttribute('name', f'added_cam_target_{cam_id}')
    new_body.setAttribute(
        'pos', f'{cam_target[0]} {cam_target[1]} {cam_target[2]}')
    new_geom = xmldoc.createElement('geom')
    geom_name = f'added_cam_target_geom_{cam_id}'
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

def add_objects(scene_name, object_info, run_id=None, material_name=None):
    # scene_name = object_info['scene_name']
    object_name = object_info['object_name']
    mesh_names = object_info['mesh_names']
    pos = object_info['pos']
    size = object_info['size']
    color = object_info['color']
    quat = object_info['quat']

    xmldoc = minidom.parse(scene_name)

    assets = xmldoc.getElementsByTagName('asset')[0]
    for mesh_ind in range(len(mesh_names)):
        new_mesh = xmldoc.createElement('mesh')
        new_mesh.setAttribute('name', f'gen_mesh_{object_name}_{mesh_ind}')
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

    if not object_info.get('mocap', False) and not object_info.get('site', False):
        new_joint = xmldoc.createElement('joint')
        new_joint.setAttribute('name', f'gen_joint_{object_name}')
        new_joint.setAttribute('class', '/')
        new_joint.setAttribute('type', 'free')
        #new_joint.setAttribute('damping', '0.001')
        new_body.appendChild(new_joint)
    world_body.appendChild(new_body)

    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

    return body_name, geom_names


# def add_objects(scene_name, object_info, run_id=None, material_name=None):
#     # scene_name = object_info['scene_name']
#     object_name = object_info['object_name']
#     mesh_names = object_info['mesh_names']
#     pos = object_info['pos']
#     size = object_info['size']
#     color = object_info['color']
#     quat = object_info['quat']

#     xmldoc = minidom.parse(scene_name)
#     # import pdb; pdb.set_trace()

#     assets = xmldoc.getElementsByTagName('asset')[0]
#     for mesh_ind in range(len(mesh_names)):
#         new_mesh = xmldoc.createElement('mesh')
#         new_mesh.setAttribute('name', f'gen_mesh_{object_name}_{mesh_ind}')
#         new_mesh.setAttribute('class', 'geom0')
#         # new_mesh.setAttribute('class', 'geom')
#         new_mesh.setAttribute('scale', f'{size[0]} {size[1]} {size[2]}')
#         new_mesh.setAttribute('file', mesh_names[mesh_ind])
#         assets.appendChild(new_mesh)

#     world_body = xmldoc.getElementsByTagName('worldbody')[0]

#     new_body = xmldoc.createElement('body')
#     body_name = f'gen_body_{object_name}'
#     new_body.setAttribute('name', body_name)
#     new_body.setAttribute('pos', f'{pos[0]} {pos[1]} {pos[2]}')
#     # new_body.setAttribute('euler', f'{rot[0]} {rot[1]} {rot[2]}')
#     new_body.setAttribute('quat', f'{quat[0]} {quat[1]} {quat[2]} {quat[3]}')
#     if object_info.get('mocap', False):
#         new_body.setAttribute('mocap', 'true')

#     geom_names = []
#     for geom_ind in range(len(mesh_names)):
#         new_geom = xmldoc.createElement('geom')
#         geom_name = f'gen_geom_{object_name}_{geom_ind}'
#         geom_names.append(geom_name)
#         new_geom.setAttribute('name', geom_name)
#         # new_geom.setAttribute('mass', '1')
#         new_geom.setAttribute('class', '/')
#         new_geom.setAttribute('type', 'mesh')
#         if not material_name is None:
#             new_geom.setAttribute('material', material_name)
#         if material_name is None:
#             if len(color) == 3:
#                 new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} 1')
#             else:
#                 new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} {color[3]}')
#         new_geom.setAttribute('mesh', f'gen_mesh_{object_name}_{geom_ind}')
#         new_body.appendChild(new_geom)

#     if not object_info.get('mocap', False):
#         new_joint = xmldoc.createElement('joint')
#         new_joint.setAttribute('name', f'gen_joint_{object_name}')
#         new_joint.setAttribute('class', '/')
#         new_joint.setAttribute('type', 'free')
#         #new_joint.setAttribute('damping', '0.001')
#         new_body.appendChild(new_joint)
#     world_body.appendChild(new_body)

#     with open(scene_name, "w") as f:
#         xmldoc.writexml(f)

#     return body_name, geom_names


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


def add_object_in_scene(object_info, scale_3d=False):
    synset_category = object_info['synset_category']
    object_mesh = trimesh.load(object_info['mesh_fname'], force='mesh')
    # Determine object size
    if scale_3d:
        scale_vec, scale_matrix = determine_object_scale(
            synset_category, object_mesh)
    else:
        scale_vec = [1, 1, 1]
        scale_matrix = np.eye(4)
    object_mesh.apply_transform(scale_matrix)
    object_bounds = object_mesh.bounds

    range_max = np.max(object_bounds[1] - object_bounds[0])
    object_size = 1 / range_max
    normalize_vec = [object_size, object_size, object_size]
    normalize_matrix = np.eye(4)
    normalize_matrix[:3, :3] *= normalize_vec
    object_mesh.apply_transform(normalize_matrix)
    obj_scale_vec = np.array(scale_vec) * np.array(normalize_vec)

    # Store the upright object in .stl file in assets
    r = R.from_euler('xyz', [(1/2)*np.pi, 0, 0], degrees=False)
    upright_mat = np.eye(4)
    upright_mat[0:3, 0:3] = r.as_matrix()
    object_mesh.apply_transform(upright_mat)
    upright_fname = object_info['mesh_names'][0]
    f = open(upright_fname, "w+")
    f.close()
    object_mesh.export(upright_fname)
    object_rot = np.random.uniform(0, 2*np.pi, 3)
    r2 = R.from_euler('xyz', object_rot, degrees=False)
    rotation_mat = np.eye(4)
    rotation_mat[0:3, 0:3] = r2.as_matrix()
    object_mesh.apply_transform(rotation_mat)
    object_bounds = object_mesh.bounds
    object_bottom = -object_bounds[0][2]

    return {
        'bounds': object_bounds,
        'size': [1, 1, 1],
        'scale': obj_scale_vec,
        'rot': object_rot,
    }


def add_object_in_scene_with_xyz(object_info, scale_3d=False):

    output_object_info = add_object_in_scene(object_info, scale_3d=False)

    # Determine object position
    object_x = np.random.uniform(
        object_info['x_sample_range'][0],
        object_info['x_sample_range'][1],
    )
    object_y = np.random.uniform(
        object_info['y_sample_range'][0],
        object_info['y_sample_range'][1],
    )
    object_z = object_info['table_height'] + object_bottom  # + 0.001
    object_xyz = [object_x, object_y, object_z]

    output_object_info.update({
        'pos': np.asarray(object_xyz),
    })
    return output_object_info


def add_table_in_scene(num_objects, table_color, table_mesh_fname, transformed_mesh_fname):
    # Choose table and table scale and add to sim
    # table_id = '97b3dfb3af4487b2b7d2794d2db4b0e7'
    # table_mesh_filename = os.path.join(shapenet_filepath, f'04379243/{table_id}/models/model_normalized.obj')
    table_mesh = trimesh.load(table_mesh_fname, force='mesh')
    # Rotate table so that it appears upright in Mujoco
    scale_mat = np.eye(4)
    r = R.from_euler('x', 90, degrees=True)
    scale_mat[0:3, 0:3] = r.as_matrix()
    table_mesh.apply_transform(scale_mat)
    table_bounds = table_mesh.bounds

    table_xyz_range = np.min(table_bounds[1, :2] - table_bounds[0, :2])
    table_size = 2*(num_objects + 2)/table_xyz_range
    table_scale = [table_size, table_size, table_size]
    # Export table mesh as .stl file
    #
    f = open(transformed_mesh_fname, "w+")
    f.close()
    table_mesh.export(transformed_mesh_fname)

    table_bounds = table_bounds*np.array([table_scale, table_scale])
    table_bottom = -table_bounds[0][2]
    # table_height = table_bounds[1][2] - table_bounds[0][2]

    # Move table above floor
    table_xyz = [0, 0, table_bottom]
    table_orientation = [0, 0, 0]

    # table_color = selected_colors[0]
    table_info = {
        'object_name': 'table',
        'bounds': table_bounds,
        'mesh_names': [transformed_mesh_fname],
        'pos': table_xyz,
        'size': table_scale,
        'color': table_color,
        'rot': table_orientation,
    }

    return table_info


def mujoco_pose_output(vec):
    position = vec[:3]
    object_quat = copy.deepcopy(vec[3:])
    object_quat[:3] = vec[3:][1:]
    object_quat[3] = vec[3:][0]
    rotation = R.from_quat(object_quat)
    return position, rotation


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

def create_cuboid_mesh(self, lx,ly,lz, save_file=None):
    wall_mesh = trimesh.creation.box((lx,ly,lz))
    # wall_mesh_file = os.path.join(self.scene_folder_path, f'assets/wall_{wall_idx}.stl')
    f = open(save_file, "w+")
    f.close()
    wall_mesh.export(save_file)

def create_walls(inner_pts, outer_pts, bottom_height=0):
    '''
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