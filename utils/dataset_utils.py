import os 
import numpy as np
import matplotlib.image as mpimg


#!/usr/bin/env python3

import math
import numpy as np

from PIL import Image as PIL_Image

import open3d as o3d


"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def generateCroppedPointCloud(depth_img, intrinsic, camera_to_world_mat, img_width, img_height):
    od_cammat = cammat2o3d(intrinsic, img_width, img_height)
    od_depth = o3d.geometry.Image(depth_img)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
    transformed_cloud = o3d_cloud.transform(camera_to_world_mat)

    return transformed_cloud

def compile_mask_files(dir_path):
    '''
    Dictionary mapping from camera index (in str form, e.g. '00039') to files of individual
    object segmentation capture by the given camera
    '''
    mask_all_d = dict()
    for seg_path in os.listdir(dir_path):
        seg_path_pre = seg_path.split('.')[0]
        l = seg_path_pre.rsplit('_')
        if len(l) == 3 and l[0] == 'segmentation':
            other_obj_masks = mask_all_d.get(l[1], [])
            other_obj_masks.append(os.path.join(dir_path, seg_path))
            mask_all_d[l[1]] = other_obj_masks
    return mask_all_d

def data_dir_list(root_dir):
    l = []
    for subdir in os.listdir(root_dir):
        if subdir.startswith('scene_'):
            subdir_path = os.path.join(root_dir, subdir)
            scene_description_dir = os.path.join(subdir_path, 'scene_description.p')
            if not os.path.exists(scene_description_dir):
                continue 
            scene_name = subdir_path.split("/")[-1]
            # if int(subdir_path[-3] ) < 2:
            #     continue
            l.append(subdir_path)

    return l 
