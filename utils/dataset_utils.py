import os 
import numpy as np
import matplotlib.image as mpimg


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

def compile_mask(mask_path_lists):
    masks = []
    for mask_path in mask_path_lists:
        mask = mpimg.imread(mask_path)
        masks.append(mask)
    
    return np.sum(np.stack(masks), axis=0)

def project_back_3d(P, world_to_camera_tf_mat, pt_2d, mult):
    '''
    pt_2d : (N, 2)
    muly : (N, )
    '''
    N = len(pt_2d)
    center_pad = np.append(pt_2d.T, np.ones(N).astype('int').reshape(1,-1), axis=0)
    center_pad = center_pad * mult

    pt_3d_camera_pre = np.linalg.inv(P) @ center_pad
    pt_3d_camera = np.append(pt_3d_camera_pre, np.ones(len(pt_2d)).astype('int').reshape(1,-1), axis=0)

    pt_3d_homo = np.linalg.inv(world_to_camera_tf_mat) @ pt_3d_camera
    pt_3d_homo = pt_3d_homo / pt_3d_homo[-1,:]
    pt_3d_homo = pt_3d_homo[:-1].T
    return pt_3d_homo

def get_mult(P, world_to_camera_tf_mat, pt_3d):
    N = len(pt_3d)
    pt_3d_homo = np.append(pt_3d.T, np.ones(N).astype('int').reshape(1,-1), axis=0) #(4,N)
    pt_3d_camera = world_to_camera_tf_mat @ pt_3d_homo #(4,N)
    pixel_coord = P @ (pt_3d_camera[:-1, :])
    mult = pixel_coord[-1, :]
    return mult 

def compile_camera_info(object_descriptions):
    object_indices = object_descriptions['object_indices']
    cam_info = object_descriptions[object_indices[0]]['object_cam_d']
    cam_d = dict()
    for cam_num, v in cam_info.items():
        cam_d_cam_num = dict()
        cam_d_cam_num['P'] = v['intrinsics']
        cam_d_cam_num['world_to_camera_mat'] = v['world_to_camera_mat']
        cam_d[cam_num] = cam_d_cam_num
    
    positions = []
    for obj_idx in object_indices:
        positions.append(object_descriptions[obj_idx]['position'])
    all_position = np.stack(positions)
    for cam_num, v in cam_d.items():
        mult = get_mult(v['P'], v['world_to_camera_mat'], all_position)
        v['mult'] = dict(zip(object_indices, mult)) 
        cam_d[cam_num] = v
    return cam_d