import numpy as np
import utils.transforms as utrans
from data_gen.datagen_utils import from_depth_img_to_pc, process_pointcloud
import torchvision

def get_pointcloud(rgb_all, depth_all, mask, mask_all, rot, img_mean, img_std):
    '''
    depth_all: PIL image of the entire scene
    mask: PIL image of the object's mask
    mask_all: PIL image of all objects' masks
    '''
    depth = np.asarray(depth_all)
    height, width = depth.shape
    obj_label = 255 - np.asarray(mask)[:,:,0]
    # all_labels = 255 - np.asarray(mask_all)[:,:,0]

    obj_points_inds = np.where(obj_label, depth, 0.0).flatten().nonzero()[0]
    #all_obj_points_inds = np.where(all_labels, depth, 0.0).flatten().nonzero()[0]
    all_ptcld = from_depth_img_to_pc(depth, cam_cx=320., cam_cy=220., fx=579.4112549695428, fy=579.4112549695428)

    obj_points, obj_mask = process_pointcloud(all_ptcld, obj_points_inds, rot)
    #all_obj_points, all_obj_mask = pc.process_pointcloud(all_ptcld, all_obj_points_inds, rot)
    
    img_rgb = utrans.normalize(torchvision.transforms.ToTensor()(rgb_all), img_mean, img_std)
    x_ind, y_ind = np.unravel_index(obj_points_inds[obj_mask], (height, width))
    obj_points_features = img_rgb.permute(1,2,0)[x_ind, y_ind].numpy()

    return obj_points, obj_points_features