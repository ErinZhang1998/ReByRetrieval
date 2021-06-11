import numpy as np
import utils.transforms as utrans
import torchvision

def from_world_to_camera_mat_to_tf(world_to_camera_mat):
    rot = world_to_camera_mat[:3,:3].T
    translation = np.linalg.inv(-world_to_camera_mat[:3,:3]) @ world_to_camera_mat[:3,3].reshape(3,1)
    return translation, rot

def make_pointcloud(depth_image, cam_cx=320.0, cam_cy=240.0, fx=579.411255, fy=579.411255, cam_scale=1.0, upsample=1):
    img_width = int(2*cam_cx)
    img_height = int(2*cam_cy)
    xmap = np.array([[j for i in range(int(upsample*img_width))] for j in range(int(upsample*img_height))])
    ymap = np.array([[i for i in range(int(upsample*img_width))] for j in range(int(upsample*img_height))])
    
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
    pc_mean=np.mean(obj_pointcloud, axis=0)

    obs_ptcld_min = np.amin(obj_pointcloud, axis=0)
    obs_ptcld_max = np.amax(obj_pointcloud, axis=0)
    scale = 4.0*float(np.max(obs_ptcld_max-obs_ptcld_min))

    obj_pointcloud = (obj_pointcloud - pc_mean) / scale
    obj_pointcloud = rot.dot(obj_pointcloud.T).T

    low=np.array([-0.5,-0.5,-0.5])
    hi=np.array([0.5,0.5,0.5])
    cloud_mask = np.argwhere(np.all(np.logical_and(obj_pointcloud>=low, obj_pointcloud<=hi), axis=1))
    obj_pointcloud=obj_pointcloud[cloud_mask][:,0,:] + 0.5

    return obj_pointcloud, cloud_mask.flatten()

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
    all_ptcld = make_pointcloud(depth)

    obj_points, obj_mask = process_pointcloud(all_ptcld, obj_points_inds, rot)
    #all_obj_points, all_obj_mask = pc.process_pointcloud(all_ptcld, all_obj_points_inds, rot)
    
    img_rgb = utrans.normalize(torchvision.transforms.ToTensor()(rgb_all), img_mean, img_std)
    x_ind, y_ind = np.unravel_index(obj_points_inds[obj_mask], (height, width))
    obj_points_features = img_rgb.permute(1,2,0)[x_ind, y_ind].numpy()

    return obj_points, obj_points_features