import numpy as np

def make_pointcloud_all_points(depth_image):
    cam_scale = 1.0

    cam_cx = 320.0
    cam_cy = 240.0
    camera_params={'fx':579.411255, 'fy':579.411255, 'img_width':640, 'img_height': 480}
    
    depth_masked = depth_image.flatten()[:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[:, np.newaxis].astype(np.float32)
    
    pt2 = depth_masked / cam_scale
    pt0 = (ymap_masked/upsample - cam_cx) * pt2 / (camera_params['fx'])
    pt1 = (xmap_masked/upsample - cam_cy) * pt2 / (camera_params['fy'])
    cloud = np.concatenate((pt0, -pt1, -pt2), axis=1)
    return cloud    