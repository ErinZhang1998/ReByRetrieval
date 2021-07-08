import numpy as np 

class ImageAnnotation(object):
    def __init__(
        self, 
        center_3d,
        center,
        bbox, 
        rgb_file_path,
        mask_file_path, 
        model_file_path,
        percentage_not_occluded,
        cam_intrinsics, 
        cam_location, 
        cam_quat, 
        width=640, 
        height=480
    ):
        '''
        model_file_path --> category_id
        rgb_file_path --> image_id

        '''
        self.center_3d = [float(item) for item in center_3d]
        self.center = [int(item) for item in center]
        self.bbox = [int(item) for item in bbox]
        self.rgb_file_path = rgb_file_path
        self.mask_file_path = mask_file_path
        self.model_file_path = model_file_path
        self.percentage_not_occluded = float(percentage_not_occluded)
        cam_intrinsics = list(np.asarray(cam_intrinsics).astype(float))
        for row in range(len(cam_intrinsics)):
            cam_intrinsics[row] = list(cam_intrinsics[row])
        self.cam_intrinsics = cam_intrinsics
        self.cam_location = list(np.asarray(cam_location).astype(float))
        self.cam_quat = list(np.asarray(cam_quat).astype(float))
        self.width = int(width)
        self.height = int(height) 

        self.id, self.image_id, self.category_id = None, None, None  
    
    def output_dict(self):
        output = {
            'center' : self.center,
            'iscrowd': 0,
            'area': 0,
            'bbox': self.bbox,
            'segmentation': None,
            'width': self.width,
            'height': self.height,
            'camera_pose': {},
            'camera_intrinsics': self.cam_intrinsics,
            'location': self.cam_location,
            'quaternion_xyzw': self.cam_quat,
            'rgb_file_path' : self.rgb_file_path,
            'mask_file_path' : self.mask_file_path,
            'model_file_path' : self.model_file_path,
            'percentage_not_occluded' : self.percentage_not_occluded,
            'center_3d' : self.center_3d,
        }
        return output