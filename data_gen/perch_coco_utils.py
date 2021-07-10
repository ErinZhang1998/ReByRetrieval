import numpy as np 

class ImageAnnotation(object):
    def __init__(
        self,
        center,
        bbox, 
        rgb_file_path,
        mask_file_path, 
        model_name,
        percentage_not_occluded,
        number_pixels,
        cam_intrinsics, 
        object_location, 
        object_quat, 
        model_annotation,
        camera_annotation,
        width=640, 
        height=480,
    ):
        '''
        model_name --> category_id
        rgb_file_path --> image_id

        '''
        self.center = [int(item) for item in center]
        self.bbox = [int(item) for item in bbox]
        self.rgb_file_path = rgb_file_path
        self.mask_file_path = mask_file_path
        self.model_name = model_name
        self.percentage_not_occluded = float(percentage_not_occluded)
        self.number_pixels = int(number_pixels)
        cam_intrinsics = list(np.asarray(cam_intrinsics).astype(float))
        for row in range(len(cam_intrinsics)):
            cam_intrinsics[row] = list(cam_intrinsics[row])
        self.cam_intrinsics = cam_intrinsics
        self.object_location = list(np.asarray(object_location).astype(float))
        self.object_quat = list(np.asarray(object_quat).astype(float))
        self.width = int(width)
        self.height = int(height) 

        self.model_annotation = model_annotation
        self.camera_annotation = camera_annotation

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
            'location': self.object_location,
            'quaternion_xyzw': self.object_quat,
            'rgb_file_path' : self.rgb_file_path,
            'mask_file_path' : self.mask_file_path,
            'model_name' : self.model_name,
            'percentage_not_occluded' : self.percentage_not_occluded,
            'number_pixels' : self.number_pixels,
            'model_annotation' : self.model_annotation,
            'camera_annotation' : self.camera_annotation,
        }
        return output