import numpy as np 

class ImageAnnotation(object):
    def __init__(self, bbox, mask_file_path, cam_intrinsics, cam_location, cam_quat, width=640, height=480):
        self.bbox = bbox
        self.mask_file_path = mask_file_path
        self.cam_intrinsics = cam_intrinsics
        self.cam_location = cam_location
        self.cam_quat = cam_quat
        self.width = width
        self.height = height 

        self.id, self.image_id, self.category_id = None, None, None 
    
    def assign_id(self, id):
        self.id = id 
    
    def assign_image_id(self, image_id):
        self.image_id = image_id 
    
    def assign_category_id(self, category_id):
        self.category_id = category_id 
    
    def output_dict(self):
        assert not self.id is None
        assert not self.image_id is None
        assert not self.category_id is None
        output = {
            'id': self.id,
            'image_id': self.image_id,
            'category_id': self.category_id,
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

        }
        return output