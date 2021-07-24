import numpy as np 
import datagen_utils

class ImageAnnotation(object):
    def __init__(self, ann):
        '''
        model_name --> category_id
        rgb_file_path --> image_id

        '''
        self.ann = ann
    
    def output_dict(self):
        output = {
            'center' : [int(item) for item in self.ann['center']],
            'iscrowd': 0,
            'area': 0,
            'bbox': [int(item) for item in self.ann['bbox']],
            'segmentation': None,
            'width': int(self.ann.get('width', 640)),
            'height': int(self.ann.get('width', 480)),
            'camera_pose': {},
            'location': [float(item) for item in np.asarray(self.ann['location'])],
            'quaternion_xyzw': [float(item) for item in np.asarray(self.ann['quaternion_xyzw'])],
            "cam_num" : self.ann['cam_num'],
            "object_idx" : self.ann['object_idx'],
            'model_name' : self.ann['model_name'],
            'percentage_not_occluded' : float(self.ann['percentage_not_occluded']),  
            'number_pixels' : int(self.ann['number_pixels']), 
            'mask_file_path' : self.ann['mask_file_path'], 
        }
        return output