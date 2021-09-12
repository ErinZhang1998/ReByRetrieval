# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from kornia import augmentation as K

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

__all__ = ["RetrievalMapper"]

class RetrievalMapper:
    def __init__(self, cfg, is_train = True):
        # augs = utils.build_augmentation(cfg, is_train)
        if is_train:
            augs = [
                T.Resize((480,640)),
                T.RandomBrightness(0.8, 1.3),
                T.RandomContrast(0.6, 1.3),
                T.RandomSaturation(0.8, 1.4),
                T.RandomLighting(0.7),
                T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
            ]
        else:
            augs = []
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augs)
        self.image_format           = cfg.INPUT.FORMAT
        self.use_instance_mask      = cfg.MODEL.MASK_ON
        self.instance_mask_format   = cfg.INPUT.MASK_FORMAT
        self.use_keypoint           = cfg.MODEL.KEYPOINT_ON
        self.keypoint_hflip_indices = None
        self.proposal_topk          = None
        self.recompute_boxes        = False
    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        return dataset_dict


# class MyColorAugmentation(T.Augmentation):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         if type(self.args.dataset_config.color_jitter.brightness) is list:
#             a,b = self.args.dataset_config.color_jitter.brightness
#             brightness = (a,b)
#         else:
#             brightness = self.args.dataset_config.color_jitter.brightness
#         if type(self.args.dataset_config.color_jitter.contrast) is list:
#             a,b = self.args.dataset_config.color_jitter.contrast
#             contrast = (a,b)
#         else:
#             contrast = self.args.dataset_config.color_jitter.contrast
#         if type(self.args.dataset_config.color_jitter.saturation) is list:
#             a,b = self.args.dataset_config.color_jitter.saturation
#             saturation = (a,b)
#         else:
#             saturation = self.args.dataset_config.color_jitter.saturation
#         if type(self.args.dataset_config.color_jitter.hue) is list:
#             a,b = self.args.dataset_config.color_jitter.hue
#             hue = (a,b)
#         else:
#             hue = self.args.dataset_config.color_jitter.hue
#         self.color_jitter_prob = self.args.dataset_config.color_jitter.prob
#         self.color_jitter_transform = K.ColorJitter(
#             brightness=brightness, 
#             contrast=contrast, 
#             saturation=saturation, 
#             hue=hue,
#             p=self.args.dataset_config.color_jitter.p,
#         )
        
#         self._init(locals())
    
    
#     def get_transform(self, image):
#         r = np.random.rand(2)
#         return T.ColorTransform(lambda x: x * r[0] + r[1] * 10)