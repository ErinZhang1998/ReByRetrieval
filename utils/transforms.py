from __future__ import division
import torch
import numpy as np
import numbers
import collections
import cv2
import copy

'''
img.shape
(480, 640, 3)
(height, width, channel)
'''

def normalize(tensor, mean, std):

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def to_tensor(pic):
    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()

class Resized(object):
    def __init__(self, width = 227, height = 227):
        self.width = width
        self.height = height
        
    
    def __call__(self, img, mask, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        h, w = img.shape[0], img.shape[1]
        # center_copy[0] = w - center_copy[0]
        center_copy[0] *= (self.width / w)
        center_copy[1] *= (self.height / h)

        resized_mask = cv2.resize(mask, (self.height,self.width), interpolation=cv2.INTER_NEAREST)
        resized_img = cv2.resize(img, (self.height,self.width), interpolation=cv2.INTER_NEAREST)
        
        return np.ascontiguousarray(resized_img), resized_mask, center_copy


def horizontal_flip(img, flip):
    if flip:
        if len(img.shape) < 3:
            img = img[:, ::-1]
        else:
            img = img[:, ::-1, :]
    return img


class RandomHorizontalFlip(object):
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, img, mask, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        h, w= img.shape[0], img.shape[1]

        if np.random.random() < self.prob:
            img = horizontal_flip(img, True)
            mask = horizontal_flip(mask, True)
            center_copy[0] = w - center_copy[0]
            return np.ascontiguousarray(img),  mask, center_copy
        
        return img, mask, center_copy


class Compose(object):
    """
    img: (480, 640, 3)
    mask: (480, 640)
    
    Example:
        >>> Mytransforms.Compose([
        >>>      Mytransforms.RandomResized(),
        >>>      Mytransforms.RandomRotate(40),
        >>>      Mytransforms.RandomCrop(368),
        >>>      Mytransforms.RandomHorizontalFlip(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, center):
        center_copy = copy.deepcopy(center)
        img_copy = copy.deepcopy(img)
        mask_copy = copy.deepcopy(mask)
        for t in self.transforms:
            img_copy, mask_copy, center_copy = t(img_copy, mask_copy, center_copy)

        return img_copy, mask_copy, center_copy