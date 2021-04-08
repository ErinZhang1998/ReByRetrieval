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
    res = copy.deepcopy(tensor)
    for i, (m, s) in enumerate(zip(mean, std)):
        if len(res.shape) == 4:
            res[:,i,:,:].sub_(m).div_(s)
        else:
            res[i,:,:].sub_(m).div_(s)
    return res

def denormalize(tensor, mean, std):
    res = copy.deepcopy(tensor)
    for i, (m, s) in enumerate(zip(mean, std)):
        if len(res.shape) == 4:
            res[:,i,:,:].mul_(s).add_(s)
        else:
            res[i,:,:].mul_(s).add_(s)
    return res

def to_tensor(pic):
    '''
    pic: numpy array of images
    (N,size,size,channel) or (size,size,channel) or any 2d array
    '''
    if len(pic.shape) == 4:
        img = torch.from_numpy(pic.transpose((0, 3, 1, 2)).copy())
    else:
        if len(pic.shape) < 3:
            img = torch.from_numpy(pic.copy())
        else:
            img = torch.from_numpy(pic.transpose((2, 0, 1)).copy())

    return img.float()

def from_tensor(pic):
    '''
    pic: torch tensor
    (N,channel, size,size) or (channel, size,size) or any 2d tensor
    '''
    if len(pic.shape) == 4:
        img = pic.permute((0,2,3,1)).numpy()
    else:
        if len(pic.shape) < 3:
            img = pic.numpy()
        else:
            img = pic.permute((1,2,0)).numpy()

    return img

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
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
            center_copy[0] = w - center_copy[0]
            return img,  mask, center_copy
        
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