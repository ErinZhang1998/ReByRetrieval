from __future__ import division
import torch
import numpy as np
import numbers
import collections
import cv2
import copy
import PIL


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

        resized_mask = cv2.resize(mask, (self.width,self.height), interpolation=cv2.INTER_NEAREST)
        resized_img = cv2.resize(img, (self.width,self.height), interpolation=cv2.INTER_NEAREST)
        
        return np.ascontiguousarray(resized_img), resized_mask, center_copy

class CropArea(object):
    def __init__(self, corners):
        self.corners = corners
        self.x0, self.y0 = np.min(self.corners, axis=0)
        self.x0, self.y0 = int(self.x0), int(self.y0)
        self.x1, self.y1 = np.max(self.corners, axis=0)
        self.x1, self.y1 = int(self.x1), int(self.y1)
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
    
    def __call__(self, img, mask, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        center_copy[0] = center_copy[0] - self.x0
        center_copy[1] = center_copy[1] - self.y0
        
        h, w = img.shape[0], img.shape[1]
        new_img = np.empty((self.height, self.width, 3), dtype=np.float32)
        new_img.fill(128)
        new_mask = np.empty((self.height, self.width), dtype=np.float32)
        new_mask.fill(0)
        new_img = img[self.y0: self.y1, self.x0: self.x1, :].copy()
        new_mask = mask[self.y0: self.y1, self.x0: self.x1].copy()
        
        return np.ascontiguousarray(new_img), new_mask, center_copy

def superimpose_image(canvas, img, mask):
    if np.max(img) <= 1:
        img = img * 255
    img_all = np.empty((img.shape[0], img.shape[1], 3), dtype=np.float32)
    img_all.fill(0)
    
    anti_mask = np.logical_not(mask > 0)
    
    for i in range(3):
        
        img_all[:,:,i][anti_mask] = canvas[:,:,i][anti_mask]
        img_all[:,:,i][mask > 0] = img[:,:,i][mask > 0]
    return np.uint8(img_all)

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

def mask_to_PIL(pil_img):
    pil_img[pil_img > 0] = 1
    pil_img_3 = np.stack([pil_img]*3,axis=2)
    pil_img_3 = 1 - pil_img_3
    PIL_image = PIL.Image.fromarray(np.uint8(pil_img_3 * 255)).convert('RGB')
    return PIL_image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, center):
        center_copy = copy.deepcopy(center)
        img_copy = copy.deepcopy(img)
        mask_copy = copy.deepcopy(mask)
        for t in self.transforms:
            img_copy, mask_copy, center_copy = t(img_copy, mask_copy, center_copy)

        return img_copy, mask_copy, center_copy


class PILCropArea(object):
    def __init__(self, corners):
        self.corners = corners
        self.x0, self.y0 = np.min(self.corners, axis=0)
        self.x0, self.y0 = int(self.x0), int(self.y0)
        self.x1, self.y1 = np.max(self.corners, axis=0)
        self.x1, self.y1 = int(self.x1), int(self.y1)
        self.width = self.x1 - self.x0
        self.height = self.y1 - self.y0
        
        self.area = (self.x0,self.y0,self.x1, self.y1)
    
    def __call__(self, img, mask, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        center_copy[0] = center_copy[0] - self.x0
        center_copy[1] = center_copy[1] - self.y0
        
        img = img.crop(self.area)
        mask = mask.crop(self.area)
        
        return img,mask, center_copy

class PILRandomHorizontalFlip(object):
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, img, mask, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        w,h = img.size

        if np.random.random() < self.prob:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            center_copy[0] = w - center_copy[0]
            return img,  mask, center_copy
        
        return img, mask, center_copy

class PILResized(object):
    def __init__(self, width = 227, height = 227):
        self.width = width
        self.height = height
        
    
    def __call__(self, img, mask, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        h, w = img.shape[0], img.shape[1]
        center_copy[0] *= (self.width / w)
        center_copy[1] *= (self.height / h)

        resized_mask = mask.resize((self.width,self.height))
        resized_img = img.resize((self.width,self.height))
        
        return resized_img, resized_mask, center_copy
