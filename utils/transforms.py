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
        
    
    def __call__(self, img, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        h, w = img.shape[0], img.shape[1]
        # center_copy[0] = w - center_copy[0]
        center_copy[0] *= (self.width / w)
        center_copy[1] *= (self.height / h)

        # resized_img = cv2.resize(img, (self.height,self.width), interpolation=cv2.INTER_CUBIC)
        resized_img = cv2.resize(img, (self.height,self.width), interpolation=cv2.INTER_NEAREST)
        
        return np.ascontiguousarray(resized_img), center_copy


def crop(img, center, offset_left, offset_up, w, h):
    center[0] -= offset_left
    center[1] -= offset_up

    height, width= img.shape[0], img.shape[1]
    new_img = np.empty((h, w, 3), dtype=np.float32)
    new_img.fill(128)

    st_x = 0
    ed_x = w
    st_y = 0
    ed_y = h
    or_st_x = offset_left
    or_ed_x = offset_left + w
    or_st_y = offset_up
    or_ed_y = offset_up + h

    # the person_center is in left
    if offset_left < 0:
        st_x = -offset_left
        or_st_x = 0
    if offset_left + w > width:
        ed_x = width - offset_left
        or_ed_x = width
    # the person_center is in up
    if offset_up < 0:
        st_y = -offset_up
        or_st_y = 0
    if offset_up + h > height:
        ed_y = height - offset_up
        or_ed_y = height

    new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()

    return np.ascontiguousarray(new_img), center

class RandomCrop(object):

    def __init__(self, size = 227, center_perturb_max = 5):
        assert isinstance(size, numbers.Number)
        self.size = [int(size), int(size)] # (w, h) (227, 227)
        self.center_perturb_max = center_perturb_max
        self.ratio_x = np.random.uniform(0, 1)
        self.ratio_y = np.random.uniform(0, 1)

    def get_params(self, img, center, output_size, center_perturb_max):
        
        x_offset = int((self.ratio_x - 0.5) * 2 * center_perturb_max)
        y_offset = int((self.ratio_y - 0.5) * 2 * center_perturb_max)
        print(x_offset, y_offset)
        center_x = center[0] + x_offset
        center_y = center[1] + y_offset

        return int(round(center_x - output_size[0] / 2)), int(round(center_y - output_size[1] / 2))

    def __call__(self, img, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        h, w= img.shape[0], img.shape[1]
        # center_copy[0] = w - center_copy[0]
        offset_left, offset_up = self.get_params(img, center_copy, self.size, self.center_perturb_max)
        return crop(img, center_copy, offset_left, offset_up, self.size[0], self.size[1])


class RandomHorizontalFlip(object):
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, img, center):
        center_copy = copy.deepcopy(center)
        center_copy = center_copy.reshape(-1,)
        h, w= img.shape[0], img.shape[1]
        # center_copy[0] = w - center_copy[0]

        if np.random.random() < self.prob:
            if len(img.shape) < 3:
                img = img[:, ::-1]
            else:
                img = img[:, ::-1, :]
            center_copy[0] = w - 1 - center_copy[0]
            return np.ascontiguousarray(img), center_copy
        
        return img, center_copy


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

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

    def __call__(self, img, center):
        center_copy = copy.deepcopy(center)
        for t in self.transforms:
            img, center_copy = t(img, center_copy)

        return img, center_copy