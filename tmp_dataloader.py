from __future__ import print_function

import imageio
import numpy as np
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import pickle 
import os 

class VOCDataset(Dataset):
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    INV_CLASS = {}
    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    # TODO: Adjust data_dir according to where **you** stored the data
    def __init__(self, split, size, data_dir='/home/ubuntu/VOCdevkit/VOC2007/'):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.size = size
        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]

        self.anno_list = self.preload_anno()

    @classmethod
    def get_class_name(cls, index):
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)

    def preload_anno(self):
        """
        :return: a list of lables. each element is in the form of [class, weight],
         where both class and weight are a numpy array in shape of [20],
        """
        # load from pickle file if one exists
        curdit = os.getcwd()
        filename = os.path.join(curdit, '{}.pkl'.format(self.split))
#         if os.path.exists(filename):
#             print("Preload from file the annotation list", filename)
#             with open(filename, 'rb') as f:
#                 return pickle.load(f)
#         else:
#             print("Fail to preload from file the annotation list", filename)
        
        label_list = []
        i = 0
        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)
            root = tree.getroot()
            # label = np.zeros(len(CLASS_NAMES)).astype('int')
            # weight = np.ones(len(CLASS_NAMES)).astype('int')
            # for obj in root.findall('object'):
            #     name = obj.find('name').text
            #     label[INV_CLASS[name]] = 1
            #     difficult = int(obj.find('difficult').text)
            #     if difficult == 1:
            #         weight[INV_CLASS[name]] = 0
            
            label = np.zeros(len(self.CLASS_NAMES)).astype('int')
            not_difficult_once = np.zeros(len(self.CLASS_NAMES)).astype('int')
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                label[self.INV_CLASS[name]] = 1
                difficult = int(obj.find('difficult').text)
                if difficult == 0:
                    not_difficult_once[self.INV_CLASS[name]] = 1
            weight = ((1-label) + not_difficult_once > 0).astype('int')
            
            label_list.append([label, weight])
            i+=1
        with open(filename, 'wb+') as f:
            pickle.dump(label_list, f)
        return label_list

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element
        image: FloatTensor in shape of (C, H, W) in scale [-1, 1].
        label: LongTensor in shape of (Nc, ) binary label
        weight: FloatTensor in shape of (Nc, ) difficult or not.
        """
        findex = self.index_list[index]
        fpath = os.path.join(self.img_dir, findex + '.jpg')
        # TODO: insert your code here. hint: read image, find the labels and weight.
        img = Image.open(fpath)#.convert('RGB')
        '''
        Add random crops and left-right flips when training, and do a center crop when testing, etc. 
        As for natural images, another common practice is to subtract the mean values of RGB images from ImageNet dataset.
        The mean values for RGB images are: [123.68, 116.78, 103.94] – sometimes, rescaling to [−1, 1] suffices
        '''
        if self.split == 'train' or self.split == 'trainval':
            trans = transforms.Compose([
                    transforms.Resize((int(1.25*self.size),int(1.25*self.size))),
                    transforms.RandomCrop((self.size,self.size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
            # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            trans = transforms.Compose([
                    transforms.Resize((int(1.25*self.size),int(1.25*self.size))),
                    transforms.CenterCrop((self.size,self.size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        
        img = trans(img)
        lab_vec,wgt_vec = self.anno_list[index]
        
        image = torch.FloatTensor(img)
        label = torch.FloatTensor(lab_vec)
        wgt = torch.FloatTensor(wgt_vec)
        return image, label, wgt
