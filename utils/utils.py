import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wandb
import io 
from PIL import Image
import PIL
import copy

class Struct:
    '''The recursive class for building and representing objects with.'''
    def __init__(self, obj):
        self.obj_dict = obj
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)
    
    def __getitem__(self, val):
        return self.__dict__[val]
    
    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def fill_in_args_from_default(my_dict, default_dict):
    filled_dict = {}
    for k,v in default_dict.items():
        try:
            myv = my_dict[k]
            filled_dict[k] = myv
        except:
            filled_dict[k] = v
    return filled_dict