import numpy as np
import os 
import time 
import datetime


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
            if isinstance(myv, dict):
                assert isinstance(v, dict)
                subdict = fill_in_args_from_default(myv, v)
                filled_dict[k] = subdict
            else:
                filled_dict[k] = myv
        except:
            filled_dict[k] = v
    return filled_dict

def get_timestamp():                                                                                          
    ts = time.time()                                                                                            
    timenow = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')                             
    return timenow 

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
