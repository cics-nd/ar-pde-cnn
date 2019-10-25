import torch
import numpy as np
import os, errno

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(*directories):
    '''
    Makes a directory if it does not exist
    '''
    for directory in list(directories):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def toNumpy(tensor):
    '''
    Converts pytorch tensor to numpy array
    '''
    return tensor.detach().cpu().numpy()