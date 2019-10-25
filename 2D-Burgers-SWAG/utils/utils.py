import os, errno

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
    Converts Pytorch tensor to numpy array
    '''
    return tensor.detach().cpu().numpy()

def toTuple(a):
    '''
    Converts array to tuple
    '''
    try:
        return tuple(toTuple(i) for i in a)
    except TypeError:
        return a