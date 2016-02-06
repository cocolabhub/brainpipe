import pickle
from scipy.io import loadmat, savemat
from os.path import splitext, isfile

__all__ = ['savefile', 'loadfile']


def savefile(name, **kwargs):
    """Save a file without carrying of extension.
    """
    name = _safetySave(name)
    fileName, fileExt = splitext(name)
    if fileExt == '.pickle':
        with open(name, 'wb') as f:
            pickle.dump(kwargs, f)
    elif fileExt == '.mat':
        data = savemat(name, kwargs)


def loadfile(name):
    """Load a file without carrying of extension. The function return
    a dictionnary data.
    """
    fileName, fileExt = splitext(name)
    if fileExt == '.pickle':
        with open(name, "rb") as f:
            data = pickle.load(f)
    elif fileExt == '.mat':
        data = loadmat(name)

    return data


def _safetySave(name):
    """Check if a file name exist. If it exist, increment it with '(x)'
    """
    k = 1
    while isfile(name):
        fname, fext = splitext(name)
        if fname.find('(')+1:
            name = fname[0:fname.find('(')+1]+str(k)+')'+fext
        else:
            name = fname+'('+str(k)+')'+fext
        k += 1
    return name
