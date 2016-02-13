import pickle
from scipy.io import loadmat, savemat
from os.path import splitext, isfile
import numpy as n

__all__ = ['savefile',
           'loadfile',
           'jobsMngmt',
           'list2index',
           'groupInList',
           'adaptsize'
           ]


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


def jobsMngmt(n_jobs, **kwargs):
    """Manage the jobs repartition between loops
    """
    def _jobsAssign(val):
        for i, k in enumerate(kwargs.keys()):
            kwargs[k] = int(val[i])
        return kwargs

    if n_jobs == 1:
        return _jobsAssign([1]*len(kwargs))
    else:
        keys = list(kwargs.keys())
        values = n.array(list(kwargs.values()))
        jobsRepartition = list(n.ones(len(keys)))
        jobsRepartition[values.argmax()] = n_jobs
        return _jobsAssign(jobsRepartition)


def list2index(dim1, dim2):
    """From two dimensions dim1 and dim2, build a list of
    tuple which combine this two list
    Example:
    for (2,3) -> [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)]
    """
    list1 = list(n.arange(dim1))*dim2
    list2 = sum([[k]*dim1 for k in range(dim2)], [])
    return list(zip(list1, list2)), list1, list2


def groupInList(x, idx):
    """Group elements in an array/list using a list of index
    Example:
    groupInList([1,2,3,4,5],[0,0,1,1,2]) = [[1,2],[3,4],[5]]
    """
    if not isinstance(x, n.ndarray):
        x = n.array(x)
    if not isinstance(idx, list):
        idx = list(idx)
    # Get the list of unique elements in idx:
    uelmt = list(set(idx))
    idx = n.array(idx)
    return [list(x[n.where(idx == k)]) for k in uelmt]


def adaptsize(x, dim, whre=None):
    """Adapt the dimension of an array depending of the tuple dim
    x : the signal for swaping axis
    dim : the dim dimensions
    whre : a list to define where to put this new dimension

    Example :
    x.shape = (2, 4001, 160)
    x2 = adaptsize(x, (4001, 2), whre=[2,1])
    x2.shape = (160, 2, 4001)
    """
    if isinstance(dim, int):
        goto = 1
        dim = [dim]
    else:
        if len(dim) < len(x.shape):
            goto = len(dim)
        else:
            goto = len(dim)-1
    if not isinstance(dim, list):
        dim = list(dim)
    if not whre:
        whre = list(n.arange(len(x.shape)))
    if isinstance(whre, int):
        whre = [whre]

    for k in range(goto):
        x = x.swapaxes(n.where(n.array(x.shape) == dim[k])[0], whre[k])
    return x
