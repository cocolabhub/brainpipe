import pickle
from scipy.io import loadmat, savemat
from os.path import splitext, isfile
import numpy as n

__all__ = ['savefile',
           'loadfile',
           'jobsMngmt',
           'list2index',
           'groupInList'
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
