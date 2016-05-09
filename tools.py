import numpy as n

__all__ = ['binarize',
           'binArray',
           'p2str',
           'list2index',
           'groupInList'
           ]


def binarize(starttime, endtime, width, step, kind='tuple'):
    """Generate a window to binarize a signal

    starttime : int
        Start at "starttime"

    endtime : int
        End at "endtime"

    width : int, optional [def : None]
        width of a single window

    step : int, optional [def : None]
        Each window will be spaced by the "step" value

    kind : string, optional, [def: 'list']
        Return either a list or a tuple
    """
    X = n.vstack((n.arange(starttime, endtime-width+step, step),
                  n.arange(starttime+width, endtime+step, step)))
    if X[1, -1] > endtime:
        X = X[:, 0:-1]
    if kind == 'array':
        return X
    if kind == 'list':
        return [[X[0][k], X[1][k]] for k in range(X.shape[1])]
    elif kind == 'tuple':
        return [(X[0][k], X[1][k]) for k in range(X.shape[1])]


def binArray(x, binList, axis=0):
    """Binarize an array

    x : array
        Array to binarize

    binList : list of tuple/list
        This list contain the index to binarize the array x

    axis : int, optional, [def: 0]
        Binarize along the axis "axis"

    -> Return the binarize x and the center of each window.
    """
    nbin = len(binList)
    x = n.swapaxes(x, 0, axis)

    xBin = n.zeros((nbin,)+x.shape[1::])
    for k, i in enumerate(binList):
        if i[1] - i[0] == 1:
            xBin[k, ...] = x[i[0], ...]
        else:
            xBin[k, ...] = n.mean(x[i[0]:i[1], ...], 0)

    return n.swapaxes(xBin, 0, axis), [(k[0]+k[1])/2 for k in binList]


def p2str(p):
    """Convert a pvalue to a string. Usefull for saving !
    """
    pStr = str(p)
    return pStr[-1]+'e-'+str(len(pStr[pStr.find('.')+1::]))


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

