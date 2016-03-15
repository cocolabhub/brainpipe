import numpy as n
from pandas import DataFrame

__all__ = [
    'meanLabel', 'nbest', 'countlist', 'extendlist', 'binarize',
    'replaceElementList', 'reorderList', 'MeltingSort', 'repDataFrame'
]

####################################################################
# - Get the mean of matrix using a vector label :
####################################################################
def meanLabel(x,y):
    # Get and check size elements :
    x = n.matrix(x)
    y = n.array(y)
    yLen = len(y)
    yUnique = n.unique(y)
    yUniqueLen = len(yUnique)
    rdim = n.arange(0,len(x.shape),1)[n.array(x.shape) == yLen]
    if len(rdim) != 0 : rdim = rdim[0]
    else: raise ValueError("None of x dimendion is "+str(N)+" length. [x] = "+str(x.shape))
    if rdim == 0: x = x.T

    xMean = n.zeros( (x.shape[0], yUniqueLen) )
    for k in range(0,yUniqueLen):
        xMean[:,k] = n.ravel(n.mean(x[:,y == yUnique[k] ], axis=1))

    return xMean


####################################################################
# - Get the n best values of a list :
####################################################################
def nbest(x, nbestval):
    nbestlist = []
    nbestidx = []
    for k in range(0, nbestval):
        x = n.array(x)
        nbestidx.extend([x.argmax()])
        nbestlist.extend([x.max()])
        x = list(x)
        x[n.array(x).argmax()] = n.array(x).min()

    return nbestlist, nbestidx

####################################################################
# - Count the number of element in a list :
####################################################################
def countlist(l,order='off'):
    listunq = n.unique(l)
    listlen = len(listunq)
    listcount = n.zeros((1,listlen))
    for k in range(0,listlen):
        listcount[0,k] = l.count(listunq[k])

    listcnt = listcount[0].astype(int)

    if order is 'off':
        return listcnt, listunq
    else:
        sortlistidx = n.flipud(n.array(listcnt).argsort())
        return listcnt[sortlistidx], listunq[sortlistidx]

####################################################################
# - Extend each element of a list :
####################################################################
def extendlist(xname,xnum=1):
    listF = []
    for k in range(0, len(xname)):
        listF.extend(list(xname[k])*xnum)

    return listF

####################################################################
# - Define a time vector :
####################################################################
def binarize(starttime,endtime,width,step, kind='list'):
    X = n.vstack( (n.arange(starttime,endtime-width,step),n.arange(starttime+width,endtime,step)) )
    if kind == 'list':
        return X
    elif kind == 'tuple':
        X2 = list(X[0]) + list(X[1])
        return [ (X2[k],X2[k+len(list(X[0]))]) for k in n.arange(0,len(list(X[0])),1) ]


####################################################################
# - Replace element in a list :
####################################################################
def replaceElementList(lst, fromList, toList):

    if len(fromList) != len(toList): raise ValueError('"fromList" and "toList" must have the same dimension')
    else: nbRpl = len(fromList)

    for k in range(0,nbRpl):
        lst = [toList[k] if x == fromList[k] else x for x in lst]
    return lst

####################################################################
# - Reorder a list :
####################################################################
def reorderList(LstDef, LstDesorder):
    LstOrder = [n.nan]*len(LstDef)

    for k in range(0,len(LstDesorder)):
        LstOrder[LstDef.index(LstDesorder[k])] = LstDesorder[k]

    LstOrderNotNan = [x for x in LstOrder if x is not n.nan]
    return LstOrderNotNan


####################################################################
# - Sort a list containing diffrent element types :
####################################################################
def MeltingSort(ObjToSort, to_replace=None, value=None, colname=None):
    # The object to sort can be either a list or a pandas Dataframe :
    if type(ObjToSort) == list: LstToSort = ObjToSort
    else:
        if colname is None:
            raise ValueError('PLease enter a name of a column for the panda Dataframe')
        else:
            LstToSort = ObjToSort[colname]
    # First, remove the part of each element wich is before/after the number :
    if to_replace != None:
        LstMelt = [x.replace(to_replace, value) for x in LstToSort]
    else: LstMelt = LstToSort
    # Separate :
    LstMeltSort, LstRest = [], []
    for k in LstMelt:
        try:
             int(k)
             LstMeltSort.append( k )
        except :
             LstRest.append( k )
    # Assemble the sorted int list with others elemnets :
    LstMeltSort = sorted(LstMeltSort, key=int)+LstRest
    # Get the new index :
    LstMeltSortCp = LstMeltSort.copy()
    LstIndex = []
    for k in LstMelt:
        LstIndex.append(LstMeltSortCp.index(k))
        LstMeltSortCp[LstMeltSortCp.index(k)] = n.nan
    # Finally manage with pandas Dataframe :
    if type(ObjToSort) == list: LstOut = LstMeltSort
    else:
#         TestPd = pd.DataFrame()
        ObjToSort['IndexCol'] = LstIndex
        ObjToSort = ObjToSort.sort(['IndexCol'])
        ObjToSort=ObjToSort.set_index([list(n.arange(ObjToSort.shape[0]))])
        LstOut = ObjToSort.drop('IndexCol', axis=1)

    return LstOut, LstIndex

####################################################################
# - Repeat a DataFrame :
####################################################################
def repDataFrame(X, rep):
    Xrep = DataFrame()
    for k in range(0,rep):
        Xrep = Xrep.append(X)
    return Xrep