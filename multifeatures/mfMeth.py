from brainpipe.classification.classification_tools import classify, classifier_choice
from brainpipe.multifeatures.sequence import sequence_inner
from brainpipe.statistic.stat_tools import binofeat
import numpy as n

__all__ = ['select_MF', 'apply_MF']

####################################################################
# - Select and return a mf model :
####################################################################


def select_MF(Id, y, clfIn, p=0.05, display=False, direction='forward',
              nbest=10, grpType='sf', **kwargs):
    """Centralize the implemented model for the multifeatures which can be
    selected using the Id Id can be a string containing multiple elements.
    [ex: Id='031']

    Parameters
    ----------
    Id : string
        '0': No selection. All the features are used
        '1': Select <p significant features using a binomial law
        '2': Select <p significant features using permutations
        '3': use 'forward'/'backward'/'exhaustive'to  select features
        '4': Select the nbest features

    y : array-like
        The target variable to try to predict in the case of
        supervised learning

    clfIn : estimator object implementing 'fit'
        The object to use to fit the data

    p : float < 1, default: 0.05
        The significiance level to select features

    display : 'on' or 'off'
        display informations

    direction : string, optional, default: 'forward'
        Use 'forward', 'backward' or 'exhaustive'

    nbest : int, optional [def: 10]
        For the Id 4, use this parameter to control the number of
        features to select

    **kwargs : dictionnary
        Optional parameter for the classify function

    Returns
    -------
    MFmeth : list
        list of selected methods

    MFstr : list
        list of the name of the selected methods

    """

    # Define all the methods :
    def submeth(Idx):
        if Idx == '0':  # Select all features
            def MFmeth(x):
                return select_all(x)
            StrMeth = 'SelectAll'

        if Idx == '1':  # use a binomial law to select features
            def MFmeth(x):
                return select_bino(x, y, p=p, classifier=clfIn,
                                   grpType=grpType, **kwargs)

            StrMeth = 'Binomial selection at p<'+str(p)

        if Idx == '2':  # use permutations to select features
            def MFmeth(x):
                return select_perm(x, y, p=p, classifier=clfIn,
                                   grpType=grpType, **kwargs)
            StrMeth = 'Permutation selection at p<'+str(p)

        # use 'forward'/'backward'/'exhaustive'to  select features
        if Idx == '3':
            clf = classifier_choice(clfIn, n_tree=kwargs['n_tree'],
                                    n_knn=kwargs['n_knn'],
                                    kern=kwargs['kern'])

            def MFmeth(x):
                return sequence_inner(clf, x, y,
                                      direction=direction,
                                      inner_folds=kwargs['n_folds'],
                                      display=display)
            StrMeth = direction+' feature selection'

        if Idx == '4':  # nbest features
            def MFmeth(x):
                return select_nbest(x, y, nbest=nbest, classifier=clfIn,
                                    grpType=grpType, **kwargs)
            StrMeth = str(nbest)+' best features'

        return MFmeth, StrMeth

    # Define a list containing the methods:
    return [submeth(k)[0] for k in Id], [submeth(k)[1] for k in Id]

####################################################################
# - Apply the selected model to the dataset :
####################################################################


def apply_MF(MFmeth, MFstr, x):
    """Apply a list of method to the features x. MFstr contain the name of each
    method"""

    idx = [k for k in range(0, x.shape[0])]

    # Each method find features and return an reduce set:
    def findAndSelect(meth, x, idxOld):
        idxNew = meth(x)
        return x[idxNew, :], [idxOld[k] for k in idxNew]

    # Apply each method and get the final set of features:
    for k in MFmeth:
        if not x.size or not idx:
            break
        x, idx = findAndSelect(k, x, idx)

    # String of method application:
    MFstrCascade = ' => '.join(MFstr)

    return x, idx, MFstrCascade

# _________________________________________________________________________
# Implemented methods:
# 0 -> select_all
# 1 -> select_bino
# 2 -> select_perm
# 3 -> sequence_inner
# _________________________________________________________________________

####################################################################
# 0 - Select all the features :
####################################################################


def select_all(x):
    """Select and return all the features"""
    return [k for k in range(0, x.shape[0])]

####################################################################
# 1 - Select significant features based on the binomial law :
####################################################################


def select_bino(x, y, p=0.05, grpType='sf', **kwargs):
    """Select <p significant features using the binomial law"""
    if x.shape[0] is not len(y):
        x = x.T
    # Classify each features :
    da, _, _, _ = classify(x, y, kind=grpType, **kwargs)
    # Get significant features :
    signifeat, _ = binofeat(y, da, p)
    # Complete signifeat for group features:
    if grpType == 'mf':
        signifeat = signifeat*x.shape[1]
    # Return list of significant features:
    return [k for k, i in enumerate(list(signifeat)) if i]

####################################################################
# 2 - Select significant features based on permutations :
####################################################################


def select_perm(x, y, p=0.05, grpType='sf', **kwargs):
    """Select <p significant features using the permutations"""

    n_perm = round(1/p)
    # Classify each features :
    _, _, _, pvalue = classify(x, y, n_perm=n_perm, kind=grpType, **kwargs)

    return [k for k in range(0, len(pvalue)) if pvalue[k] < p]

####################################################################
# 3 - Select nbest features :
####################################################################


def select_nbest(x, y, nbest=10, grpType='sf', **kwargs):
    """Select nbest features"""

    # Classify each features :
    da, _, _, _ = classify(x, y, kind=grpType, **kwargs)
    return list(n.ravel(da.T).argsort()[-nbest:][::-1])
