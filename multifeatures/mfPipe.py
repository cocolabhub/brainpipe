from brainpipe.classification.classification_tools import (crossval_choice,
                                                           classifier_choice,
                                                           classifier_string)
from brainpipe.multifeatures.mfMeth import select_MF, apply_MF
from brainpipe.statistic.stat_tools import binostatinv
import pandas as pd
import numpy as n
import itertools


__all__ = ['mf']

####################################################################
# - Conpute Multi-features :
####################################################################


def mf(x, y, Id=0, clfIn=0, clfOut=0, p=0.05,
        n_knn=3, n_tree=100, kern='rbf', cvIn='sss', n_foldsIn=10, repIn=1,
        cvOut='skfold', n_foldsOut=3, repOut=10, probOccur='rpercent',
        display=True, threshold=None, direction='forward', nbest=10,
        grpList=None, grpCombine=False, grpAs='single'):
    """Compute the multi-features contain in an array x, using the target
    vector y. The Id serve to combine MF methods.

    Parameters
    ----------

    x : array-like
        The features. Dimension [nfeat x ntrial]

    y : array-like
        The target variable to try to predict in the case of
        supervised learning

    Id : string
        '0': No selection. All the features are used
        '1': Select <p significant features using a binomial law
        '2': Select <p significant features using permutations
        '3': use 'forward'/'backward'/'exhaustive'to  select features
        '4': select 'nbest' features based on evaluation

    grpList : list, optional, [def: None]
        Specify

    clf : estimator object implementing 'fit'
        The object to use to fit the data

    p : float < 1, default: 0.05
        The significiance level to select features

    threshold : float, default: None
        variable equivalent to p. If threshold is defined, the programm
        will search for the p value associed

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
    # Get size elements :
    nfeat, ntrial = x.shape
    if grpList is None:
        grpList, grpCombine = ['0'] * nfeat, False
    if threshold is not None:
        p = binostatinv(y, threshold)
    if grpAs == 'single':
        grpType = 'sf'
    elif grpAs == 'group':
        grpType = 'mf'

    # Manage group of features :
    groupinfo = combineInfo(x, grpList, grpCombine=grpCombine)
    # if display:
    #     print('-> '+str(len(list(groupinfo['feature'])))+' combinations found')

    # Define classifier option :
    clfOutMod = classifier_choice(
        clfOut, n_tree=n_tree, n_knn=n_knn, kern=kern)

    # Keep the info :
    setup = {'p': p, 'cvOut': cvOut, 'n_foldsOut': n_foldsOut,
             'repOut': repOut, 'cvIn': cvIn, 'n_foldsIn': n_foldsIn,
             'repIn': repIn, 'n_tree': n_tree, 'kern': kern,
             'probOccur': probOccur, 'direction': direction, 'nbest': nbest}

    # Run the MF model for each combinaition:
    da, prob = [], []
    for k in range(len(groupinfo)):
        if display:
            print('=> Group : ' + groupinfo['feature'].iloc[k]+' ('+str(
                k)+'/'+str(len(groupinfo))+')', end='\r')
        daComb, probComb, MFstr = MFcv(
            x[groupinfo['idx'].iloc[k], :], y, Id, clfOutMod, clfIn=clfIn,
            display=display, grpType=grpType, **setup)
        da.append(daComb), prob.append(probComb)

    # Get final info on the classifier used :
    setup['clfIn'], _ = classifier_string(
        clfIn, n_tree=n_tree, n_knn=n_knn, kern=kern)
    setup['clfOut'], _ = classifier_string(
        clfOut, n_tree=n_tree, n_knn=n_knn, kern=kern)

    # Complete the pandas Dataframe about group decoding
    groupinfo['da'], groupinfo['occurrence'] = [
        sum(k) / len(k) for k in da], prob

    return da, prob, MFstr, groupinfo, setup

####################################################################
# - Conpute the mf on separate training/testing :
####################################################################


def MFcv(x, y, Id, clfOut, p=0.05, clfIn=0, cvOut='skfold', n_foldsOut=3,
         repOut=10, cvIn='skfold', n_foldsIn=10, repIn=1,
         n_tree=100, n_knn=10, kern='rbf', probOccur='rpercent', display=True,
         direction='forward', nbest=10, grpType='sf'):
    """cross-validation multifeatures """
    idxCvOut, da = [], []
    for k in range(0, repOut):
        # Define variables and outer cross-validation :
        predCv, yTestCv = [], []
        cvOuti = crossval_choice(
            y, cvkind=cvOut, n_folds=n_foldsOut, rndstate=k)
        for train_index, test_index in cvOuti:
            # Get training and testing sets:
            xTrain, xTest, yTrain, yTest = x[:, train_index], x[
                :, test_index], y[train_index], y[test_index]

            # Get the MF model:
            MFmeth, MFstr = select_MF(Id, yTrain, clfIn, p=p, display=display,
                                      cvkind=cvIn, rep=repIn,
                                      n_folds=n_foldsIn, n_tree=n_tree,
                                      n_knn=n_knn, kern=kern, grpType=grpType,
                                      direction=direction, nbest=nbest)

            # Apply the MF model:
            xCv, idxCvIn, MFstrCascade = apply_MF(MFmeth, MFstr, xTrain)

            # Select the xTest features:
            xTest = xTest[idxCvIn, :]

            # Classify:
            if xCv.size:
                predCv.extend(clfOut.fit(xCv.T, yTrain).predict(xTest.T))
            else:
                predCv.extend(n.array([None] * len(test_index)))

            # Keep info:
            idxCvOut.extend(idxCvIn), yTestCv.extend(yTest)

        # Get the decoding accuracy :
        da.append(100 * sum([1 if predCv[k] == yTestCv[k]
                             else 0 for k in range(0, len(predCv))
                             ]) / len(predCv))

    prob = occurProba(idxCvOut, list(range(0, x.shape[0])), kind=probOccur)

    return da, prob, MFstrCascade

####################################################################
# - Generate combinaition of features and get info :
####################################################################


def combineInfo(x, grpList, grpCombine=True):

    nfeat, ntrial = x.shape
    gpFeat = pd.DataFrame({'group': grpList})
    gp = gpFeat.groupby(['group'])
    if grpCombine:
        start, stop = 1, None
    else:
        start, stop = 1, 2  # len(gp.groups.keys())-1

    seen = set()
    seen_add = seen.add
    unqOrder = [x for x in grpList if not (x in seen or seen_add(x))]
    idxGp = [gp.groups[k] for k in unqOrder]
    idxComb = getCombi(idxGp, start=start, stop=stop, kind=int)
    featNameComb = getCombi(
        unqOrder, start=start, stop=stop, kind=str, sep=' + ')

    return pd.DataFrame({'feature': featNameComb, 'idx': idxComb})


####################################################################
# - Combine string and int elements :
####################################################################
def getCombi(stuff, start=1, stop=None, kind=int, sep=''):
    allComb = []
    if stop is None:
        stop = len(stuff) + 1
    for L in range(start, stop):
        for subset in itertools.combinations(stuff, L):
            if kind == str:
                allComb.extend([sep.join(subset)])
            elif kind == int:
                t = []
                [t.extend(k) for k in subset]
                allComb.extend([t])
    return allComb


####################################################################
# - Compute the occurence probability :
####################################################################
def occurProba(x, ref, kind='percent'):
    if kind == 'percent':
        return [100 * x.count(k) / len(x) if x else 0 for k in ref]
    if kind == 'rpercent':
        return [round(100 * x.count(k) / len(x)) if x else 0 for k in ref]
    if kind == 'count':
        return [x.count(k) if x else 0 for k in ref]
