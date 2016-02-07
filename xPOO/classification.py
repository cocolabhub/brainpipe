import numpy as n

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import (StratifiedKFold, KFold,
                                      StratifiedShuffleSplit, ShuffleSplit,
                                      cross_val_score, permutation_test_score)

__all__ = ['classify',
           'defClf',
           'defCv'
           ]


class classify(object):
    def __init__(self):
        if isinstance(clf, (int, str)):
            pass

    def __new__(self):
        pass

    def fit(self):
        pass

    def stat(self):
        pass

    def plot(self):
        pass


class defClf(object):
    """Choose a classifier and switch easyly between classifiers
    implemented in scikit-learn.

    Parameters
    ----------
    y : array
        The vector label

    clf : int or string, optional, [def : 0]
        Define a classifier. Use either an integer or a string
        [Example : classifier='lda' or 0].
        Choose between :
            - 0 / 'lda': Linear Discriminant Analysis (LDA)
            - 1 / 'svm' : Support Vector Machine (SVM)
            - 2 / 'linearsvm' : Linear SVM
            - 3 / 'nusvm' : Nu SVM
            - 4 / 'nb' : Naive Bayesian
            - 5 / 'knn' : k-Nearest Neighbor
            - 6 / 'rf' : Random Forest
            - 7 / 'lr' : Logistic Regression
            - 8 / 'qda' : Quadratic Discriminant Analysis

    kern : string, optional, [def : 'rbf']
        Kernel of the 'svm' classifier

    n_knn : int, optional, [def : 10]
        Number of neighbors for the 'knn' classifier

    n_tree : int, optional, [def : 100]
        Number of trees for the 'rf' classifier

    **kwargs : optional arguments. To define other parameters,
    see the description of scikit-learn.

    Return
    ----------
    A scikit-learn classification object with two supplementar arguments :
        - lgStr : long description of the classifier
        - shStr : short description of the classifier
    """

    def __init__(self, y, clf='lda', kern='rbf', n_knn=10, n_tree=100,
                 priors=False, **kwargs):
        pass

    def __str__(self):
        return self.lgStr

    def __new__(self, y, clf='lda', kern='rbf', n_knn=10, n_tree=100,
                priors=False, **kwargs):

        # Default value for priors :
        priors = n.array([1/len(n.unique(y))]*len(n.unique(y)))

        if isinstance(clf, str):
            clf = clf.lower()

        # LDA :
        if clf == 'lda' or clf == 0:
            clfObj = LinearDiscriminantAnalysis(
                priors=priors, **kwargs)
            clfObj.lgStr = 'Linear Discriminant Analysis'
            clfObj.shStr = 'LDA'

        # SVM : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
        elif clf == 'svm' or clf == 1:
            clfObj = SVC(kernel=kern, probability=True, **kwargs)
            clfObj.lgStr = 'Support Vector Machine (kernel=' + kern + ')'
            clfObj.shStr = 'SVM-' + kern

        # Linear SVM:
        elif clf == 'linearsvm' or clf == 2:
            clfObj = LinearSVC(**kwargs)
            clfObj.lgStr = 'Linear Support Vector Machine'
            clfObj.shStr = 'LSVM'

        # Nu SVM :
        elif clf == 'nusvm' or clf == 3:
            clfObj = NuSVC(**kwargs)
            clfObj.lgStr = 'Nu Support Vector Machine'
            clfObj.shStr = 'NuSVM'

        # Naive Bayesian :
        elif clf == 'nb' or clf == 4:
            clfObj = GaussianNB(**kwargs)
            clfObj.lgStr = 'Naive Baysian'
            clfObj.shStr = 'NB'

        # KNN :
        elif clf == 'knn' or clf == 5:
            clfObj = KNeighborsClassifier(n_neighbors=n_knn, **kwargs)
            clfObj.lgStr = 'k-Nearest Neighbor (neighbor=' + str(n_knn) + ')'
            clfObj.shStr = 'KNN-' + str(n_knn)

        # Random forest :
        elif clf == 'rf' or clf == 6:
            clfObj = RandomForestClassifier(n_estimators=n_tree, **kwargs)
            clfObj.lgStr = 'Random Forest (tree=' + str(n_tree) + ')'
            clfObj.shStr = 'RF-' + str(n_tree)

        # Logistic regression :
        elif clf == 'lr' or clf == 7:
            clfObj = LogisticRegression(**kwargs)
            clfObj.lgStr = 'Logistic Regression'
            clfObj.shStr = 'LogReg'

        # QDA :
        elif clf == 'qda' or clf == 8:
            clfObj = QuadraticDiscriminantAnalysis(**kwargs)
            clfObj.lgStr = 'Quadratic Discriminant Analysis'
            clfObj.shStr = 'QDA'

        return clfObj


class defCv(object):
    """Choose a cross_validation (CV) and switch easyly between
    CV implemented in scikit-learn.

    Parameters
    ----------
    y : array
        The vector label

    cvtype : string, optional, [def : skfold]
        Define a cross_validation. Choose between :
            - 'skfold' : Stratified k-Fold
            - 'kfold' : k-fold
            - 'sss' : Stratified Shuffle Split
            - 'ss' : Shuffle Split

    n_folds : integer, optional, [def : 10]
        Number of folds

    rndstate : integer, optional, [def : 0]
        Define a random state. Usefull to replicate a result

    rep : integer, optional, [def : 10]
        Number of repetitions

    **kwargs : optional arguments. To define other parameters,
    see the description of scikit-learn.

    Return
    ----------
    A scikit-learn classification object with two supplementar arguments :
        - lgStr : long description of the cross_validation
        - shStr : short description of the cross_validation
    """

    def __init__(self, y, cvtype='skfold', n_folds=10, rndstate=0, rep=10,
                 **kwargs):
        pass

    def __str__(self):
        return self.lgStr

    def __new__(self, y, cvtype='skfold', n_folds=10, rndstate=0, rep=10,
                **kwargs):
        y = n.ravel(y)

        # Stratified k-fold :
        if cvtype == 'skfold':
            cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True,
                                 random_state=rndstate, **kwargs)
            cv.lgStr = str(rep)+'-times, '+str(n_folds)+' Stratified k-folds'
            cv.shStr = str(rep)+'rep x'+str(n_folds)+' '+cvtype

        # k-fold :
        elif cvtype == 'kfold':
            cv = KFold(len(y), n_folds=n_folds, shuffle=True,
                       random_state=rndstate, **kwargs)
            cv.lgStr = str(rep)+'-times, '+str(n_folds)+' k-folds'
            cv.shStr = str(rep)+'rep x'+str(n_folds)+' '+cvtype

        # Shuffle stratified k-fold :
        elif cvtype == 'sss':
            cv = StratifiedShuffleSplit(y, n_iter=n_folds,
                                        test_size=1/n_folds,
                                        random_state=rndstate, **kwargs)
            cv.lgStr = str(rep)+'-times, test size 1/' + \
                str(n_folds)+' Shuffle Stratified Split'
            cv.shStr = str(rep)+'rep x'+str(n_folds)+' '+cvtype

        # Shuffle stratified :
        elif cvtype == 'ss':
            cv = ShuffleSplit(len(y), n_iter=rep, test_size=1/n_folds,
                              random_state=rndstate, **kwargs)
            cv.lgStr = str(rep)+'-times, test size 1/' + \
                str(n_folds)+' Shuffle Stratified'
            cv.shStr = str(rep)+'rep x'+str(n_folds)+' '+cvtype

        return cv


def checkXY(x, y):
    """Prepare the inputs x and y
    """
    pass
