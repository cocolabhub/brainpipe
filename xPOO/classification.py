import numpy as n

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.cross_validation import (StratifiedKFold, KFold,
                                      StratifiedShuffleSplit, ShuffleSplit)

__all__ = ['defClf', 'defCv']


class defClf(object):
    """Choose a classifier and switch easyly between classifiers
    implemented in scikit-learn.

    Parameters
    ----------
    y : array
        The vector label

    classifier : int or string, optional, [def : 0]
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

    def __init__(self, y, classifier='lda', kern='rbf', n_knn=10, n_tree=100,
                 priors=False, **kwargs):
        pass

    def __new__(self, y, classifier='lda', kern='rbf', n_knn=10, n_tree=100,
                priors=False, **kwargs):

        # Default value for priors :
        priors = n.array([1/len(n.unique(y))]*len(n.unique(y)))

        if isinstance(classifier, str):
            classifier = classifier.lower()

        # LDA :
        if classifier == 'lda' or classifier == 0:
            clf = LinearDiscriminantAnalysis(
                priors=priors, **kwargs)
            clf.lgStr, clf.shStr = 'Linear Discriminant Analysis', 'LDA'

        # SVM : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
        elif classifier == 'svm' or classifier == 1:
            clf = SVC(kernel=kern, probability=True, **kwargs)
            clf.lgStr = 'Support Vector Machine (kernel=' + kern + ')'
            clf.shStr = 'SVM-' + kern

        # Linear SVM:
        elif classifier == 'linearsvm' or classifier == 2:
            clf = LinearSVC(**kwargs)
            clf.lgStr, clf.shStr = 'Linear Support Vector Machine', 'LSVM'

        # Nu SVM :
        elif classifier == 'nusvm' or classifier == 3:
            clf = NuSVC(**kwargs)
            clf.lgStr, clf.shStr = 'Nu Support Vector Machine', 'NuSVM'

        # Naive Bayesian :
        elif classifier == 'nb' or classifier == 4:
            clf = GaussianNB(**kwargs)
            clf.lgStr, clf.shStr = 'Naive Baysian', 'NB'

        # KNN :
        elif classifier == 'knn' or classifier == 5:
            clf = KNeighborsClassifier(n_neighbors=n_knn, **kwargs)
            clf.lgStr = 'k-Nearest Neighbor (neighbor=' + str(n_knn) + ')'
            clf.shStr = 'KNN-' + str(n_knn)

        # Random forest :
        elif classifier == 'rf' or classifier == 6:
            clf = RandomForestClassifier(n_estimators=n_tree, **kwargs)
            clf.lgStr = 'Random Forest (tree=' + str(n_tree) + ')'
            clf.shStr = 'RF-' + str(n_tree)

        # Logistic regression :
        elif classifier == 'lr' or classifier == 7:
            clf = LogisticRegression(**kwargs)
            clf.lgStr, clf.shStr = 'Logistic Regression', 'LogReg'

        # QDA :
        elif classifier == 'qda' or classifier == 8:
            clf = QuadraticDiscriminantAnalysis(**kwargs)
            clf.lgStr, clf.shStr = 'Quadratic Discriminant Analysis', 'QDA'

        return clf


class defCv(object):
    """Choose a cross_validation (CV) and switch easyly between
    CV implemented in scikit-learn.

    Parameters
    ----------
    y : array
        The vector label

    kind : string, optional, [def : skfold]
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

    def __init__(self, y, kind='skfold', n_folds=10, rndstate=0, rep=10,
                 **kwargs):
        pass

    def __new__(self, y, kind='skfold', n_folds=10, rndstate=0, rep=10,
                **kwargs):
        y = n.ravel(y)

        # Stratified k-fold :
        if kind == 'skfold':
            cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True,
                                 random_state=rndstate, **kwargs)
            cv.lgStr = str(rep)+'-times, '+str(n_folds)+' Stratified k-folds'
            cv.shStr = str(rep)+'rep x'+str(n_folds)+' '+kind

        # k-fold :
        elif kind == 'kfold':
            cv = KFold(len(y), n_folds=n_folds, shuffle=True,
                       random_state=rndstate, **kwargs)
            cv.lgStr = str(rep)+'-times, '+str(n_folds)+' k-folds'
            cv.shStr = str(rep)+'rep x'+str(n_folds)+' '+kind

        # Shuffle stratified k-fold :
        elif kind == 'sss':
            cv = StratifiedShuffleSplit(y, n_iter=n_folds,
                                        test_size=1/n_folds,
                                        random_state=rndstate, **kwargs)
            cv.lgStr = str(rep)+'-times, test size 1/' + \
                str(n_folds)+' Shuffle Stratified Split'
            cv.shStr = str(rep)+'rep x'+str(n_folds)+' '+kind

        # Shuffle stratified :
        elif kind == 'ss':
            cv = ShuffleSplit(len(y), n_iter=rep, test_size=1/n_folds,
                              random_state=rndstate, **kwargs)
            cv.lgStr = str(rep)+'-times, test size 1/' + \
                str(n_folds)+' Shuffle Stratified'
            cv.shStr = str(rep)+'rep x'+str(n_folds)+' '+kind

        return cv
