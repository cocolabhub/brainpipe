import numpy as n
from sklearn import cross_validation, svm, naive_bayes, \
    neighbors, ensemble, metrics, linear_model, discriminant_analysis



####################################################################
# - Main classify samples :
####################################################################
def classify(x, y, classifier='lda', kern='rbf', n_folds=10, rep=10, kind='sf', n_jobs=1, n_knn=3, n_perm=0, n_tree=100,
             cvkind='skfold'):
    "da, all_scores, permutation_scores, pvalue"
    # Check format :
    x = checkfeat(x,y)
    n_epoch, n_feat = x.shape
    priors = n.array([1/len(n.unique(y))]*len(n.unique(y)))

    # - Classifier's choice :
    if (type(classifier) is int) | (type(classifier) is str):
        clf = classifier_choice(classifier, kern=kern, n_knn=n_knn, n_tree=n_tree, priors=priors)
    else : clf = classifier

    # - Cross validation definition :
    if kind == 'mf' and n_perm == 0:  # Multi feature classification
        da, all_scores, cv_model = classify_fcn(x, y, clf, n_folds=n_folds, rep=rep, n_jobs=n_jobs, cvkind=cvkind)
    elif kind == 'sf' and n_perm == 0:  # Single features classification
        da = n.zeros((1, n_feat))
        all_scores = n.zeros((rep, n_folds, n_feat))
        for k in range(0, n_feat):
            da[:, k], all_scores[:, :, k], cv_model = classify_fcn(x[:, k], y, clf, n_folds=n_folds, rep=rep,
                                                                   n_jobs=n_jobs, cvkind=cvkind)

    # Statistical evaluation :
    if n_perm == 0:
        permutation_scores, pvalue = 0, [[0]]
    else:
        all_scores = 0
        cv_model = crossval_choice(y, cvkind=cvkind, n_folds=n_folds, rndstate=0)
        if kind == 'mf':  # Multi feature classification
            da, permutation_scores, pvalue = cross_validation.permutation_test_score(clf, x, y, scoring="accuracy",
                                                                                     cv=cv_model, n_permutations=n_perm,
                                                                                     n_jobs=n_jobs)
        elif kind == 'sf':  # Single features classification
            permutation_scores = n.zeros((n_perm, n_feat))
            da = n.zeros((1, n_feat))
            pvalue = n.zeros((1, n_feat))
            for k in range(0, n_feat):
                da[0, k], permutation_scores[:, k], pvalue[0, k] = cross_validation.permutation_test_score(clf, x[:, k], y,
                                                                                                           scoring="accuracy",
                                                                                                           cv=cv_model,
                                                                                                           n_permutations=n_perm,
                                                                                                           n_jobs=n_jobs)

    return 100*da, 100*all_scores, permutation_scores, pvalue#list(pvalue[0])


####################################################################
# - Sub classification fonction :
####################################################################
def classify_fcn(x, y, clf, n_folds=10, rep=10, n_jobs=1, cvkind='skfold'):
    all_scores = n.zeros((rep, n_folds))

    for k in range(0, rep):
        # Shuffling
        cv_model = crossval_choice(y, cvkind=cvkind, n_folds=n_folds, rndstate=k, rep=rep)
        all_scores[k, :] = cross_validation.cross_val_score(clf, x, y, cv=cv_model, n_jobs=n_jobs)

    da = all_scores.mean()

    return da, all_scores, cv_model


####################################################################
# - Check the format of data :
####################################################################
def checkfeat(x,y):
    x, y = n.matrix(x), n.ravel(y)
    nbtrials = len(y)
    dim1, dim2 = x.shape
    diffe = dim1 - nbtrials
    if  diffe is not 0:
        x = x.T
    return x

####################################################################
# - Load a basic classifier :
####################################################################
def classifier_choice(classifier='lda', kern='rbf', n_knn=10, n_tree=100,priors=None):

    if classifier == 'lda' or classifier == 0:  # --- LDA ---
        clf = discriminant_analysis.LinearDiscriminantAnalysis(priors=priors)
    elif classifier == 'SVC' or classifier == 1:  # --- SVM ---
        clf = svm.SVC(kernel=kern,probability=True)  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
    elif classifier == 'LinearSVC' or classifier == 2:
        clf = svm.LinearSVC()
    elif classifier == 'NuSVC' or classifier == 3:
        clf = svm.NuSVC()
    elif classifier == 'GaussianNB' or classifier == 4:  # --- NB ---
        clf = naive_bayes.GaussianNB()
    elif classifier == 'knn' or classifier == 5:  # --- KNN ---
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_knn)
    elif classifier == 'rf' or classifier == 6:  # --- Random Forest ---
        clf = ensemble.RandomForestClassifier(n_estimators=n_tree)
    elif classifier == 'lr' or classifier == 7:  # --- Logistic regression ---
        clf = linear_model.LogisticRegression()
    elif classifier == 'qda' or classifier == 8:  # --- QDA ---
        clf = discriminant_analysis.QuadraticDiscriminantAnalysis()

    return clf


def classifier_string(classifier='lda', kern='rbf', n_knn=10, n_tree=100):
    if classifier == 'lda' or classifier == 0:  # --- LDA ---
        clfstr, short = 'Linear Discriminant Analysis', 'LDA'
    elif classifier == 'SVC' or classifier == 1:  # --- SVM ---
        clfstr = 'Support Vector Machine (kernel=' + kern + ')'
        short = 'SVM-' + kern
    elif classifier == 'LinearSVC' or classifier == 2:
        clfstr, short = 'Linear Support Vector Machine', 'LSVM'
    elif classifier == 'NuSVC' or classifier == 3:
        clfstr, short = 'Nu Support Vector Machine', 'NuSVM'
    elif classifier == 'GaussianNB' or classifier == 4:  # --- NB ---
        clfstr, short = 'Naive Baysian', 'NB'
    elif classifier == 'knn' or classifier == 5:  # --- KNN ---
        clfstr = 'k-Nearest Neighbor (neighbor=' + str(n_knn) + ')'
        short = 'KNN-' + str(n_knn)
    elif classifier == 'rf' or classifier == 6:  # RandomForest
        clfstr = 'Random Forest (tree=' + str(n_tree) + ')'
        short = 'RF-' + str(n_tree)
    elif classifier == 'lr' or classifier == 7:  # --- Logistic regression ---
        clfstr, short = 'Logistic Regression', 'LogReg'
    elif classifier == 'qda' or classifier == 8:  # --- QDA ---
        clfstr, short = 'Quadratic Discriminant Analysis', 'QDA'
    return clfstr, short

####################################################################
# - Choose a cross_validation :
####################################################################
def crossval_choice(y, cvkind='skfold', n_folds=10, rndstate=0, rep=10):
    y = n.ravel(y)
    if cvkind == 'skfold':
        cv_model = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=rndstate)
    elif cvkind == 'kfold':
        cv_model = cross_validation.KFold(len(y), n_folds=n_folds, shuffle=True, random_state=rndstate)
    elif cvkind == 'sss':
        cv_model = cross_validation.StratifiedShuffleSplit(y, n_iter=n_folds, test_size=1/n_folds, random_state=rndstate)
    elif cvkind == 'ss':
        cv_model = cross_validation.ShuffleSplit(len(y), n_iter=rep, test_size=1/n_folds, random_state=rndstate)

    return cv_model



####################################################################
# - Compare classifiers and get the predicted labels :
####################################################################
def classifier_comparison_pred(xtrain, ytrain, xtest, ytest, clf):
    nbclassifier = len(clf)
    pred = n.zeros((len(ytest),nbclassifier))
    for k in range(0,nbclassifier):
        clf[k].fit(xtrain, ytrain)  # Train your model
        pred[:,k] = clf[k].predict(xtest)  # Test your model
    return pred


####################################################################
# - Compare classifiers and get the decoding accuracy :
####################################################################
def classifier_comparison(x, y, clf, cvkind='skfold', n_folds=10):
    # Define the cross-validation model :
    cv_model = crossval_choice(y, cvkind=cvkind, n_folds=n_folds)
    nbclassifier = len(clf)
    da = n.zeros((1,nbclassifier))
    for k in range(0,nbclassifier):
        da[0,k] = n.mean(cross_validation.cross_val_score(clf[k], x, y, cv=cv_model))

    return da


####################################################################
# - Eval prediction accuracy :
####################################################################
def evalpred(pred,test):
    bool_pred = n.array(pred) == n.array(test)
    return sum(bool_pred.astype(int)) / len(test)


####################################################################
# - Time generalization :
####################################################################
def timegeneralization(x, y, classifier=0, n_knn=10, n_tree=100, cvkind=None, n_folds=10, rndstate=0, kern='rbf'):

    # - Define array for single or multi features :
    xnbfeat = len(x.shape)
    if xnbfeat == 2:
        nbtrials, nbtime = x.shape
    elif xnbfeat == 3:
        nbtrials, nbtime, _ = x.shape

    # - Classifier's choice :
    if (type(classifier) is int) | (type(classifier) is str):
        clf = classifier_choice(classifier, kern=kern, n_knn=n_knn, n_tree=n_tree)
    else : clf = classifier

    # - Define a cross-validation or not :
    if cvkind is not None:
        cvmodel = crossval_choice(y, cvkind=cvkind, n_folds=n_folds, rndstate=rndstate)

    da = n.zeros([nbtime,nbtime])
    for k in range(0,nbtime): # Training dimension
        xx = checkfeat(x[:,k],y)
        for i in range(0,nbtime): # Tsting dimension
            xy = checkfeat(x[:,i],y)
            if k == i and cvkind is not None:
                da[i,k] = n.mean(cross_validation.cross_val_score(clf, xx, y, cv=cvmodel))
            else:
                clf.fit(xx,y)
                pred = clf.predict(xy)
                da[i,k] = metrics.accuracy_score(y, pred)
    return da


####################################################################
# - Get the optimal number of tree for Random Forest :
####################################################################
def optimal_tree_nb(x, y, treerange=range(64, 137, 8), n_folds=10, cvkind='skfold'):
    print('-> Optimized the number of trees. From', str(treerange[0]),'to',str(treerange[-1]),'step',str(treerange[1]-treerange[0]))
    treescores = n.zeros((n_folds, len(treerange)))
    # Define the cross-validation model :
    cv_model = crossval_choice(y, cvkind=cvkind, n_folds=n_folds)
    # Compute for each number of tree :
    for k in range(0, len(treerange)):
        clf = classifier_choice(6, n_tree=treerange[k])
        treescores[:, k] = cross_validation.cross_val_score(clf, x, y, cv=cv_model)

    treescoresM = n.mean(treescores, 0)
    optiNbtree = treerange[treescoresM.argmax()]
    print('-> Optimal number of trees :',str(optiNbtree))
    return optiNbtree, treescoresM


####################################################################
# - Get the optimal number of neighbor for knn :
####################################################################
def optimal_neighbor_nb(x, y, n_range=range(5, 100, 5), n_folds=10, cvkind='skfold'):
    print('-> Optimized the number of neighbor. From', str(n_range[0]),'to',str(n_range[-1]),'step',str(n_range[1]-n_range[0]))
    lenrange = len(n_range)
    nscores = n.zeros((n_folds, lenrange))
    # Define the cross-validation model :
    cv_model = crossval_choice(y, cvkind=cvkind, n_folds=n_folds)
    # Compute for each number of neighbor :
    for k in range(0,lenrange):
        clf = classifier_choice('knn', n_knn=n_range[k])
        nscores[:, k] = cross_validation.cross_val_score(clf, x, y, cv=cv_model)

    nscoresM = n.mean(nscores,0)
    optineighbor = n_range[nscoresM.argmax()]
    print('-> Optimal number of neighbor :',str(optineighbor))

    return optineighbor, nscoresM
