import numpy as n
from sklearn import cross_validation
from itertools import combinations


def sequence(estimator, x, y, direction='forward', inner_folds=10, outer_folds=10, stratified=True, n_jobs=1,
             display=True, criterion=1):
    """Find an optimal sequence of features using a forward, backward or exhaustive sequence.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    x : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like
        The target variable to try to predict in the case of
        supervised learning.

    direction : string, optional, default: 'forward'
        Use 'forward', 'backward' or 'exhaustive'

    inner_folds & outer_folds : integer, optional, default : 10
        Respectively the number of folds for the inner and outer classification.

    stratified : boolean optional
        Whether the task is a classification task, in which case stratified KFold will be used.

    n_jobs : integer, optional, default : 1
        The number of CPUs to use to do the computation. -1 means 'all CPUs'.

    display : boolean
        Print the evolution of the features selection.

    criterion : integer, optional, default : 1
        If criterion is 0, the features selection won't stop and every features are going to used, but just re-order.

    Returns
    -------

    score_sequence : floating point
        Score obtain with the sequence.

    selected_features : list of integer
        List of the selected features for each fold.
    """
    # - Adapt variable :
    x = n.matrix(x)
    y = n.ravel(y)

    # - Define the outer cross-validation :
    cv_outer = cross_validation.check_cv(outer_folds, X=x, y=y, classifier=stratified)
    # - Initialize variables :
    k = 0
    selected_features = []
    predictions = []
    test_set = []
    for train_index, test_index in cv_outer:
        if display:
            print('-> Fold', k + 1)
        # - Define outer train set and test set :
        x_train, x_test = x[train_index, :], x[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        # - Run the features selection on the train set :
        feat_sequence = n.array(
            sequence_inner(estimator, x_train, y_train, direction=direction, inner_folds=inner_folds,
                           stratified=stratified, n_jobs=n_jobs, display=display, criterion=criterion))
        # - Run the prediction :
        estimator.fit(x_train[:, feat_sequence], y_train)  # Train your model
        pred = estimator.predict(x_test[:, feat_sequence])  # Test your model
        # - Retain all selected features and update:
        predictions.extend(pred)
        selected_features.append(list(feat_sequence))
        test_set.extend(y_test)
        k += 1

    # - Get the score of the selected sequence of features :
    bool_pred = n.array(predictions) == n.array(test_set)
    score_sequence = sum(bool_pred.astype(int)) / len(test_set)

    return score_sequence, selected_features


def sequence_inner(estimator, x, y, direction='forward', inner_folds=10, stratified=True, n_jobs=1, display=True,
                   criterion=1):
    """Inner sequence, inside the outer cross-validation."""
    # 1 - Get size elements and create default list of features :
    y = n.ravel(y)
    if x.shape[0]!=len(y): x = x.T
    n_epoch, n_feat = x.shape

    cv_inner = cross_validation.check_cv(inner_folds, X=x, y=y, classifier=stratified)
    default_list = list(range(n_feat))#[[k] for k in range(n_feat)]#
    if direction == 'forward':
        features_list = []
        direction_word = 'added'
        combination = default_list
    elif direction == 'backward':
        features_list = list(range(n_feat))
        direction_word = 'removed'
        combination = [default_list]

    if direction is not 'exhaustive':
        # 2 - Classify all features to find the best one :
        old_score = 0
        new_score = 0.05
        k = 0

        while k <= n_feat - 1 and old_score <= new_score:

            old_score = new_score
            combinationN = [[k] if type(k) is int else k for k in combination ] # Check combination type

            new_score, ind_score, all_scores = _classify_combination(estimator, x, y, combinationN, cv=cv_inner,
                                                                     n_jobs=n_jobs)
            # Update :
            features_list.append(default_list[ind_score])  # add the best features
            del default_list[ind_score]  # remove it from the default list

            k += 1
            if display:
                print('Step', k, '- feature', direction_word, ':', features_list[-1], '|| DA =', new_score)

            # generate new combinations
            if direction == 'forward':
                combination = _generate_combination([features_list], default_list, direction)
                selected_features = features_list
            elif direction == 'backward':
                if len(default_list) > 1:
                    combination = _generate_combination([default_list], default_list, direction)
                else:
                    combination = default_list
                selected_features = default_list
            if criterion == 0:
                old_score = new_score

        return selected_features

    elif direction == 'exhaustive':
        combination = _generate_combination(list(range(n_feat)), [], direction)
        if display:
            print(len(combination), 'combinations found')
        new_score, ind_score, all_scores = _classify_combination(estimator, x, y, combination, cv=cv_inner,
                                                                 n_jobs=n_jobs)
        return list(combination[ind_score])


def _generate_combination(permanent_feat, other_feat, direction):
    """Generate the combination for the forward, backward models."""
    if direction == 'forward':
        combination = [(x + [y]) for x in permanent_feat for y in other_feat if x != y]
    elif direction == 'backward':
        combination = [(list(set(x).difference([y]))) for x in permanent_feat for y in other_feat if x != y]
    elif direction == 'exhaustive':
        combination = []
        for i in range(1, len(permanent_feat) + 1):
            all_comb_for_L = list(combinations(permanent_feat, i))
            combination.extend(all_comb_for_L)

    return combination


def _classify_combination(estimator, x, y, combination, cv=None, n_jobs=1):
    """Classify each combination and return scores and the best one location."""
    # - Get combination size :
    n_comb = len(combination)
    # - Classify each combination :
    scores_combi = n.zeros((1, n_comb))
    for k in range(0, n_comb):
        scores_combi[0, k] = 100*n.mean(cross_validation.cross_val_score(estimator, x[:, combination[k]], y, cv=cv, n_jobs=n_jobs))

    return scores_combi.max(), scores_combi.argmax(), scores_combi