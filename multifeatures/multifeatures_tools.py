import numpy as n
import brainpipe.classification.classification_tools as ct


####################################################################
# - Get significant features based on permutations :
####################################################################
def signifeat_permutations(x, y, threshold=None, classifier='lda', kern='rbf', n_folds=10, n_jobs=1, n_knn=10,
                           n_perm=100,
                           n_tree=100, cvkind='skfold'):
    "Get significant features based on permutation test"
    if threshold is None:
        threshold = 1 / n_perm

    _, nfeat = x.shape
    signifeat = n.array(range(0, nfeat))
    da, _, _, pvalue = ct.classify(x, y, classifier=classifier, kern=kern, n_folds=n_folds, rep=1,
                                   kind='sf', n_jobs=n_jobs, n_knn=n_knn, n_perm=n_perm, n_tree=n_tree,
                                   cvkind=cvkind)
    upper_features = n.squeeze(pvalue < threshold)
    return signifeat[upper_features], n.squeeze(pvalue), n.squeeze(da)


####################################################################
# - Classify combination of features :
####################################################################
def classify_combination(x, y, combination, classifier='lda', kern='rbf', n_folds=10, rep=1, n_jobs=1,
                         n_knn=3, n_perm=0, n_tree=100,
                         cvkind='skfold'):
    """Classify each combination and return scores and the best one location."""
    # - Get combination size :
    n_comb = len(combination)

    # - Classify each combination :
    da = n.zeros((1, n_comb))
    for k in range(0, n_comb):
        da[0, k], all_scores, permutation_scores, pvalue = ct.classify(x[:, combination[k]], y, classifier=classifier,
                                                                       kern=kern,
                                                                       n_folds=n_folds, rep=rep, kind='mf',
                                                                       n_jobs=n_jobs,
                                                                       n_knn=n_knn, n_perm=n_perm, n_tree=n_tree,
                                                                       cvkind=cvkind)

    return da, all_scores, permutation_scores, pvalue


####################################################################
# - Transform a string list to number :
####################################################################
def IdtoNb(Id, nb_pear_id=1):
    "Function to convert a list of string Id's in number"

    Id_unique_str = list(n.unique(Id))
    Id_ref_nb = range(0, len(Id_unique_str))
    Id_nb = [0] * 0

    for k in range(0, len(Id)):
        Id_nb.append([Id_unique_str.index(Id[k])] * nb_pear_id)

    return n.squeeze(n.array(Id_nb).reshape(1, len(Id) * nb_pear_id))


####################################################################
# - Compute combinations using a list :
####################################################################
def Id_combinations(Id_nb):
    "Get a combinaition of Id"

    Id_unique = n.unique(Id_nb)
    Id_combinaisons = [0] * 0
    Id_bool2int = n.array(range(0, len(Id_nb)))
    for k in Id_unique:
        Id_idx = Id_nb == k
        Id_combinaisons.append(list(Id_bool2int[Id_idx]))

    return Id_combinaisons


####################################################################
# -  Transform to a bidimentional problem :
####################################################################
def bidimtransform(x, y, ybi, classifier=0, cvkind='skfold', n_folds=10, p=0):
    nbtrans = len(ybi)
    nbepoch, nbfeat = x.shape
    flist = n.array(range(0, nbfeat))
    da = n.zeros((nbtrans, nbfeat))
    signifeat = []
    for i in range(0, nbtrans):
        # First, get index for selected item :
        ylocidx = yloc(y, ybi[i])

        # Select y and x index :
        ynb = y[ylocidx]
        xnb = x[ylocidx, :]

        # Classify each x :
        da[i, :], _, _, _ = ct.classify(xnb, ynb, classifier=classifier, kind='sf', rep=1, cvkind=cvkind,
                                        n_folds=n_folds)

        # Get significant features?
        if p is not 0:
            th = ct.binostat(ynb, p)
            signifeat.extend([flist[100 * da[i, :] >= th]])

    return da, signifeat


def yloc(y, nb):
    epochlist = n.array(range(0, len(y)))
    yloclist = []
    for k in range(0, len(nb)):
        loc = y == nb[k]
        yloclist.extend(epochlist[loc])

    return yloclist
