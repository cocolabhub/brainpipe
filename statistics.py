from scipy.stats import binom
import numpy as n

__all__ = [
            'binostat',
            'binostatinv',
            'binofeat',
            'perm2pval',
            'permIntraClass'
          ]


def binostat(y, p):
    """Get the significant decoding accuracy associated to an y vector label

    Parameters
    ----------
    y  : array
        The vector label

    p : int / float
        p-value
    """
    y = n.ravel(y)
    nbepoch = len(y)
    nbclass = len(n.unique(y))
    return binom.ppf(1 - p, nbepoch, 1 / nbclass) * 100 / nbepoch


def binostatinv(y, da):
    """Get the p-value associated to an y vector label and a decoding accuracy.

    Parameters
    ----------
    y  : array
        The vector label

    da : array
        The decoding accuracy array
    """
    y = n.ravel(y)
    nbepoch = len(y)
    nbclass = len(n.unique(y))
    return 1 - binom.cdf(nbepoch * da / 100, nbepoch, 1 / nbclass)


def binofeat(y, da, p):
    """Get the significants features using the binomial law.

    Parameters
    ----------
    y  : array
        The vector label

    p : int / float
        p-value

    da : array
        The decoding accuracy array
    """
    th = binostat(y, p)
    signifeat = da >= th
    try:
        return signifeat[0], th
    except:
        return [signifeat], th


def perm2pval(da, daperm):
    """Return the p-value deduced from permutations

    Parameters
    ----------
    da  : 1D array or list
        The decoding accuracy

    daperm : array
        Decoding accuracy of permutations
    """
    nfeat = len(da)
    if daperm.shape[0] is not nfeat:
        daperm = daperm.T
    nperm = daperm.shape[1]
    pvalue = []
    for k in range(nfeat):
        score = da[k]
        permutations = daperm[k, :]
        pvalue.append((n.sum(permutations >= score) + 1) / nperm)
    return pvalue


def permIntraClass(x, y, rndstate):
    """Generate intra permutations of x
    """
    ntrials, nfeat = x.shape
    yClass = list(set(y))

    rndstate = n.random.RandomState(rndstate)
    xCperm = n.zeros(x.shape)
    for j in yClass:
        idxClass = n.where(y == j)[0]
        for l in range(nfeat):
            xSub = n.ravel(x[idxClass, l])
            xCperm[idxClass, l] = rndstate.permutation(xSub)
    return xCperm
