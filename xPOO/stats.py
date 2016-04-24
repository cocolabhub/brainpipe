import numpy as np
from scipy.stats import binom

__all__ = ['binomial', 'permutation']


class binomial(object):

    """Binomial law for machine learning.

    Parameters
    ----------
    y  : array
        The vector label

    Methods
    ----------
    -> p2da : transform p-values to decoding accuracies
    -> da2p : transform decoding accuracies to p-value
    -> signifeat : get significants features
    """

    def __init__(self, y):
        self._y = np.ravel(y)
        self._nbepoch = len(y)
        self._nbclass = len(np.unique(y))

    def da2p(self, da):
        """Get the p-value associated to the y vector label.

        Parameter
        ----------
        da : int / float / list /array [0 <= da <= 100]
            The decoding accuracy array. Ex : da = [75, 33, 25, 17].

        Return
        ----------
        p : np.ndarray
            The p-value associate to each decoding accuracy
        """
        if not isinstance(da, np.ndarray):
            da = np.array(da)
        if (da.max() > 100) or (da.min() < 0):
            raise ValueError('Consider 0<=da<=100')
        return 1 - binom.cdf(self._nbepoch * da / 100, self._nbepoch, 1 / self._nbclass)

    def p2da(self, p):
        """Get the significant decoding accuracy associated to an y vector label

        Parameter
        ----------
        p : int / float / list /array [0 <= p < 1]
            p-value. Ex : p = [0.05, 0.01, 0.001, 0.00001]

        Return
        ----------
        da : np.ndarray
            The decoding accuracy associate to each p-value
        """
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        if (p.max() >= 1):
            raise ValueError('Consider 0<=p<1')
        return binom.ppf(1 - p, self._nbepoch, 1 / self._nbclass) * 100 / self._nbepoch

    def signifeat(self, feat, th):
        """Get significants features.

        Parameter
        ----------
        feat : array
            Array containing either decoding accuracy (da) either p-values

        th : int / float
            The threshold in order to find significants features.

        Return
        ----------
        index : array
            The index corresponding to significants features

        signi_feat : array
            The significants features
        """
        # If feat is p-values :
        if (feat.min() >= 0) and (feat.max() < 1) and (th >= 0) and (th < 1):
            index = np.where(feat < th)
        # If feat is da :
        elif (feat.min() >= 0) and (feat.max() <= 100) and (th >= 0) and (th <= 100):
            index = np.where(feat >= th)
        # Any other case :
        else:
            raise ValueError('Cannot recognize the type of feat.')

        return index, feat[index]


class permutation(object):

    """Eval statistical significiancy of permutations. Calculation is optimized
    for n-dimentional array.

    Parameter
    ----------
    n_perm : int
        Number of permutations

    Methods
    ----------
    -> perm2p : transform permutations into p-values
    """

    def __init__(self, n_perm):
        self.n_perm = n_perm

    def perm2p(self, perm, th, pdim=0, twotails=False):
        """Get the associated p-value of a permutation distribution

        Parameter
        ----------
        perm : array
            Array of permutations. One dimension must be n_perm

        th : int / float
            Threshold to get the p-value from the distribution

        pdim : int, optional, [def : 0]
            Integer to locate where is the dimension of perm equal to n_perm

        twotails : bool, optional, [def : False]
            Define if the calculation of p-value must take into account the
            two tails of the permutation distribution

        Return
        ----------
        pvalue : array
            Array of associated p-values
        """
        n_perm = self.n_perm
        # Check perm dimensions :
        perm, dim = self._permreshape(perm, n_perm, pdim)
        # Get the permutation function :
        fcn = self._permfcn(perm, th, n_perm, twotails)
        return fcn(perm, th, n_perm)

    def _evalp(self, perm, th, n_perm, kind='sup'):
        """Compute p-values from either superior or inferior part of the distribution.
        """
        # Superior part of the distribution :
        if kind is 'sup':
            return (np.sum(perm >= np.max([th, -th]), axis=0)) / n_perm
        # Inferior part of the distribution :
        elif kind is 'inf':
            return (np.sum(perm <= np.min([th, -th]), axis=0)) / n_perm
        return evalp

    def _permfcn(self, perm, th, n_perm, twotails):
        """Return either function for calculation of one or two tails
        """
        # One tail :
        if not twotails:
            def evalp(perm, th, n_perm):
                return self._evalp(perm, th, n_perm, kind='sup')
        # Two tails :
        else:
            def evalp(perm, th, n_perm):
                pleft = self._evalp(perm, th, n_perm, kind='inf')
                pright = self._evalp(perm, th, n_perm, kind='sup')
                return pleft + pright
        return evalp

    def _permreshape(self, perm, n_perm, pdim):
        """Force the first dimension to be n_perm
        """
        # Get variables :
        dim = np.array(perm.shape)
        check = list(np.where(dim == n_perm)[0])
        # Check if n_perm is in permutation's dimension :
        if not check:
            raise ValueError(
                'None of the permutations dimensions correspond to '+str(
                                                                 n_perm))
        # Check if n_perm is at pdim :
        if dim[pdim] != n_perm:
            raise ValueError(
                'The dimension '+str(pdim)+' of permutation must be '+str(
                                                                  n_perm))
        # Force the first dimension to be n_perm :
        if dim[0] != n_perm:
            perm = np.swapaxes(perm, 0, pdim)

        return perm, dim
