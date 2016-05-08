import numpy as np
from scipy.stats import binom
from itertools import product

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
    -> generate : generate random datasets and permutations
    -> perm2p : transform permutations into p-values
    """

    def __init__(self, n_perm):
        self.n_perm = n_perm
        
    def __str__(self):
        return 'ok'
        
    def generate(self, mu=0, sigma=100, shift=0.1, dim=(5, 5)):
        """
        """
        import pylab as P
        perm = mu + sigma*P.randn(self.n_perm, *dim)
        data = mu +shift + sigma*P.randn(*dim)
        return data, perm

    def perm2p(self, data, perm, threshold=0.05, twotails=False, maxstat=-2):
        """Get the associated p-value of a permutation distribution

        Parameter
        ----------
        data : array
            Array of real data. The shape must be (d1, d2, ..., dn)

        perm : array
            Array of permutations. The shape must be (n_perm, d1, d2, ..., dn)

        threshold : int / float
            Every values upper to threshold are going to be set to 1.

        twotails : bool, optional, [def : False]
            Define if the calculation of p-value must take into account the
            two tails of the permutation distribution
            
        maxstat : int, optional, [def : -2]
            Correct p-values with maximum statistique. maxstat correspond to
            the dimension of perm for correction. Use -1 to correct through all
            dimensions. Otherwise, use d1, d2, ... or dn to correct through a
            specific dimension. If maxstat is -2, not correction is applied.

        Return
        ----------
        pvalue : array
            Array of associated p-values
        """
        n_perm = self.n_perm
        # Check data type :
        if isinstance(data, (int, float)):
            data = np.matrix(data)
        if perm.ndim <= data.ndim:
            perm = perm[..., np.newaxis]
        # Check permutations shape :
        psh, dsh = np.array(perm.shape), np.array(data.shape)
        cond = int(psh[0] - n_perm)
        if (cond is not 0) or not np.array_equal(psh[1::], dsh):
            raise ValueError('perm must have a shape of '+str(tuple([n_perm]+list(dsh))))
        # Get maxstat correction :
        perm = self._maxstat(data, perm, maxstat)
        # Loop on data values :
        dataLoop = product(*tuple([range(k) for k in data.shape]))
        # Get the permutation function :
        fcn = self._permfcn(perm, n_perm, twotails)
        # Apply the function :
        pval = np.empty_like(data)
        pval = np.zeros(data.shape)
        for k in dataLoop:
            pval[k] = fcn(perm[(np.arange(n_perm), *k)], data[k], n_perm)
        # Replace 0 by /n_perm :
        pval[np.where(pval == 0)] = 1/n_perm
        # Threshold results :
        # pval[np.where(pval >= threshold)] = 1
        return pval

    def _evalp(self, perm, th, n_perm, kind='sup'):
        """Compute p-values from either superior or inferior part of the distribution.
        """
        # Superior part of the distribution :
        if kind is 'sup':
            return (np.sum(perm >= np.max([th, -th]))) / n_perm
        # Inferior part of the distribution :
        elif kind is 'inf':
            return (np.sum(perm <= np.min([th, -th]))) / n_perm
        return evalp

    def _permfcn(self, perm, n_perm, twotails):
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
    
    def _maxstat(self, data, perm, maxstat):
        """Correction by the maximum statistique
        """
        # No correction :
        if maxstat == -2:
            permR = perm
        # Max through all dimensions :
        elif maxstat == -1:
            permR = perm.max()*np.ones(perm.shape)
        # Max through specific dimension :
        elif (maxstat >= 0) and (maxstat < data.ndim):
            permR = np.max(perm, axis=maxstat+1, keepdims=True)
            psh, pRsh = perm.shape, permR.shape
            permR = np.tile(permR, [int(psh[k]/pRsh[k]) for k in range(perm.ndim)])
        # Any other values :
        else:
            raise ValueError('maxstat must be an integer between -2 and '+str(data.ndim-1))
        return permR


class multcomp(object):

    """Correct p-values with multiple comparisons.

    Parameter
    ----------
    n_perm : int
        Number of permutations

    Methods
    ----------
    -> bonferroni : bonferroni correction
    -> fdr : False Discovery Rate (Fieldtrip's Matlab code adaptation)
    """

    def __init__(self, ):
        pass

    def bonferroni(self, p, q, dim=-1):
        """
        """
        if dim == -1:
            fact = len(np.ravel(p))
        else:
            fact = p.shape[dim]
        return fact*p
    
    def fdr(self, p, q):
        """
        """
        # Thresholding function :
        def c(V):
            # See Genovese, Lazar and Holmes (2002) page 872, second column, first paragraph
            if V < 1000:
                # Compute it exactly
                s = np.sum(1./np.arange(1, V+1, 1))
            else:
                # Approximate it
                s = np.log(V) + 0.57721566490153286060651209008240243104215933593992359880576723488486772677766467093694706329174674951463144724980708248096050401448654283622417399764492353625350033374293733773767394279259525824709491600873520394816567
            return s
        # Convert the input into a row vector and sort
        pback = p.copy()
        dim = p.shape
        p = np.ravel(p)
        idx = np.argsort(p, axis=0)
        p = p[idx]
        V = len(p)
        # Compute the threshold probability for each voxel
        pi = (np.arange(1, V+1)/V)  * q / c(V)
        h = p <= pi
        # Undo the sorting and convert the output back into the original format
        idxrev = np.argsort(idx)
        h = h[idxrev].reshape(dim)
        pback[~h] = 1
        return pback, h