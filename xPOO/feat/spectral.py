from .filtering import fextract, docfilter
from brainpipe.xPOO.tools import binarize, binArray
from brainpipe.xPOO.visualization.cmon_plt import ndplot
from .utils._feat import (_manageWindow, _manageFrequencies, normalize,
                          _checkref)

from joblib import Parallel, delayed
import numpy as np
from itertools import product

__all__ = ['sigfilt', 'amplitude', 'power', 'phase']


# ------------------------------------------------------------
# DOCUMENTATION
# ------------------------------------------------------------
docsignal = """norm : int, optional [def : 0]
        Number to choose the normalization method
            0 : No normalisation
            1 : Substraction
            2 : Division
            3 : Substract then divide
            4 : Z-score

    baseline : tuple/list of int [def: (1,1)]
        Define a window to normalize the power

    split : int or list of int, optional [def: None]
        Split the frequency band f in "split" band width.
        If f is a list which contain couple of frequencies,
        split is a list too.
"""

supfilter = """
    **kwargs : supplementar arguments for filtering
""" + docfilter


# ------------------------------------------------------------
# MAIN SPECTRAL CLASS
# ------------------------------------------------------------
class _spectral(ndplot):
    """This class is optimized for 3D arrays.

    Parameters
    ----------
    sf : int
        Sampling frequency

    npts : int
        Number of points of the time serie

    f : tuple/list
        List containing the couple of frequency bands to extract spectral
        informations. Example : f=[ [2,4], [5,7], [60,250] ]

    window : tuple, list, None, optional [def: None]
        List/tuple: [100,1500]
        List of list/tuple: [(100,500),(200,4000)]
        None and the width and step parameters will be considered

    width : int, optional [def : None]
        width of a single window.

    step : int, optional [def : None]
        Each window will be spaced by the "step" value.

    time : list/array, optional [def: None]
        Define a specific time vector

    """

    def __init__(self, sf, npts, kind, f, baseline, norm, method, window,
                 width, step, split, time, **kwargs):
        self._window, self.xvec = _manageWindow(
            npts, window=window, width=width, step=step, time=time)
        self.f, self._fSplit, self._fSplitIndex = _manageFrequencies(
            f, split=split)

        # Get variables :
        self._baseline = baseline
        self._norm = norm
        self._width = width
        self._step = step
        self._split = split
        self._nf = len(self.f)
        self._sf = sf
        self._npts = npts
        self.yvec = [round((k[0]+k[1])/2) for k in self.f]
        self._kind = kind
        self._fobj = fextract(method, kind, **kwargs)
        # ndplot.__init__(self, self._kind)

    def __str__(self):
        extractStr = str(self._fobj)
        powStr = '{ki}(norm={no}, step={st}, width={wi}, split={sp},\n'
        powStr = powStr.format(ki=self._kind, no=self._norm, st=self._step,
                               wi=self._width, sp=self._split)
        return powStr+extractStr+')'

    def get(self, x, statmeth='permutation', twotails=False, maxstat=-2,
            n_perm=200, n_jobs=-1):
        """Get the spectral informations of the signal x.

        Parameters
        ----------
        x : array
            Data. x should have a shape of
            (n_electrodes x n_pts x n_trials)

        statmeth : string, optional, [def : 'permutation']
            Method to evaluate the statistical significiancy. To get p-values,
            the program will compare real values with a defined baseline. As a
            consequence, the 'norm' parameter should not be equal to zero.
            Implemented methods to get p-values :
                - 'permutation' : randomly shuffle real data with baseline.
                Control the numbe of permutations with the n_perm parameter.
                For example, if n_perm = 1000, this mean that minimum p-values
                will be 0.001.
                - 'wilcoxon' : Wilcoxon signed-rank test
                - 'kruskal' : Kruskal-Wallis H-test

        twotails : bool, optional, [def : False]
            For the permutation method, get p-values from the two tails of the
            distribution.

        maxtstat : integer, optional, [def -2]
            Correct p-values with maximum statistique. maxstat correspond to
            the dimension of perm for correction. Use -1 to correct through all
            dimensions. Otherwise, use d1, d2, ... or dn to correct through a
            specific dimension. If maxstat is -2, not correction is applied.

        n_perm : integer, optional, [def : 200]
            Number of permutations for assessing statistical significiancy.

        n_jobs : integer, optional, [def : -1]
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all cores.

        Returns
        ----------
        xF : array
            The un/normalized feature of x, with a shape of
            (n_frequency x n_electrodes x n_window x n_trials)
        """
        # Get variables :
        self._statmeth = statmeth
        self._n_perm = n_perm
        self._2t = twotails
        self._mxst = maxstat
        # Check input size :
        if len(x.shape) == 2:
            x = x[np.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]
        # Check statistical method :
        _checkref('statmeth', statmeth, ['permutation', 'wilcoxon', 'kruskal'])

        data = Parallel(n_jobs=n_jobs)(
            delayed(_get)(x[k, ...], self) for k in range(nfeat))
        xF, pvalues = zip(*data)
        return np.swapaxes(np.array(xF), 0, 1), np.swapaxes(np.array(pvalues), 0, 1)

    @staticmethod
    def freqvec(fstart, fend, fwidth, fstep):
        """Define a frequency vector

        Parameters
        ----------
        fstart : int
            Starting frequency

        fend : int
            Ending frequency

        fwidth : int
            Width for each band

        fstep : int
            Step between bands
        """
        return binarize(fstart, fend, fwidth, fstep, kind='list')


# ------------------------------------------------------------
# FILTERED SIGNAL
# ------------------------------------------------------------
class sigfilt(_spectral):

    """Extract the filtered signal. """
    __doc__ += _spectral.__doc__ + docsignal + supfilter

    def __init__(self, sf, npts, f=[60, 200], baseline=(1, 2), norm=0,
                 window=None, width=None, step=None, split=None, time=None,
                 **kwargs):
        _spectral.__init__(self, sf, npts, 'signal', f, baseline, norm,
                           'filter', window, width, step, split, time,
                           **kwargs)


# ------------------------------------------------------------
# AMPLITUDE
# ------------------------------------------------------------
class amplitude(_spectral):

    """Extract the amplitude of the signal. """
    __doc__ += _spectral.__doc__ + docsignal
    __doc__ += """
    method : string
        Method to transform the signal. Possible values are:
            - 'hilbert' : apply a hilbert transform to each column
            - 'hilbert1' : hilbert transform to a whole matrix
            - 'hilbert2' : 2D hilbert transform
            - 'wavelet' : wavelet transform
    """ + supfilter

    def __init__(self, sf, npts, f=[60, 200], baseline=(1, 2), norm=0,
                 method='hilbert1', window=None, width=None, step=None,
                 split=None, time=None, **kwargs):
        _checkref('method', method, ['hilbert', 'hilbert1', 'hilbert2',
                  'wavelet'])
        _spectral.__init__(self, sf, npts, 'amplitude', f, baseline, norm,
                           method, window, width, step, split, time,
                           **kwargs)


# ------------------------------------------------------------
# POWER
# ------------------------------------------------------------
class power(_spectral):

    """Extract the power of the signal. """
    __doc__ += _spectral.__doc__ + docsignal
    __doc__ += """
    method : string
        Method to transform the signal. Possible values are:
            - 'hilbert' : apply a hilbert transform to each column
            - 'hilbert1' : hilbert transform to a whole matrix
            - 'hilbert2' : 2D hilbert transform
            - 'wavelet' : wavelet transform
    """ + supfilter

    def __init__(self, sf, npts, f=[60, 200], baseline=(1, 2), norm=0,
                 method='hilbert1', window=None, width=None, step=None,
                 split=None, time=None, **kwargs):
        _checkref('method', method, ['hilbert', 'hilbert1', 'hilbert2',
                  'wavelet'])
        _spectral.__init__(self, sf, npts, 'power', f, baseline, norm,
                           method, window, width, step, split, time,
                           **kwargs)


# ------------------------------------------------------------
# PHASE
# ------------------------------------------------------------
class phase(_spectral):

    """Extract the phase of a signal. """
    __doc__ += _spectral.__doc__
    __doc__ += """method : string
        Method to transform the signal. Possible values are:
            - 'hilbert' : apply a hilbert transform to each column
            - 'hilbert1' : hilbert transform to a whole matrix
            - 'hilbert2' : 2D hilbert transform
    """ + supfilter

    def __init__(self, sf, npts, f=[60, 200], method='hilbert', window=None,
                 width=None, step=None, time=None, **kwargs):
        _checkref('method', method, ['hilbert', 'hilbert1', 'hilbert2'])
        _spectral.__init__(self, sf, npts, 'phase', f, None, 0, method,
                           window, width, step, None, time, **kwargs)


# ------------------------------------------------------------
# SUB GET FUNCTION
# ------------------------------------------------------------
def _get(x, self):
    """Sub get function.

    Get the spectral info of x.
    """
    bsl, norm, n_perm = self._baseline, self._norm, self._n_perm
    maxstat, twotails = self._mxst, self._2t

    # Get the filter properties and apply:
    fMeth = self._fobj.get(self._sf, self._fSplit, self._npts)
    xF = self._fobj.apply(x, fMeth)
    nf, npts, nt = xF.shape

    # Normalize power :
    if norm is not 0:
        xFm = np.mean(xF[:, bsl[0]:bsl[1], :], 1)
        baseline = np.tile(xFm[:, np.newaxis, :], [1, xF.shape[1], 1])
        xFn = normalize(xF, baseline, norm=norm)
    else:
        baseline = np.zeros(xF.shape)

    # Mean Frequencies :
    xF, _ = binArray(xF, self._fSplitIndex, axis=0)
    xFn, _ = binArray(xFn, self._fSplitIndex, axis=0)

    # Mean time :
    if self._window is not None:
        xF, xvec = binArray(xF, self._window, axis=1)
        xFn, xvec = binArray(xFn, self._window, axis=1)

    # Statistical evaluation :
    if norm is not 0:
        pvalues = _evalstat(xF, xFn, xFm, self._statmeth, n_perm, norm,
                            maxstat, twotails)
    else:
        pvalues = None

    return xFn, pvalues


def _evalstat(x, xn, bsl, meth, n_perm, norm, maxstat, twotails):
    """Statistical evaluation of features

    [x] = [xn] = (nFce, npts, nTrials)
    [bsl] = (nFce, nTrials)
    """
    # Get shape of xF :
    nf, npts, nt = x.shape
    pvalues = np.ones((nf, npts))

    # Permutations :
    if meth == 'permutation':
        from brainpipe.xPOO.stats import permutation
        # Pre-define permutations :
        pObj = permutation(n_perm)
        perm = np.zeros((n_perm, nf, npts))
        # For each permutation :
        for p in range(n_perm):
            # Get 1D iterations :
            ite = product(range(nf), range(npts))
            permT = np.random.permutation(2*nt)
            for f, pts in ite:
                bs, xs = bsl[f, :], x[f, pts, :]
                # Reshape data :
                subX = np.vstack((bsl[f, :], x[f, pts, :])).reshape(2*nt,)
                # Shuffle data :
                subX = subX[permT].reshape(nt, 2)
                # Normalize data :
                subX = normalize(subX[:, 0], subX[:, 1], norm=norm)
                # Get mean of data :
                perm[p, f, pts] = np.mean(subX)
        # Get final pvalues :
        pvalues = pObj.perm2p(np.mean(xn, 2), perm, twotails=twotails,
                              maxstat=maxstat)

    # Wilcoxon test :
    elif meth == 'wilcoxon':
        from scipy.stats import wilcoxon
        # Get iterations :
        ite = product(range(nf), range(npts))
        # Compute wilcoxon :
        for k, i in ite:
            _, pvalues[k, i] = wilcoxon(x[k, i, :], bsl[k, :])

    # Kruskal-Wallis :
    elif meth == 'kruskal':
        from scipy.stats import kruskal
        # Get iterations :
        ite = product(range(nf), range(npts))
        # Compute Kruskal-Wallis :
        for k, i in ite:
            _, pvalues[k, i] = kruskal(x[k, i, :], bsl[k, :])

    return pvalues
