from .filtering import fextract, docfilter
from brainpipe.xPOO.tools import binarize, binArray
from .utils._feat import (_manageWindow, _manageFrequencies, normalize,
                          _checkref)

from joblib import Parallel, delayed
import numpy as np

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
class _spectral(object):
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

    def __str__(self):
        extractStr = str(self._fobj)
        powStr = '{ki}(norm={no}, step={st}, width={wi}, split={sp},\n'
        powStr = powStr.format(ki=self._kind, no=self._norm, st=self._step,
                               wi=self._width, sp=self._split)
        return powStr+extractStr+')'

    def get(self, x, n_perm=200, n_jobs=-1):
        """Get the spectral informations of the signal x.

        Parameters
        ----------
        x : array
            Data. x should have a shape of
            (n_electrodes x n_pts x n_trials)

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
        if len(x.shape) == 2:
            x = x[np.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]

        xF = Parallel(n_jobs=n_jobs)(
            delayed(_get)(x[k, ...], self) for k in range(nfeat))
        return np.swapaxes(np.array(xF), 0, 1)

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
    bsl = self._baseline
    # Get the filter properties and apply:
    fMeth = self._fobj.get(self._sf, self._fSplit, self._npts)
    xF = self._fobj.apply(x, fMeth)

    # Normalize power :
    if self._norm is not 0:
        xF = normalize(xF, np.tile(np.mean(xF[:, bsl[0]:bsl[1], :], 1)[
          :, np.newaxis, :], [1, xF.shape[1], 1]), norm=self._norm)

    # Mean Frequencies :
    xF, _ = binArray(xF, self._fSplitIndex, axis=0)

    # Mean time :
    if self._window is not None:
        xF, xvec = binArray(xF, self._window, axis=1)

    return xF
