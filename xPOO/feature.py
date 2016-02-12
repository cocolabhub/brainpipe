import numpy as n
from brainpipe.xPOO._utils._feat import (_manageWindow, _manageFrequencies,
                                         normalize, _featC)
from brainpipe.xPOO._utils._plot import _2Dplot
from brainpipe.xPOO.tools import binarize, binArray
from brainpipe.xPOO.filtering import fextract
from brainpipe.xPOO._utils._filtering import _apply_method, _get_method
from joblib import Parallel, delayed

__all__ = [
    'power',
    'phase',
    'cfc'
]


# ----------------------------------------------------------------------------
#                                   POWER
# ----------------------------------------------------------------------------

class power(_featC):

    """Compute the power of multiple signals.

    Parameters
    ----------
    sf : int
        Sampling frequency

    f : list
        List containing the couple of frequency bands.
        Each couple can be either a list or a tuple.

    npts : int
        Number of points of the time serie

    norm : int, optional [def : 0]
        Number to choose the normalization method
            0 : No normalisation
            1 : Substraction
            2 : Division
            3 : Substract then divide
            4 : Z-score

    baseline : tuple/list of int [def: (1,1)]
        Define a window to normalize the power

    method : string
        Method to transform the signal. Possible values are:
            - 'hilbert' : apply a hilbert transform to each column
            - 'hilbert1' : hilbert transform to a whole matrix
            - 'hilbert2' : 2D hilbert transform
            - 'wavelet' : wavelet transform
            - 'filter' : filtered signal

    window : tuple, list, None, optional [def: None]
        List/tuple: [100,1500]
        List of list/tuple: [(100,500),(200,4000)]
        None and the width and step paameters will be considered

    width : int, optional [def : None]
        width of a single window

    step : int, optional [def : None]
        Each window will be spaced by the "step" value

    split : int or list of int, optional [def: None]
        Split the frequency band f in "split" band width.
        If f is a list which contain couple of frequencies,
        split is a list too.

    time : list/array, optional [def: None]
        Define a specific time vector

    **kwargs : additional arguments for filtering
        See of description of the filtsig module

    Method
    ----------
    get : compute the power of a signal
    freqvec : define a frequency vector
    tf : time-frequency map
    plot : simple power plot
    tfplot : time-frequency plot
    """

    def __init__(self, sf, f, npts, baseline=(1, 2), norm=0, method='hilbert1',
                 window=None, width=None, step=None, split=None, time=None,
                 **kwargs):

        # Define windows and frequency :
        self.filter = fextract(kind='power', method=method, **kwargs)
        self.window, self.xvec = _manageWindow(
            npts, window=window, width=width, step=step, time=time)
        self.f, self.fSplit, self._fSplitIndex = _manageFrequencies(
            f, split=split)

        # Get variables :
        self.baseline = baseline
        self.norm = norm
        self.width = width
        self.step = step
        self.split = split
        self._nf = len(self.f)
        self._sf = sf
        self._npts = npts
        self.yvec = [round((k[0]+k[1])/2) for k in self.f]
        self.featKind = 'Power'

    def __str__(self):
        extractStr = str(self.filter)
        powStr = 'Power(norm='+str(self.norm)+', step='+str(
            self.step)+', width='+str(self.width)+', split='+str(
            self.split)+',\n'

        return powStr+extractStr+')'

    def get(self, x, n_jobs=-1):
        if len(x.shape) == 2:
            x = x[n.newaxis, ...]
        nfeat = x.shape[0]

        xF = Parallel(n_jobs=n_jobs)(
            delayed(_get)(x[k, ...], self) for k in range(nfeat))

        return n.squeeze(xF)

    def tf(self, x, f=None, n_jobs=-1):
        """Compute the Time-frequency map

        x : array
            - If x is a 2D array of size (npts, ntrial) and f, the frequency
            vector has a length of nf, the "tf" method will return the mean
            time-frequency map of size (nf, npts).
            - If x is a 3D array of size (N, npts, ntrial), tf is calculated
            for each N and a list of length N is return and each element of it
            have a size of (nf, npts).

        f : tuple/list, optional, [def: None]
            f is to specified a particular frequency vector for the tf.
            It must be: f=(fstart, fend, fwidth, fstep)
        """
        # Define variables :
        if len(x.shape) == 2:
            x = x[n.newaxis, ...]
        nfeat = x.shape[0]
        bkp_win, bkp_norm = self.window, self.norm
        window = self.window
        self.window = None

        # Define a frequency a vector :
        if f is not None:
            self.f = binarize(f[0], f[1], f[2], f[3], kind='list')
            self._nf = len(self.f)
            _, self.fSplit, self._fSplitIndex = _manageFrequencies(
                self.f, split=None)
            self.yvec = [((k[0]+k[1])/2).astype(int) for k in self.f]

        # Compute tf in parallele :
        tF = Parallel(n_jobs=n_jobs)(
            delayed(_tf)(x[k, ...], self, bkp_win,
                         bkp_norm) for k in range(nfeat))

        return tF

    def tfplot(self, tf, title='Time-frequency map', xlabel='Time',
               ylabel='Frequency', cblabel='Power modulations', interp=(1, 1),
               **kwargs):
        return _2Dplot(tf, self.xvec, self.yvec, title=title,
                       xlabel=xlabel, ylabel=ylabel,
                       cblabel=cblabel, interp=interp, **kwargs)


def _get(x, self):
    """Sub-tf function
    """
    # Get the filter properties and apply:
    fMeth = self.filter.get(self._sf, self.fSplit, self._npts)
    xF = self.filter.apply(x, fMeth)

    # Normalize power :
    if self.norm is not 0:
        xF = normalize(xF, n.tile(n.mean(xF[:, self.baseline[0]:self.baseline[
            1], :], 1)[:, n.newaxis, :], [1, xF.shape[1], 1]), norm=self.norm)

    # Mean Frequencies :
    xF, _ = binArray(xF, self._fSplitIndex, axis=0)

    # Mean time :
    if self.window is not None:
        xF, xvec = binArray(xF, self.window, axis=1)

    return xF


def _tf(x, self, bkp_win, bkp_norm):
    """Sub-tf function
    """
    # Get the power :
    self.window, self.norm = None, 0
    tf = self.get(x, n_jobs=1)
    self.window, self.norm = bkp_win, bkp_norm

    # Normalize the power or not :
    if (self.norm != 0) or (self.baseline != (1, 1)):
        X = n.mean(tf, 2)
        Y = n.matlib.repmat(n.mean(X[:, self.baseline[0]:self.baseline[
            1]], 1), X.shape[1], 1).T
        tfn = normalize(X, Y, norm=self.norm)
    else:
        tfn = n.mean(tf, 2)

    # Mean time :
    if self.window is not None:
        tfn, _ = binArray(tfn, bkp_win, axis=1)

    return tfn


# ----------------------------------------------------------------------------
#                                   PHASE
# ----------------------------------------------------------------------------

class phase(_featC):

    """Compute the phase of multiple signals.

    Parameters
    ----------
    sf : int
        Sampling frequency

    f : list
        List containing the couple of frequency bands.
        Each couple can be either a list or a tuple.

    npts : int
        Number of points of the time serie

    method : string
        Method to transform the signal. Possible values are:
            - 'hilbert' : apply a hilbert transform to each column
            - 'hilbert1' : hilbert transform to a whole matrix
            - 'hilbert2' : 2D hilbert transform

    window : tuple, list, None, optional [def: None]
        List/tuple: [100,1500]
        List of list/tuple: [(100,500),(200,4000)]
        None and the width and step paameters will be considered

    width : int, optional [def : None]
        width of a single window

    step : int, optional [def : None]
        Each window will be spaced by the "step" value

    time : list/array, optional [def: None]
        Define a specific time vector

    **kwargs : additional arguments for filtering
        See of description of the filtsig module

    Method
    ----------
    get : compute the power of a signal
    freqvec : define a frequency vector
    tfmap : time-frequency map
    plot : simple power plot
    tfplot : time-frequency plot
    """

    def __init__(self, sf, f, npts, method='hilbert', window=None,
                 width=None, step=None, time=None, **kwargs):
        self.filter = fextract(kind='phase', method=method, **kwargs)
        self.window, self.xvec = _manageWindow(
            npts, window=window, width=width, step=step)
        self.f, _, _ = _manageFrequencies(f, split=None)
        self.width = width
        self.step = step
        self._nf = len(self.f)
        self.__sf = sf
        self._npts = npts
        self.fMeth = self.filter.get(sf, self.f, npts)
        self.yvec = [round((k[0]+k[1])/2) for k in self.f]
        self.featKind = 'Phase'

    def __str__(self):
        extractStr = str(self.filter)
        phaStr = 'Phase(step='+str(self.step)+', width='+str(
            self.width)+',\n'

        return phaStr+extractStr+')'

    def _get(self, x):
        npts, ntrial = x.shape
        # Get power :
        xF = self.filter.apply(x, self.fMeth)
        # Mean time :
        if self.window is not None:
            xF, _ = binArray(xF, self.window, axis=1)
        return xF


class cfc(object):

    def __init__(self):
        pass
