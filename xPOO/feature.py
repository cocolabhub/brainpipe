import numpy as n

from brainpipe.xPOO._utils._feat import (_manageWindow, _manageFrequencies,
                                         normalize)
from brainpipe.xPOO._utils._plot import _plot, _2Dplot
from brainpipe.xPOO.tools import binarize, binArray
from brainpipe.xPOO.filtering import fextract
from brainpipe.xPOO._utils._filtering import _apply_method, _get_method
from brainpipe.xPOO._utils._system import groupInList, list2index, adaptsize
from brainpipe.xPOO.cfc.methods import *

from joblib import Parallel, delayed


__all__ = [
    'power',
    'TF',
    'phase',
    'cfc'
]


# ----------------------------------------------------------------------------
#                                   POWER
# ----------------------------------------------------------------------------
class _powerDoc(object):

    """
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
        width of a single window.

    step : int, optional [def : None]
        Each window will be spaced by the "step" value.

    split : int or list of int, optional [def: None]
        Split the frequency band f in "split" band width.
        If f is a list which contain couple of frequencies,
        split is a list too.

    time : list/array, optional [def: None]
        Define a specific time vector

    **kwargs : additional arguments for filtering
        See of description of the filtsig module
    """


class power(_powerDoc):

    """Compute the power of multiple signals.

    Parameters
    ----------
    sf : int
        Sampling frequency

    npts : int
        Number of points of the time serie

    f : tuple/list, optional, [def : [60,200]]
        List containing the couple of frequency bands.
        Each couple can be either a list or a tuple.
        Example : f=[ [2,4], [5,7], [60,250] ] will compute
        the power in three frequency bands
    """
    __doc__ += _powerDoc.__doc__

    def __init__(self, sf, npts, f=[60, 200], baseline=(1, 2), norm=0,
                 method='hilbert1', window=None, width=None, step=None,
                 split=None, time=None, **kwargs):
        self = feat_init(self, sf, f, npts, baseline, norm, method,
                         window, width, step, split, time, 'power', **kwargs)

    def __str__(self):
        extractStr = str(self.filter)
        powStr = 'Power(norm='+str(self.norm)+', step='+str(
            self.step)+', width='+str(self.width)+', split='+str(
            self.split)+',\n'

        return powStr+extractStr+')'

    def get(self, x, n_jobs=-1):
        """Get the power of the signal x. This method is optimized
        for 3D matrix x.

        Parameters
        ----------
        x : array
            - If x is a 2D array of size (npts, ntrial) and f, the frequency
            vector has a length of nf, the "get" method will return a
            (nf, npts, ntrial) array
            - If x is a 3D array of size (N, npts, ntrial), power is calculated
            for each N and a list of length N is return and each element of it
            have a size of (nf, npts, ntrial).

        n_jobs : integer
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all the cores.
        """
        if len(x.shape) == 2:
            x = x[n.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]

        xF = Parallel(n_jobs=n_jobs)(
            delayed(_get)(x[k, ...], self) for k in range(nfeat))

        return n.squeeze(xF)

    def plot(self, x, title=' feature', xlabel='Time',
             ylabel=' modulations', **kwargs):
        """Simple plot
        """
        return _plot(self.xvec, x, title=self.featKind+title, xlabel=xlabel,
                     ylabel=self.featKind+ylabel,
                     **kwargs)


class TF(_powerDoc):

    """Compute the time-frequency map of multiple signals.

    Parameters
    ----------
    sf : int
        Sampling frequency

    npts : int
        Number of points of the time serie

    f : tuple/list, optional, [def : (2, 200, 20, 10)]
        Define the frequency vector to compute the time-frequency maps.
        This tuple is define like thos f=(fstart, fend, fwidth, fstep)
        where :
        fstart, fend : starting and ending frequency
        fwidth, fstep : sliding frequency window of length fwidth and
                        fstep sliding. For example, if fwidth=20 and
                        fstep=10, this mean there is a 50% covering
    """
    __doc__ += _powerDoc.__doc__

    def __init__(self, sf, npts, f=(2, 200, 20, 10), baseline=(1, 2), norm=0,
                 method='hilbert1', window=None, width=None, step=None,
                 time=None, **kwargs):
        f = binarize(f[0], f[1], f[2], f[3], kind='list')
        self = feat_init(self, sf, f, npts, baseline, norm, method,
                         window, width, step, None, time, 'power', **kwargs)

    def __str__(self):
        extractStr = str(self.filter)
        powStr = 'tf(norm='+str(self.norm)+', step='+str(
            self.step)+', width='+str(self.width)+', split='+str(
            self.split)+',\n'

        return powStr+extractStr+')'

    def get(self, x, n_jobs=-1):
        """Get the time frequency map of a signal x. This method is optimized
        for 3D matrix x.

        Parameters
        ----------
        x : array
            - If x is a 2D array of size (npts, ntrial) and f, the frequency
            vector has a length of nf, the "get" method will return a
            (nf, npts, ntrial) array
            - If x is a 3D array of size (N, npts, ntrial), power is calculated
            for each N and a list of length N is return and each element of it
            have a size of (nf, npts, ntrial).

        n_jobs : integer
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all the cores.
        """
        if len(x.shape) == 2:
            x = x[n.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]

        # Compute tf in parallele :
        bkp_win, bkp_norm = self.window, self.norm
        tF = Parallel(n_jobs=n_jobs)(
            delayed(_tf)(x[k, ...], self, bkp_win,
                         bkp_norm) for k in range(nfeat))

        return tF

    def plot(self, tf, title='Time-frequency map', xlabel='Time',
             ylabel='Frequency', cblabel='Power modulations', interp=(1, 1),
             **kwargs):
        return _2Dplot(tf, self.xvec, self.yvec, title=title,
                       xlabel=xlabel, ylabel=ylabel,
                       cblabel=cblabel, interp=interp, **kwargs)


def _tf(x, self, bkp_win, bkp_norm):
    """Sub-tf function

    Compute the tf for a single x.
    """
    # Get the power :
    self.window, self.norm = None, 0
    tf = _get(x, self)
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


def _get(x, self):
    """Sub get function.

    Get the power of a signal x.
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


def feat_init(self, sf, f, npts, baseline, norm, method, window, width, step,
              split, time, kind, **kwargs):
    """Initialize power objects.

    I defined a function to initialize power objects because the parallel
    cumputing doesn't accept objects initialize using an othr class.
    May be not esthetic but it works...
    """
    # Define windows and frequency :
    self.filter = fextract(kind=kind, method=method, **kwargs)
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
    self.featKind = kind

    return self


# ----------------------------------------------------------------------------
#                                   PHASE
# ----------------------------------------------------------------------------

class phase(object):

    """Compute the phase of multiple signals.

    Parameters
    ----------
    sf : int
        Sampling frequency

    npts : int
        Number of points of the time serie

    f : tuple/list, optional, [def : [60,200]]
        List containing the couple of frequency bands.
        Each couple can be either a list or a tuple.
        Example : f=[ [2,4], [5,7], [60,250] ] will compute
        the phase in three frequency bands
    """
    __doc__ += _powerDoc.__doc__

    def __init__(self, sf, npts, f=[60, 200], method='hilbert', window=None,
                 width=None, step=None, time=None, **kwargs):
        self = feat_init(self, sf, f, npts, None, 0, method,
                         window, width, step, None, time, 'phase', **kwargs)

    def __str__(self):
        extractStr = str(self.filter)
        phaStr = 'Phase(step='+str(self.step)+', width='+str(
            self.width)+',\n'

        return phaStr+extractStr+')'

    def get(self, x, n_jobs=-1):
        """Get the phase of the signal x. This method is optimized
        for 3D matrix x.

        Parameters
        ----------
        x : array
            - If x is a 2D array of size (npts, ntrial) and f, the frequency
            vector has a length of nf, the "get" method will return a
            (nf, npts, ntrial) array
            - If x is a 3D array of size (N, npts, ntrial), phase is calculated
            for each N and a list of length N is return and each element of it
            have a size of (nf, npts, ntrial).

        n_jobs : integer
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all the cores.
        """
        if len(x.shape) == 2:
            x = x[n.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]

        xF = Parallel(n_jobs=n_jobs)(
            delayed(_get)(x[k, ...], self) for k in range(nfeat))

        return n.squeeze(xF)

    def plot(self, x, title=' feature', xlabel='Time',
             ylabel=' modulations', **kwargs):
        """Simple plot
        """
        return _plot(self.xvec, x, title=self.featKind+title, xlabel=xlabel,
                     ylabel=self.featKind+ylabel,
                     **kwargs)


# ----------------------------------------------------------------------------
#                          CROSS-FREQUENCY COUPLING
# ----------------------------------------------------------------------------


class cfc(object):

    def __init__(self, sf, npts, Id='114', phase=[2, 4], amplitude=[60, 200],
                 method_phase='hilbert', method_amp='hilbert',
                 cycle=(3, 6), nbins=18, window=None, width=None, step=None,
                 time=None, **kwargs):

        # Define windows and frequency :
        self.pha = fextract(kind='phase', method=method_phase,
                            cycle=cycle[0], **kwargs)
        self.amp = fextract(kind='amplitude', method=method_amp,
                            cycle=cycle[1], **kwargs)
        self.window, self.xvec = _manageWindow(npts, window=window,
                                               width=width, step=step,
                                               time=time)
        self.pha.f, _, _ = _manageFrequencies(phase, split=None)
        self.amp.f, _, _ = _manageFrequencies(amplitude, split=None)
        if self.window is None:
            self.window = [(0, npts)]
            self.xvec = [0, npts]

        # Get variables :
        self.Id = Id
        self.nbins = nbins
        self.width = width
        self.step = step
        self._nPha = len(self.pha.f)
        self._nAmp = len(self.amp.f)
        self._sf = sf
        self._npts = npts

    def get(self, x, n_jobs=-1, xPha=None, xAmp=None):
        """Get the cfc mesure of an input signal.
        """
        # Check input variables :
        if xPha is None:
            xPha = x
        if xAmp is None:
            xAmp = x
        npts, ntrial = xPha.shape
        W = self.window
        nwin = len(W)

        # Get the filter for phase/amplitude properties :
        phaMeth = self.pha.get(self._sf, self.pha.f, self._npts)
        ampMeth = self.amp.get(self._sf, self.amp.f, self._npts)

        # Filt the phase and amplitude :
        xPha = self.pha.apply(xPha, phaMeth)
        xAmp = self.amp.apply(xAmp, ampMeth)

        claIdx, listWin, listTrial = list2index(nwin, ntrial)

        # Run the classification :
        uCfc = Parallel(n_jobs=n_jobs)(delayed(_cfcGet)(
            n.squeeze(xPha[:, W[k[0]][0]:W[k[0]][1], k[1]]),
            n.squeeze(xAmp[:, W[k[0]][0]:W[k[0]][1], k[1]]),
            self) for k in claIdx)
        uCfc = n.array(groupInList(uCfc, listWin))

        return uCfc


def _cfcGet(pha, amp, self):

    # Get the cfc model :
    Model, Sur, Norm, ModelStr, SurStr, NormStr = CfcSettings(self.Id)

    return Model(n.matrix(pha), n.matrix(amp))
