import numpy as n

from brainpipe.xPOO._utils._feat import (_manageWindow, _manageFrequencies,
                                         normalize)
# from brainpipe.xPOO._utils._plot import _plot, _2Dplot
from brainpipe.xPOO.tools import binarize, binArray
from brainpipe.xPOO.filtering import fextract
from brainpipe.xPOO._utils._filtering import _apply_method, _get_method
from brainpipe.xPOO._utils.S import adaptsize

from brainpipe.xPOO.cfc.methods import *
from brainpipe.xPOO.cfc._cfc import *
from brainpipe.xPOO._utils._preferedphase import *

from joblib import Parallel, delayed
from psutil import cpu_count


__all__ = [
    'power',
    'TF',
    'phase',
    'pac',
    'plv',
    'preferedphase',
    'pentropy'
]


# ----------------------------------------------------------------------------
#                                   POWER
# ----------------------------------------------------------------------------


class TF(object):

    """Compute the time-frequency map (TF) of multiple signals. Use the get()
    method to compute the TF.

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
                 method='hilbert1', rejection=False, window=None, width=None,
                 step=None, time=None, **kwargs):
        f = binarize(f[0], f[1], f[2], f[3], kind='list')
        self = _simpleInit(self, sf, f, npts, baseline, norm, method,
                           window, width, step, None, time, 'power', **kwargs)
        self.rejection = rejection

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
            Data for computing the TF. x should have a shape of
            (n_electrodes x n_pts x n_trials)

        n_jobs : integer, optional, [def : -1]
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all cores.

        Returns
        ----------
        tF : array
            The un/normalized time-frequency map of x, with a shape of
            (n_frequency x n_electrodes x n_window)

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

        return n.array(tF)

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

    # Reject some values :
    if self.rejection:
        th = n.mean(tf) + 2*n.std(tf)
        tf[tf > th] = n.nan

    # Normalize the power or not :
    if (self.norm != 0) or (self.baseline != (1, 1)):
        X = n.nanmean(tf, 2)
        Y = n.matlib.repmat(n.nanmean(X[:, self.baseline[0]:self.baseline[
            1]], 1), X.shape[1], 1).T
        tfn = normalize(X, Y, norm=self.norm)
    else:
        tfn = n.nanmean(tf, 2)

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


# ----------------------------------------------------------------------------
#                          Prefered-phase
# ----------------------------------------------------------------------------


class preferedphase(object):

    """For a given amplitude, get the prefered phase. The amplitude is bined,
    relatively to the phase. Use the get() method to compute the prefered
    phase.

    Parameters
    ----------
    sf : int
        Sampling frequency

    npts : int
        Number of points of the time serie

    nbins : integer, optional, [def : 72]
        Number of bins to binarized the phase.
    """
    __doc__ += _cfcDoc.__doc__
    __doc__ += _winDoc.__doc__

    def __init__(self, sf, npts, nbins=72,
                 pha_f=[2, 4], pha_meth='hilbert', pha_cycle=3,
                 amp_f=[60, 200], amp_meth='hilbert', amp_cycle=6,
                 window=None, width=None, step=None,
                 time=None, **kwargs):

        # Initalize pac object :
        pha_kind = 'phase'
        amp_kind = 'amplitude'
        self = _cfcInit(self, pha_f, pha_kind, pha_meth, pha_cycle,
                        amp_f, amp_kind, amp_meth, amp_cycle,
                        sf, npts, window, width, step, time, **kwargs)
        self.nbins = nbins
        step = 2*n.pi/nbins
        self._vecbin = binarize(0, 2*n.pi+step, step, step)
        if len(self._vecbin) > nbins:
            self._vecbin = self._vecbin[0:-1]
        self.centerbin = [n.mean(k) for k in self._vecbin]
        self.nbins = len(self._vecbin)

    def get(self, xpha, xamp, n_perm=200, n_jobs=-1):
        """Get the normalized cfc mesure between an xpha and xamp signals.

        Parameters
        ----------
        xpha : array
            Signal for phase. The shape of xpha should be :
            (n_electrodes x n_pts x n_trials)

        xamp : array
            Signal for amplitude. The shape of xamp should be :
            (n_electrodes x n_pts x n_trials)

        n_perm : integer, optional, [def : 200]
            Number of permutations for assessing reliability of the prefered
            phase.

        n_jobs : integer, optional, [def : -1]
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all the cores.

        If the same signal is used (example : xpha=x and xamp=x), this mean
        the program compute a local prefered phase.

        Returns
        ----------
        ncfc : array
            The unormalized cfc mesure. The size of ncfc depends of parameters
            but in general it is :
            (n_phase x n_amplitude x n_electrodes x n_windows x n_trials)

        pvalue : array
            The associated p-values. The size of pvalue depends of parameters
            but in general it is :
            (n_phase x n_amplitude x n_electrodes x n_windows)
        """
        # Check the inputs variables :
        xpha, xamp = _cfcCheck(xpha, xamp, self._npts)
        self.n_perm = n_perm
        N = xpha.shape[0]

        # Manage jobs repartition :
        if (N < cpu_count()) and (n_jobs != 1):
            surJob = n_jobs
            elecJob = 1
        elif (N >= cpu_count()) and (n_jobs != 1):
            surJob = 1
            elecJob = n_jobs
        else:
            surJob, elecJob = 1, 1

        # Get the unormalized prefered phase :
        pdph = Parallel(n_jobs=elecJob)(delayed(_pfdph)(
            xpha[k, ...], xamp[k, ...], surJob, self) for k in range(N))
        abin, pvalue = zip(*pdph)

        # Compute pvalue :
        if n_perm == 0:
            pvalue = None
        else:
            pvalue = n.array(pvalue)

        return n.array(abin), pvalue


def _cfcInit(self, pha_f, pha_kind, pha_meth, pha_cycle,
             amp_f, amp_kind, amp_meth, amp_cycle,
             sf, npts, window, width, step, time, **kwargs):
    # Define windows and frequency :
    self.pha = fextract(kind=pha_kind, method=pha_meth,
                        cycle=pha_cycle, **kwargs)
    self.amp = fextract(kind=amp_kind, method=amp_meth,
                        cycle=amp_cycle, **kwargs)
    self.window, self.xvec = _manageWindow(npts, window=window,
                                           width=width, step=step,
                                           time=time)
    self.pha.f, _, _ = _manageFrequencies(pha_f, split=None)
    self.amp.f, _, _ = _manageFrequencies(amp_f, split=None)
    if self.window is None:
        self.window = [(0, npts)]
        self.xvec = [0, npts]

    # Get variables :
    self.width = width
    self.step = step
    self._nPha = len(self.pha.f)
    self._nAmp = len(self.amp.f)
    self._sf = sf
    self._npts = npts
    self._nwin = len(self.window)

    return self


# ----------------------------------------------------------------------------
#                          Permutation entropie
# ----------------------------------------------------------------------------


class pentropy(object):

    """
    """

    def __init__(self):
        pass

    def get(self):
        """
        """
        pass
