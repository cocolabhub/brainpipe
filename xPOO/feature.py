import numpy as n

from brainpipe.xPOO._utils._feat import (_manageWindow, _manageFrequencies,
                                         normalize)
from brainpipe.xPOO._utils._plot import _plot, _2Dplot
from brainpipe.xPOO.tools import binarize, binArray
from brainpipe.xPOO.filtering import fextract
from brainpipe.xPOO._utils._filtering import _apply_method, _get_method
from brainpipe.xPOO._utils._system import adaptsize

from brainpipe.xPOO.cfc.methods import *
from brainpipe.xPOO.cfc._cfc import *

from joblib import Parallel, delayed
from psutil import cpu_count

from itertools import product


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
        List containing the couple of frequency bands. Each couple can be
        either a list or a tuple. Example : f=[ [2,4], [5,7], [60,250] ]
        will compute the power in three frequency bands
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
            Data for computing power. x should have a shape of
            (n_electrodes x n_pts x n_trials)

        n_jobs : integer, optional, [def : -1]
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all cores.

        Returns
        ----------
        xF : array
            The un/normalized power of x, with a shape of
            (n_frequency x n_electrodes x n_window x n_trials)

        """
        if len(x.shape) == 2:
            x = x[n.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]

        xF = Parallel(n_jobs=n_jobs)(
            delayed(_get)(x[k, ...], self) for k in range(nfeat))

        return n.swapaxes(n.array(xF), 0, 1)

    def plot(self, x, title=' feature', xlabel='Time',
             ylabel=' modulations', **kwargs):
        """Simple plot
        """
        return _plot(self.xvec, x, title=self.featKind+title, xlabel=xlabel,
                     ylabel=self.featKind+ylabel,
                     **kwargs)


class TF(_powerDoc):

    """Compute the time-frequency map (TF) of multiple signals.

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
    cumputing doesn't accept objects initialize using an other class.
    May be not esthetic but it works...
    """
    # Define windows and frequency :
    self.filter = fextract(kind=kind, method=method, **kwargs)
    self.window, self.xvec = _manageWindow(npts, window=window, width=width,
                                           step=step, time=time)
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
            Data for computing phase. x should have a shape of
            (n_electrodes x n_pts x n_trials)

        n_jobs : integer, optional, [def : -1]
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all cores.

        Returns
        ----------
        tF : array
            The un/normalized time-frequency map of x, with a shape of
            (n_frequency x n_electrodes x n_window x n_trials)
        """
        if len(x.shape) == 2:
            x = x[n.newaxis, ...]
        if x.shape[1] != self._npts:
            raise ValueError('The second dimension must be '+str(self._npts))
        nfeat = x.shape[0]

        xF = Parallel(n_jobs=n_jobs)(
            delayed(_get)(x[k, ...], self) for k in range(nfeat))

        return n.swapaxes(n.array(xF), 0, 1)

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


class pac(object):

    """Compute the phase-amplitude coupling (pac) either in local or
    distant coupling.

    Parameters
    ----------
    sf : int
        Sampling frequency

    npts : int
        Number of points of the time serie

    Id : string, optional, [def : '114']
        The Id correspond to the way of computing cfc. Id is composed with
        three digits [ex : Id='210']
            - First digit : refer to the method of cfc. Here is the list of
            cfc mesure implemented :
                '1' - Modulation Index (See Canolty, 2006)
                '2' - Kullback-Leibler Distance (See Tort, 2010)
                '3' - Phase synchrony
                '4' - Amplitude PSD
                '5' - Heights Ratio
                '6' - ndPAC (See Ozkurt, 2012)
            - Second digit : cfc mesures are usually sensible to noise. In
            consequence, the second digit refer to the method for computing
            surrogates. List of surrogates methods :
                '0' - No surrogates
                '1' - Shuffle phase values
                '2' - Time lag
                '3' - Swap phase/amplitude through trials
                '4' - Swap amplitude
                '5' - circular shifting
            - Third digit : after computing surrogates, the true cfc mesure
            will be normalized by surrogates. So the third digit refer to the
            way of normalizing cfc by surrogates :
                '0' - No normalization
                '1' - Substraction : substract the mean of surrogates
                '2' - Divide : divide by the mean of surrogates
                '3' - Substract then divide : substract then divide by the mean
                of surrogates
                '4' - Z-score : substract the mean and divide by the deviation
                of the surrogates
        So, if Id='123', this mean that cfc will be evaluate using the
        Modulation Index ('1'), then surrogates will be find by introducing a
        time lag ('2') and finally, the true cfc value will be normalized by
        substracting then dividing by the mean of surrogates.

    pha_f : tuple/list, optional, [def : [2,4]]
            List containing the couple of frequency bands for the phase.
            Each couple can be either a list or a tuple. Example :
            f=[ [2,4], [5,7], [60,250] ] will compute the phase in three
            frequency bands.

    pha_meth : string, optional, [def : 'hilbert']
        Method for the phase extraction.

    pha_cycle : integer, optional, [def : 3]
        Number of cycles for filtering the phase.

    amp_f : tuple/list, optional, [def : [60,200]]
            List containing the couple of frequency bands for the amplitude.
            Each couple can be either a list or a tuple.

    amp_meth : string, optional, [def : 'hilbert']
        Method for the amplitude extraction.

    amp_cycle : integer, optional, [def : 6]
        Number of cycles for filtering the amplitude.

    nbins : integer, optional, [def : 18]
        Some cfc method (like Kullback-Leibler Distance or Heights Ratio) need
        a binarization of the phase. nbins control the number of bins.

    window : tuple, list, None, optional [def: None]
        List/tuple: [100,1500]
        List of list/tuple: [(100,500),(200,4000)]
        None of the width and step parameters will be considered

    width : int, optional [def : None]
        width of a single window.

    step : int, optional [def : None]
        Each window will be spaced by the "step" value.

    time : list/array, optional [def: None]
        Define a specific time vector


    Methods
    ----------
    -> get : get the cfc without statistique (no surrogates and no
    normalization just the true cfc mesure)

    -> statget : get the normalized cfc

    Contributor: Juan LP Soto
    """

    def __init__(self, sf, npts, Id='114',
                 pha_f=[2, 4], pha_meth='hilbert', pha_cycle=3,
                 amp_f=[60, 200], amp_meth='hilbert', amp_cycle=6,
                 nbins=18, window=None, width=None, step=None,
                 time=None, **kwargs):

        # Define windows and frequency :
        self.pha = fextract(kind='phase', method=pha_meth,
                            cycle=pha_cycle, **kwargs)
        self.amp = fextract(kind='amplitude', method=amp_meth,
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
        self.Id = Id
        _, _, _, ModelStr, SurStr, NormStr = CfcSettings(Id, nbins)
        self.model = ['Method : '+ModelStr, 'Surrogates : '+SurStr,
                      'Normalization : '+NormStr]
        self.nbins = nbins
        self.width = width
        self.step = step
        self._nPha = len(self.pha.f)
        self._nAmp = len(self.amp.f)
        self._sf = sf
        self._npts = npts
        self._nwin = len(self.window)

    def __str__(self):
        phafilt = 'Phase : '+str(self.pha)
        ampfilt = 'Amplitude : '+str(self.amp)
        met = self.model[0]+',\n'+self.model[1]+',\n'+self.model[2]+',\n'
        cfcStr = 'Crossfrequency Coupling(step='+str(self.step)+', width='+str(
            self.width)+', Id='+self.Id+', nbins='+str(self.nbins)+',\n'+met

        return cfcStr+phafilt+',\n'+ampfilt+')'

    def get(self, xpha, xamp, n_jobs=-1):
        """Get the true cfc mesure between an xpha and xamp signals.

        Parameters
        ----------
        xpha : array
            Signal for phase. The shape of xpha should be :
            (n_electrodes x n_pts x n_trials)

        xamp : array
            Signal for amplitude. The shape of xamp should be :
            (n_electrodes x n_pts x n_trials)

        n_jobs : integer, optional, [def : -1]
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all cores.

        If the same signal is used (example : xpha=x and xamp=x), this mean
        the program compute a local cfc.

        Returns
        ----------
        ucfc : array
            The unormalized cfc mesure. The size of ucfc depends of parameters
            but in general it is :
            (n_phase x n_amplitude x n_electrodes x n_windows x n_trials)
        """
        # Check the inputs variables :
        xpha, xamp = _cfcCheck(xpha, xamp, self._npts)
        N = xpha.shape[0]

        # Get the unormalized cfc:
        uCfc = Parallel(n_jobs=n_jobs)(delayed(_cfcFilt)(
            xpha[k, ...], xamp[k, ...], self) for k in range(N))

        return adaptsize(n.array(uCfc), (2, 3, 4, 1, 0))

    def statget(self, xpha, xamp, n_jobs=-1, n_perm=200):
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
            Number of permutations for normalizing the cfc mesure.

        n_jobs : integer, optional, [def : -1]
            Control the number of jobs for parallel computing. Use 1, 2, ...
            depending of your number or cores. -1 for all the cores.

        If the same signal is used (example : xpha=x and xamp=x), this mean
        the program compute a local cfc.

        Returns
        ----------
        ncfc : array
            The unormalized cfc mesure. The size of ncfc depends of parameters
            but in general it is :
            (n_electrodes x n_windows x n_trials x n_amplitude x n_phase)

        pvalue : array
            The associated p-values. The size of pvalue depends of parameters
            but in general it is :
            (n_electrodes x n_windows x n_trials x n_amplitude x n_phase)
        """
        # Check the inputs variables :
        xpha, xamp = _cfcCheck(xpha, xamp, self._npts)
        self.n_perm = n_perm
        self.p = 1/n_perm
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

        # Get the unormalized cfc and surogates:
        cfcsu = Parallel(n_jobs=elecJob)(delayed(_cfcFiltSuro)(
            xpha[k, ...], xamp[k, ...], surJob, self) for k in range(N))
        uCfc, Suro, mSuro, stdSuro = zip(*cfcsu)
        uCfc = n.array(uCfc)
        Suro = n.array(Suro)
        mSuro = n.array(mSuro)
        stdSuro = n.array(stdSuro)

        # Normalize each cfc:
        _, _, Norm, _, _, _ = CfcSettings(self.Id)
        nCfc = Norm(uCfc, mSuro, stdSuro)

        # Confidence interval :
        pvalue = n.array([_cfcPvalue(nCfc[k, ...], Suro[
            k, ...]) for k in range(N)])
        nsz = (2, 3, 4, 1, 0)

        return adaptsize(nCfc, nsz), adaptsize(pvalue, nsz)


# ----------------------------------------------------------------------------
#                          Phase-locking value
# ----------------------------------------------------------------------------


class plv(object):

    """Compute the phase synchrony between two signals.
    """

    def __init__(self):
        pass

    def get(self):
        """
        """
        pass

    def statget(self):
        """
        """
        pass

# ----------------------------------------------------------------------------
#                          Prefered-phase
# ----------------------------------------------------------------------------


class preferedphase(object):

    """For a given amplitude, get the prefered phase activation.
    """

    def __init__(self):
        pass

    def get(self):
        """
        """
        pass

    def statget(self):
        """
        """
        pass


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

    def statget(self):
        """
        """
        pass
