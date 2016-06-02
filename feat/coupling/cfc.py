from joblib import Parallel, delayed
from psutil import cpu_count
import numpy as np
from itertools import product
from scipy.special import erfinv

from brainpipe.feat.utils._feat import (_manageWindow, _manageFrequencies,
                                        _checkref)
from brainpipe.feat.filtering import fextract, docfilter
from brainpipe.feat.coupling.pac._pac import *
from brainpipe.feat.coupling.pac.pacmeth import *
from brainpipe.visu.cmon_plt import tilerplot
from brainpipe.tools import binarize
from brainpipe.statistics import perm_2pvalue
from brainpipe.feat.utils._feat import normalize
from brainpipe.feature import power, phase, sigfilt
from brainpipe.visual import addLines

__all__ = ['pac', 'PhaseLockedPower']


windoc = """
        window: tuple/list/None, optional [def: None]
            List/tuple: [100,1500]
            List of list/tuple: [(100,500),(200,4000)]
            None and the width and step parameters will be considered

        width: int, optional [def: None]
            width of a single window.

        step: int, optional [def: None]
            Each window will be spaced by the "step" value.

        time: list/array, optional [def: None]
            Define a specific time vector

    """

Footnotes = """

    .. rubric:: Footnotes
    .. [#f1] `Canolty et al, 2006 <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2628289/>`_
    .. [#f2] `Tort et al, 2010 <http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2941206/>`_
    .. [#f3] `Ozkurt et al, 2012 <http://www.ncbi.nlm.nih.gov/pubmed/22531738/>`_
    .. [#f4] `Bahramisharif et al, 2013 <http://www.jneurosci.org/content/33/48/18849.short/>`_

"""


class _coupling(tilerplot):

    """
    """

    def __init__(self, pha_f, pha_kind, pha_meth, pha_cycle,
                 amp_f, amp_kind, amp_meth, amp_cycle,
                 sf, npts, window, width, step, time, **kwargs):
        # Define windows and frequency :
        self._pha = fextract(kind=pha_kind, method=pha_meth,
                             cycle=pha_cycle, **kwargs)
        self._amp = fextract(kind=amp_kind, method=amp_meth,
                             cycle=amp_cycle, **kwargs)
        self._window, self._xvec = _manageWindow(npts, window=window,
                                                 width=width, step=step,
                                                 time=time)
        self._pha.f, _, _ = _manageFrequencies(pha_f, split=None)
        self._amp.f, _, _ = _manageFrequencies(amp_f, split=None)
        if self._window is None:
            self._window = [(0, npts)]
            self.xvec = [0, npts]

        # Get variables :
        self._width = width
        self._step = step
        self._nPha = len(self._pha.f)
        self._nAmp = len(self._amp.f)
        self._sf = sf
        self._npts = npts
        self._nwin = len(self._window)
        self.pha = [np.mean(k) for k in self._pha.f]
        self.amp = [np.mean(k) for k in self._amp.f]


class pac(_coupling):

    """Compute the phase-amplitude coupling (pac) either in local or
    distant coupling. PAC require three things:

        - Main method for compute it
        - Surrogates to correct the true pac value
        - A normalization method to correct true pas value bu surrogates

    Contributor: Juan LP Soto.

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

    Kargs:
        Id: string, optional, [def: '113']
            The Id correspond to the way of computing pac. Id is composed of
            three digits [ex : Id='210']

                * First digit: refer to the pac method:

                    - '1': Modulation Index [#f1]_
                    - '2': Kullback-Leibler Distance [#f2]_
                    - '3': Heights Ratio
                    - '4': Phase synchrony
                    - '5': ndPAC [#f3]_
                    - '6': Amplitude PSD [NOT IMPLEMENTED YET]

                * Second digit: refer to the method for computing surrogates:

                    - '0': No surrogates
                    - '1': Swap trials phase/amplitude [#f2]_
                    - '2': Swap trials amplitude [#f4]_
                    - '3': Shuffle phase values
                    - '4': Time lag [#f1]_ [NOT IMPLEMENTED YET]
                    - '5': circular shifting [NOT IMPLEMENTED YET]

                * Third digit: refer to the normalization method for correction:

                    - '0': No normalization
                    - '1': Substract the mean of surrogates
                    - '2': Divide by the mean of surrogates
                    - '3': Substract then divide by the mean of surrogates
                    - '4': Z-score

            So, if Id='143', this mean that pac will be evaluate using the
            Modulation Index ('1'), then surrogates will be find by introducing a
            time lag ('4') and finally, the true pac value will be normalized by
            substracting then dividing by the mean of surrogates.

        pha_f: tuple/list, optional, [def: [2,4]]
                List containing the couple of frequency bands for the phase.
                Example: f=[ [2,4], [5,7], [60,250] ]

        pha_meth: string, optional, [def: 'hilbert']
            Method for the phase extraction.

        pha_cycle: integer, optional, [def: 3]
            Number of cycles for filtering the phase.

        amp_f: tuple/list, optional, [def: [60,200]]
                List containing the couple of frequency bands for the amplitude.
                Each couple can be either a list or a tuple.

        amp_meth: string, optional, [def: 'hilbert']
            Method for the amplitude extraction.

        amp_cycle: integer, optional, [def: 6]
            Number of cycles for filtering the amplitude.

        nbins: integer, optional, [def: 18]
            Some pac method (like Kullback-Leibler Distance or Heights Ratio) need
            a binarization of the phase. nbins control the number of bins.

    """
    __doc__ += windoc + docfilter + Footnotes

    def __init__(self, sf, npts, Id='113', pha_f=[2, 4], pha_meth='hilbert',
                 pha_cycle=3, amp_f=[60, 200], amp_meth='hilbert', amp_cycle=6,
                 nbins=18, window=None, width=None, step=None, time=None, **kwargs):
        # Check pha and amp methods:
        _checkref('pha_meth', pha_meth, ['hilbert', 'hilbert1', 'hilbert2'])
        _checkref('amp_meth', amp_meth, ['hilbert', 'hilbert1', 'hilbert2'])

        # Check the type of f:
        if (len(pha_f) == 4) and isinstance(pha_f[0], (int, float)):
            pha_f = binarize(
                pha_f[0], pha_f[1], pha_f[2], pha_f[3], kind='list')
        if (len(amp_f) == 4) and isinstance(amp_f[0], (int, float)):
            amp_f = binarize(
                amp_f[0], amp_f[1], amp_f[2], amp_f[3], kind='list')
        self.xvec = []

        # Initalize pac object :
        self.Id = Id
        me = Id[0]
        # Manage settings :
        #   1 - Choose if we extract phase or amplitude :
        #       - Methods using phase // amplitude :
        if me in ['1', '2', '3', '5', '6']:
            pha_kind, amp_kind = 'phase', 'amplitude'
        #       - Methods using phase // phase :
        elif me in ['4']:
            pha_kind, amp_kind = 'phase', 'phase'
        #   2 - Specific case of Ozkurt :
        if me == '5':
            Id = '500'
        # Initialize cfc :
        _coupling.__init__(self, pha_f, pha_kind, pha_meth, pha_cycle,
                           amp_f, amp_kind, amp_meth, amp_cycle,
                           sf, npts, window, width, step, time, **kwargs)
        # Get pac model :
        _, _, _, ModelStr, SurStr, NormStr = CfcSettings(Id, nbins)
        self.model = ['Method : '+ModelStr, 'Surrogates : '+SurStr,
                      'Normalization : '+NormStr]
        self._nbins = nbins

    def __str__(self):
        phafilt = 'Phase : '+str(self._pha)
        ampfilt = 'Amplitude : '+str(self._amp)
        met = self.model[0]+',\n'+self.model[1]+',\n'+self.model[2]+',\n'
        cfcStr = 'Crossfrequency Coupling(step='+str(self._step)+', width='+str(
            self._width)+', Id='+self.Id+', nbins='+str(self._nbins)+',\n'+met

        return cfcStr+phafilt+',\n'+ampfilt+')'

    def get(self, xpha, xamp, n_perm=200, p=0.05, matricial=True, n_jobs=-1):
        """Get the normalized cfc mesure between an xpha and xamp signals.

        Args:
            xpha: array
                Signal for phase. The shape of xpha should be :
                (n_electrodes x n_pts x n_trials)

            xamp: array
                Signal for amplitude. The shape of xamp should be :
                (n_electrodes x n_pts x n_trials)

        Kargs:
            n_perm: integer, optional, [def: 200]
                Number of permutations for normalizing the cfc mesure.

            p: float, optional, [def: 0.05]
                p-value for the statistical method of Ozkurt 2012.

            n_jobs: integer, optional, [def: -1]
                Control the number of jobs for parallel computing. Use 1, 2, ..
                depending of your number or cores. -1 for all the cores.

            If the same signal is used (example : xpha=x and xamp=x), this mean
            the program compute a local cfc.

        Returns:
            ncfc: array
                The unormalized cfc mesure of size :
                (n_amplitude x n_phase x n_electrodes x n_windows x n_trials)

            pvalue: array
                The associated p-values of size :
                (n_amplitude x n_phase x n_electrodes x n_windows)
        """
        # Check the inputs variables :
        xpha, xamp = _cfcCheck(xpha, xamp, self._npts)
        self.n_perm = n_perm
        self._matricial = matricial
        if n_perm != 0:
            self.p = 1/n_perm
        else:
            self.p = None
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
        uCfc = np.array(uCfc)

        # Permutations ans stat:
        if self.Id[0] is not '5':
            # Compute permutations :
            if self.n_perm is not 0:
                Suro, mSuro, stdSuro = np.array(
                    Suro), np.array(mSuro), np.array(stdSuro)

                # Normalize each cfc:
                _, _, Norm, _, _, _ = CfcSettings(self.Id)
                nCfc = Norm(uCfc, mSuro, stdSuro)

                # Confidence interval :
                pvalue = perm_2pvalue(uCfc.mean(2), np.rollaxis(Suro.mean(2), 4),
                                      self.n_perm, tail=1)

                return nCfc.transpose(3, 4, 0, 1, 2), pvalue.transpose(2, 3, 0, 1)
            else:
                return uCfc.transpose(3, 4, 0, 1, 2), None
        elif self.Id[0] is '5':
            # Ozkurt threshold :
            xlim = (erfinv(1-p)**2)
            # Set to zero non-significant values:
            idxUn = np.where(uCfc <= 2*xlim)
            uCfc[idxUn] = 0
            return uCfc.transpose(3, 4, 0, 1, 2), None


class PhaseLockedPower(object):

    """Extract phase-locked power and visualize shifted time-frequency map
    according to phase peak.

    Args:
        sf: int
            Sampling frequency

        npts: int
            Number of points of the time serie

    Kargs:
        f: tuple/list, optional, [def: (2, 200, 10, 5)]
            The frequency vector (fstart, fend, fwidth, fstep)

        pha: tuple/list, optional, [def: [8, 13]]
            Frequency for phase.

        time: array/list, optional, [def: None]
            The time vector to use

        baseline: tuple/list, optional, [def: None]
            Location of baseline (in sample)

        norm: integer, optional, [def: None]
            Normalize method
                - 0: No normalisation
                - 1: Substraction
                - 2: Division
                - 3: Substract then divide
                - 4: Z-score
        powArgs: any supplementar arguments are directly passed to the power
        function.
    """

    def __init__(self, sf, npts, f=(2, 200, 10, 5), pha=[8, 13], time=None,
                 baseline=None, norm=None, **powArgs):
        # Define objects:
        self._normBck = norm
        self._baseline = baseline
        self._powObj = power(
            sf, npts, f=f, baseline=baseline, norm=0, time=time, **powArgs)
        self._phaObj = phase(sf, npts, f=pha)
        self._sigObj = sigfilt(sf, npts, f=pha)

    def get(self, x, cue):
        """Get power phase locked

        Args:
            x: array
                Data of shape (npt, ntrials)

            cue: integer
                Cue to align time-frequency maps.

        Returns:
            xpow, xpha, xsig: repectively realigned power, phase and filtered
            signal
        """
        # Find cue according to define time vector
        self._cue = cue
        xvec = self._powObj.xvec
        cue = np.abs(np.array(xvec)-cue).argmin()
        self._cueIdx = cue
        # Extact power, phase and filtered signal:
        xpow = np.squeeze(self._powObj.get(x)[0])
        xpha = np.squeeze(self._phaObj.get(x)[0])
        xsig = np.squeeze(self._sigObj.get(x)[0])
        # Re-align:
        xpha_s, xpow_s, xsig_s = np.empty_like(
            xpha), np.empty_like(xpow), np.empty_like(xsig)
        nTrials = xsig.shape[1]
        for k in range(nTrials):
            # Get shifting:
            move = self._PeakDetection(xsig[:, k], cue)
            # Apply shifting:
            xpha_s[:, k] = self._ShiftSignal(np.matrix(xpha[:, k]), move)
            xsig_s[:, k] = self._ShiftSignal(np.matrix(xsig[:, k]), move)
            xpow_s[:, :, k] = self._ShiftSignal(xpow[:, :, k], move)
        xpow_s = np.mean(xpow_s, 2)
        # Normalize mean power:
        if self._normBck is not 0:
            bsl = self._baseline
            xFm = np.mean(xpow_s[:, bsl[0]:bsl[1]], 1)
            baseline = np.tile(xFm[:, np.newaxis], [1, xpow_s.shape[1]])
            xpow_s = normalize(xpow_s, baseline, norm=self._normBck)

        return xpow_s, xpha_s, xsig_s

    def tflockedplot(self, xpow, sig, cmap='viridis', vmin=None, vmax=None,
                     ylim=None, alpha=0.3, kind='std', vColor='r',
                     sigcolor='slateblue', fignum=0):
        """Plot realigned time-frequency maps.

        Args:
            xpow, sig: output of the get() method. sig can either be the phase
            or the filtered signal.

        Kargs:
            cmap: string, optional, [def: 'viridis']
                The colormap to use

            vmin, vmax: int/float, otpional, [def: None, None]
                Limits of the colorbar

            ylim: tuple/list, optional, [def: None]
                Limit for the plot of the signal

            alpha: float, optional, [def: 0.3]
                Transparency of deviation/sem

            kind: string, optional, [def: 'std']
                Choose between 'std' or 'sem' to either display standard
                deviation or standard error on the mean for the signal plot

            vColor: string, optional, [def: 'r']
                Color of the vertical line which materialized the choosen cue

            sigcolor: string, optional, [def: 'slateblue']
                Color of the signal

            fignum: integer, optional, [def: 0]
                Number of the figure

        Returns:
            figure, axes1 (TF plot), axes2 (signal plot), axes3 (colorbar)
        """
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        xvec = self._powObj.xvec
        yvec = self._powObj.yvec

        fig = plt.figure(fignum, figsize=(8, 9))
        gs = gridspec.GridSpec(11, 11)
        ax1 = plt.subplot(gs[0:-2, 0:-1])
        ax2 = plt.subplot(gs[-2::, 0:-1])
        ax3 = plt.subplot(gs[1:-3, -1])

        # TF:
        im = ax1.imshow(xpow, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[xvec[0], xvec[-1], yvec[-1], yvec[0]])
        addLines(ax1, vLines=[self._cue], vShape=['-'], vWidth=[3],
                 vColor=[vColor])
        ax1.set_xticks([])
        ax1.set_xticklabels('')
        ax1.set_ylabel('Frequency (hz)')
        ax1.invert_yaxis()
        ax1.tick_params(axis='both', which='both', top='off', right='off')
        ax1.axis('tight')

        cb = plt.colorbar(im, cax=ax3)
        cb.set_ticks(cb.get_clim())
        cb.set_label('Power modulations', labelpad=-10)

        # Signal:
        xm = sig.mean(1)
        if kind == 'std':
            x2add = sig.std(1)
        elif kind == 'sem':
            x2add = sig.std(1)/np.sqrt(len(xm)-1)
        xlow, xhigh = xm-x2add, xm+x2add
        ax = ax2.plot(xvec, xm, lw=2, color=sigcolor)
        ax2.fill_between(xvec, xlow, xhigh, alpha=alpha,
                         color=ax[0].get_color())
        ax2.set_yticks(ax2.get_ylim())
        ax2.tick_params(axis='both', which='both', top='off', right='off')
        ax2.set_xlabel('Time')
        if ylim is not None:
            ax2.set_ylim(ylim)
        else:
            ax2.axis('tight')
        addLines(ax2, vLines=[self._cue], vShape=['-'], vWidth=[3],
                 vColor=[vColor])

        return plt.gcf(), ax1, ax2, ax3

    @staticmethod
    def _PeakDetection(sig, cue):
        """Detect peaks in a signal and return the shifting length
        corresponding to the defined cue
        sig: vector
        cue: integer (in sample)
        """
        peaks = []
        for k in range(len(sig)-1):
            if (sig[k-1] < sig[k]) and (sig[k] > sig[k+1]):
                peaks.append(k)
        minPeak = peaks[np.abs(np.array(peaks)-cue).argmin()]
        return minPeak-cue

    @staticmethod
    def _ShiftSignal(sig, move):
        """
        """
        npts = sig.shape[1]
        sigShift = np.zeros(sig.shape)
        if move >= 0:
            sigShift[:, 0:npts-move] = sig[:, move::]
        elif move < 0:
            sigShift[:, np.abs(move)::] = sig[:, 0:npts-np.abs(move)]
        return sigShift
