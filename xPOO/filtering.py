"""
Design a filter, filt a signal, extract the phase, amplitude or power
"""

import numpy as n
from scipy.signal import filtfilt, butter, bessel, hilbert, hilbert2, detrend
from brainpipe.xPOO._utils._filtering import fir_order, fir1, morlet

__all__ = [
    'fdesign',
    'fextract'
]

__author__ = 'Etienne Combrisson'


class fdesign(object):

    """Design a filter

    Parameters
    ----------
    filtname : string, optional [def : 'fir1']
        Name of the filter. Possible values are:
            - 'fir1' : Window-based FIR filter design
            - 'butter' : butterworth filter
            - 'bessel' : bessel filter

    cycle : int, optional [def : 3]
        Number of cycle to use for the filter. This parameter
        is only avaible for the 'fir1' method

    order : int, optional [def : 3]
        Order of the 'butter' or 'bessel' filter

    axis : int, optional [def : 0]
        Filter accross the dimension 'axis'
    """

    def __init__(self, filtname='fir1', cycle=3, order=3, axis=0):
        if filtname not in ['fir1', 'butter', 'bessel', 'wavelet']:
            raise ValueError('No "filtname" called "'+str(filtname)+'"'
                             ' is defined. Choose between "fir1", "butter", '
                             '"bessel"')
        self.filtname = filtname
        self.cycle = cycle
        self.order = order
        self.axis = axis

    def __str__(self):
        if self.filtname == 'fir1':
            filtStr = 'Filter(name='+self.filtname+', cycle='+str(
                self.cycle)+', axis='+str(self.axis)+')'
        else:
            filtStr = 'Filter(name='+self.filtname+', order='+str(
                self.order)+', axis='+str(self.axis)+')'
        return filtStr

    def _getFiltDesign(self, sf, f, npts):
        """Get the designed filter
        sf : sample frequency
        f : frequency vector/list [ex : f = [2,4]]
        npts : number of points
        """
        if type(f) != n.ndarray:
            f = n.array(f)
        if self.filtname == 'fir1':
            fOrder = fir_order(sf, npts, f[0], cycle=self.cycle)
            b, a = fir1(fOrder, f/(sf / 2))
        elif self.filtname == 'butter':
            b, a = butter(self.order, [(2*f[0])/sf,
                                       (2*f[1])/sf], btype='bandpass')
            fOrder = None
        elif self.filtname == 'bessel':
            b, a = bessel(self.order, [(2*f[0])/sf,
                                       (2*f[1])/sf], btype='bandpass')
            fOrder = None

        def filSignal(x): return filtfilt(b, a, x, padlen=fOrder,
                                          axis=self.axis)
        return filSignal


class fextract(fdesign):

    """Extract informations from a signal

    Parameters
    ----------
    method : string
        Method to transform the signal. Possible values are:
            - 'hilbert' : apply a hilbert transform to each column
            - 'hilbert1' : hilbert transform to a whole matrix
            - 'hilbert2' : 2D hilbert transform
            - 'wavelet' : wavelet transform
            - 'filter' : filtered signal

    kind : string
        Type of information to extract to the transformed signal.
        Possible values are:
            - 'signal' : return the transform signal
            - 'phase' : phase of the the transform signal
            - 'amplitude' : amplitude of the transform signal
            - 'power' : power of the transform signal

    filtname : string, optional [def : 'fir1']
        Name of the filter. Possible values are:
            - 'fir1' : Window-based FIR filter design
            - 'butter' : butterworth filter
            - 'bessel' : bessel filter

    cycle : int, optional [def : 3]
        Number of cycle to use for the filter. This parameter
        is only avaible for the 'fir1' method

    order : int, optional [def : 3]
        Order of the 'butter' or 'bessel' filter

    axis : int, optional [def : 0]
        Filter accross the dimension 'axis'

    dtrd : bool, optional [def : Flase]
        Detrend the signal

    wltWidth : int, optional [def : 7]
        Width of the wavelet

    wltCorr : int, optional [def : 3]
        Correction of the edgde effect of the wavelet

    Method
    ----------
    getMeth : get the list of methods
        sf : sample frequency
        f : frequency vector/list [ex : f = [ [2,4], [5,7], [8,13] ]]
        npts : number of points
    -> Return a list of methods. The length of the list depend on the
    length of the frequency list "f".

    applyMeth : apply the list of methods
        x : array signal, [x] = npts x ntrials
        fMeth : list of methods
    -> Return a 3D array nFrequency x npts x ntrials

    """

    def __init__(self, method, kind, filtname='fir1', cycle=3, order=3,
                 axis=0, dtrd=False, wltWidth=7, wltCorr=3):
        if method not in ['hilbert', 'hilbert1', 'hilbert2', 'wavelet',
                          'filter']:
            raise ValueError('No "method" called "'+str(method)+'" is defined.'
                             ' Choose between "hilbert", "hilbert1", '
                             '"hilbert2", "wavelet", "filter"')
        if kind not in ['signal', 'phase', 'amplitude', 'power']:
            raise ValueError('No "kind" called "'+str(self.kind)+'"'
                             ' is defined. Choose between "signal", "phase", '
                             '"amplitude", "power"')
        self.method = method
        self.kind = kind
        self.wltWidth = wltWidth
        self.wltCorr = wltCorr
        self.dtrd = dtrd
        self.filtname = filtname
        self.cycle = cycle
        self.order = order
        self.axis = axis
        super().__init__(filtname=filtname, cycle=cycle, order=order,
                         axis=axis)

    def __str__(self):
        filtStr = super().__str__()
        if self.method == 'wavelet':
            supStr = ', wavelet(width='+str(
                self.wltWidth)+', correction='+str(self.wltCorr)+')'
        else:
            supStr = ''

        return 'Extract(kind='+self.kind+', method='+self.method+', detrend='+str(
            self.dtrd)+supStr+',\n'+filtStr+')'

    def _getTransform(self, sf, f, npts):

        fDesign = self._getFiltDesign(sf, f, npts)

        if self.method == 'hilbert':      # Hilbert method
            def hilb(x):
                xH = n.zeros(x.shape)*1j
                xF = fDesign(x)
                for k in range(0, x.shape[1]):
                    xH[:, k] = hilbert(xF[:, k])
                return xH
            return hilb
        elif self.method == 'hilbert1':   # Hilbert method 1
            def hilb1(x): return hilbert(fDesign(x))
            return hilb1
        elif self.method == 'hilbert2':   # Hilbert method 2
            def hilb2(x): return hilbert2(fDesign(x))
            return hilb2
        elif self.method == 'wavelet':    # Wavelet method
            def wav(x): return morlet(x, sf, (f[0]+f[1])/2,
                                      wavelet_width=self.wltWidth)
            return wav
        elif self.method == 'filter':     # Filter the signal
            def fm(x): return fDesign(x)
            return fm

    def _getKind(self):

        if self.kind == 'signal':        # Unmodified signal
            def sig_k(x): return x
            return sig_k
        elif self.kind == 'phase':       # phase of the filtered signal
            def phase_k(x): return n.angle(x)
            return phase_k
        elif self.kind == 'amplitude':   # amplitude of the filtered signal
            def amp_k(x): return abs(x)
            return amp_k
        elif self.kind == 'power':       # power of the filtered signal
            def pow_k(x): return n.square(abs(x))
            return pow_k

    def getMeth(self, sf, f, npts):
        """Get the methods
        sf : sample frequency
        f : frequency vector/list [ ex : f = [[2,4],[5,7]] ]
        npts : number of points
        -> Return a list of methods
        """
        if type(f[0]) == int:
            f = [f]
        xKind = self._getKind()
        fmeth = []
        for k in f:
            def fme(x, fce=k): return xKind(self._getTransform(
                sf, fce, npts)(x))
            fmeth.append(fme)
        return fmeth

    def applyMeth(self, x, fMeth):
        """Apply the methods
        x : array signal
        fMeth : list of methods
        -> 3D array of the transform signal
        """
        npts, ntrial = x.shape
        nFce = len(fMeth)
        xf = n.zeros((nFce, npts, ntrial))

        # Detrend the signal :
        if self.dtrd:
            x = detrend(x, axis=0)

        # Apply methods :
        for k in range(0, nFce):  # For each frequency in the tuple
            xf[k, ...] = fMeth[k](x)

        # Correction for the wavelet (due to the wavelet width):
        if (self.method == 'wavelet') and (self.wltCorr is not None):
            w = 3*self.wltWidth
            xf[:, 0:w, :] = xf[:, w+1:2*w+1, :]
            xf[:, npts-w:npts, :] = xf[:, npts-2*w-1:npts-w-1, :]

        return xf
