"""
Design a filter, filt a signal, extract the phase, amplitude or power
"""

import numpy as np
from psutil import cpu_count

from .utils import _filt

__all__ = [
    'filter',
    'fdesign'
]


class filter(object):

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
        self._filtname = filtname
        self._cycle = cycle
        self._order = order
        self._axis = axis

    def __str__(self):
        if self._filtname == 'fir1':
            filtStr = 'Filter(name='+self._filtname+', cycle='+str(
                self._cycle)+', axis='+str(self._axis)+')'
        else:
            filtStr = 'Filter(name='+self._filtname+', order='+str(
                self._order)+', axis='+str(self._axis)+')'
        return filtStr


class fdesign(filter):

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

    dtrd : bool, optional [def : False]
        Detrend the signal

    wltWidth : int, optional [def : 7]
        Width of the wavelet

    wltCorr : int, optional [def : 3]
        Correction of the edgde effect of the wavelet

    Method
    ----------
    get : get the list of methods
        sf : sample frequency
        f : frequency vector/list [ex : f = [ [2,4], [5,7], [8,13] ]]
        npts : number of points
    -> Return a list of methods. The length of the list depend on the
    length of the frequency list "f".

    apply : apply the list of methods
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
            raise ValueError('No "kind" called "'+str(kind)+'"'
                             ' is defined. Choose between "signal", "phase", '
                             '"amplitude", "power"')
        self._method = method
        self._kind = kind
        self._wltWidth = wltWidth
        self._wltCorr = wltCorr
        self._dtrd = dtrd
        super().__init__(filtname=filtname, cycle=cycle, order=order,
                         axis=axis)

    def __str__(self):
        filtStr = super().__str__()
        if self._method == 'wavelet':
            supStr = ', wavelet(width='+str(
                self._wltWidth)+', correction='+str(self._wltCorr)+')'
        else:
            supStr = ''

        return 'Extract(kind='+self._kind+', method='+self._method+', detrend='+str(
            self._dtrd)+supStr+',\n'+filtStr+')'

    def filt(self, x, sf, f, n_jobs=-1):
        """"""
        _bksz = x.shape
        # Adapt frequency vector :
        if type(f[0]) == int:
            f = [f]
        # Adapt shape of x to 2d array:
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        elif len(x.shape) == 3:
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
        # Get the filter :
        xf = _filt(x, sf, f, x.shape[0], self._filtname, self._cycle,
                   self._order, self._axis, self._method, self._dtrd,
                   self._kind, self._wltCorr, self._wltWidth, n_jobs=n_jobs)
        # Reshape xf :
        if len(_bksz) == 3:
            xf = xf.reshape(xf.shape[0], _bksz[0], _bksz[1], _bksz[2])
        if len(xf.shape) == 2:
            xf = xf[np.newaxis, ...]

        return xf

    def _splitArray(self, x, sf, f, n_jobs=-1):
        """"""
        N = x.shape[0]

        if (n_jobs == -1) or (n_jobs == 0) or (n_jobs > cpu_count()):
            njob = cpu_count()
        else:
            njob = n_jobs

        try:
            step = int((N-N % njob)/njob)
            if (step == 0) or (step == 1):
                idx = [list(range(N))]
            else:
                idx = [[k+step*q for k in range(step)] for q in range(
                                                                  int(N/step))]
                sup = list(range(idx[-1][-1]+1, N))
                if sup:
                    idx[-1].extend(sup)
        except:
            idx = [list(range(N))]

        xjobs = [x[k, ...] for k in idx]
        self._split = {'split'+str(k): [str(i.shape)] for k, i in enumerate(xjobs)}
        return [k.reshape(k.shape[0]*k.shape[1], k.shape[2]) for k in xjobs]
