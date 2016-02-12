"""
Design a filter, filt a signal, extract the phase, amplitude or power
"""

import numpy as n
from brainpipe.xPOO._utils._filtering import (fir_order,
                                              fir1,
                                              morlet,
                                              _get_method,
                                              _apply_method)

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

    def get(self, sf, f, npts):
        """Get the methods
        sf : sample frequency
        f : frequency vector/list [ ex : f = [[2,4],[5,7]] ]
        npts : number of points
        -> Return a list of methods
        """
        if type(f[0]) == int:
            f = [f]
        fMeth = _get_method(sf, f, npts, self.filtname, self.cycle, self.order,
                            self.axis, self.method, self.wltWidth, self.kind)
        return fMeth

    def apply(self, x, fMeth):
        """Apply the methods
        x : array signal
        fMeth : list of methods
        -> 3D array of the transform signal
        """
        return _apply_method(x, fMeth, self.dtrd, self.method,
                             self.wltCorr, self.wltWidth)
