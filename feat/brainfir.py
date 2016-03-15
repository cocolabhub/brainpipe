import numpy as n
import numpy.matlib as m
import math
from scipy.signal import hilbert, hilbert2, filtfilt, butter, bessel, detrend

__all__ = [
    'fir_order', 'fir_filt', 'filtvec', 'filter_design'
]

# ####################################################################
# # - Filt the signal either like phase or amplitude :
# ####################################################################


def filtvec(x, sf, fce, kind, cycle=3, filtname='fir1', order=3, axis=0,
            fmeth='hilbert', dtrd=None, wavelet_width=7, wavelet_correction=3):

    timeL, nbTrials = x.shape
    nFce = len(fce)
    xf = n.zeros((nFce, timeL, nbTrials))

    # Detrend the signal :
    if dtrd is not None:
        x = detrend(x, axis=0)

    # Define a filtering method :
    def filtvec_filter(fmeth, x, sf, timeL, cycle=cycle, filtname=filtname,
                       order=order, axis=axis, wavelet_width=wavelet_width):
        if fmeth == 'hilbert':        # Hilbert method
            def fvecfilt(fce):
                xH, xF = n.zeros(x.shape)*1j, filter_design(sf, timeL,
                                                            n.array(fce),
                                                            filtname=filtname,
                                                            cycle=cycle,
                                                            order=order,
                                                            axis=axis)(x)
                for k in range(0, x.shape[1]):
                    xH[:, k] = hilbert(xF[:, k])
                return xH
        if fmeth == 'hilbert1':       # Hilbert method
            def fvecfilt(fce):
                return hilbert(filter_design(sf, timeL, n.array(fce),
                               filtname=filtname, cycle=cycle, order=order,
                               axis=axis)(x))
        if fmeth == 'hilbert2':       # Hilbert2 method
            def fvecfilt(fce):
                return hilbert2(filter_design(sf, timeL, n.array(fce),
                                filtname=filtname, cycle=cycle, order=order,
                                axis=axis)(x))
        elif fmeth == 'wavelet':      # Wavelet method
            def fvecfilt(fce):
                return morlet(x, sf, (fce[0]+fce[1])/2,
                              wavelet_width=wavelet_width)
        elif fmeth == 'filter':       # Filter the signal
            def fvecfilt(fce):
                return filter_design(sf, timeL, n.array(fce), filtname=filtname,
                                     cycle=cycle, order=order, axis=axis)(x)
        return fvecfilt

    fvecfilt = filtvec_filter(fmeth, x, sf, timeL, cycle=cycle,
                              filtname=filtname, order=order, axis=axis,
                              wavelet_width=wavelet_width)

    # Define the desired final object :
    def filtvec_meth(x):
        if kind == 'signal':        # Unmodified signal
            def fvecmeth(x): return x
        elif kind == 'phase':       # phase of the filtered signal
            def fvecmeth(x): return n.angle(x)
        elif kind == 'amplitude':   # amplitude of the filtered signal
            def fvecmeth(x): return abs(x)
        elif kind == 'power':       # power of the filtered signal
            def fvecmeth(x): return n.square(abs(x))
        return fvecmeth

    fvecmeth = filtvec_meth(x)

    # Apply the method to the object :
    for k in range(0, nFce):  # For each frequency in the tuple
        xf[k, ...] = fvecmeth(fvecfilt(fce[k]))

    # Correction for the wavelet (due to the wavelet width):
    if (fmeth == 'wavelet') and (wavelet_correction is not None):
        w = 3*wavelet_width
        xf[:, 0:w, :] = xf[:, w+1:2*w+1, :]
        xf[:, timeL-w:timeL, :] = xf[:, timeL-2*w-1:timeL-w-1, :]

    return xf


####################################################################
# - Get the filter order :
####################################################################
def fir_order(Fs, sizevec, flow, cycle=3):
    filtorder = cycle * (Fs // flow)

    if (sizevec < 3 * filtorder):
        filtorder = (sizevec - 1) // 3

    return int(filtorder)


####################################################################
# - Separe for odd/even case :
####################################################################
# Odd case
def NoddFcn(F, M, W, L):  # N is odd
    # Variables :
    b0 = 0
    m = n.array(range(int(L + 1)))
    k = m[1:len(m)]
    b = n.zeros(k.shape)

    # Run Loop :
    for s in range(0, len(F), 2):
        m = (M[s + 1] - M[s]) / (F[s + 1] - F[s])
        b1 = M[s] - m * F[s]
        b0 = b0 + (b1 * (F[s + 1] - F[s]) + m / 2 * (
            F[s + 1] * F[s + 1] - F[s] * F[s])) * abs(
            n.square(W[round((s + 1) / 2)]))
        b = b + (m / (4 * math.pi * math.pi) * (
            n.cos(2 * math.pi * k * F[s + 1]) - n.cos(2 * math.pi * k * F[s])
            ) / (k * k)) * abs(n.square(W[round((s + 1) / 2)]))
        b = b + (F[s + 1] * (m * F[s + 1] + b1) * n.sinc(2 * k * F[s + 1]) - F[
            s] * (m * F[s] + b1) * n.sinc(2 * k * F[s])) * abs(n.square(
                W[round((s + 1) / 2)]))

    b = n.insert(b, 0, b0)
    a = (n.square(W[0])) * 4 * b
    a[0] = a[0] / 2
    aud = n.flipud(a[1:len(a)]) / 2
    a2 = n.insert(aud, len(aud), a[0])
    h = n.concatenate((a2, a[1:] / 2))

    return h


# Even case
def NevenFcn(F, M, W, L):  # N is even
    # Variables :
    k = n.array(range(0, int(L) + 1, 1)) + 0.5
    b = n.zeros(k.shape)

    # # Run Loop :
    for s in range(0, len(F), 2):
        m = (M[s + 1] - M[s]) / (F[s + 1] - F[s])
        b1 = M[s] - m * F[s]
        b = b + (m / (4 * math.pi * math.pi) * (n.cos(2 * math.pi * k * F[
            s + 1]) - n.cos(2 * math.pi * k * F[s])) / (
            k * k)) * abs(n.square(W[round((s + 1) / 2)]))
        b = b + (F[s + 1] * (m * F[s + 1] + b1) * n.sinc(2 * k * F[s + 1]) - F[
            s] * (m * F[s] + b1) * n.sinc(
            2 * k * F[s])) * abs(n.square(W[round((s + 1) / 2)]))

    a = (n.square(W[0])) * 4 * b
    h = 0.5 * n.concatenate((n.flipud(a), a))

    return h


####################################################################
# - Filt the signal :
####################################################################

def firls(N, F, M):
    # Variables definition :
    W = n.ones(round(len(F) / 2))
    N += 1
    F /= 2
    L = (N - 1) / 2

    Nodd = bool(N % 2)

    if Nodd:  # Odd case
        h = NoddFcn(F, M, W, L)
    else:  # Even case
        h = NevenFcn(F, M, W, L)

    return h


####################################################################
# - Compute the window :
####################################################################

def fir1(N, Wn):
    # Variables definition :
    nbands = len(Wn) + 1
    ff = n.array((0, Wn[0], Wn[0], Wn[1], Wn[1], 1))

    f0 = n.mean(ff[2:4])
    L = N + 1

    mags = n.array(range(nbands)) % 2
    aa = n.ravel(m.repmat(mags, 2, 1), order='F')

    # Get filter coefficients :
    h = firls(L - 1, ff, aa)

    # Apply a window to coefficients :
    Wind = n.hamming(L)
    b = n.matrix(h.T * Wind)
    c = n.matrix(n.exp(-1j * 2 * math.pi * (f0 / 2) * n.array(range(L))))
    b = b / abs(c * b.T)

    return n.ndarray.squeeze(n.array(b)), 1


# ####################################################################
# # - Filt the signal :
# ####################################################################
def fir_filt(x, Fs, Fc, fOrder):
    (b, a) = fir1(fOrder, Fc / (Fs / 2))
    return filtfilt(b, a, x, padlen=fOrder)


# ####################################################################
# # - Design a filter :
# ####################################################################
def filter_design(sf, N, f, filtname='fir1', cycle=3, order=3, axis=0):
    if filtname == 'fir1':
        fOrder = fir_order(sf, N, f[0], cycle=cycle)
        b, a = fir1(fOrder, f/(sf / 2))
    elif filtname == 'butter':
        b, a = butter(order, [(2*f[0])/sf, (2*f[1])/sf], btype='bandpass')
        fOrder = None
    elif filtname == 'bessel':
        b, a = bessel(order, [(2*f[0])/sf, (2*f[1])/sf], btype='bandpass')
        fOrder = None

    def FiltDesign(x): return filtfilt(b, a, x, padlen=fOrder, axis=axis)

    return FiltDesign

# ####################################################################
# # - Morlet :
# ####################################################################


def morlet(x, Fs, f, wavelet_width=7):
    dt = 1/Fs
    sf = f/wavelet_width
    st = 1/(2*n.pi*sf)
    N, nepoch = x.shape

    t = n.arange(-3.5*st, 3.5*st, dt)

    A = 1/(st*n.sqrt(n.pi))**(1/2)
    m = A*n.exp(-n.square(t)/(2*st**2))*n.exp(1j*2*n.pi*f*t)

    xMorlet = n.zeros((N, nepoch))
    for k in range(0, nepoch):
        y = 2*n.abs(n.convolve(x[:, k], m))/Fs
        xMorlet[:, k] = y[
            int(n.ceil(len(m)/2))-1:int(len(y)-n.floor(len(m)/2))]

    return xMorlet
