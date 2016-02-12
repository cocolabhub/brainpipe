import numpy as n
from numpy.matlib import repmat
from math import pi
from scipy.signal import filtfilt
from scipy.signal import filtfilt, butter, bessel, hilbert, hilbert2, detrend

__all__ = [
    'fir_order',
    'fir_filt',
    'morlet',
]


def _apply_method(x, fMeth, dtrd, method, wltCorr, wltWidth):
    npts, ntrial = x.shape
    nFce = len(fMeth)
    xf = n.zeros((nFce, npts, ntrial))

    # Detrend the signal :
    if dtrd:
        x = detrend(x, axis=0)

    # Apply methods :
    for k in range(nFce):  # For each frequency in the tuple
        xf[k, ...] = fMeth[k](x)

    # Correction for the wavelet (due to the wavelet width):
    if (method == 'wavelet') and (wltCorr is not None):
        w = 3*wltWidth
        xf[:, 0:w, :] = xf[:, w+1:2*w+1, :]
        xf[:, npts-w:npts, :] = xf[:, npts-2*w-1:npts-w-1, :]

    return xf


def _get_method(sf, f, npts, filtname, cycle, order, axis, method, wltWidth,
                kind):
    """Get a list of functions of combinaitions: kind // transformation // design
    """
    # Get the kind (power, phase, signal, amplitude)
    fcnKind = _getKind(kind)
    fmeth = []
    for k in f:
        def fme(x, fce=k):
            return fcnKind(_getTransform(sf, fce, npts, method, wltWidth,
                                         filtname, cycle, order, axis)(x))
        fmeth.append(fme)
    return fmeth


def _getFiltDesign(sf, f, npts, filtname, cycle, order, axis):
    """Get the designed filter
    sf : sample frequency
    f : frequency vector/list [ex : f = [2,4]]
    npts : number of points
    - 'fir1'
    - 'butter'
    - 'bessel'
    """

    if type(f) != n.ndarray:
        f = n.array(f)

    # fir1 filter :
    if filtname == 'fir1':
        fOrder = fir_order(sf, npts, f[0], cycle=cycle)
        b, a = fir1(fOrder, f/(sf / 2))

    # butterworth filter :
    elif filtname == 'butter':
        b, a = butter(order, [(2*f[0])/sf, (2*f[1])/sf], btype='bandpass')
        fOrder = None

    # bessel filter :
    elif filtname == 'bessel':
        b, a = bessel(order, [(2*f[0])/sf, (2*f[1])/sf], btype='bandpass')
        fOrder = None

    def filtSignal(x):
        return filtfilt(b, a, x, padlen=fOrder, axis=axis)

    return filtSignal


def _getTransform(sf, f, npts, method, wltWidth, *arg):
    """Return a fuction which contain a transformation
    - 'hilbert'
    - 'hilbert1'
    - 'hilbert2'
    - 'wavelet'
    """
    # Get the design of the filter :
    fDesign = _getFiltDesign(sf, f, npts, *arg)

    # Hilbert method
    if method == 'hilbert':
        def hilb(x):
            xH = n.zeros(x.shape)*1j
            xF = fDesign(x)
            for k in range(x.shape[1]):
                xH[:, k] = hilbert(xF[:, k])
            return xH
        return hilb

    # Hilbert method 1
    elif method == 'hilbert1':
        def hilb1(x): return hilbert(fDesign(x))
        return hilb1

    # Hilbert method 2
    elif method == 'hilbert2':
        def hilb2(x): return hilbert2(fDesign(x))
        return hilb2

    # Wavelet method
    elif method == 'wavelet':
        def wav(x): return morlet(x, sf, (f[0]+f[1])/2, wavelet_width=wltWidth)
        return wav

    # Filter the signal
    elif method == 'filter':
        def fm(x): return fDesign(x)
        return fm


def _getKind(kind):
    """Return a function to modify or not, the original signal.
    The implemented functions are:
    - 'signal' : original signal
    - 'phase' : phase of the signal
    - 'amplitude' : amplitude of the signal
    - 'power' : power of the signal
    """
    # Unmodified signal
    if kind == 'signal':
        def sig_k(x): return x
        return sig_k

    # Phase of the signal
    elif kind == 'phase':
        def phase_k(x): return n.angle(x)
        return phase_k

    # Amplitude of the signal
    elif kind == 'amplitude':
        def amp_k(x): return abs(x)
        return amp_k

    # Power of the signal
    elif kind == 'power':
        def pow_k(x): return n.square(abs(x))
        return pow_k


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
        b = b + (m / (4 * pi * pi) * (
            n.cos(2 * pi * k * F[s + 1]) - n.cos(2 * pi * k * F[s])
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
        b = b + (m / (4 * pi * pi) * (n.cos(2 * pi * k * F[
            s + 1]) - n.cos(2 * pi * k * F[s])) / (
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
    aa = n.ravel(repmat(mags, 2, 1), order='F')

    # Get filter coefficients :
    h = firls(L - 1, ff, aa)

    # Apply a window to coefficients :
    Wind = n.hamming(L)
    b = n.matrix(h.T * Wind)
    c = n.matrix(n.exp(-1j * 2 * pi * (f0 / 2) * n.array(range(L))))
    b = b / abs(c * b.T)

    return n.ndarray.squeeze(n.array(b)), 1


####################################################################
# - Filt the signal :
####################################################################
def fir_filt(x, Fs, Fc, fOrder):
    (b, a) = fir1(fOrder, Fc / (Fs / 2))
    return filtfilt(b, a, x, padlen=fOrder)


####################################################################
# - Morlet :
####################################################################
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
