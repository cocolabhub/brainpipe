import numpy as n
from brainpipe.feature.brainfir import fir_filt, fir_order, filtvec
from scipy.signal import hilbert

__all__ = [
    'phase'
]

####################################################################
# - Get the phase either for an array or a matrix :
####################################################################
# def phase(x, N, fs, fc, winCenter=None, winLength=0, cycle=3):

#     # If no center frequency vec is specified, then it will return all the timepoints :
#     if winCenter is None : winCenter = n.arange(0, N, 1)
#     else : winCenter = n.array(winCenter).astype(int)

#     # Get size elements :
#     x = n.matrix(x)
#     fc = n.array(fc)
#     ndim = len(x.shape)

#     # Check size :
#     if ndim == 1:
#         npts, ncol = len(x), 1
#     elif ndim == 2:
#         rdim = n.arange(0,len(x.shape),1)[n.array(x.shape) == N]
#         if len(rdim) != 0 : rdim = rdim[0]
#         else: raise ValueError("None of x dimendion is "+str(N)+" length. [x] = "+str(x.shape))
#         npts, ncol = x.shape[rdim], x.shape[1-rdim]
#     if x.shape[0] != npts: x = x.T

#     # Get the filter order :
#     fOrder = fir_order(fs, npts, fc[0], cycle = cycle)

#     # Compute the phase for each colums :
#     xF = n.zeros((npts,ncol))
#     for k in range(0,ncol):
#         xF[:,k] = n.angle(hilbert(fir_filt(n.array(x[:,k]).T, fs, fc, fOrder)))

#     # Define the window vector :
#     winVec = n.vstack((winCenter-winLength/2,winCenter+winLength/2)).astype(int)
#     nbWin = winVec.shape[1]

#     # Bin the phase :
#     if winLength == 0:
#         xShape = xF[list(winCenter),:]
#     elif winLength != 0:
#         xShape = n.zeros((nbWin,ncol))
#         for k in range(0,nbWin):
#             print(winVec[0,k],winVec[1,k])
#             xShape[k,:] = n.mean(xF[ winVec[0,k]:winVec[1,k], : ],0)

#     return xShape

def phase(x, fs, fc, window=None, winCenter=None, winLength=0, **kwargs):
    # -----------------------------------------------------------------------
    # Check input arguments :
    # -----------------------------------------------------------------------
    timeL, trials = x.shape
    # Number of frequencies :
    if type(fc)==tuple:fc=[fc]
    nfc = len(fc)
    # Phase bining (or not :D)
    if (winCenter is not None) or (winLength!=0): 
        window = [(k-winLength/2,k+winLength/2) for k in winCenter]
    if window is not None: 
        window = [tuple(n.array((k[0],k[1])).astype(int)) for k in window]

    # -----------------------------------------------------------------------
    # Extract phase :
    # -----------------------------------------------------------------------
    xF = filtvec(x, fs, fc, 'phase', **kwargs)

    # -----------------------------------------------------------------------
    # Bin the phase :
    # -----------------------------------------------------------------------
    if (window is None) : xShape = xF
    else:
        nbWin = len(window)
        xShape = n.zeros((nfc, nbWin, trials))
        for k in range(0,nbWin):
            xShape[:,k,:] = n.mean(xF[ :, window[k][0]:window[k][1], : ],1)

    return xShape