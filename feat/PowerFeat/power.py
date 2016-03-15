import numpy as n
from .. import brainfir
from brainpipe.feature.normalization import normalize
from brainpipe.system.tools import binarize

__all__ = [
    'power', 'tfmap', 'freqvec'
]

####################################################################
# - Compute power :
####################################################################


def power(x, fs, fc, baseline=(1, 1), norm=0, window=None, edge=None,
          width=None, step=None, split=None, **kwargs):

    # -----------------------------------------------------------------------
    # Check input arguments :
    # -----------------------------------------------------------------------
    timeL, trials = x.shape
    # Number of frequencies :
    if type(fc) == tuple:
        fc = [fc]
    nfc = len(fc)
    # Split the power :
    if (split is None) or (type(split) == int):
        split = [split]
    if len(split) != nfc:
        split = split*nfc
    fcSplit, fcSplitIndex = [], []
    for k in range(0, nfc):
        if split[k] == None:
            fcSplit.append(fc[k]), fcSplitIndex.append(k)
        elif type(split[k]) == int:
            f = n.arange(fc[k][0], fc[k][1]+split[k], split[k])
            splitList = [(f[i], f[i+1]) for i in range(0, len(f)-1)]
            fcSplit = fcSplit + splitList
            fcSplitIndex = fcSplitIndex + [k]*len(splitList)
    # Binarize the power :
    if type(window) == tuple:
        window = [window]
    else:
        if width is not None:
            if step is None:
                step = round(width/2)
            window = binarize(0, timeL, width, step, kind='tuple')
    if (window is not None):
        nWin = len(window)

    # -----------------------------------------------------------------------
    # Extract power :
    # -----------------------------------------------------------------------
    xF = n.square(brainfir.filtvec(x, fs, fcSplit, 'amplitude', **kwargs))

    # -----------------------------------------------------------------------
    # Remove the edge effect :
    # -----------------------------------------------------------------------
    if edge == 'on':
        xF[:, 0, :], xF[:, -1, :] = xF[:, 1, :], xF[:, -2, :]

    # -----------------------------------------------------------------------
    # Normalize the power :
    # -----------------------------------------------------------------------
    if (norm != 0) or (baseline != (1, 1)):
        xFn = normalize(xF, n.tile(n.mean(xF[:, baseline[0]:baseline[1], :], 1)
                        [:, n.newaxis, :], [1, xF.shape[1], 1]), norm=norm)
    else:
        xFn = xF

    # -----------------------------------------------------------------------
    # Mean the splitted matrix :
    # -----------------------------------------------------------------------
    xFsplit = n.zeros((nfc, x.shape[0], x.shape[1]))
    for k in range(0, nfc):
        if len(n.where(n.array(fcSplitIndex) == k)[0]) != 1:
            xFsplit[k, :, :] = n.mean(
                xFn[n.where(n.array(fcSplitIndex) == k), :, :], 1)
        else:
            xFsplit[k, :, :] = xFn[n.where(n.array(fcSplitIndex) == k), :, :]

    # -----------------------------------------------------------------------
    # Binarize the power :
    # -----------------------------------------------------------------------
    if window is not None:
        xFsplit = n.swapaxes(xFsplit, 1, 2)
        xFsplitWin = n.zeros((nfc, trials, nWin))
        for k in range(0, nWin):
            xFsplitWin[:, :, k] = n.mean(
                xFsplit[:, :, window[k][0]:window[k][1]], 2)
        xFsplit = n.swapaxes(xFsplitWin, 1, 2)

    return n.squeeze(xFsplit)


####################################################################
# - Define a frequency vector (for TF analysis) :
####################################################################
def freqvec(fstart, fend, fwidth, fstep):
    ConcatFreq = [list(n.arange(fstart, fend-fwidth+fstep, fstep)),
                  list(n.arange(fstart+fwidth, fend+fstep, fstep))]
    return [(ConcatFreq[0][k], ConcatFreq[1][k]) for k in
            range(0, len(ConcatFreq[0]))]


####################################################################
# - Time Frequency Analysis :
####################################################################
def tfmap(x, fs, fvec=(5, 250, 10, 5), fc=None, norm=0, baseline=(1, 1),
          **kwargs):
    if fc is None:
        fc = freqvec(fvec[0], fvec[1], fvec[2], fvec[3])

    # Compute the tf:
    tf = power(x, fs, fc, norm=0, baseline=(1, 1), **kwargs)

    # Return the normalized mean TF:
    if (norm != 0) or (baseline != (1, 1)):
        X = n.mean(tf, 2)
        Y = n.matlib.repmat(
            n.mean(X[:, baseline[0]:baseline[1]], 1), X.shape[1], 1).T
        tfn = normalize(X, Y, norm=norm)
    else:
        tfn = n.mean(tf, 2)

    return tf, tfn
