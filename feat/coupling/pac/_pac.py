import numpy as n
from itertools import product
from joblib import Parallel, delayed

from brainpipe._utils._system import (groupInList, list2index,
                                      adaptsize)
from brainpipe.cfc.methods import *

__all__ = [
            '_cfcCheck',
            '_cfcPvalue',
            '_cfcFiltSuro'
          ]


def _cfcFiltSuro(xPha, xAmp, surJob, self):
    """SUP: Get the cfc and surrogates

    The function return:
        - The unormalized cfc
        - All the surrogates (for pvalue)
        - The mean of surrogates (for normalization)
        - The deviation of surrogates (for normalization)
    """
    # Check input variables :
    npts, ntrial = xPha.shape
    W = self._window
    nwin = len(W)

    # Get the filter for phase/amplitude properties :
    phaMeth = self._pha.get(self._sf, self._pha.f, self._npts)
    ampMeth = self._amp.get(self._sf, self._amp.f, self._npts)

    # Filt the phase and amplitude :
    xPha = self._pha.apply(xPha, phaMeth)
    xAmp = self._amp.apply(xAmp, ampMeth)

    # 2D loop trick :
    claIdx, listWin, listTrial = list2index(nwin, ntrial)

    # Get the unormalized cfc :
    uCfc = [_cfcGet(n.squeeze(xPha[:, W[k[0]][0]:W[k[0]][1], k[1]]),
                    n.squeeze(xAmp[:, W[k[0]][0]:W[k[0]][1], k[1]]),
                    self.Id, self._nbins) for k in claIdx]
    uCfc = n.array(groupInList(uCfc, listWin))

    # Run surogates on each window :
    if self.n_perm != 0:
        Suro = Parallel(n_jobs=surJob)(delayed(_cfcGetSuro)(
            xPha[:, k[0]:k[1], :], xAmp[:, k[0]:k[1], :],
            self.Id, self.n_perm, self._nbins) for k in self._window)
        mSuro = [n.mean(k, 3) for k in Suro]
        stdSuro = [n.std(k, 3) for k in Suro]
    else:
        Suro, mSuro, stdSuro = None, None, None

    return uCfc, Suro, mSuro, stdSuro


def _cfcGet(pha, amp, Id, nbins):
    """Compute the basic cfc model
    """
    # Get the cfc model :
    Model, _, _, _, _, _ = CfcSettings(Id, nbins=nbins)

    return Model(n.matrix(pha), n.matrix(amp), nbins)


def _cfcGetSuro(pha, amp, Id, n_perm, nbins):
    """Compute the basic cfc model
    """
    # Get the cfc model :
    Model, Sur, _, _, _, _ = CfcSettings(Id, nbins=nbins)

    return Sur(pha, amp, Model, n_perm)


def _cfcCheck(xPha, xAmp, npts):
    """Manage xPha and xAmp size
    """
    if xPha.shape == xAmp.shape:
        if len(xPha.shape) == 2:
            xPha = xPha[n.newaxis, ...]
            xAmp = xAmp[n.newaxis, ...]
        if xPha.shape[1] != npts:
            raise ValueError('Second dimension must be '+str(npts))
    else:
        raise ValueError('xPha and xAmp must have the same size')

    return xPha, xAmp


def _cfcPvalue(nCfc, perm):
    """Get the pvalue of the cfc using permutations
    """
    nCfc, perm = n.mean(nCfc, 1), n.mean(perm, 1)
    nW, nA, nP = nCfc.shape
    nperm = perm.shape[3]

    pvalue = n.ones(nCfc.shape)
    for i, k, l in product(range(nW), range(nA), range(nP)):
        pv = (n.sum(perm[i, k, l, :] >= nCfc[i, k, l])) / nperm
        if pv == 0:
            pvalue[i, k, l] = 1/nperm
        else:
            pvalue[i, k, l] = pv
    return pvalue
