import numpy as n
from itertools import product
from joblib import Parallel, delayed

from brainpipe.tools import binarize

__all__ = ['_pfdph']


def _pfdph(xPha, xAmp, n_jobs, self):
    """
    """
    # Check input variables :
    npts, nTrial = xPha.shape
    n_perm = self.n_perm

    # Get the filter for phase/amplitude properties :
    phaMeth = self.pha.get(self._sf, self.pha.f, npts)
    ampMeth = self.amp.get(self._sf, self.amp.f, npts)

    # Filt the phase and amplitude :
    xPha = self.pha.apply(xPha, phaMeth)+n.pi
    xAmp = self.amp.apply(xAmp, ampMeth)

    # Get the binarized amplitude :
    binamp = _subPfdph(xPha, xAmp, self._vecbin, self.window)

    # Compute surrogates :
    if n_perm != 0:
        # Generate n_perm phase shuffle :
        perm = [n.random.permutation(npts) for k in range(n_perm)]

        # Get the binarized amplitude corresponding to permutations :
        suro = Parallel(n_jobs=n_jobs)(delayed(_subPfdph)(
            xPha, xAmp[:, k, :], self._vecbin, self.window) for k in perm)

        # Get maximum amplitude :
        suro = n.mean(n.array(suro).max(axis=5), 4)
        abin = n.mean(binamp.max(axis=4), 3)

        # Get pvalue :
        pvalue = _pvalue(abin, suro)
    else:
        pvalue = None

    return binamp, pvalue


def _subPfdph(xPha, xAmp, vecbin, window):
    """
    """
    # Get elements size :
    nPha, nAmp, nWin = xPha.shape[0], xAmp.shape[0], len(window)
    nBin, nTrial = len(vecbin), xPha.shape[2]

    # Run the loop on nPha, nWin and nTrial :
    loo = product(range(nPha), range(nWin), range(nTrial))
    binamp = n.zeros((nPha, nAmp, nWin, nTrial, nBin))
    for nph, nw, nt in loo:
        # Get the window :
        win1, win2 = window[nw][0], window[nw][1]
        pha = n.ravel(n.squeeze(xPha[nph, win1:win2, nt]))
        amp = xAmp[:, win1:win2, nt]

        # Binarize amplitude relatively to the phase :
        for k, i in enumerate(vecbin):
            # Find where phase take vecbin values :
            pL = n.where((pha >= i[0]) & (pha < i[1]))[0]

            # Binarize amplitude :
            binamp[nph, :, nw, nt, k] = amp[:, pL].sum(axis=1)

    return binamp


def _pvalue(abin, suro):
    """
    """
    # Get element size :
    nPerm, nPha, nAmp, nWin = suro.shape

    # Loop :
    loo = product(range(nPha), range(nAmp), range(nWin))
    pvalue = n.zeros((nPha, nAmp, nWin))
    for nph, nam, nw in loo:
        pv = (n.sum(suro[:, nph, nam, nw] >= abin[nph, nam, nw])) / nPerm
        if pv == 0:
            pvalue[nph, nam, nw] = 1/nPerm
        else:
            pvalue[nph, nam, nw] = pv

    return pvalue
