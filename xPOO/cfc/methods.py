import numpy as n
from itertools import product
from brainpipe.xPOO.tools import binarize

__all__ = [
    'CfcSettings',
]


# ----------------------------------------------------------------------------
#                            ID to CFC MODEL
# ----------------------------------------------------------------------------
def CfcSettings(Id, nbins=18, n_perm=200, tlag=[0, 0]):
    """From an Id, get the model of cfc composed with:
    - Method : how to compute cfc
    - Surrogates : method for computing surrogates
    - Normalization : how to normalize the cfc with surrogates

    For each of the three components, this function return a string and a
    function to apply.
    """
    # Define the method of PAC :
    [CfcModel, CfcModelStr] = CfcMethodList(int(Id[0]), nbins=nbins)

    # Define the way to compute surrogates :
    [CfcSur, CfcSurStr] = CfcSurrogatesList(int(Id[1]), CfcModel,
                                            n_perm=n_perm, tlag=tlag)

    # Define the way to normalize the Cfc with surrogates :
    [CfcNorm, CfcNormStr] = CfcNormalizationList(int(Id[2]))

    return CfcModel, CfcSur, CfcNorm, CfcModelStr, CfcSurStr, CfcNormStr


# ----------------------------------------------------------------------------
#                                 METHODS
# ----------------------------------------------------------------------------

def CfcMethodList(Id, nbins=18):
    """List of methods to compute the cfc. This list include methods for
    Phase-Amplitude, phase-phase or amplitude-amplitude coupling. Here's the
    list of the implemented methods :
    - Modulation Index
    - Kullback-Leibler Distance
    - Phase synchrony
    - Amplitude PSD
    - Heights Ratio
    - ndPAC

    Each method take at least a pha and amp array with the respective
    dimensions:
    pha.shape = (Nb phase     x Time points)
    amp.shape = (Nb amplitude x Time points)
    And each method should return a (Nb amplitude x Nb phase)
    """
    # Modulation Index (Canolty, 2006)
    if Id == 1:
        def CfcModel(pha, amp, *arg):
            return ModulationIndex(pha, amp)
        CfcModelStr = 'Modulation Index (Canolty, 2006)'

    # Kullback-Leiber divergence (Tort, 2010)
    elif Id == 2:
        def CfcModel(pha, amp, nbins=nbins):
            return KullbackLeiblerDistance(pha, amp, nbins)
        CfcModelStr = 'Kullback-Leibler Distance ['+str(
            nbins)+' bins] (Tort, 2010)'

    # Heights ratio
    elif Id == 3:
        def CfcModel(pha, amp, nbins=nbins):
            return HeightsRatio(pha, amp, nbins)
        CfcModelStr = 'Heights ratio ['+str(nbins)+' bins]'

    # Phase synchrony
    elif Id == 4:
        def CfcModel(pha, amp, *arg):
            return PhaseSynchrony(pha, amp)
        CfcModelStr = 'Phase synchrony'

    # ndPac (Ozkurt, 2012)
    elif Id == 5:
        def CfcModel(pha, amp, *arg):
            return ndCfc(pha, amp)
        CfcModelStr = 'Normalized direct Pac (Ozkurt, 2012)'
#         Cfcsup.CfcstatMeth = 1

    return CfcModel, CfcModelStr


def ModulationIndex(pha, amp):
    """Modulation index (Canolty, 2006)

    Method :
    abs(amplitude x exp(phase)) <-> sum modulations of the
    complex radius accross time. MI = resultant radius
    """
    return n.array(abs(amp*n.exp(1j*pha).T)/pha.shape[1])


def KullbackLeiblerDistance(pha, amp, nbins):
    """Kullback Leibler Distance (Tort, 2010)
    """
    # Get the phase locked binarized amplitude :
    abin, abinsum = _kl_hr(pha, amp, nbins)
    abin = n.divide(abin, n.rollaxis(abinsum, 0, start=3))
    abin[abin == 0] = 1
    abin = abin * n.log2(abin)

    return (1 + abin.sum(axis=2)/n.log2(nbins))


def HeightsRatio(pha, amp, nbins):
    """Heights Ratio
    """
    # Get the phase locked binarized amplitude :
    abin, abinsum = _kl_hr(pha, amp, nbins)
    M, m = abin.max(axis=2), abin.min(axis=2)
    MDown = M.copy()
    MDown[MDown == 0] = 1

    return (M-m)/MDown


def _kl_hr(pha, amp, nbins):
    nPha, npts, nAmp = *pha.shape, amp.shape[0]
    step = 2*n.pi/nbins
    vecbin = binarize(-n.pi, n.pi+step, step, step)
    if len(vecbin) > nbins:
        vecbin = vecbin[0:-1]

    abin = n.zeros((nAmp, nPha, nbins))
    for k, i in enumerate(vecbin):
        # Find where phase take vecbin values :
        pL, pC = n.where((pha >= i[0]) & (pha < i[1]))

        # Matrix to do amp x binMat :
        binMat = n.zeros((npts, nPha))
        binMat[pC, pL] = 1
        meanMat = n.matlib.repmat(binMat.sum(axis=0), nAmp, 1)
        meanMat[meanMat == 0] = 1

        # Multiply matrix :
        abin[:, :, k] = n.divide(n.dot(amp, binMat), meanMat)
    abinsum = n.array([abin.sum(axis=2) for k in range(nbins)])

    return abin, abinsum

# ----------------------------------------------------------------------------
#                                 SURROGATES
# ----------------------------------------------------------------------------


def CfcSurrogatesList(Id, CfcModel, n_perm=200, tlag=[0, 0]):
    """List of methods to compute surrogates.

    The surrogates are used to normalized the cfc value. It help to determine
    if the cfc is reliable or not. Usually, the surrogates used the same cfc
    method on surrogates data.
    Here's the list of methods to compute surrogates:
    - No surrogates
    - Shuffle phase values
    - Time lag
    - Swap phase/amplitude through trials
    - Swap amplitude
    - circular shifting

    Each method should return the surrogates, the mean of the surrogates and
    the deviation of the surrogates.
    """
    # No surrogates
    if Id == 0:
        def CfcSuroModel(pha, amp, CfcModel, n_perm):
            return (None, None, None)
        CfcSuroModelStr = 'No surrogates'

    # Shuffle phase values
    elif Id == 1:
        def CfcSuroModel(pha, amp, CfcModel, n_perm):
            return CfcShuffle(pha, amp, CfcModel, n_perm=n_perm)
        CfcSuroModelStr = 'Shuffle phase values'

    # Introduce a time lag
    elif Id == 2:
        def CfcSuroModel(pha, amp, CfcModel, n_perm, tlag):
            return CfcTimeLag(pha, amp, CfcModel, n_perm=n_perm, tlag=tlag)
        CfcSuroModelStr = 'Time lag on amplitude between ['+int(
            tlag[0])+';'+int(tlag[1])+'] , (Canolty, 2006)'

    # Swap phase/amplitude through trials
    elif Id == 3:
        def CfcSuroModel(pha, amp, CfcModel, n_perm):
            return CfcTrialSwap(pha, amp, CfcModel, n_perm=n_perm)
        CfcSuroModelStr = 'Swap phase/amplitude through trials (Tort, 2010)'

    # Swap ampliude
    elif Id == 4:
        def CfcSuroModel(pha, amp, CfcModel, n_perm):
            return CfcAmpSwap(pha, amp, CfcModel, n_perm=n_perm)
        CfcSuroModelStr = 'Swap amplitude, (Bahramisharif, 2013)'

    # Circular shifting
    elif Id == 5:
        def CfcSuroModel(pha, amp, CfcModel, n_perm):
            return CfcCircShift(pha, amp, CfcModel, n_perm=n_perm)
        CfcSuroModelStr = 'Circular shifting'

    return CfcSuroModel, CfcSuroModelStr


def CfcShuffle(xfP, xfA, CfcModel, n_perm=200):
    """Shuffle the phase values. For each shuffle phase distribution,
    we compute the cfc using the cfc method.
    """
    nPha, timeL, nbTrials = xfP.shape
    nAmp = xfA.shape[0]

    perm = [n.random.permutation(timeL) for k in range(n_perm)]
    # Compute surrogates :
    Suro = n.zeros((nbTrials, nAmp, nPha, n_perm))
    for k in range(nbTrials):
        CurPha, curAmp = xfP[:, :, k], n.matrix(xfA[:, :, k])
        for i in range(n_perm):
            # Randomly permutate phase values :
            CurPhaShuffle = CurPha[:, perm[i]]
            # compute new Cfc :
            Suro[k, :, :, i] = CfcModel(n.matrix(CurPhaShuffle), curAmp)

    return Suro


# ----------------------------------------------------------------------------
#                               NORMALIZATION
# ----------------------------------------------------------------------------
def CfcNormalizationList(Id):
    """List of the normalization methods.

    Use a normalization to normalize the true cfc value by the surrogates.
    Here's the list of the normalization methods :
    - No normalization
    - Substraction : substract the mean of surrogates
    - Divide : divide by the mean of surrogates
    - Substract then divide : substract then divide by the mean of surrogates
    - Z-score : substract the mean and divide by the deviation of the
                surrogates

    The normalized method only return the normalized cfc.
    """
    # No normalisation
    if Id == 0:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            return uCfc
        CfcNormModelStr = 'No normalisation'

    # Substraction
    if Id == 1:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            return ucfc-SuroMean
        CfcNormModelStr = 'Substract the mean of surrogates'

    # Divide
    if Id == 2:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            return uCfc/SuroMean
        CfcNormModelStr = 'Divide by the mean of surrogates'

    # Substract then divide
    if Id == 3:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            SuroMean[SuroMean == 0] = 1
            return (uCfc-SuroMean)/SuroMean
        CfcNormModelStr = 'Substract then divide by the mean of surrogates'

    # Z-score
    if Id == 4:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            SuroStd[SuroStd == 0] = 1
            return (uCfc-SuroMean)/SuroStd
        CfcNormModelStr = 'Z-score'

    return CfcNormModel, CfcNormModelStr
