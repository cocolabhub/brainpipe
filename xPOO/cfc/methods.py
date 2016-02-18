import numpy as n

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
        def CfcModel(pha, amp):
            return ModulationIndex(pha, amp)
        CfcModelStr = 'Modulation Index (Canolty, 2006)'

    # Kullback-Leiber divergence (Tort, 2010)
    elif Id == 2:
        def CfcModel(pha, amp, nbins):
            return KullbackLeiblerDistance(pha, amp, nbins=nbins)
        CfcModelStr = 'Kullback-Leibler Distance (Tort, 2010) ['+str(
            nbins)+'bins]'

    # Phase synchrony
    elif Id == 3:
        def CfcModel(pha, amp):
            return PhaseSynchrony(pha, amp)
        CfcModelStr = 'Phase synchrony'

    # Amplitude PSD
    elif Id == 4:
        CfcModelStr = 'Amplitude PSD'

    # Heights ratio
    elif Id == 5:
        def CfcModel(pha, amp, nbins):
            return HeightsRatio(pha, amp, nbins=nbins)
        CfcModelStr = 'Heights ratio'

    # ndPac (Ozkurt, 2012)
    elif Id == 6:
        def CfcModel(pha, amp):
            return ndCfc(pha, amp)
        CfcModelStr = 'Normalized direct Pac (Ozkurt, 2012)'
#         Cfcsup.CfcstatMeth = 1

    return CfcModel, CfcModelStr


def ModulationIndex(pha, amp):
    """Modulation index (Canolty, 2006)
    """
    return n.array(abs(amp*n.exp(1j*pha).T)/pha.shape[1])


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
    Suro = n.zeros((nAmp, nPha, nbTrials, n_perm))
    for k in range(nbTrials):
        CurPha, curAmp = xfP[:, :, k], n.matrix(xfA[:, :, k])
        for i in range(n_perm):
            # Randpmly permutate phase values :
            CurPhaShuffle = CurPha[:, perm[i]]
            # compute new Cfc :
            Suro[:, :, k, i] = CfcModel(n.matrix(CurPhaShuffle), curAmp)
    # Return surrogates,mean & deviation for each surrogate distribution :
    return Suro, n.mean(Suro, 3), n.std(Suro, 3)


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
            return (uCfc-SuroMean)/SuroMean
        CfcNormModelStr = 'Substract then divide by the mean of surrogates'

    # Z-score
    if Id == 4:
        def CfcNormModel(uCfc, SuroMean, SuroStd):
            return (uCfc-SuroMean)/SuroStd
        CfcNormModelStr = 'Z-score: substract the mean and divide by the deviation of the surrogates'

    return CfcNormModel, CfcNormModelStr
