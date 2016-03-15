import numpy as n
# from scipy.linalg.blas import dgemm

####################################################################
# - List of Cfc implemented methods :
####################################################################
def CfcMethodList(Id, nbins=18):
    if Id == 1:     # Modulation Index (Canolty, 2006)
        def CfcModel(pha,amp): return ModulationIndex(pha,amp)
        CfcModelStr = 'Modulation Index (Canolty, 2006)'
    elif Id == 2:   # Kullback-Leiber divergence (Tort, 2010)
        def CfcModel(pha,amp,nbins): return KullbackLeiblerDistance(pha,amp,nbins=nbins)
        CfcModelStr = 'Kullback-Leibler Distance (Tort, 2010) ['+str(nbins)+'bins]'
    elif Id == 3:    # Phase synchrony
        def CfcModel(pha,amp): return PhaseSynchrony(pha,amp)
        CfcModelStr = 'Phase synchrony'
    elif Id == 4:   # Amplitude PSD
        CfcModelStr = 'Amplitude PSD'
    elif Id == 5:   # Heights ratio
        def CfcModel(pha,amp,nbins): return HeightsRatio(pha,amp,nbins=nbins)
        CfcModelStr = 'Heights ratio';
    elif Id == 6:   # ndPac (Ozkurt, 2012)
        def CfcModel(pha,amp): return ndCfc(pha,amp)
        CfcModelStr = 'Normalized direct Pac (Ozkurt, 2012)'
#         Cfcsup.CfcstatMeth = 1

    return CfcModel, CfcModelStr

####################################################################
# 1 - Modulation Index :
####################################################################
def ModulationIndex(pha,amp):
    # [pha] = Nb phase x Time points
    # [amp] = Nb amplitude x Time points
    # return a Nb amplitude x Nb phase coupling
    # return abs(dgemm(alpha=1.0, a=amp, b=n.real(n.exp(1j*pha)), trans_b=True))/pha.shape[1]
    return abs(amp*n.exp(1j*pha).T)/pha.shape[1]