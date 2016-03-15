import numpy as n
# from brainpipe.feature.brainfir import filtvec
from .CfcMehods import *
from .CfcSurrogates import *
from .CfcNormalization import CfcNormalizationList
from .. import brainfir

__all__ = [
    'CrossFrequencyCoupling', 'CfcVectors'
]

####################################################################
# - Compute the cross-frequency coupling (Cfc) :
####################################################################
def CrossFrequencyCoupling(x, sf, phase, amplitude, Id='114', xPha=None, xAmp=None, nbins=18, p=0.005,
                           display=True, window=None, cycle=(3,6), filtname='fir1', order=3, fmeth='hilbert',
                           wavelet_width=7, dtrd=None, wavelet_correction=3):
    # [x] = Time points x trials

    #----------------------------------------------------------------------------------
    # 1 - Check and initialize variables:
    #----------------------------------------------------------------------------------
    if type(phase) == tuple: phase = [phase]
    if type(amplitude) == tuple: amplitude = [amplitude]
    if (xPha == None) & (xAmp == None): xPha, xAmp = x, x
    if xPha.shape != xAmp.shape: raise ValueError("The signals used for phase and amplitude must have the same length")
    nPha, nAmp, timeL, nbTrials, surrogates = len(phase), len(amplitude), xPha.shape[0], xPha.shape[1], round(1/p)
    if window == None: window, nWin = (0,timeL), 1
    else: nWin = len(window)
    if type(window) == tuple: window = [window]

    #----------------------------------------------------------------------------------
    # 2 - Get the model of Cfc (Cfc method, surrogates & normalization):
    #----------------------------------------------------------------------------------
    CfcModel, CfcSur, CfcNorm, CfcModelStr, CfcSurStr, CfcNormStr = CfcSettings(Id)

    #----------------------------------------------------------------------------------
    # 3 - Display informations:
    #----------------------------------------------------------------------------------
    if display == True:
        print('-> Input dimension: [x] =',int(timeL),'time pts x ',int(nbTrials),'trials')
        print('-> Bands:',int(nPha),'phases,',int(nAmp),'amplitudes,',int(nPha*nAmp),'Cross-Frequency Coupling computed')
        if filtname == 'fir1': print('-> Filter:',filtname,'filter with (cycle phase x amplitude) =',cycle)
        else: print('-> Filter:',filtname,'filter order',int(order))
        print('-> Cfc method:',CfcModelStr)
        print('-> Surrogates method:',CfcSurStr,'[',int(surrogates),' surrogates ]')
        print('-> Normalization:',CfcNormStr)

    #----------------------------------------------------------------------------------
    # 4 - Extract phase, amplitude and Cfc of all trials:
    #----------------------------------------------------------------------------------
    xfP, xfA, uCfc = n.zeros((nPha,timeL,nbTrials)), n.zeros((nAmp,timeL,nbTrials)), n.zeros((nAmp,nPha,nbTrials,nWin))
    xfP = brainfir.filtvec(xPha, sf, phase, 'phase', cycle=cycle[0], filtname=filtname, order=order, axis=0, dtrd=dtrd)
    xfA = brainfir.filtvec(xAmp, sf, amplitude, 'amplitude', cycle=cycle[1], filtname=filtname, order=order, axis=0, fmeth=fmeth, wavelet_width=wavelet_width, dtrd=dtrd, wavelet_correction=wavelet_correction)
    for k in range(0,nbTrials):
        for i in range(0,nWin):
            uCfc[:,:,k,i] = CfcModel(n.matrix(xfP[:,window[i][0]:window[i][1],k]),n.matrix(xfA[:,window[i][0]:window[i][1],k]))

    #----------------------------------------------------------------------------------
    # 5 - Compute surrogates:
    #----------------------------------------------------------------------------------
    SuroAll, SuroMean, SuroStd = [0]*nWin, n.zeros((nAmp,nPha,nbTrials,nWin)), n.zeros((nAmp,nPha,nbTrials,nWin))
    for i in range(0,nWin):
        SuroAll[i], SuroMean[:,:,:,i], SuroStd[:,:,:,i] = CfcSur(xfP[:,window[i][0]:window[i][1],:],xfA[:,window[i][0]:window[i][1],:],CfcModel,surrogates=surrogates)

    #----------------------------------------------------------------------------------
    # 6 - Normalize Cfc with surrogates:
    #----------------------------------------------------------------------------------
    nCfc = CfcNorm(uCfc,SuroMean,SuroStd)

    #----------------------------------------------------------------------------------
    # 7 - Get the confidance interval of the nCfc values:
    #----------------------------------------------------------------------------------
    pCfc = n.zeros(nCfc.shape)
    for k in range(0,nWin):
        for i in range(0,nAmp):
            for j in range(0,nPha):
                for l in range(0,nbTrials):
                    pCfc[i,j,l,k] = Get_pCfc(nCfc[i,j,l,k],SuroAll[k][i,j,l,:])

    return nCfc, pCfc, uCfc, [SuroAll, SuroMean, SuroStd]

####################################################################
# - Get the pvalue of the nCfc :
####################################################################
def Get_pCfc(toeval,distribution):
    nSuro = distribution.shape[0]
    nbUpperCfc = len(n.arange(0,nSuro)[distribution > toeval])
    if nbUpperCfc != 0: return 1/nbUpperCfc
    else: return 1/nSuro

####################################################################
# - Get the Cfc model from the Id variable :
####################################################################
def CfcSettings(Id,nbins=18,surrogates=200,tlag=[0,0]):
    # Define the method of PAC :
    [CfcModel, CfcModelStr] = CfcMethodList(int(Id[0]),nbins=nbins)
    # Define the way to compute surrogates :
    [CfcSur, CfcSurStr] = CfcSurrogatesList(int(Id[1]),CfcModel,surrogates=surrogates,tlag=tlag)
    # Define the way to normalize the Cfc with surrogates :
    [CfcNorm, CfcNormStr] = CfcNormalizationList(int(Id[2]))

    return CfcModel, CfcSur, CfcNorm, CfcModelStr, CfcSurStr, CfcNormStr

####################################################################
# - Generate phase and amplitude vectors :
####################################################################
def CfcVectors(phase=(2,30,2,1), amplitude=(10,200,10,5)):
    # Get values from tuple:
    pStart, pEnd, pWidth, pStep = phase[0], phase[1], phase[2], phase[3]
    aStart, aEnd, aWidth, aStep = amplitude[0], amplitude[1], amplitude[2], amplitude[3]
    # Generate two array for phase and amplitude :
    pDown, pUp = n.arange(pStart-pWidth/2, pEnd-pWidth/2+1, pStep), n.arange(pStart+pWidth/2, pEnd+pWidth/2+1, pStep)
    aDown, aUp = n.arange(aStart-aWidth/2, aEnd-aWidth/2+1, aStep), n.arange(aStart+aWidth/2, aEnd+aWidth/2+1, aStep)
    # Generate the center frequency vector :
    pVec, aVec = (pUp+pDown)/2, (aUp+aDown)/2
    # Generate the tuple for the CrossFrequencyCoupling function :
    pTuple = [(pDown[k],pUp[k]) for k in range(0,pDown.shape[0])]
    aTuple = [(aDown[k],aUp[k]) for k in range(0,aDown.shape[0])]

    return pVec, aVec, pTuple, aTuple
