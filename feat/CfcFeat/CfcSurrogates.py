import numpy as n

####################################################################
# - List of Cfc surrogates methods :
####################################################################
def CfcSurrogatesList(Id, CfcModel,surrogates=200, tlag=[0,0]):
    if Id == 0: # No surrogates
        def CfcSuroModel(pha,amp,CfcModel,surrogates): return (None,None,None)
        CfcSuroModelStr = 'No surrogates'
    elif Id == 1: # Shuffle phase values
        def CfcSuroModel(pha,amp,CfcModel,surrogates): return CfcShuffle(pha,amp,CfcModel,surrogates=surrogates)
        CfcSuroModelStr = 'Shuffle phase values'
    elif Id == 2: # Introduce a time lag
        def CfcSuroModel(pha,amp,CfcModel,surrogates,tlag): return CfcTimeLag(pha,amp,CfcModel,surrogates=surrogates,tlag=tlag)
        CfcSuroModelStr = 'Time lag on amplitude between ['+int(tlag[0])+';'+int(tlag[1])+'] , (Canolty, 2006)'
    elif Id == 3: # Swap phase/amplitude through trials
        def CfcSuroModel(pha,amp,CfcModel,surrogates): return CfcTrialSwap(pha,amp,CfcModel,surrogates=surrogates)
        CfcSuroModelStr = 'Swap phase/amplitude through trials (Tort, 2010)'
    elif Id == 4: # Swap ampliude
        def CfcSuroModel(pha,amp,CfcModel,surrogates): return CfcAmpSwap(pha,amp,CfcModel,surrogates=surrogates)
        CfcSuroModelStr = 'Swap amplitude, (Bahramisharif, 2013)'
    elif Id == 5: # Circular shifting
        def CfcSuroModel(pha,amp,CfcModel,surrogates): return CfcCircShift(pha,amp,CfcModel,surrogates=surrogates)
        CfcSuroModelStr = 'Circular shifting'

    return CfcSuroModel, CfcSuroModelStr

####################################################################
# 1 - Shuffle phase values :
####################################################################
def CfcShuffle(xfP,xfA,CfcModel,surrogates=200):
    nPha, timeL, nbTrials = xfP.shape
    nAmp = xfA.shape[0]

    perm = [n.random.permutation(timeL) for k in range(0,surrogates)]
    # Compute surrogates :
    Suro = n.zeros((nAmp,nPha,nbTrials,surrogates))
    for k in range(0,nbTrials):
        CurPha, curAmp = xfP[:,:,k], n.matrix(xfA[:,:,k])
        for i in range(0,surrogates):
            # Randpmly permutate phase values :
            CurPhaShuffle = CurPha[:,perm[i]]
            # compute new Cfc :
            Suro[:,:,k,i] = CfcModel(n.matrix(CurPhaShuffle),curAmp)
    # Return surrogates,mean & deviation for each surrogate distribution :
    return Suro, n.mean(Suro,3), n.std(Suro,3)