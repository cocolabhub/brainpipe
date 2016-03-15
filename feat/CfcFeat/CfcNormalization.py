
####################################################################
# - List of Cfc normalization implemented methods :
####################################################################
def CfcNormalizationList(Id):
    if Id == 0: # No normalisation
        def CfcNormModel(uCfc,SuroMean,SuroStd): return uCfc
        CfcNormModelStr = 'No normalisation'
    if Id == 1: # Substraction
        def CfcNormModel(uCfc,SuroMean,SuroStd): return ucfc-SuroMean
        CfcNormModelStr = 'Substract the mean of surrogates'
    if Id == 2: # Divide
        def CfcNormModel(uCfc,SuroMean,SuroStd): return uCfc/SuroMean
        CfcNormModelStr = 'Divide by the mean of surrogates'
    if Id == 3: # Substract then divide
        def CfcNormModel(uCfc,SuroMean,SuroStd): return (uCfc-SuroMean)/SuroMean
        CfcNormModelStr = 'Substract then divide by the mean of surrogates'
    if Id == 4: # Z-score
        def CfcNormModel(uCfc,SuroMean,SuroStd): return (uCfc-SuroMean)/SuroStd
        CfcNormModelStr = 'Z-score: substract the mean and divide by the deviation of the surrogates'

    return CfcNormModel, CfcNormModelStr