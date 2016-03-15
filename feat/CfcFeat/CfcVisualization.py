from ...vizualisation.tools_plot import mapplot
import numpy as n


#---------------------------------------------------------------------
# Plot CfcMap :
#---------------------------------------------------------------------
def CfcMap(nCfc, title='Cross-frequency coupling', cbLab='Cross-frequency coupling', interp=(0.01,0.01), 
	xlabel='Frequency for phase (Hz)', ylabel='Frequency for amplitude (Hz)', figName='Cfc Map', phase=None, amplitude=None, **kwargs):
	#phase=None, amplitude=None, vmin=None, vmax=None, cmap='jet', axes=None, labelsize='auto', ticksize='auto', fullscreen='off',savefig=None, figsize=(50,50)):
    """Ok c'est cool"""

    return mapplot(nCfc, title=title, cbLab=cbLab, interp=interp, xlabel=xlabel, ylabel=ylabel,figName=figName, x=phase, y=amplitude, **kwargs)
    	# x=phase, y=amplitude, vmin=vmin, vmax=vmax, cmap=cmap, axes=axes, labelsize=labelsize, ticksize=ticksize,
     #    fullscreen=fullscreen,savefig=savefig,figsize=figsize)
