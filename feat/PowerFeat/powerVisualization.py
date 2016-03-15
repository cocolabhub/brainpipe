from ...vizualisation.tools_plot import mapplot
import numpy as n


#---------------------------------------------------------------------
# Plot PowerMap :
#---------------------------------------------------------------------
def powerMap(nCfc, time=None, frequency=None, interp=None, xlabel='Time (ms)', ylabel='Frequency (Hz)',figName='Power Map', 
             title='Time frequency analysis', cbLab='Power modulation',**kwargs):
# vmin=None, vmax=None, cmap='jet', axes=None, labelsize='auto',ticksize='auto',,fullscreen='off',savefig=None
    """Ok c'est cool"""

    return mapplot(nCfc, x=time, y=frequency, interp=interp, xlabel=xlabel, ylabel=ylabel,figName=figName, title=title, cbLab=cbLab,**kwargs)
    #vmin=vmin, vmax=vmax, cmap=cmap, axes=axes, labelsize=labelsize,ticksize=ticksize,,fullscreen=fullscreen,savefig=savefig
