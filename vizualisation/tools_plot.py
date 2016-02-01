import matplotlib.pyplot as plt
import numpy as n
import seaborn as sns
import brainpipe.system.interpolation as interpo

__all__ = [
    'BorderPlot', 'plotLine', 'mapinterpolation', 'mapplot',
]

####################################################################
# - Time generalization plot :
####################################################################
def BorderPlot(x, y=None, time=None, linewidth=6, condition=None, palette=None, xlabel='', fullscreen='off',
               ylabel='', title='', despine='on', vLines=[], vColor=None, vShape=None, vWidth=None,
               hLines=[], hColor=None, hWidth=None, hShape=None, vmin=None, vmax=None, tight=True):
    # [x] = nbtrials x timepoints

    # ---------------------------------------------------------
    # Default elements part :
    # ---------------------------------------------------------

    if y is None: y = n.ones((x.shape[0]))

    # Get size elements :
    yUnique = n.unique(y)
    yUniqueLen = len(yUnique)
    yPearClass = len(n.arange(0,len(y),1)[y == yUnique[0]])

    # Check size elements :
    rdim = n.arange(0,len(x.shape),1)[n.array(x.shape) == len(y)]
    if len(rdim) != 0 : rdim = rdim[0]
    else: raise ValueError("None of x dimendion is "+str(len(y))+" length. [x] = "+str(x.shape))
    if rdim == 1: x = x.T
    nbtrials, nbpoints = x.shape

    if time is None: time = n.arange(0,nbpoints,1)

    # Define the default palette :
    if palette is None:
        palette = ['b', 'g', 'r', 'darkorange', 'purple', 'gold', 'dimgray', 'k']

    # Build a 3D matrix of conditions :
    if yUniqueLen == 1:
        xr = x
    else:
        xr = n.zeros((yPearClass,nbpoints,yUniqueLen))
        for k in range(0,yUniqueLen):
            xr[:,:,k] = x[y == yUnique[k] ]
    if vmin is None:
        vmin = xr.min()
    if vmax is None:
        vmax = xr.max()

    # ---------------------------------------------------------
    # Plot part :
    # ---------------------------------------------------------
    # fig = plt.figure('Border plot')
    # ax = plt.gca()#fig.add_subplot(111)
    # Draw BorderPlot :
    sns.set(palette=palette, context='poster', style='white')
    q = sns.tsplot(data=xr, time=time, condition=condition,linewidth=linewidth)
    q.set_ylim([vmin,vmax])
    if tight:
        plt.autoscale(tight=True)
    q.set_xlabel(xlabel), q.set_ylabel(ylabel), q.set_title(title)

    xl, yl = q.get_xlim(), q.get_ylim()
    if despine == 'on': sns.despine()

    # Add vertical/horizontal lines :
    if (vLines!=[]) or (hLines!=[]):
        q = plotLine(vLines=vLines, vWidth=vWidth, vColor=vColor, vShape=vShape,
                     hLines=hLines, hWidth=hWidth, hColor=hColor, hShape=hShape,
                     xaxis=list(xl), yaxis=list(yl),axes=plt.gca())

    # Fullscreen mode :
    if fullscreen == 'on':
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    return q


####################################################################
# - Time generalization plot :
####################################################################
def plotLine(vLines=[], vColor=None, vShape=None, vWidth=None, hLines=[], hColor=None, hWidth=None, hShape=None,
             xaxis=None, yaxis=None, fit='off', axes=None):

    if axes == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axes

    # Get the number of vertical and horizontal lines :
    nV = len(vLines)
    nH = len(hLines)

    # Define the limit of the axis :
    if xaxis is None :
        if nV == 0: xaxis = [-1,1]
        if nV != 0 and nV > 1: xaxis = [ n.array(vLines).min(), n.array(vLines).max() ]
        if nV == 1: xaxis = [ vLines[0]-1, vLines[0]+1 ]
    if yaxis is None :
        if nH == 0: yaxis = [-1,1]
        if nH != 0 and nH > 1: yaxis = [ n.array(hLines).min(), n.array(hLines).max() ]
        if nH == 1: yaxis = [ hLines[0]-1, hLines[0]+1 ]

    # Define the color :
    if vColor is None: vColor = ['gray']*nV
    if hColor is None: hColor = ['black']*nH

    # Define the width :
    if vWidth is None: vWidth = [1]*nV
    if hWidth is None: hWidth = [1]*nH

    # Define the shape :
    if vShape is None: vShape = ['--']*nV
    if hShape is None: hShape = ['-']*nH

    # Plot Verticale lines :
    for k in range(0,nV):
            ax.plot((vLines[k], vLines[k]), (yaxis[0], yaxis[1]), vShape[k],  color=vColor[k],
                         linewidth=vWidth[k] )
    # Plot Horizontal lines :
    for k in range(0,nH):
            ax.plot((xaxis[0], xaxis[1]), (hLines[k], hLines[k]), hShape[k],  color=hColor[k],
                         linewidth=hWidth[k] )

    if fit == 'on':
        plt.gca().set_xlim(n.array(xaxis))
        plt.gca().set_ylim(n.array(yaxis))

    return plt.gca()


def mapinterpolation(data,x=None,y=None,interpx=1,interpy=1):
    # Get data size :
    dim2, dim1 = data.shape
    ndim2, ndim1 = dim2/interpy, dim1/interpx
    # Define xticklabel and yticklabel :
    if x is None: x = n.arange(0,dim1,interpx)
    if y is None: y = n.arange(0,dim2,interpy)
    # Define the meshgrid :
    Xi,Yi = n.meshgrid(n.arange(0,dim1-1,interpx),n.arange(0,dim2-1,interpy))
    # 2D interpolation :
    datainterp = interpo.interp2(data,Xi,Yi)
    # Linearly interpolate vectors :
    xvecI = n.linspace(x[0],x[-1],datainterp.shape[0])
    yvecI = n.linspace(y[0],y[-1],datainterp.shape[1])

    return datainterp, xvecI, yvecI

#---------------------------------------------------------------------
# map fcn :
#---------------------------------------------------------------------
def mapplot(signal, x=None, y=None, vmin=None, vmax=None, cmap='jet', axes=None, labelsize='auto', ticksize='auto', ajust='on', cb='on',
              interp=None, xlabel='', ylabel='',figName='', title='', cbLab='',fullscreen='off',savefig=None, figsize=(50,50), typePlot='multi'):

    if (len(signal.shape) == 2) or (typePlot == 'single'):
        return _mapplot(signal, x=x, y=y, vmin=vmin, vmax=vmax, cmap=cmap, axes=axes, labelsize=labelsize,
              interp=interp, xlabel=xlabel, ylabel=ylabel,figName=figName, ticksize=ticksize,
             title=title,cbLab=cbLab,fullscreen=fullscreen,savefig=savefig, cb=cb, figsize=figsize)
    elif len(signal.shape) == 3:
        nSubplot = signal.shape[2]
        
        # Handle with input arguments :
        # Title :
        if type(title)==str: title = [title]*nSubplot
        # vmin & vmax :
        if vmin==None: vmin=[round(signal.min())]*nSubplot
        if vmax==None: vmax=[int(n.floor(signal.max()))]*nSubplot
        if (type(vmin)==int) or (type(vmin)==float): vmin=[vmin]*nSubplot
        if (type(vmax)==int) or (type(vmax)==float): vmax=[vmax]*nSubplot
        
        fig = plt.figure(figName,figsize=figsize)
        for k in range(0,nSubplot):
            if ajust=='on':
                if k > 0: ylabel=''
                cbl='on'
                # if k!=nSubplot-1: cbl=None
                # else: cbl=cb
                ytl='on'
                # if k > 0: ytl=None
                # else: ytl='on' 
            if ajust=='off': cbl, ytl = 'on', 'on'

            ax = fig.add_subplot(1,nSubplot,k+1)
            _mapplot(signal[:,:,k],vmin=vmin[k],vmax=vmax[k],title=title[k],savefig=None,axes=ax, x=x,y=y,cmap=cmap,ticksize=ticksize,
                interp=interp, xlabel=xlabel, ylabel=ylabel, labelsize=labelsize,cbLab=cbLab,fullscreen=fullscreen, cb=cbl, yticklabel=ytl)
            
        # Save the figure:
        if type(savefig)==str: fig.savefig(savefig, bbox_inches='tight')
            
        return plt.gcf()


def _mapplot(signal, x=None, y=None, vmin=None, vmax=None, cmap='jet', axes=None, labelsize='auto', ticksize='auto', figsize=(50,50),
              interp=None, xlabel='', ylabel='',figName='', title='', cbLab='',fullscreen='off',savefig=None, cb='on', yticklabel='on'):


    #---------------------------------------------------------------------
    # Manage input variables :
    #---------------------------------------------------------------------
    if y is None: y = n.arange(0,signal.shape[0])
    if x is None: x = n.arange(signal.shape[1])
    if interp != None:
        signali, xvec, yvec = mapinterpolation(signal,interpx=interp[0],interpy=interp[1],x=x,y=y)
    else:
        signali, xvec, yvec = signal, x, y
    # try:
    #      plt.style.use('brainpipe')
    # except :
    #      plt.style.use('ggplot')
    if axes is None:
        fig = plt.figure(figName,figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axes
    if vmin is None: vmin=signali.min()
    if vmax is None: vmax=signali.max()

    #---------------------------------------------------------------------
    # Imshow the CfcMap : 
    #---------------------------------------------------------------------
    im = ax.imshow(signali, cmap=cmap, extent=[xvec[0],xvec[-1],yvec[-1],yvec[0]], aspect='auto', vmin=vmin, vmax=vmax)
    
    AxesPos = ax.get_position()
    def textResize(txt):
        return txt*n.exp(AxesPos.width-0.9)

    if labelsize == 'auto': sizeLab = textResize(24)
    else: sizeLab = labelsize
    if ticksize == 'auto': sizeCb = textResize(18)
    else: sizeCb = ticksize
    ax.invert_yaxis()
    ax.set_xlabel(xlabel,size=sizeLab)
    ax.set_ylabel(ylabel,size=sizeLab)
    ax.set_title(title, y=1.02, size=sizeLab)
    plt.grid(b=None)
    forceAspect(ax,aspect=1)
    if yticklabel is None: ax.set_yticklabels('')

    #---------------------------------------------------------------------
    # Colorbar properties :
    #---------------------------------------------------------------------
    if cb is None: 
        cb = plt.colorbar(im, shrink=0, pad=0.02)
        cb.ax.set_yticklabels('')
    else: 
        cb = plt.colorbar(im, shrink=AxesPos.width+0.05, pad=0.02)
        cb.set_ticks([vmin,vmax])#n.arange(round(vmin),n.floor(vmax)+0.01))
        cb.ax.set_aspect(10)
        cb.set_label(cbLab,size=sizeCb, labelpad=-5)
        cb.ax.tick_params(labelsize=sizeLab) 

    #---------------------------------------------------------------------
    # Other parameters :
    #---------------------------------------------------------------------
    AxesPos = ax.get_position()
    ax.set_position([AxesPos.x0, 0.12, AxesPos.width, AxesPos.height])
    ax.tick_params(labelsize=sizeCb)
    # Fullscreen mode :
    if fullscreen == 'on':
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    fig = plt.gcf()
    
    return fig

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
