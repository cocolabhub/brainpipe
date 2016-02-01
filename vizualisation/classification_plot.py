import matplotlib.pyplot as plt
import seaborn as sns
import numpy as n
from matplotlib.colors import Normalize
from brainpipe.vizualisation.tools_plot import mapplot, mapinterpolation
import matplotlib.lines as mlines

####################################################################
# - Permutations plot :
####################################################################
def permutations_plot(y, da, pvalue, permutation_scores):
    nb_class = len(n.unique(y))
    plt.hist(permutation_scores, 20, label='Permutation scores')
    ylim = plt.ylim()
    plt.plot(2 * [da], ylim, '--g', linewidth=3,
             label='Classification Score'
                   ' (pvalue %s)' % pvalue)
    plt.plot(2 * [1. / nb_class], ylim, '--k', linewidth=3, label='Luck')

    plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Score')
    plt.show()


####################################################################
# - Time generalization plot :
####################################################################
def timegeneralizationMap(da, time=None, interp=None, xlabel='Generalization time', ylabel='Training time', cmap='plasma',
                          figName='Time generalization map', title='Time generalization map', cbLab='Decoding accuracy(%)',
                          blurred=None, typePlot='single', cue=None, vmin=None, vmax=None,  pvalue=None, figsize=(12.8,8.8), 
                          **kwargs):
    
    # Interpolate the map :
    if time is None: time = n.arange(0,da.shape[0])
    if interp != None:
        da, time, _ = mapinterpolation(da,interpx=interp[0],interpy=interp[1],x=time,y=time)
        
    # Define a blurred=(threshold,alpha)
    if vmin is None: vmin=da.min()
    if vmax is None: vmax=da.max()
    if blurred is not None:
        dam, daM = vmin,vmax#da.min(), da.max()
        norm = Normalize(dam, daM)
        daNorm = plt.get_cmap(cmap)(norm(da))
        daNorm[da < blurred[0], -1] = blurred[1]
        typePlot='single'
    else: daNorm = da
        
    fig = mapplot(daNorm, x=time, y=time, interp=None, xlabel=xlabel, ylabel=ylabel,figName=figName, title=title, cbLab=cbLab,
                   cmap=cmap, typePlot=typePlot, vmin=vmin, vmax=vmax, figsize=figsize, **kwargs)
    ax = plt.gca()
    ax.plot((time[0], time[-1]), (time[0], time[-1]), '-',linewidth=1.5, color='k')
    
    
    # Define a pvalue=(threshold,label,color,linewidth):
    if pvalue != None: 
        if type(pvalue)!=list: pvalue=[pvalue]
        pval, pcol, ppatch = [], [], []
        for k in range(0,len(pvalue)): 
            pval.append(pvalue[k][0]), pcol.append(pvalue[k][2])
            ppatch.append(mlines.Line2D([], [], color=pvalue[k][2], label=pvalue[k][1]))

        ax.contour(time,time,da,pval,colors=pcol)
        plt.legend(handles=ppatch, loc=2, fancybox=True, frameon=True, shadow=True, 
           borderpad=0.3, labelspacing=0.05, handletextpad=0.3, fontsize=20, borderaxespad=0.37)
            
    # Define a cue=(timeIndex,string,color,linewidth):
    if cue != None: 
        if type(cue)!=list: cue=[cue]
        Label = []
        for k in range(0,len(cue)): 
            plt.plot((cue[k][0], cue[k][0]), (time[0], time[-1]), '-',linewidth=cue[k][3], color=cue[k][2])
            plt.text(cue[k][0]-120, -400, cue[k][1], rotation=90, color='w', size=20, weight='bold')
            Label.append(cue[k][0])
#         plt.autoscale(tight=True)
        ax.set_xticklabels(Label), ax.set_yticklabels(Label), ax.set_xticks(Label), ax.set_yticks(Label)

    plt.autoscale(tight=True)
    plt.grid(b=None)

    return plt.gcf()