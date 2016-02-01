import numpy as n
import matplotlib.pyplot as plt

__all__ = ['addLines', 'BorderPlot']


class _pltutils(object):

    def __init__(self, ax, title='', xlabel='', ylabel='', xlim=[], ylim=[],
                 xticks=[], yticks=[], xticklabels=[], yticklabels=[],
                 style='seaborn-poster'):

        if not hasattr(self, '_xType'):
            self._xType = int
        if not hasattr(self, '_yType'):
            self._yType = int
        if not n.array(xlim).size:
            xlim = list(ax.get_xlim())
        if not n.array(ylim).size:
            ylim = list(ax.get_ylim())
        if not n.array(xticks).size:
            xticks = n.array(ax.get_xticks()).astype(self._xType)
        if not n.array(yticks).size:
            yticks = n.array(ax.get_yticks()).astype(self._yType)
        if not xticklabels:
            xticklabels = xticks
        if not yticklabels:
            yticklabels = yticks
        ax.set_title(title, y=1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.style.use(style)


class addLines(object):

    def __init__():
        pass

    def __new__(self, ax,
                vLines=[], vColor=None, vShape=None, vWidth=None,
                hLines=[], hColor=None, hWidth=None, hShape=None):
        # Get the axes limits :
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())

        # Get the number of vertical and horizontal lines :
        nV = len(vLines)
        nH = len(hLines)

        # Define the color :
        if not vColor:
            vColor = ['gray']*nV
        if not hColor:
            hColor = ['black']*nH

        # Define the width :
        if not vWidth:
            vWidth = [1]*nV
        if not hWidth:
            hWidth = [1]*nH

        # Define the shape :
        if not vShape:
            vShape = ['--']*nV
        if not hShape:
            hShape = ['-']*nH

        # Plot Verticale lines :
        for k in range(0, nV):
            ax.plot((vLines[k], vLines[k]), (ylim[0], ylim[1]), vShape[k],
                    color=vColor[k], linewidth=vWidth[k])
        # Plot Horizontal lines :
        for k in range(0, nH):
            ax.plot((xlim[0], xlim[1]), (hLines[k], hLines[k]), hShape[k],
                    color=hColor[k], linewidth=hWidth[k])


class BorderPlot(_pltutils):

    def __init__():
        pass

    def __new__(self, time, x, y=n.array([]), kind='sem', color='', alpha=0.2,
                linewidth=2, legend='', ncol=1, loc=0, **kwargs):

        self.xType = type(time[0])
        self.yType = type(x[0, 0])

        # Check arguments :
        if x.shape[1] == len(time):
            x = x.T
        npts, dev = x.shape
        if not y.size:
            y = n.array([0]*dev)
        yClass = n.unique(y)
        nclass = len(yClass)
        if not color:
            color = ['darkblue', 'darkgreen', 'darkred',
                     'darkorange', 'purple', 'gold', 'dimgray', 'k']
        else:
            if type(color) is not list:
                color = [color]
            if len(color) is not nclass:
                color = color*nclass
        if not legend:
            legend = ['']*dev
        else:
            if type(legend) is not list:
                legend = [colegendcolorlor]
            if len(legend) is not nclass:
                legend = legend*nclass

        # For each class :
        for k in yClass:
            _BorderPlot(time, x[:, n.where(y == k)[0]], color[k], kind=kind,
                        alpha=alpha, linewidth=linewidth, legend=legend[k])
        ax = plt.gca()
        ax.legend(loc=loc, frameon=False, ncol=ncol)
        plt.axis('tight')

        _pltutils.__init__(self, ax, **kwargs)

        return plt.gca()


def _BorderPlot(time, x, color, kind='sem', alpha=0.2, legend='', linewidth=2):
    npts, dev = x.shape
    # Get the deviation/sem :
    xStd = n.std(x, axis=1)
    if kind is 'sem':
        xStd = xStd/n.sqrt(npts-1)
    xMean = n.mean(x, 1)
    xLow, xHigh = xMean-xStd, xMean+xStd

    # Plot :
    ax = plt.plot(time, xMean, color=color, label=legend, linewidth=linewidth)
    plt.fill_between(time, xLow, xHigh, alpha=alpha, color=ax[0].get_color())
