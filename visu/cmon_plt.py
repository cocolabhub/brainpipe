import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from brainpipe.visu._interp import mapinterpolation
from warnings import warn


__all__ = ['addLines', 'BorderPlot', 'tilerplot']


class _pltutils(object):

    """
    **kwargs:
        title:
            title of plot [def: '']

        xlabel:
            label of x-axis [def: '']

        ylabel:
            label of y-axis [def: '']

        xlim:
            limit of the x-axis [def: [], current limit of x]

        ylim:
            limit of the y-axis [def: [], current limit of y]

        xticks:
            ticks of x-axis [def: [], current x-ticks]

        yticks:
            ticks of y-axis [def: [], current y-ticks]

        xticklabels:
            label of the x-ticks [def: [], current x-ticklabels]

        yticklabels:
            label of the y-ticks [def: [], current y-ticklabels]

        style:
            style of the plot [def: None]

    """

    def __init__(self, ax, title='', xlabel='', ylabel='', xlim=[], ylim=[],
                 xticks=[], yticks=[], xticklabels=[], yticklabels=[],
                 style=None):

        if not hasattr(self, '_xType'):
            self._xType = int
        if not hasattr(self, '_yType'):
            self._yType = int
        # Axes ticks :
        if np.array(xticks).size:
            ax.set_xticks(xticks)
        if np.array(yticks).size:
            ax.set_yticks(yticks)
        # Axes ticklabels :
        if xticklabels:
            ax.set_xticklabels(xticklabels)
        if yticklabels:
            ax.set_yticklabels(yticklabels)
        # Axes labels :
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        # Axes limit :
        if np.array(xlim).size:
            ax.set_xlim(xlim)
        if np.array(ylim).size:
            ax.set_ylim(ylim)
        ax.set_title(title, y=1.02)
        # Style :
        if style:
            plt.style.use(style)


class tilerplot(object):

    """Automatic tiler plot for 1, 2 and 3D data.
    """

    def plot1D(self, fig, y, x=None, maxplot=10, figtitle='',
               subdim=None, transpose=False, color='b', **kwargs):
        """Simple one dimentional plot

        Args:
            y: array
                Data to plot. y can either have one, two or three dimensions.
                If y is a vector, it will be plot in a simple window. If y is
                a matrix, all values inside are going to be superimpose. If y
                is a 3D matrix, the first dimension control the number of subplots.

            x: array, optional, [def: None]
                x vector for plotting data.

        Kargs:
            figtitle: string, optional, [def: '']
                Add a name to your figure

            subdim: tuple, optional, [def: None]
                Force subplots to be subdim=(n_colums, n_rows)

            maxplot: int, optional, [def: 10]
                Control the maximum number of subplot to prevent very large plot.
                By default, maxplot is 10 which mean that only 10 subplot can be
                defined.

            transpose: bool, optional, [def: False]
                Invert subplot (row <-> column)

            color: string, optional, [def: 'b']
                Color of the plot

            kwargs:
                Supplementar arguments to control each suplot:
                title, xlabel, ylabel (which can be list for each subplot)
                xlim, ylim, xticks, yticks, xticklabels, yticklabels, style.
        """
        # Fig properties:
        self._fig = self._figmngmt(fig, figtitle=figtitle, transpose=transpose)
        # Check y shape :
        y = self._checkarray(y)
        if x is None:
            x = np.arange(y.shape[1])
        # Get default for title, xlabel and ylabel:
        kwout, kwargs = self._completeLabels(kwargs, y.shape[0], 'title',
                                             'xlabel', 'ylabel', default='')
        # Plotting function :

        def _fcn(y, k):
            plt.plot(x, y, color=color)
            plt.axis('tight')
            _pltutils(plt.gca(), kwout['title'][k], kwout['xlabel'][k],
                      kwout['ylabel'][k], **kwargs)
        # Run function for each yi :
        return self._subplotND(y, _fcn, maxplot, subdim)

    def plot2D(self, fig, y, xvec=None, yvec=None, cmap='inferno',
               colorbar=True, cbticks='auto', vmin=None, vmax=None, cblabel='',
               sharex=False, sharey=False, subdim=None, mask=None,
               interpolation='none', resample=(0, 0), under=None, over=None,
               figtitle='', transpose=False, maxplot=10, **kwargs):
        """Plot y as an image

        Args:
            fig: figure
                A matplotlib figure where plotting

            y: array
                Data to plot. y can either have one, two or three dimensions.
                If y is a vector, it will be plot in a simple window. If y is
                a matrix, all values inside are going to be superimpose. If y
                is a 3D matrix, the first dimension control the number of subplots.

        Kargs:
            xvec, yvec: array, optional, [def: None]
                Vectors for y and x axis of each picture

            cmap: string, optional, [def: 'inferno']
                Choice of the colormap

            colorbar: bool, optional, [def: True]
                Add or not a colorbar to your plot

            vmin, vmax: int/float, optional, [def: None]
                Control minimum and maximum of the image

            cblabel: string, optional, [def: '']
                Label for the colorbar

            subdim: tuple, optional, [def: None]
                Force subplots to be subdim=(n_colums, n_rows)

            interpolation: string, optional, [def: 'none']
                Plot interpolation

            resample: tuple, optional, [def: (0, 0)]
                Interpolate the map for a specific dimension. If (0.5, 0.1),
                this mean that the programme will insert one new point on x-axis,
                and 10 new points on y-axis. Pimp you map and make it sooo smooth.

            figtitle: string, optional, [def: '']
                Add a name to your figure

            maxplot: int, optional, [def: 10]
                Control the maximum number of subplot to prevent very large plot.
                By default, maxplot is 10 which mean that only 10 subplot can be
                defined.

            transpose: bool, optional, [def: False]
                Invert subplot (row <-> column)

            kwargs:
                Supplementar arguments to control each suplot:
                title, xlabel, ylabel (which can be list for each subplot)
                xlim, ylim, xticks, yticks, xticklabels, yticklabels, style.
        """

        # Fig properties:
        self._fig = self._figmngmt(fig, figtitle=figtitle, transpose=transpose)

        # Share axis:
        if sharex:
            self._fig.subplots_adjust(hspace=0)
        # Mask properties:
        if (mask is not None):
            if not (mask.shape == y.shape):
                warn('The shape of mask '+str(mask.shape)+' must be the same '
                     'of y '+str(y.shape)+'. Mask will be ignored')
                mask = None
            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
        else:
            mask = []

        # Check y shape :
        y = self._checkarray(y)
        if xvec is None:
            xvec = np.arange(y.shape[-1])
        if yvec is None:
            yvec = np.arange(y.shape[1])

        l0, l1, l2 = y.shape

        # Resample data:
        if resample != (0, 0):
            yi = []
            maski = []
            for k in range(l0):
                yT, xvec, yvec = mapinterpolation(y[k, ...], x=xvec, y=yvec,
                                                  interpx=resample[0],
                                                  interpy=resample[1])
                yi.append(yT)
                if np.array(mask).size:
                    maskT, _, _ = mapinterpolation(mask[k, ...], x=xvec, y=yvec,
                                                   interpx=resample[0],
                                                   interpy=resample[1])
                    maski.append(maskT)
            y = np.array(yi)
            mask = maski
            del yi, yT
        # Get default for title, xlabel and ylabel:
        kwout, kwargs = self._completeLabels(kwargs, y.shape[0], 'title',
                                             'xlabel', 'ylabel', default='')

        # Plotting function :
        def _fcn(y, k, mask=mask):
            # Get a mask for data:
            if np.array(mask).size:
                mask = np.array(mask)
                norm = Normalize(vmin, vmax)
                y = plt.get_cmap(cmap)(norm(y))
                y[..., 3] = mask[k, ...]
            # Plot picture:
            im = plt.imshow(y, aspect='auto', cmap=cmap,
                            interpolation=interpolation, vmin=vmin, vmax=vmax,
                            extent=[xvec[0], xvec[-1], yvec[-1], yvec[0]])
            # Manage under and over:
            if (under is not None) and (isinstance(under, str)):
                im.cmap.set_under(color=under)
            if (over is not None) and (isinstance(over, str)):
                im.cmap.set_over(color=over)

            plt.axis('tight')
            ax = plt.gca()
            _pltutils(ax, kwout['title'][k], kwout['xlabel'][k],
                      kwout['ylabel'][k], **kwargs)
            if colorbar:
                cb = plt.colorbar(im, shrink=0.7, pad=0.01, aspect=10)
                if cbticks == 'auto':
                    cb.set_ticks(im.colorbar.get_clim())
                elif cbticks is None:
                    pass
                else:
                    cb.set_ticks(cbticks)
                cb.set_label(cblabel, labelpad=-10)

            ax.invert_yaxis()
        axAll = self._subplotND(y, _fcn, maxplot, subdim, sharex, sharey)
        fig = plt.gcf()
        fig.tight_layout()

        # Run function for each yi :
        return fig, axAll

    def plotcustom(self, fig, y, fcn, maxplot=10, subdim=None):
        """
        """
        self._fig = fig
        self._transpose = False
        return self._subplotND(y, fcn, maxplot, subdim)

    def _figmngmt(self, fig, figtitle='', transpose=False):
        # Change title:
        if figtitle:
            fig.suptitle(figtitle, fontsize=14, fontweight='bold')
        self._transpose = transpose
        return fig

    def _checkarray(self, y):
        """Check input shape
        """
        # Vector :
        if y.ndim == 1:
            y = y[np.newaxis, ..., np.newaxis]
        # 2D array :
        elif y.ndim == 2:
            y = y[np.newaxis]
        # more than 3D array :
        elif y.ndim > 3:
            raise ValueError('array to plot should not have more than '
                             '3 dimensions')
        return y

    def _subplotND(self, y, fcn, maxplot, subdim, sharex, sharey):
        """Manage subplots
        """
        axall = []
        L = y.shape[0]
        if L <= maxplot:
            fig = self._fig
            # If force subdim:
            if not subdim:
                if L < 4:
                    ncol, nrow = L, 1
                else:
                    ncol = round(np.sqrt(L)).astype(int)
                    nrow = round(L/ncol).astype(int)
                    while nrow*ncol < L:
                        nrow += 1
            else:
                nrow, ncol = subdim
            # Sublots:
            if self._transpose:
                backup = ncol
                ncol = nrow
                nrow = backup
            self._nrow, self._ncol = nrow, ncol
            for k in range(L):
                fig.add_subplot(nrow, ncol, k+1)
                fcn(y[k, ...], k)
                ax = plt.gca()
                # Share-y axis:
                if sharey and (k % ncol == 0):
                    pass
                else:
                    if sharey:
                        ax.set_yticklabels([])
                        ax.set_ylabel('')
                # Share-x axis:
                if sharex and (k < (nrow-1)*ncol):
                    ax.set_xticklabels([])
                    ax.set_xlabel('')
                axall.append(plt.gca())
                plt.gca().grid('off')
            return axall
        else:
            raise ValueError('Warning : the "maxplot" parameter prevent to a'
                             'large number of plot. To increase the number'
                             ' of plot, change "maxplot"')

    def _completeLabels(self, kwargs, L, *arg, default=''):
        """Function for completing title, xlabel, ylabel...
        """
        kwlst = list(kwargs.keys())
        kwval = list(kwargs.values())
        kwout = {}
        # For each arg:
        for k in arg:
            # If empty set to default :
            if k not in kwlst:
                kwout[k] = [default]*L
            else:
                val = kwargs[k]
                # If not empty and is string:
                if isinstance(val, str):
                    kwout[k] = [val]*L
                # If not empty and is string:
                elif isinstance(val, list):
                    # Check size:
                    if len(val) == L:
                        kwout[k] = val
                    else:
                        warn('The length of "'+k+'" must be '+str(L))
                        kwout[k] = [val[0]]*L
                # remove the key:
                kwargs.pop(k, None)

        return kwout, kwargs

# tilerplot.plot1D.__doc__ += _pltutils.__doc__
# tilerplot.plot2D.__doc__ += _pltutils.__doc__


class addLines(object):

    """Add vertical and horizontal lines to an existing plot.

    Args:
        ax: matplotlib axes
            The axes to add lines. USe for example plt.gca()

    Kargs:
        vLines: list, [def: []]
            Define vertical lines. vLines should be a list of int/float

        vColor: list of strings, [def: ['gray']]
            Control the color of the vertical lines. The length of the
            vColor list must be the same as the length of vLines

        vShape: list of strings, [def: ['--']]
            Control the shape of the vertical lines. The length of the
            vShape list must be the same as the length of vLines

        hLines: list, [def: []]
            Define horizontal lines. hLines should be a list of int/float

        hColor: list of strings, [def: ['black']]
            Control the color of the horizontal lines. The length of the
            hColor list must be the same as the length of hLines

        hShape: list of strings, [def: ['-']]
            Control the shape of the horizontal lines. The length of the
            hShape list must be the same as the length of hLines

    Return:
        The current axes

    Example:
        >>> # Create an empty plot:
        >>> plt.plot([])
        >>> plt.ylim([-1, 1]), plt.xlim([-10, 10])
        >>> addLines(plt.gca(), vLines=[0, -5, 5, -7, 7], vColor=['k', 'r', 'g', 'y', 'b'],
        >>>          vWidth=[5, 4, 3, 2, 1], vShape=['-', '-', '--', '-', '--'],
        >>>          hLines=[0, -0.5, 0.7], hColor=['k', 'r', 'g'], hWidth=[5, 4, 3],
        >>>          hShape=['-', '-', '--'])

    """

    def __init__(self, ax,
                 vLines=[], vColor=None, vShape=None, vWidth=None,
                 hLines=[], hColor=None, hWidth=None, hShape=None):
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

        return plt.gca()


class BorderPlot(_pltutils):

    """Plot a signal with it associated deviation. The function plot the
    mean of the signal, and the deviation (std) or standard error on the mean
    (sem) in transparency.

    Args:
        time: array/limit
            The time vector of the plot (len(time)=N)

        x: numpy array
            The signal to plot. One dimension of x must be the length of time
            N. The other dimension will be consider to define the deviation.
            For example, x.shape = (N, M)

    Kargs:
        y: numpy array, optional, [def: None]
            Label vector to separate the x signal in diffrent classes. The
            length of y must be M. If no y is specified, the deviation will be
            computed for the entire array x. If y is composed with integers
            Example: y = np.array([1,1,1,1,2,2,2,2]), the function will
            geneate as many curve as the number of unique classes in y. In this
            case, two curves are going to be considered.

        kind: string, optional, [def: 'sem']
            Choose between 'std' for standard deviation and 'sem', standard
            error on the mean (wich is: std(x)/sqrt(N-1))

        color: string or list of strings, optional
            Specify the color of each curve. The length of color must be the
            same as the length of unique classes in y.

        alpha: int/float, optional [def: 0.2]
            Control the transparency of the deviation.

        linewidth: int/float, optional, [def: 2]
            Control the width of the mean curve.

        legend: string or list of strings, optional, [def: '']
            Specify the label of each curve and generate a legend. The length
            of legend must be the same as the length of unique classes in y.

        ncol: integer, optional, [def: 1]
            Number of colums for the legend

        kwargs:
            Supplementar arguments to control each suplot:
            title, xlabel, ylabel (which can be list for each subplot)
            xlim, ylim, xticks, yticks, xticklabels, yticklabels, style.
    Return:
        The axes of the plot.
    """
    # __doc__ += _pltutils.__doc__

    def __init__(self, time, x, y=None, kind='sem', color='',
                 alpha=0.2, linewidth=2, legend='', ncol=1, **kwargs):
        pass

    def __new__(self, time, x, y=None, kind='sem', color='', alpha=0.2,
                linewidth=2, legend='', ncol=1, axes=None, **kwargs):

        self.xType = type(time[0])
        self.yType = type(x[0, 0])

        # Check arguments :
        if x.shape[1] == len(time):
            x = x.T
        npts, dev = x.shape
        if y is None:
            y = np.array([0]*dev)
        yClass = np.unique(y)
        nclass = len(yClass)
        if not color:
            color = ['darkblue', 'darkgreen', 'darkred',
                     'darkorange', 'purple', 'gold', 'dimgray', 'black']
        else:
            if type(color) is not list:
                color = [color]
            if len(color) is not nclass:
                color = color*nclass
        if not legend:
            legend = ['']*dev
        else:
            if type(legend) is not list:
                legend = [legend]
            if len(legend) is not nclass:
                legend = legend*nclass

        # For each class :
        for k in yClass:
            _BorderPlot(time, x[:, np.where(y == k)[0]], color[k], kind,
                        alpha, legend[k], linewidth, axes)
        ax = plt.gca()
        plt.axis('tight')

        _pltutils.__init__(self, ax, **kwargs)

        return plt.gca()


def _BorderPlot(time, x, color, kind, alpha, legend, linewidth, axes):
    npts, dev = x.shape
    # Get the deviation/sem :
    xStd = np.std(x, axis=1)
    if kind is 'sem':
        xStd = xStd/np.sqrt(npts-1)
    xMean = np.mean(x, 1)
    xLow, xHigh = xMean-xStd, xMean+xStd

    # Plot :
    if axes is None:
        axes = plt.gca()
    plt.sca(axes)
    ax = plt.plot(time, xMean, color=color, label=legend, linewidth=linewidth)
    plt.fill_between(time, xLow, xHigh, alpha=alpha, color=ax[0].get_color())
