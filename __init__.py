# from brainpipe.classification import classification_tools
# from brainpipe.classification.multifeatures import multifeatures_tools, sequence

# print('ok chef')
# from brainpipe.stat import classification_stat
# from brainpipe.vizualisation import classification_plot#, tools_plot
# from brainpipe.physiology import brainstructure
# from brainpipe.commonth.braindata import *

# from brainpipe.feature.CfcFeat import *
# from brainpipe.feature.Power import *
# from brainpipe.feature.Phase import *
# from brainpipe.feature.brainfir import *
# from brainpipe.feature.loadfeatures import *
# from brainpipe.feature.normalization import *

# from brainpipe.system.tools import *
# from brainpipe.vizualisation.tools_plot import *


# __all__ = [
#     'CrossFrequencyCoupling'
# ]

from brainpipe.xPOO.feature import phase, power
from brainpipe.xPOO.bpstudy import study
from brainpipe.xPOO.visualization.cmon_plt import addLines, BorderPlot

__repr__ = 'brainpipe'