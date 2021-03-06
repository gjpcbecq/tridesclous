from .version import version as __version__
from .dataio import DataIO
from .tools import *
from .peakdetector import *
from .waveformextractor import *
from .clustering import Clustering
from .peeler import Peeler
from .filter import SignalFilter

from .spikesorter import SpikeSorter

from .mpl_plot import *

try:
    from .gui import *
except ImportError:
    import logging
    logging.warning('INteractive GUI not availble')
    pass

