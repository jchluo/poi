# -*- coding: utf-8 -*-

"""
POI library
~~~~~~~~~~~~~~~~~~~~~
usage:

"""

__title__ = 'poi recommender lib'
__version__ = '1.0.0'
__author__ = 'jchluo'

# load data
from .loader import load_checkins, load_locations 
# models
from .wmf import WMF
from .bpr import BPR
from .userbase import UserBase
from .kde import distance 
from .powerlaw import PowerLaw
#eval
from .models import Evaluation

# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())
