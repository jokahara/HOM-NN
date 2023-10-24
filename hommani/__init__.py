# -*- coding: utf-8 -*-

#from .utils import EnergyShifter
#from .nn import ANIModel, Ensemble, SpeciesConverter
from .ani_model import CustomAniNet
from . import ani_model
from . import cross_validate
from . import datasets
from . import ensemble
from pkg_resources import get_distribution, DistributionNotFound
#import warnings

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

__all__ = ['CustomAniNet', 'ani_model', 'cross_validate', 'datasets', 'ensemble']
