# This file is a part of the `ewald-allegro` package. Please see LICENSE and README at the root for information on using it.
from ._version import __version__
from . import _compile
from . import _extern

# Re-export the model for easy access
from .model import AllegroModel
from .model.ewald_allegro_v2 import EwaldAllegroModelV2

__all__ = ["__version__", "_compile", "_extern", "AllegroModel", "EwaldAllegroModelV2"]
