# calibration/__init__.py

"""
Calibration subpackage.

Contains modules for:
  - master bias creation (mkmasterbias)
  - bias correction (bias_correction)
  - master dark creation (mkmasterdark)
  - dark correction (dark_correction)
  - master flats creation (mkmasterflats)
  - flat correction for science frames (flat_correction)
"""

from . import mkmasterbias
from . import bias_correction
from . import mkmasterdark
from . import dark_correction
from . import mkmasterflats
from . import flat_correction

__all__ = [
    "mkmasterbias",
    "bias_correction",
    "mkmasterdark",
    "dark_correction",
    "mkmasterflats",
    "flat_correction",
]
