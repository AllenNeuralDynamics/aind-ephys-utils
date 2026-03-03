"""Pure, testable operations used by `.ephys` accessors.

Functions in this package operate on xarray objects and should avoid side
effects.
"""

from .align import align
from .baseline import baseline
from .bin import bin
from .normalize import normalize
from .pseudopop import pseudopop
from .psth import psth
from .reduce import reduce
from .restrict import restrict
from .smooth import smooth

__all__ = [
    "align",
    "baseline",
    "bin",
    "normalize",
    "psth",
    "pseudopop",
    "reduce",
    "restrict",
    "smooth",
]
