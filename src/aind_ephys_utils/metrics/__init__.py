"""Higher-level metrics for exploratory analysis.

The names re-exported here are the main entry points; the full surface
is reachable via the ``metrics.ripley`` submodule.
"""

from __future__ import annotations

from .latency import spike_latency
from .ripley import compare_ripley_l, ripley_k, ripley_k_envelope

__all__ = [
    "spike_latency",
    # ripley — main entry points
    "ripley_k",
    "ripley_k_envelope",
    "compare_ripley_l",
]
