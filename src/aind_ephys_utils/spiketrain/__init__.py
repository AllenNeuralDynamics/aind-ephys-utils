"""Raw-array spike-train transforms and utilities.

Functions in this subpackage operate directly on numpy arrays and lists
of spike-time arrays (no xarray coupling).  The names re-exported here
are the main entry points; the full surface is reachable via the
submodules (``spiketrain.epochs``, ``spiketrain.spike_utils``).
"""

from __future__ import annotations

from .epochs import (
    add_random_offset_with_wrap,
    concat_event_windows,
    concat_spikes_in_epochs,
)
from .spike_utils import (
    align_to_events,
    align_to_events_exclude_trigger,
    build_synchrony_index,
    count_coincident_per_train,
    count_coincident_units,
    exclude_near,
    near_events,
)

__all__ = [
    # epochs
    "concat_spikes_in_epochs",
    "concat_event_windows",
    "add_random_offset_with_wrap",
    # spike_utils
    "exclude_near",
    "near_events",
    "count_coincident_per_train",
    "build_synchrony_index",
    "count_coincident_units",
    "align_to_events",
    "align_to_events_exclude_trigger",
]
