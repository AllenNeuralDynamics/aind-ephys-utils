"""Naming conventions used by `aind_ephys_utils`.

This module centralizes canonical dimension names, coordinate names, and
attribute keys used across the library.

Keeping conventions stable makes it easier to write composable analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ----------------------------
# Canonical dimension names
# ----------------------------
TRIAL: Final[str] = "trial"
UNIT: Final[str] = "unit"
TIME: Final[str] = "time"
CHANNEL: Final[str] = "channel"
SPIKE: Final[str] = "spike"  # only for explicit spike-dimension formats (rare)


# ----------------------------
# Common coordinate names
# ----------------------------
EVENT: Final[str] = "event"  # events dimension in an events object
EVENT_TIME_VAR: Final[str] = "t"  # events["t"] = time per (trial, event)

# ----------------------------
# Default column names
# ----------------------------
SPIKE_TIMES_COL: Final[str] = "spike_times"
TRIAL_START_COL: Final[str] = "start_time"
TRIAL_END_COL: Final[str] = "end_time"

# ----------------------------
# Standard attrs keys
# ----------------------------
ATTR_TIME_UNIT: Final[str] = "ephys.time_unit"  # e.g. "s"
ATTR_TIMEBASE: Final[str] = "ephys.timebase"  # e.g. "session" or "trial"
ATTR_KIND: Final[str] = (
    "ephys.kind"  # "spikes_ragged" | "continuous" | "binned"
)
ATTR_HISTORY: Final[str] = "ephys.history"  # list[str]
ATTR_VALID_INTERVALS: Final[str] = "ephys.valid_intervals"  # list[tuple[float,float]]


@dataclass(frozen=True)
class Conventions:
    """
    Central place to define naming conventions. Make this stable and boring.
    """

    trial: str = TRIAL
    unit: str = UNIT
    time: str = TIME
    channel: str = CHANNEL

    event: str = EVENT
    event_time_var: str = EVENT_TIME_VAR

    spike_times_col: str = SPIKE_TIMES_COL
    trial_start_col: str = TRIAL_START_COL
    trial_end_col: str = TRIAL_END_COL

    attr_time_unit: str = ATTR_TIME_UNIT
    attr_timebase: str = ATTR_TIMEBASE
    attr_kind: str = ATTR_KIND
    attr_history: str = ATTR_HISTORY
    attr_valid_intervals: str = ATTR_VALID_INTERVALS

    default_time_unit: str = "s"


C = Conventions()
