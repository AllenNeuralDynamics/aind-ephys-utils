"""Higher-level metrics for exploratory analysis.

The names re-exported here are the main entry points; the full surface
(measurement helpers, buffers, etc.) is reachable via the submodules
``metrics.ccg`` and ``metrics.connectivity``.
"""

from __future__ import annotations

from .ccg import (
    NN_to_pair_vec,
    ccg,
    ccg_allpairs_sparse,
    ccg_between_sets_sparse,
    ccg_trial_paired,
    ccg_trial_surrogates,
    clip_spikes_to_trials,
    measure_fwhm,
    measure_prominence,
    monte_carlo_pvalue,
    pair_vec_to_NN,
    rescale_ccgs,
    smooth_ccgs,
)
from .connectivity import (
    build_shape_prefilter,
    run_surrogates,
    run_two_stage_mc,
)
from .latency import spike_latency

__all__ = [
    "spike_latency",
    # ccg — main entry points
    "ccg",
    "ccg_allpairs_sparse",
    "ccg_between_sets_sparse",
    "clip_spikes_to_trials",
    "ccg_trial_paired",
    "ccg_trial_surrogates",
    "monte_carlo_pvalue",
    "measure_fwhm",
    "measure_prominence",
    "smooth_ccgs",
    "rescale_ccgs",
    "pair_vec_to_NN",
    "NN_to_pair_vec",
    # connectivity — main entry points
    "build_shape_prefilter",
    "run_surrogates",
    "run_two_stage_mc",
]
