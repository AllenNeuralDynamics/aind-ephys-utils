"""Higher-level metrics for exploratory analysis.

The names re-exported here are the main entry points; the full surface
(measurement helpers, buffers, etc.) is reachable via the submodules
``metrics.ccg`` and ``metrics.connectivity``.
"""

from __future__ import annotations

from .ccg import (
    CCGCounts,
    NN_to_pair_vec,
    TrialSegments,
    ccg,
    ccg_allpairs_sparse,
    ccg_between_sets_sparse,
    ccg_trial_paired,
    ccg_trial_surrogates,
    clip_spikes_to_trials,
    clip_to_window,
    compute_ccg_counts,
    covariance_density,
    cross_intensity,
    directional_excess,
    excess_density,
    fold_over_baseline,
    jitter_spikes_times,
    legacy_auto_normalized,
    measure_fwhm,
    measure_prominence,
    monte_carlo_pvalue,
    normalized_covariance,
    pair_correlation,
    pair_vec_to_NN,
    peak_contributing_spikes,
    rescale_ccgs,
    smooth_ccgs,
    to_dense,
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
    "TrialSegments",
    "clip_spikes_to_trials",
    "clip_to_window",
    "ccg_trial_paired",
    "ccg_trial_surrogates",
    "monte_carlo_pvalue",
    "measure_fwhm",
    "measure_prominence",
    "peak_contributing_spikes",
    "jitter_spikes_times",
    "smooth_ccgs",
    "rescale_ccgs",
    "CCGCounts",
    "compute_ccg_counts",
    "cross_intensity",
    "covariance_density",
    "normalized_covariance",
    "pair_correlation",
    "excess_density",
    "fold_over_baseline",
    "legacy_auto_normalized",
    "directional_excess",
    "to_dense",
    "pair_vec_to_NN",
    "NN_to_pair_vec",
    # connectivity — main entry points
    "build_shape_prefilter",
    "run_surrogates",
    "run_two_stage_mc",
]
