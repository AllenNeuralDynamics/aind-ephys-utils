# CCG Implementation Review and Recommended Fixes

## Scope

This review covers the current `aind-ephys-utils` CCG implementation, with emphasis on correctness of lag counting and trial pairing, normalization behavior, trial-window handling, edge cases and input validation, performance and memory efficiency, and tests needed to make the implementation trustworthy.

The core sparse two-pointer CCG algorithm appears fundamentally sound for the common case of sorted point-process data, integer-multiple lag windows, and common trial windows. The more serious issues are in the surrounding trial-pairing, normalization, and wrapper logic.

## Verification status

Every finding below was re-checked against `src/aind_ephys_utils/metrics/ccg.py` at commit `4f81a07` by running counterexamples, not by reading alone. Each item carries a **Verification** block with the observed numbers.

Items 1-5 correspond to items 1-5 of the companion evaluation. **Item 6 is new** — it was found during verification and appears in neither original document. Items 7-18 are the original 6-17, shifted by one.

| # | Finding | Verdict |
|---|---------|---------|
| 1 | `ccg_trial_paired` ignores advertised normalization modes | Confirmed — also accepts invalid mode strings silently |
| 2 | Trial fast path keyed on duration, not alignment | Confirmed — spurious peaks at ±1.0 s in a case whose true answer is 0 |
| 3 | Variable-window `corrcoef` mixes clipped and unclipped counts | Confirmed |
| 4 | Non-integer `max_lag / bin_size` under-searches the outer bin | Confirmed — 30/30 brute-force mismatches at ratios 1.51, 1.6, 2.51 |
| 5 | `window=` clips for multi-trial but not single-trial input | Confirmed — 2 coincidences returned where 1 is correct |
| 6 | **Dense lower-triangle mirror is wrong for non-identity pairings** | **New — confirmed** |
| 7 | `corrcoef` is not bounded in `[-1, 1]` | Confirmed — 1.0533 from a two-spike construction |
| 8 | Explicit single-trial `pairs=` computes the Cartesian product | Confirmed — 10 requested pairs cost 100 CCGs |
| 9 | Explicit `(i, i)` ignores `exclude_zero_lag_autocorr` | Confirmed |
| 10-12 | Missing input validation; `clip_to_window` can expand | Confirmed by inspection |
| 13-18 | Performance and redundancy items | Confirmed by inspection; 18 (`rescale_ccgs_zero_mean`) is exactly redundant |

---

# Production exposure

Two facts bound how much any of this matters today.

**The analysis capsule does not import this copy.** `ccg-analysis-capsule/code/ccg_session.py` imports from `pl_inputs_analysis.ccg`, a near-duplicate module (the wrapper differs; the kernel and the entire trial-paired path are the same code). Every fix has to land in both copies, and the PL-inputs copy is the one that produced the existing results.

**The production configuration avoids most of the defects.** `NORMALIZE = "corrcoef"`, `BIN_SIZE = 0.0005`, `MAX_LAG = 0.250` — an exactly integer lag ratio of 500 — and the pipeline always runs `clip_spikes_to_trials` followed by `clip_to_window(trial_segs_raw, (-pre.min(), post.min()))`, which forces uniform `pre`/`post` and never expands beyond available data. That closes items 1, 2, 3, 4, 5 and 12 for the main analysis. The surrogate path passes an explicit `pairs=` array, so the dense mirror of item 6 is not exercised either.

What survives:

- **Item 7** (`corrcoef` definition) affects every reported number, because `corrcoef` is what production uses.
- **Item 9** (explicit `(i, i)` zero-lag).
- **Item 12** (`clip_to_window` expansion) has a live caller: the phase-stability block at `ccg_session.py:2567` calls `clip_to_window(trial_segs_raw, (p_lo, p_hi))` with fixed windows rather than the common minimum, so a trial whose support is narrower than `(p_lo, p_hi)` gets its duration overstated and its normalization corrupted.

These are latent API traps rather than grounds to distrust existing results — with the exception of item 7, which is a claim-versus-reality problem in numbers that already exist.

---

# Priority 0: Correctness defects

## 1. `ccg_trial_paired` silently ignores most advertised normalization modes

### Problem

`ccg_trial_paired` advertises support for:

- `"none"`
- `"counts"`
- `"rate"`
- `"conditional"`
- `"unbiased"`
- `"corrcoef"`

However, the implementation only treats `"corrcoef"` specially. Other modes effectively return the raw histogram.

This means that, for trial-paired data, `"none"`, `"counts"`, `"rate"`, `"conditional"`, and `"unbiased"` can all produce the same output. The high-level `ccg()` wrapper therefore also silently returns incorrect results for these normalization choices when operating on multi-trial data.

### Verification

Four trials, two units, identity pairing:

```text
none           max=4 sum=32
counts         max=4 sum=32
rate           max=4 sum=32
conditional    max=4 sum=32
unbiased       max=4 sum=32
corrcoef       max=0.0306564 sum=-0.0181814
bogus_mode     max=4 sum=32
```

Every mode except `corrcoef` is bit-identical to `none`. Note the last row: `normalize` is never validated in this path, so a typo silently returns raw counts.

### Recommended fix

**Decision: do not implement the missing modes — stop advertising them.** `ccg_trial_paired` and `ccg_trial_surrogates` should accept only `"none"` and `"corrcoef"` and raise `ValueError` on anything else, with docstrings narrowed to match.

Sharing a normalization dispatcher with the single-session path (the original recommendation) is the wrong shape here. `"rate"`, `"conditional"` and `"unbiased"` are defined against a single observation duration `T`; for trial-paired data they would need the per-pair total overlap duration and, for `"unbiased"`, a lag-dependent overlap summed across trial pairs. That is different mathematics, not a shared code path, and nothing in the pipeline asks for it.

### Required tests

- Each of `"counts"`, `"rate"`, `"conditional"`, `"unbiased"` and an invalid string raises `ValueError`.
- `"none"` returns raw counts.
- `"corrcoef"` matches a reference implementation.

---

## 2. Trial clipping fast path is incorrect for equal-duration but differently aligned trials

### Problem

The implementation uses a condition equivalent to:

```python
skip_clipping = all(ts.durations == ts.durations[0])
```

Equal trial duration does **not** imply identical alignment-relative support.

Example:

```text
trial A: pre=0.2, post=0.8
trial B: pre=0.8, post=0.2
```

Both trials last 1.0 s, but after pairing/shuffling their common alignment-relative overlap is only:

```text
[-0.2, +0.2]
```

If clipping is skipped solely because durations are equal, spikes outside the true overlap can contribute to the CCG. This can create entirely spurious peaks.

### Recommended fix

The fast path should only be taken when the alignment-relative trial windows are identical.

For example:

```python
skip_clipping = (
    np.all(ts.pre == ts.pre[0])
    and np.all(ts.post == ts.post[0])
)
```

Use an appropriate floating-point tolerance if these values are derived rather than exact. Equivalently, compare the actual alignment-relative left/right boundaries.

### Verification

Two trials, `durations = [1.0, 1.0]`, `pre = [0.2, 0.8]`, `post = [0.8, 0.2]`, with spikes placed only outside the shared overlap `[-0.2, +0.2]` (trial 0 at +0.5, trial 1 at -0.5):

```text
pairing=identity: total coincidences=2 at lags=[0.]      # correct
pairing=swapped:  total coincidences=2 at lags=[-1., 1.] # should be 0
```

Under the swapped pairing the overlap collapses to `[-0.2, +0.2]`, which contains no spikes at all, yet the kernel reports two coincidences a full second apart.

### Required regression test

Construct two equal-duration trials with different pre/post splits and spikes outside their shared overlap.

Expected result:

```text
0 coincidences
```

Test both identity pairing and shuffled/non-identity pairing.

---

## 3. Variable-window `corrcoef` normalization uses inconsistent spike counts

### Problem

For variable overlap windows, the CCG histogram is correctly computed using spikes clipped to each pair's actual overlap.

However, the expected-count / normalization terms are based on original full-trial spike counts, e.g. through values derived from:

```python
trial_counts = ts.counts
```

This creates an inconsistency:

```text
observed CCG:
    uses clipped spikes

expected / normalization:
    uses unclipped spikes
```

The problem is especially serious after trial shuffling, because the target trial must be evaluated under the overlap window defined by the particular source-target trial pairing.

A related issue exists for cached per-trial autocorrelation terms: an autocorrelation computed under trial `k`'s native overlap cannot simply be reindexed and reused for another pairing if the required overlap window differs.

### Recommended fix

For every paired trial:

1. determine the actual common alignment-relative interval,
2. compute or retrieve spike counts clipped to that interval for both units,
3. compute normalization terms from those clipped counts,
4. compute autocorrelation-related terms under the same effective interval.

If performance matters, precompute counts by unique overlap window rather than using the original untrimmed trial counts.

### Simplifying option

If supporting arbitrary variable trial windows is not essential, explicitly require trial data to be clipped to a common window before `corrcoef` normalization and validate that invariant.

### Verification

Two trials with `durations = [1.0, 1.5]` (so the fast path is off), `pre = [0.2, 0.8]`, `post = [0.8, 0.7]`, swapped pairing. Overlap resolves to `[-0.2, 0.7]` for both trials:

```text
trial 0: full counts n_i=3 n_j=3 | clipped n_i=3 n_j=2
trial 1: full counts n_i=3 n_j=3 | clipped n_i=2 n_j=3
```

The histogram sees the clipped counts; `trial_counts = ts.counts` feeds the expected-count term the unclipped 3s in both trials.

### Required tests

Compare against a deliberately slow reference implementation for variable pre/post windows, unequal trial durations, non-identity pairing, and spikes both inside and outside the resulting overlap.

---

## 4. Outer lag bins can be incompletely searched when `max_lag / bin_size` is non-integer

### Problem

The implementation determines the number of lag bins approximately as:

```python
half = round(max_lag / bin_size)
```

but the two-pointer search bound is based on the original `max_lag`, approximately:

```text
max_lag + bin_size / 2
```

When `round(max_lag / bin_size)` rounds upward, the returned outer bin can extend beyond the interval actually searched.

Example:

```text
bin_size = 1 ms
max_lag = 1.6 ms
```

The output lag centers become approximately:

```text
[-2, -1, 0, 1, 2] ms
```

The +2 ms bin spans approximately `[1.5, 2.5) ms`, but a search bound derived from `1.6 + 0.5 = 2.1 ms` fails to examine valid differences from roughly 2.1 to 2.5 ms.

### Recommended fix

Choose one of two designs.

#### Preferred: require exact lag-bin compatibility

Require `max_lag / bin_size` to be integer-valued within tolerance, and raise a clear error otherwise.

#### Alternative: derive the search window from the actual returned bins

If:

```python
half = round(max_lag / bin_size)
```

then derive the candidate search extent from:

```python
half * bin_size + bin_size / 2
```

rather than from the original `max_lag`.

### Verification

30 randomized spike-train pairs per ratio, compared against a brute-force `t_j - t_i` histogram using the same left-closed bin rule:

```text
ratio= 1.00 half=1 nbins=3:  0/30 mismatch
ratio= 1.20 half=1 nbins=3:  0/30 mismatch
ratio= 1.49 half=1 nbins=3:  0/30 mismatch
ratio= 1.51 half=2 nbins=5: 30/30 mismatch
ratio= 1.60 half=2 nbins=5: 30/30 mismatch
ratio= 2.00 half=2 nbins=5:  0/30 mismatch
ratio= 2.49 half=2 nbins=5:  0/30 mismatch
ratio= 2.51 half=3 nbins=7: 30/30 mismatch
```

Mismatches appear exactly when `round(max_lag / bin_size)` rounds up, as predicted. Both the positive and negative outer bins are truncated, since `j_start` also advances past spikes below `ti - window_margin`.

### Required property test

Compare against brute-force pairwise time differences over randomized spike trains for ratios including:

```text
1.0
1.2
1.49
1.51
1.6
2.0
2.49
2.51
```

The sparse kernel and brute-force histogram must agree exactly.

---

## 5. `window=` behaves differently for single-trial and multi-trial input

### Problem

For multi-trial input, the requested window is used to clip spikes.

For single-trial input, the spike arrays can remain untrimmed while the requested window is used only as an observation-duration / metadata value.

This can produce CCG events from spikes outside the requested window.

Example:

```text
unit A spikes: [0.1, 2.0]
unit B spikes: [0.105, 2.005]
window: (0, 1)
```

Only the first pair should contribute. If both pairs are counted, the wrapper is violating its own requested observation window.

This is particularly dangerous because output metadata can still report the requested window, making the result look correct.

### Recommended fix

Apply the same clipping semantics in the single-trial and multi-trial paths. The wrapper should either always clip spikes to `window`, or explicitly define `window` as metadata-only and never clip. The former is much less surprising.

### Verification

Unit A `[0.1, 2.0]`, unit B `[0.105, 2.005]`, `window=(0, 1)`:

```text
1 trial,  window=(0,1): coincidences=2   # wrong; attrs["window"] still reports (0.0, 1.0)
2 trials, window=(0,1): coincidences=1   # multi-trial path clips correctly
```

### Required regression test

Use spike events both inside and outside the requested single-trial window and assert that only in-window events contribute. Also verify parity between equivalent one-trial input and one-element multi-trial input.

---

## 6. Dense lower-triangle mirror is wrong for non-identity trial pairings

*Not in the original review — found while verifying the other items.*

### Problem

`ccg_trial_paired`, and `_scatter` inside `ccg_trial_surrogates`, fill the lower triangle by reversing the upper-triangle histogram:

```python
C[i, j, :] = h
if i != j:
    C[j, i, :] = h[::-1]
else:
    C[i, i, :] = 0.5 * (h + h[::-1])
```

`h[::-1]` is the exact `(j, i)` CCG only under the identity pairing. Under a pairing σ, `h` counts

```text
(unit i, trial k) x (unit j, trial σ(k))
```

whereas the true `(j, i)` CCG under σ counts

```text
(unit j, trial k) x (unit i, trial σ(k))
```

Reversing the first gives the CCG for **σ⁻¹**, not σ. The two coincide only when σ is an involution — which the identity is, and which a random derangement generally is not.

The diagonal has the same problem in a subtler form: `0.5 * (h + h[::-1])` averages the σ and σ⁻¹ autocorrelograms.

### Verification

Three units, six trials, σ = `[1 2 0 4 5 3]` (two 3-cycles, not an involution):

```text
dense C[1,0] == per-pair (1,0) under sigma      : False   (sums 53 vs 61)
dense C[1,0] == per-pair (1,0) under sigma^-1   : True
identity pairing: dense C[1,0] == per-pair (1,0): True
```

### Consequence

σ⁻¹ is a derangement whenever σ is, so each surrogate draw remains a valid sample from the null and a shuffle-corrected test is not biased. The defect is that the returned matrix is not the pairing the caller requested: the upper and lower triangles describe two different surrogates, and the caller has no way to know. Only the dense (`pairs=None`) path is affected; the per-pair path applies no mirror.

### Recommended fix

Pick one and state it in the docstring:

1. compute the `(j, i)` histogram explicitly when σ is not an involution;
2. leave the lower triangle unfilled for non-identity pairings; or
3. keep the mirror and document that the lower triangle carries the σ⁻¹ CCG.

Option 1 doubles the dense-path kernel work; option 2 is cheapest and hardest to misuse.

### Required regression test

With a non-involutive pairing (e.g. a 3-cycle), assert that `C[j, i]` from the dense path equals the per-pair `(j, i)` result for whichever pairing the chosen semantics promise. Assert exact mirror equality under the identity pairing.

---

# Priority 1: Statistical/API concerns

## 7. `"corrcoef"` should not currently be documented as bounded in `[-1, 1]`

### Problem

The current normalization can exceed 1 even for simple sparse examples.

A construction with one spike in each train can produce values slightly above 1, and other sparse examples can produce larger excursions.

The use of a denominator involving something like:

```python
sqrt(abs(auto_i * auto_j))
```

is also suspicious if the goal is a true Pearson-like correlation coefficient. Taking `abs()` may make the expression numerically defined, but it does not establish the mathematical properties of a correlation coefficient.

### Recommended action

Before changing the code, clarify the intended statistic. Determine whether this mode is meant to reproduce a published estimator, a Julia reference implementation, a normalized covariance-like CCG, or a true bounded correlation coefficient.

Then either rederive/fix the estimator so the stated properties hold, or rename/document it so it does not promise `[-1, 1]`.

### Verification

One spike in each train, 1-s observation window, centre bin:

```text
bin_size=0.05: 1.053324
bin_size=0.001: 1.001001
```

Worst |value| over 200 sparse random pairs (1-5 spikes each, 10-ms bins): **1.0111**.

Two mechanisms are at work. A kernel and edge-correction mismatch between numerator and denominator produces mild excursions that scale with `bin_size / duration` — invisible at production settings. Separately, the factor-of-2 auto term drives the denominator through zero at a firing rate of `1 / bin_size`, and `sqrt(abs(...))` hides the sign flip: worst `max|C|` over 400 random draws at 50 ms bins is **157**. The root cause is that the auto term reuses a one-sided pair count inside a two-sided correlogram: `C[0]` is not 1 even when the two trains are identical. Fully diagnosed in `corrcoef-auto-term.md`, which also covers the Julia sites.

`SpikeAnalysis.jl` lives in this monorepo, so fixtures for the Julia reference can be generated directly rather than reconstructed from the derivation.

### Required tests

If the statistic is intended to be a true correlation coefficient:

```python
np.nanmax(np.abs(ccg)) <= 1 + tolerance
```

must hold across broad randomized tests.

If it is intentionally unbounded, tests should instead lock it against the trusted reference implementation.

No test can be written until this is decided: asserting the bound today merely encodes the docstring's claim rather than a decision.

---

## 8. Explicit single-trial `pairs=` can do far more work than requested

### Problem

For explicit pairs, the single-trial wrapper can gather all unique source units and target units, compute the Cartesian product between those sets, and then select only the requested cells.

Example:

```text
500 requested one-to-one pairs
500 unique source units
500 unique target units
```

can lead to approximately 250,000 pair calculations instead of 500.

### Verification

Instrumenting `ccg_between_sets_sparse` while requesting 10 disjoint one-to-one pairs across 20 units:

```text
requested 10 pairs -> kernel invoked with 10x10 = 100 CCGs
```

Cost is quadratic in the length of the pair list.

### Recommended fix

Add a dedicated explicit-pair sparse kernel, e.g.:

```python
ccg_pairs_sparse(...)
```

that directly computes only the requested `(i, j)` pairs. Use the same explicit-pair path consistently for single-trial and trial-paired calls.

---

## 9. Explicit autocorrelation-pair behavior is inconsistent across paths

### Problem

When explicit pairs are routed through the between-set implementation, requested `(i, i)` pairs may not honor the same `exclude_zero_lag_autocorr` behavior as other code paths.

### Recommended fix

Centralize autocorrelation handling so every path applies the same zero-lag exclusion rule.

### Verification

An explicit `("a", "a")` pair, unit `a` holding three spikes:

```text
1-trial  exclude=True  zero-lag=3.0
1-trial  exclude=False zero-lag=3.0
2-trial  exclude=True  zero-lag=0.0
2-trial  exclude=False zero-lag=6.0
```

The single-trial path ignores the flag entirely because `ccg_between_sets_sparse` has no such parameter.

### Required test

For an explicit `(i, i)` pair:

- with exclusion enabled: zero-lag bin must be removed,
- with exclusion disabled: zero-lag self-count must be present.

Test both one-trial and trial-paired APIs.

---

# Priority 2: Input validation and invariant checking

## 10. Sorted spike times are assumed but should be validated at the correct boundary

The two-pointer algorithm requires sorted spike times. If public APIs accept arbitrary NumPy arrays, unsorted input should either raise a clear error or be sorted explicitly.

Prefer validation over implicit sorting if preserving upstream mistakes is useful for debugging.

---

## 11. Validate basic numeric parameters

Public CCG entry points should reject:

```text
bin_size <= 0
max_lag < 0
NaN / inf spike times
nonpositive trial durations
invalid trial bounds
alignment outside the corresponding trial
invalid pair indices
```

These checks are cheap relative to the CCG computation and make failures much easier to diagnose.

---

## 12. `clip_to_window()` should not silently expand beyond available data

### Problem

If the requested window is larger than the originally stored trial support, the function cannot reconstruct spikes that were previously clipped away. Nevertheless, it may be possible to produce metadata that claims the larger duration.

That creates incorrect downstream normalization.

### Verification

`clip_to_window` re-slices the existing CSR and then unconditionally sets `durations = right - left`, `pre = -left`, `post = right` for every trial, with no comparison against the incoming `ts.pre` / `ts.post`.

This has a live caller: `ccg_session.py:2567` passes fixed phase windows rather than the common minimum.

### Recommended fix

Require:

```text
requested window ⊆ original available window
```

for every trial. Otherwise raise an error.

---

# Performance and memory improvements

These are lower priority than the correctness issues above.

## 13. Avoid an intermediate histogram allocation for every pair

The session-wide pair kernel currently appears to allocate/return a histogram for each pair and then copy it into a preallocated output buffer.

If possible, call the accumulator form directly:

```python
_ccg_two_pointer_accum(..., out_buf[k])
```

This avoids one allocation/copy per pair. Expected benefit is modest but straightforward.

---

## 14. Make the CSR representation canonical for trial data

### Current pattern

Trial preprocessing creates many small arrays:

```text
unit × trial → individual NumPy arrays
```

and then builds a second flattened CSR representation. This means spike data and Python object overhead can effectively be stored twice.

For hundreds of units × hundreds/thousands of trials, the number of tiny ndarray objects can become substantial.

### Recommended design

Use the CSR-like representation as the canonical internal form:

```text
flat spike values
offsets / indptr
trial metadata
```

Only create per-unit/per-trial array views lazily when required by user-facing APIs or debugging.

Benefits:

- lower Python object overhead,
- less duplicated data,
- simpler Numba integration,
- more predictable memory behavior.

---

## 15. Remove output buffers that are never consumed

Review trial-kernel outputs such as:

```text
out_nspikes_i
out_nspikes_j
```

Confirmed: both are written by `_ccg_all_pairs_trials` and never read anywhere in the module. Similarly, `_CCGBuffers` always allocates `auto_per_trial` even when normalization does not need it, and the surrogate path allocates two of them.

---

## 16. Trial overlap duration can often be computed once per pairing

Overlap duration depends on the paired trials, not on the neuron pair.

If the same trial pairing is used for every neuron pair, compute overlap duration per trial pair once outside the unit-pair loop. Do not repeatedly accumulate the same duration for every neuron pair.

Confirmed: `out_dur[p] += od` runs inside the pair x trial loop and yields an identical total for every `p`, and the only consumer is the `total_dur <= 0` guard in the scatter loop.

---

## 17. Avoid unconditional `astype(np.float64)` copies

If spike arrays are already `float64`, prefer:

```python
np.asarray(x, dtype=np.float64)
```

or equivalent logic that avoids unnecessary copying.

---

## 18. `rescale_ccgs_zero_mean()` appears redundant

If the implementation is simply:

```text
subtract mean
then min-max scale
```

then the mean subtraction cancels algebraically under min-max scaling.

Confirmed: max absolute difference against `rescale_ccgs` is exactly `0` on random input. Remove it, or make it a documented alias.

---

# Possible larger algorithmic improvement

## Population-wide sweep for true all-pairs CCGs

The existing pairwise sparse algorithm is well matched to selected pairs, relatively sparse firing, and short lag windows.

For a full all-pairs CCG over a large population, however, each unit's spike list gets rescanned once for every partner.

A possible alternative is:

1. merge all spikes into one globally time-sorted stream,
2. retain a unit ID for each spike,
3. use a moving global time window of `±max_lag`,
4. enumerate only event pairs that actually fall within that temporal neighborhood,
5. accumulate into `(unit_i, unit_j, lag_bin)` histograms.

Roughly:

```text
O(total_spikes + nearby_spike_pairs)
```

rather than repeatedly scanning each spike train for each unit pair.

The difficult part is efficient parallel histogram accumulation, so this should be benchmarked rather than assumed better.

Recommendation:

- keep the current two-pointer implementation as the default for explicit/sparse pairs,
- investigate a population sweep only for large true-all-pairs workloads.

---

# Test strategy

The most important improvement after fixing the known issues is to add reference/property tests.

## A. Brute-force CCG reference

Implement a deliberately slow reference:

```python
for ti in spikes_i:
    for tj in spikes_j:
        dt = tj - ti
        if dt belongs to a lag bin:
            histogram[bin(dt)] += 1
```

Use this only in tests.

Randomly generate sorted spike trains and compare the optimized kernel exactly against it.

Cover:

- empty trains,
- single spikes,
- dense bursts,
- coincident spikes,
- spikes exactly at bin boundaries,
- positive and negative lags,
- autocorrelations,
- different bin sizes,
- non-integer `max_lag / bin_size`.

---

## B. Symmetry property

For cross-correlation:

```text
CCG(i, j, lag) == CCG(j, i, -lag)
```

Test this over randomized inputs.

---

## C. One-trial vs one-element multi-trial equivalence

Equivalent data represented as a single session and as one trial should produce the same result when the same window and normalization semantics are requested.

This should catch wrapper divergence.

---

## D. Trial-pair reference implementation

Create a deliberately simple Python implementation that, for every requested trial pair:

1. computes the exact common alignment-relative interval,
2. clips each spike train to that interval,
3. computes the brute-force CCG,
4. computes normalization from the clipped data,
5. sums/averages as defined.

Compare optimized trial code against this reference for randomized pre values, post values, trial durations, trial pairings, and spike counts.

---

## E. Test every advertised normalization mode

Do not rely only on "finite output" tests. For each mode a path actually supports, create a small dataset where the expected transformation is analytically obvious — and for each mode it does *not* support, assert that it raises. Per item 1 the trial-paired path should advertise only `"none"` and `"corrcoef"`, so its test matrix is two supported modes plus a rejection test covering `"counts"`, `"rate"`, `"conditional"`, `"unbiased"` and an invalid string.

---

## F. Corrcoef reference/property tests

First decide what `"corrcoef"` is intended to mean.

Then either:

### If reproducing an existing implementation

Store fixtures from the trusted reference and require close numerical agreement.

### If intended to be a true coefficient

Test mathematical properties including:

```text
finite when defined
symmetric under appropriate reversal
bounded by [-1, 1]
correct behavior for identical / independent trains
```

---

# Suggested implementation order

Items split into those with an obvious correct answer and those needing a semantic decision first. Tests for the first group can be written immediately; tests for the second cannot, because the assertion *is* the decision.

## Phase 1a — correctness, no judgement call required

- [ ] Reject unsupported and invalid normalization modes in `ccg_trial_paired` / `ccg_trial_surrogates` (item 1).
- [ ] Fix trial fast-path condition to require constant `pre` **and** constant `post`, not merely constant duration (item 2).
- [ ] Fix or reject non-integer `max_lag / bin_size` (item 4).
- [ ] Make single-trial `window=` actually clip spikes (item 5).
- [ ] Add public-input validation (items 10, 11).
- [ ] Prevent `clip_to_window()` from expanding beyond available data (item 12).
- [ ] Add regression tests for the above.

## Phase 1b — correctness, requires a decision first

Each of these is written up in `judgement-call-items.md` with the options, their costs, a recommendation, and the test that becomes writable once the call is made.

- [ ] **Item 3** — either recompute normalization from overlap-clipped counts, or require a common window for `corrcoef` and validate that invariant.
- [ ] **Item 6** — choose mirror semantics for non-identity pairings: compute `(j, i)` properly, leave the lower triangle unfilled, or document it as σ⁻¹.
- [ ] **Item 7** — decide what `"corrcoef"` promises before touching it: reproduce the Julia estimator, or rename/document so it does not claim `[-1, 1]`.
- [ ] **Item 9** — decide whether an explicit `(i, i)` pair is an autocorrelogram (honour the flag) or a literal cross-correlation of a train with itself (ignore it), then make all paths agree.

## Phase 2 — API consistency

- [ ] Add direct explicit-pair sparse computation (item 8).
- [ ] Apply every fix to `pl_inputs_analysis/ccg.py` as well — that is the copy the capsule imports.

## Phase 3 — performance

- [ ] Accumulate directly into preallocated pair histograms.
- [ ] Make CSR the canonical trial-data representation.
- [ ] Remove unused buffers.
- [ ] Precompute trial-pair overlap durations once.
- [ ] Avoid unnecessary dtype copies.
- [ ] Remove or clarify redundant rescaling functions.

## Phase 4 — optional algorithm experiments

- [ ] Benchmark a global population spike sweep for large all-pairs workloads.
- [ ] Compare against the current pairwise sparse implementation on realistic Neuropixels-scale data.

---

# Overall assessment

The core sparse two-pointer algorithm is a good foundation and should probably remain the primary implementation.

For the common case:

```text
sorted spike times
1 ms bins
max_lag exactly divisible by bin size
identical trial windows
raw counts / carefully validated normalization
```

the event-counting path appears strong.

The primary risk is not the sparse counting algorithm itself. It is that several higher-level features currently imply broader generality than the surrounding bookkeeping actually supports. Every defect found sits in that gap: each one is reachable only by using a documented parameter in a way the production pipeline happens never to use it.

That also means the existing results stand. The production configuration — integer lag ratio, `clip_to_window` to the common minimum, explicit `pairs=`, `corrcoef` — routes around items 1-6. Item 7 is the one finding that touches numbers already reported, and it is a question about what the statistic means rather than a coding error.

Recommended approach:

1. keep the sparse kernel,
2. tighten invariants,
3. unify normalization and clipping semantics,
4. add brute-force reference/property tests,
5. optimize only after those behaviors are locked down.
