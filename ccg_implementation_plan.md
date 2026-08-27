# CCG normalization: implementation plan

Consolidates `ccg_implementation_review.md`, `judgement-call-items.md`,
`corrcoef-auto-term.md`, and the four design documents in `~/Downloads`.
Supersedes the recommendations in those documents where they differ.

Line references are `aind-ephys-utils/src/aind_ephys_utils/metrics/ccg.py`
unless stated. `PL-inputs-analysis/src/pl_inputs_analysis/ccg.py` is a
near-duplicate carrying every defect below; it is the copy
`ccg-analysis-capsule` imports.

---

## Decisions already settled

| Question | Decision |
|---|---|
| Exposure convention | **(a) clip to the shared interval.** Guarantees lag-symmetric exposure; (b) makes surrogates directionally biased. |
| Trial-paired normalization modes | Reject anything but `none` / the corrcoef family; do not implement `rate`/`conditional`/`unbiased` there. |
| `[-1, 1]` claim | Drop it. The estimand is a density; the bound is a category error. |
| Pearson / kernel correlation | Defer. Different estimand, no current requirement. |
| Primary effect size | Integrated directional excess `eps(W)`; density normalizations are shape statistics. |
| Surrogate inference | Raw counts **minus the pairing-dependent `E`**, compared against an unrestricted derangement null. Constrained shuffles are a secondary tool, not the remedy for slow drift. |
| `corrcoef` name | Deprecate → `legacy_auto_normalized`, alias retained one release. |
| Baseline naming | `conditional_uniform` for the current per-trial form; reserve `stationary` for the global `lam_i lam_j Q_b`. |
| Baseline default | Explicit in the library; `shift_predictor` at the pipeline level only. |
| Dense lower triangle under non-identity pairings | **NaN.** The dense `(N,N,B)` view is a convenience wrapper; under a general sigma it cannot fill the lower triangle without a second kernel pass. Callers needing the other direction use the pair-major result plus the flip rule below. |
| Pair-major returns | Extend to `ccg_allpairs_sparse` and `ccg_between_sets_sparse`, not just the trial path. |
| Self-pairs / ACGs | **Membership and display are separate concerns.** Self-pairs are `(i,i)` entries in the pair list; the zero-lag blanking is a projection policy, never applied in the compute path. |
| `include_self` default | **False.** |
| Legacy auto-term fix timing | **Phase 5**, bundled with the rename, so numbers move and fixtures regenerate once. The identical-trains test lands in Phase 1 as `expectedFailure` so the defect is recorded meanwhile. |

## The flip rule

Verified exactly, for raw counts and through the corrcoef normalization, and
under heterogeneous trial windows:

```
h_ji^sigma  ==  reverse( h_ij^{sigma^-1} )         NOT reverse( h_ij^sigma )
```

The naive mirror is off by 6 counts / 0.039 corrcoef on a non-involutive
pairing. Identity is the special case where `sigma^-1 == sigma`, which is why
the current mirror is correct there and only there. `E_ji^sigma ==
E_ij^{sigma^-1}` falls out of the same reindexing, and the exposure sum
`sum_k |W_k n W_sigma(k)|` is invariant under `sigma -> sigma^-1`.

**Consequence:** the kernel is free to compute whichever direction of a pair it
prefers, provided it records the direction and the sanctioned flip
`(i, j, sigma) -> (j, i, sigma^-1)` plus lag reversal is the only way to obtain
the other. Inverting the pairing is one `argsort`, `O(n_trials)` **per
pairing**, not per pair; for identity pairings it is free.

Why this matters: the CPU two-pointer is `O(n1 + n2 + K)` — nearly symmetric,
measured 2.30 ms vs 2.44 ms at a 200:1 length asymmetry, so ~6%. But a CUDA
kernel with a thread per source spike and a `searchsorted` into the target is
`O(n_source * log n_target + K)`, which is strongly asymmetric:

```
2k vs 400k spikes   short as source ~37k   long as source ~4.4M   118x
5k vs  50k spikes   short as source ~78k   long as source ~615k   7.9x
```

`K` and therefore atomic traffic are direction-independent, so the entire
asymmetry lands in search and launch overhead. Canonicalizing on "shorter unit
leads" is worth real money there. The flip rule is what makes that safe.

## Self-pairs and ACGs

### The kernel does not need to know

Verified: `_ccg_two_pointer(u, u, ...)` needs no special casing — it produces an
exactly symmetric histogram, centre bin `n` self-pairs plus near pairs. The two
`i == j` branches in the scatter are:

1. **zero-lag blanking** — a display policy;
2. **`0.5 * (h + h[::-1])`** — measured a genuine **no-op** under identity
   pairing, and under a 3-cycle it silently averages `h^sigma` with
   `h^{sigma^-1}` (max deviation 13 counts). That is the same sigma/sigma-inverse
   conflation as the lower-triangle mirror, appearing on the diagonal, and it
   gets the same treatment: `to_dense` owns it, NaN for non-involutive pairings.

So membership is sufficient for correctness. The kernel needs to know a pair is
a self-pair only to take the symmetry optimization below.

### Blanking is a display policy

The flag currently means three different things:

| path | `(i,i)` zero lag with the flag set |
|---|---|
| dense diagonal, session (`_scatter_and_normalize`) | `0` — blanks *after* subtracting `E` |
| dense diagonal, trial (`ccg_trial_paired`) | `-E/denom` — blanks *before* |
| explicit `pairs=[(i,i)]`, single trial | unblanked (`n`) |

Measured on identical data: `+0.000000` vs `-0.139134`. All three follow from
applying the blank inside the compute path at whichever point each
implementation chose. `CCGCounts` should hold raw counts always; blanking
happens at projection, after normalization, where it has one meaning. Rename
`exclude_zero_lag_autocorr` to a `zero_lag=` option on the projection — the
current name bakes "autocorr" into what is a self-pair display policy.

### `include_self=False`

Two arguments, the second decisive:

- **Aggregate-statistics hazard.** Anything reducing over the returned pair set
  — FDR, a peak-value histogram, a ranking — silently includes `N` self-pairs
  with real refractory and burst structure that is not connectivity.
  `ccg_session.py:600` already builds `for j in range(i+1, N)`, excluding the
  diagonal by hand.
- **Different parameters.** In the pipeline the two objects are computed at
  `BIN_SIZE=0.0005 / MAX_LAG=0.250` and `ACG_BIN_SIZE=0.001 / ACG_MAX_LAG=1.0`.
  A shared sweep would give ACGs at ±250 ms with 0.5 ms bins — too narrow for
  the lick-rate structure the wide window exists for. Bundling is not merely
  unnecessary, it is unusable.

### `acg()` and the symmetry optimization

Self-pairs are symmetric, so `acg()` computes only lags >= 0 with a
forward-only sweep — about 2x cheaper. Julia already does this in
`acorr_discrete_validonly` (`edges = 0:binsize:maxdiff`). Mirror into the
full-length buffer immediately rather than carrying variable-length entries:
the counting saving is kept and the output shape stays uniform.

Free invariant for the test suite: `acg(u)` and `(i,i)` routed through the
general pair path must agree bin for bin.

### The ACG is diagnostic, not inferential

It does not change whether a CCG feature is significant — the surrogate handles
that, and trial shuffling preserves each unit's within-trial autocorrelation, so
burst-driven chance coincidence is already in the null. It changes what a
significant feature *means*. The cases where the CCG alone is ambiguous:

1. **shared rhythm** — lick rate or theta entrainment produces CCG side-lobes
   that are a product of two rate oscillations, not coupling;
2. **oversplitting** — sharp zero-lag peak plus an ACG with no refractory dip;
3. **burst replication** — a bursty source replicates one synaptic event at the
   burst ISI;
4. **dip interpretation** — a trough after a peak may be target refractoriness
   rather than inhibition.

All four apply to the pairs that survive screening, not the ~10^5 swept. So
ACGs belong on-demand in the diagnostic panel, which is what
`ccg_session.py:2393` already does.

### Terminology

Keep CCG / ACG. "Correlogram" is a term of art for the difference histogram
(Perkel, Gerstein & Moore, 1967) and names the *object*, not a statistic. What
overclaimed was `corrcoef`, the normalization name. The object keeps its
conventional name; the statistics get precise ones.

---

## Trial-pair support contract

**Contract:** within every compared source-target observation pair, the two
observations must have matching support. Different pairs may have different
support. Arbitrary unequal-support comparisons stay out of scope.

**The condition is support equality, not duration equality:**

```
pre_k == pre_sigma(k)   AND   post_k == post_sigma(k)
```

`d_k == d_sigma(k)` is **not** sufficient — `[-0.2, 0.8]` and `[-0.8, 0.2]`
both have `d = 1.0` and share only `[-0.2, +0.2]`. That is review item 2's bug
restated at the contract level; keep the two spellings distinct.

### This reduces scope in three places

1. **Item 3 collapses to a validation.** For the identity pairing source and
   target are the same trial, so support matches trivially and the overlap clip
   is a no-op. For a non-identity pairing the contract requires matching
   support, so the clip is *again* a no-op. The clipped-vs-unclipped count
   mismatch therefore cannot arise for legal input. Item 3 becomes "validate
   the contract at entry and raise", not "rebuild `auto_sum_paired` from
   clipped counts."
2. **Exposure caches by scalar duration.** Each pair's exposure is
   `Q_b = int_bin (d_k - |tau|)_+ dtau`, symmetric, fixed by that trial's own
   support. Cache by unique `d`; no four-parameter geometry cache.
3. **Exposure invariance is automatic, not a check.** Trial `k` may only pair
   with trials of matching support, so pair `k`'s exposure is fixed by `k`
   regardless of partner and `sum_k Q_{d_k}` is identical under every allowed
   `sigma`. The raw-count/Welford optimization is therefore always valid under
   the contract — this replaces the earlier global-common-window precondition.

**Practical note.** Trial durations in the pipeline vary continuously
(`ccg_session.py` prints a duration *range*), so no two raw trials share exact
support and every non-identity pairing would be illegal before
`clip_to_window`. The support-class machinery is correct and worth building,
but this data will only ever exercise the single-class case. Do not
over-engineer it.

---

## Surrogate shuffle modes

Global permutation preserves each unit's within-trial structure and repeatable
trial-locked modulation, but destroys **slow shared across-trial** state —
drift, arousal, engagement, satiety. If both units co-vary slowly across
trials, the observed same-trial CCG exceeds a globally shuffled surrogate with
no within-trial coupling at all.

Mechanically this is the `E[b]` term: expected counts go as
`sum_k n_i(k) n_j(sigma(k))`. Identity gives the positively-correlated sum, a
global shuffle the uncorrelated one, and the gap *is* the across-trial rate
covariance.

Measured — two units independent within every trial, sharing only a slow drift
(period 40 trials, across-trial count correlation 0.796):

```
global derangement   mean(obs - surr) = +80.46 counts/bin   mean z = +3.50
block shuffle (20)   mean(obs - surr) = +16.00 counts/bin   mean z = +0.73
block shuffle (10)   mean(obs - surr) = +17.89 counts/bin   mean z = +0.87
```

A z of +3.5 for an uncoupled pair. Blockwise cuts it ~5x; the residual +16
supports describing this as **reducing sensitivity to** slow non-stationarity,
never **correcting** it.

### Modes to provide

All modes intersect with support compatibility, and optionally with an
experimental stratum. **Nothing is removed** — `global` stays supported and
stays the default.

| mode | definition | notes |
|---|---|---|
| `global` | any support-compatible derangement | current behaviour; the default |
| `circular_offset` | `sigma(k) = k + delta`, `delta != 0`, within a compatible block | one-to-one by construction, `sigma^-1` is offset `-delta`, composes directly with the flip rule. Implement first. |
| `within_block` | permute within contiguous blocks of size `B` | block size defines the preserved timescale |
| `local` | `|sigma(k) - k| <= L`, `sigma(k) != k` | must still be a valid derangement, not per-trial sampling. Defer. |

`np.roll(identity, offset)` — already used for the display shift predictor at
`ccg_session.py:646` — is exactly `circular_offset`.

### Provenance

The shuffle scheme is part of the scientific definition of the null and must be
recorded on the result and reported in Methods:

```
surrogate_method = "trial_permutation"
shuffle_scope    = "within_block"
block_size       = 20
support_stratified = true
```

---

## Phase 1 — correctness, no API change

Everything here is behaviour-preserving except where a current behaviour is
wrong. No new public names.

### 1a. Bookkeeping fixes (tests already written and red)

`tests/test_ccg_regression.py` has 20 tests, 29 currently-failing assertions
covering exactly these.

| # | Fix | Site |
|---|---|---|
| 1 | Raise on unsupported/invalid `normalize` in the trial paths | `ccg_trial_paired:1255`, `ccg_trial_surrogates:1563` |
| 2 | `skip_clipping` must require constant `pre` **and** `post`, not equal durations | `:1390`, `:1624`; also `uniform_ol` at `:1665` |
| 4 | Derive the two-pointer search margin from `half*bin_size`, or reject non-integer `max_lag/bin_size` | `_ccg_two_pointer_accum` |
| 5 | Single-trial `window=` must clip spikes | `_spike_list_for_units:745` |
| 10-12 | Validate sortedness, `bin_size > 0`, `max_lag >= 0`, finite spike times; `clip_to_window` must reject expansion | `clip_spikes_to_trials:1077`, `clip_to_window:1126`, public entry points |

Item 2 is the same defect the exposure design doc warns about in its §7
`[-0.2, 0.8]` / `[-0.8, 0.2]` example — the fast path is taken in exactly the
case that breaks it.

**Exit criterion:** `tests/test_ccg_regression.py` green.

### 1b. Fixes needing new tests

| # | Fix |
|---|---|
| 3 | **Validate the support contract at entry and raise on violation.** Under the contract the overlap clip is a no-op, so the clipped-vs-unclipped count mismatch cannot arise — this replaces the earlier plan of rebuilding `auto_sum_paired` from clipped counts. Keep the clip as an asserted no-op or delete it. |
| 6 | Interim NaN guard on the dense lower triangle for non-identity pairings, and on the diagonal's `0.5*(h+h[::-1])`. Subsumed by `to_dense` in Phase 2b. |
| 9 | Move zero-lag blanking out of the compute path; `CCGCounts` holds raw counts. Removes all three inconsistent meanings at once. |
| — | Add a common-window guard at the **surrogate** entry point: raw-count surrogate inference is only valid when exposure is pairing-independent. |

Item 3 is the one with a measured statistical consequence: on variable
windows it leaves the observed CCG exact while biasing every surrogate
downward, shifting the median null p-value from 0.652 to 0.100.

### 1c. Legacy auto term — deferred to Phase 5

**Decision: land the fix in Phase 5 with the rename, not here.** It changes
nothing downstream (see below), so there is no cost to waiting, and bundling it
with the rename means reported numbers move and cross-language fixtures
regenerate exactly once.

Land the identical-trains regression test **now**, marked
`@unittest.expectedFailure` with a pointer to this plan, so the defect is
recorded in the suite and an accidental fix shows up as an unexpected success.

The specification below is what Phase 5 implements.

Replace the one-sided width-`w` count plus doubled expectation with the
autocorrelogram's own two-sided centre bin:

```
A_u = (2 * count_auto_first(u, w/2) - n) - ec_shape[centre] * n^2
```

Free: same helper, halved argument; and on paths that compute the diagonal the
raw value is already in `out_buf[centre]`.

| Repo | Sites |
|---|---|
| aind-ephys-utils | `coeff = 2` at `:349`; `out_auto[i,k] = basecount - 2.0*ec` at `:998` |
| PL-inputs-analysis | same two sites |
| SpikeAnalysis.jl | `sp_corrs.jl:151-154` and `:515-518` pass `auto=false`. **Leave `acorr_discrete_validonly:438-448` alone — it is already correct.** |

Also replace `sqrt(abs(A_u*A_v))` with a raise (or NaN) on `A <= 0`. Julia's
bare `sqrt` already throws; the `abs()` is a port-side deviation.

**Also add a Julia guard**: `xcorr_discrete_normed` takes one `durs[subno]` and
has no overlap machinery, so a permuted `vs` with heterogeneous durations is
currently silently wrong. Validate common support or document the requirement.

**Regression test (both languages):** two distinct units with identical spike
times give `C[0] == 1` exactly. Parameterize over bin size and over Poisson /
regular / bursty / refractory trains — the candidate fixes disagree in
different regimes.

**Breaks `crossval_julia_python.py`.** Regenerate fixtures after both languages
land.

### 1d. Cleanups (independent, any time)

- delete `rescale_ccgs_zero_mean:862` — algebraically identical to `rescale_ccgs`
- remove `out_nspikes_i` / `out_nspikes_j` — written, never read
- hoist `out_dur` accumulation out of the pair loop — identical for every pair
- allocate `auto_per_trial` only when the normalization needs it

---

## Phase 2 — result object and deterministic transforms

No behaviour change; new API alongside the old.

1. **Name and factor the exposure.** `_expected_counts_shape:363` already
   computes `Q_b / T^2` exactly — verified to ~1e-16 against numerical
   integration, side bins and centre bin. Expose `Q_b` as a named quantity
   computed once per lag geometry. **This is a rename and factoring, not a
   rewrite.**

   Under convention (a), exposure is parameterized by the scalar overlap
   duration per trial pair, so cache by unique `d` — the four-parameter
   geometry cache the design doc proposes is not needed.

2. **`CCGCounts` result object**: `counts` (pair-major int64), `expected`,
   `lags`, `exposure` (when a single geometry applies), pair indices,
   `pairing`, observation metadata. Do not require a universal `exposure[lag]`
   field.

2b. **Dense projection is a separate API**, not something the compute functions
   do inline. Compute and transforms stay pair-major end to end:

```
counts = compute_ccg_counts(...)             # pair-major, always
values = normalized_covariance(counts)       # pair-major transform
dense  = to_dense(values, counts, fill=nan)  # explicit, opt-in projection
```

   `to_dense` owns the mirror policy in one place: it reads `counts.pairing`,
   mirrors via the flip rule when the pairing is an involution, and fills
   `NaN` otherwise. `ccg_allpairs_sparse` and `ccg_between_sets_sparse` keep
   their current signatures as `to_dense(compute(...))` shims.

   Precedent exists: `pair_vec_to_NN` / `NN_to_pair_vec` (`:2370`, `:2417`)
   already do this projection for per-pair *scalars*, and are exported.

   **Note a latent trap.** `pair_vec_to_NN` defaults `mirror=True`, and
   `connectivity.py:420` depends on it for the p-value matrix. That is correct
   today only because the reduction is `np.max(C, axis=-1)` over all lags and
   `max(reverse(x)) == max(x)`. A directional statistic — max over positive
   lags, the natural `i -> j` test — would make the mirror silently wrong in
   the same sigma-inverse way. Fold this helper under the same policy owner and
   make the mirror validity a stated precondition rather than a default.

   Benefits: the 2 GB `(N,N,B)` materialisation never happens unless asked for;
   the mirror rule has exactly one implementation and one docstring; the
   transforms stay fused and pair-major as the implementation design calls for;
   and the NaN decision becomes a projection policy rather than a wart in the
   compute path.

3. **Transforms** as fused pair-major passes with `out=`:
   `cross_intensity`, `covariance_density`, `normalized_covariance`,
   `pair_correlation`, `excess_density`, `fold_over_baseline`,
   `directional_excess(W)`, `legacy_auto_normalized`.

   Much of the target architecture already holds: the kernel returns integer
   counts, `out_buf` is already pair-major, auto terms are already per-unit,
   and `_CCGBuffers` already does buffer reuse.

4. **`directional_excess(W)` is the headline number** —
   `sum_{b in W}(H_b - E_b)/n_i`, bin-invariant (0.2989 / 0.2984 / 0.2995 /
   0.2973 across an 8x bin range against a true 0.30). Report `W` alongside it.
   Peak-height density statistics are resolution-dependent and belong to shape,
   not magnitude.

---

## Phase 3 — baseline layer

Separate `baseline=` from `statistic=`.

```
baseline = "conditional_uniform"   # the current per-trial form -- keep it
baseline = "stationary"            # global lam_i lam_j Q_b
baseline = "shift_predictor"
baseline = "surrogate_mean"
baseline = <supplied expected-count array>
```

Keep the per-trial `E_b = s_b * sum_k n_i(k) n_j(sigma(k))` form. It conditions
on per-trial counts and is strictly better than a global rate model for
task-aligned data. `stationary` is a *reference*, not a correction: on aligned
trials it removes mean rate but not shared task modulation, and the difference
is large — a `+0.088` mean offset for units with no pairwise relationship,
against a `+0.027` residual.

Library default explicit; `shift_predictor` default only in the connectivity
pipeline.

---

## Phase 4 — surrogate refactor

Move surrogate inference to raw-count space with online statistics.

- Do not normalize each surrogate. `_scatter` currently does.
- Welford per `(pair, lag)`, or reduce each shuffle to a scalar window
  statistic immediately and keep only the scalar null.
- **Keep the analytic `E`; it is what makes an unrestricted derangement
  sufficient.** An earlier draft of this plan called `E` *unnecessary* on the
  grounds that the empirical null already contains the per-trial count
  structure. That holds only when the shuffle is constrained. Measured, and
  see "Subtracting `E` versus constraining the shuffle" below: keeping `E` and
  deranging globally rejects slow drift *better* than a block shuffle does,
  and without the block shuffle's loss of sensitivity.

  Subtracting `E` is not normalizing — no auto terms, no `sqrt`, no
  pairing-dependent denominator — so Welford in count space is unaffected. The
  pairing-dependent part is one small matmul on trial counts that never touches
  spikes: **0.036 ms per surrogate, ~4% of the surrogate's cost**, and
  `with_expected=True` was not measurably slower than `False` end to end
  (0.79 vs 0.85 ms).
- **Precondition: the support contract, not a global common window.** Every
  surrogate pairing must satisfy matched source-target support. When that holds,
  exposure is invariant under every allowed permutation automatically, so
  raw-count comparison is valid without further checks. Validate the contract
  rather than inferring safety from the fact that trials are paired.
- **Add the shuffle modes** above, with `global` retained as the default and
  the scheme recorded in surrogate provenance. With `E` retained, the
  constrained modes are no longer the remedy for slow drift; they cover the
  narrower case where the non-stationarity is in the fine-timescale structure
  rather than in the rates, which `E` cannot reach.
- **Split `with_expected`.** It currently gates the expected-count term *and*
  the auto terms together. This design needs only the former, so the cheap
  thing should be cheap explicitly rather than by accident.

### Subtracting `E` versus constraining the shuffle

`E_sigma = s_b * sum_k n_i(k) n_j(sigma(k))` is pairing-dependent, so it
*tracks* the across-trial count covariance rather than cancelling it: identity
gives the correlated sum, a derangement the uncorrelated one. Subtracting it
removes the drift contribution from the observation and from every surrogate
alike, so the comparison is clean without constraining `sigma` at all.

A constrained shuffle instead puts the drift *into* the null. That works, but
it cannot distinguish slow drift from slow genuine coupling, so it nulls both.

Mean per-bin z, 60 trials, 5 seeds, 60 surrogates:

| fixture | raw + global | raw + block(10) | **E-sub + global** |
|---|---|---|---|
| drift, no coupling | 2.72 | 0.76 | **0.11** |
| drift + real 5 ms coupling | 4.73 | 2.01 | 2.02 |
| time-locked, no coupling | -0.14 | -0.17 | -0.14 |
| time-locked + real coupling | 1.28 | 0.98 | **1.29** |

Two things to read off it. `E`-subtraction rejects the drift false positive
better than the block shuffle (0.11 against 0.76), and it keeps full
sensitivity to a real coupling where the block shuffle loses about a quarter
of it (1.29 against 0.98). The `drift + coupling` row is convergent evidence
that ~2.0 is the true signal level: `E`-sub and block agree there, while
`raw + global` reports 4.73, inflated by the drift it failed to remove.

**The two mechanisms are complementary, not alternatives.**

| confound | handled by |
|---|---|
| across-trial count covariance | the analytic `E` — exactly, at no sensitivity cost |
| within-trial time-locked co-modulation | the surrogate null — any `sigma != identity` preserves both PSTHs |

`E` cannot see the second: conditioning on per-trial counts says nothing about
*when* in a trial those spikes fall. The null cannot see the first unless
`sigma` is constrained. Together, with an unrestricted derangement, both are
covered and nothing is given up.

**Caveat.** These are synthetic fixtures — one drift shape at one strength,
one coupling, 60 trials. The reason to believe the result is the algebra, not
the fixture; confirm on a real session before changing the pipeline. The
support contract still gates every non-identity pairing either way, so
`clip_to_window` remains a precondition.

---

## Phase 5 — deprecation, auto-term fix, and capsule migration

- **land the 1c auto-term fix** (specified above) across both Python copies and
  both Julia cross sites, and flip the `expectedFailure` marker on the
  identical-trains test. Doing it here means reported numbers move and
  cross-language fixtures regenerate once, alongside the rename
- `"corrcoef"` → `DeprecationWarning`, aliasing `legacy_auto_normalized`
- update ~5 real call sites: `ccg_session.py:119` (`NORMALIZE`),
  `pair_figures.py`, `oversplit_786867.py` (2), plus the workbench diagnostic
- update six figure labels — `ccg_session.py:1206, 1221, 1254, 2637, 2759` and
  `oversplit_786867.py:293` all say "Peak corrcoef". These are where the
  misleading name reaches a reader
- migrate `PL-inputs-analysis/tests/crossval_julia_python.py` to raw-count +
  metadata fixtures, which decouples counting from normalization and is
  simpler than the current comparison

---

## Open decision

**Should `connectivity.py` switch its default shuffle?** Today inference uses
global derangements (`connectivity.py:167`, mirrored in `helpers.py:659`) while
the *display* shift predictor already uses circular offsets
(`ccg_session.py:214`, `SHIFT_OFFSETS = [1, -1]`) — the drift-safe scheme is on
the path that needs it less.

Adding the modes is uncontroversial and removes nothing. Changing the inference
default is a change to the null model: it would move p-values on real data and
needs its own validation pass, not a quiet default flip. Recommend shipping the
modes first with `global` unchanged, then deciding with real-session numbers in
hand.

---

## Deferred

`kernel_correlation` (bounded Pearson of kernel-filtered processes). Well
specified, cheap to add later, no current requirement. If it is implemented:
the post-hoc convolution route gives an effective kernel of `rect (*) tri` and
is only valid when the raw bin is much smaller than the kernel width; the exact
version needs triangular scatter inside the two-pointer loop. The auto terms
must use the same kernel, or the numerator/denominator mismatch returns.

---

## Acceptance tests

Beyond `tests/test_ccg_regression.py`:

1. unequal-duration observation windows supported
2. equal-duration, differently-aligned windows supported (review item 2)
3. analytic expected counts agree with simulation for heterogeneous windows —
   **only writable once the convention is fixed; (a) and (b) give different
   right answers**
4. changing the trial pairing changes expected counts when geometries differ
5. common-window fast path numerically identical to the general path — partly
   covered by `test_swapped_pairing_excludes_out_of_overlap_spikes` and
   `test_unequal_durations_clip_correctly`
6. existing exposure/edge-correction values unchanged (they are already exact)
7. identical trains give `C[0] == 1` under the repaired legacy statistic
8. `directional_excess(W)` invariant across bin sizes
9. Python and Julia agree on raw counts plus metadata
10. different observation pairs may have different support; every individual
    pair must have matching support; unequal-support pairs raise clearly
11. equal-duration but differently-aligned supports are rejected as
    incompatible (the `d_k == d_sigma(k)` trap)
12. constrained shuffles are one-to-one, contain no identity pair, never cross
    block boundaries, and respect max displacement
13. circular-offset inversion agrees with the `sigma^-1` flip rule
14. surrogate metadata records the shuffle scheme and its parameters
15. **statistical:** two units with no fine-timescale coupling but shared slow
    drift show an apparent effect under global permutation in raw-count space,
    substantially reduced under a block-constrained one; an injected
    same-trial fast coupling remains detectable under the constrained null
16. **statistical:** the same pair, with `E` subtracted, shows no apparent
    effect under an *unrestricted* derangement, and the injected coupling is
    detected at full strength — i.e. `E`-subtraction dominates the constrained
    shuffle on both axes

---

## Sequencing and rough effort

| Step | Scope | Rough size |
|---|---|---|
| 1a | one Python file x2 copies; tests exist | 1-2 days |
| 1b | same, plus new tests | 2-3 days |
| 1c | test only (`expectedFailure`); the fix itself moves to Phase 5 | ~0 |
| 1d | mechanical | half day |
| 2 | new module structure, both Python copies | 4-6 days |
| 3 | baseline layer + Julia parity | 3-4 days |
| 4 | surrogate refactor + Welford, **plus shuffle modes and support contract** | 4-5 days |
| 5 | deprecation, capsule, crossval migration, **plus the 1c auto-term fix across Python x2 + Julia** | 4-6 days |

Phases 1a/1b/1d can land independently and immediately. Phases 2-5 are the
redesign proper and can be paused after any of them — with one caveat: the
auto-term fix rides in Phase 5, so stopping before it leaves the legacy
statistic uncorrected. That is acceptable because nothing downstream depends on
it (see Risks), but if the redesign is shelved, pull 1c forward as a standalone
cross-language change (2-3 days).

## Risks

- **All magnitudes quoted are synthetic.** Re-measure on a real session before
  deciding how much the legacy auto-term fix matters. Its effect is
  rate-dependent, so it perturbs cross-pair ranking — check whether the
  oversplitting hand-off or any published figure ranks pairs by absolute
  `corrcoef`.
- **Two Python copies must stay in step**, or the capsule silently keeps the
  old behaviour.
- **1c changes reported numbers** (~0.25-0.5% at pipeline settings) and
  nothing else. Traced every consumer: `p_adjusted` unchanged (denominator
  cancels); `raw_peak_val > MIN_PEAK` unchanged (`MIN_PEAK = 0.0`, sign only);
  `peak_z = (raw_peak_val - surr_mean)/surr_std` unchanged (all three scale
  together); CSV sort order unchanged (sorted by `surr_z`); prominence ratio
  unchanged (both terms scale). Only the `peak_corrcoef_raw` and
  `peak_corrcoef_shift` columns move. The 6.7% -> 2.7% comparability gain buys
  nothing here because nothing ranks on raw corrcoef — it is a library-
  correctness change, not a pipeline one.
- **Two sites rely on a lag-symmetric reduction for mirror validity.**
  `connectivity.py:420` and `ccg_session.py:739` both call
  `pair_vec_to_NN(..., mirror=True)`, correct only because their reductions are
  `np.max` over a lag-symmetric mask and `max(reverse(x)) == max(x)`. A
  directional statistic — max over positive lags, the natural `i -> j` test —
  breaks both silently. Fold this helper under the `to_dense` policy owner.
- Phase 2's `CCGCounts` is the point of no return for API churn; everything
  before it is a bug fix.
