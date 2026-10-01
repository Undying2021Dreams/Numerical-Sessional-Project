# RUN REPORT — Milestone M3: Jacobian and the solver

> Written for a reader with no access to the repository. Numbers, not adjectives.
> This report also covers the M2-correction work done at the start of this session,
> per review feedback on `handoffs/RUN_M2.md` (M2 itself was already accepted).
> `DECISIONS.md` is attached alongside this report per the reviewer's explicit request
> — M3 has several threshold/tolerance choices where the reasoning matters as much as
> the value.

---

## 0. M2 corrections applied this session (before M3 started)

The reviewer accepted M2, independently re-derived our corrected late-time-slope
formula from eq. (3) and confirmed it reproduces our numbers exactly, then asked five
follow-ups plus specified M3's exact scope. Full derivations in `DECISIONS.md` D-M2-7
through D-M2-10.

| # | Ask | Resolution |
|---|---|---|
| Q2 | The "slope" question had two correct, different answers, not one wrong one | Implemented the **Patlak plot**: `C_T(t)/C_P(t)` vs normalised time `int_0^t C_P ds / C_P(t)` is exactly linear with slope `Ki = K1 k3/(k2+k3)` (classical, no `mu_4` correction) at late time. Fitted via our own `src.qr.lstsq` on frames `t>=3.5 min` (13 of 25 points): slope relative error `9.95e-4` to `1.56e-2` across the four regions (frontal/temporal/occipital tightest; white_matter — smallest `k2+k3` — measurably worse, R²=0.997 vs 0.9996-0.9999 elsewhere, explained not hidden). This is now flagged as an M5 primary result (course outcome CO2: linear least-squares regression). |
| Q1 | Derivative had no independent verification | Added a central finite-difference cross-check: max relative error **1.12e-8** across all 4 regions, 5 test times (`t=0.5` to `57.5` min), step `h=1e-4` chosen via the standard optimal-step argument. |
| Q3 | Radau vs RK45 | `Radau` is now the standing default for ODE-based Track B cross-checks (reasoning: well-separated timescales, `mu` spans a 1270x range vs tissue rates `k2+k3` — an explicit method must resolve the fastest mode everywhere, an implicit one need not). |
| Q4 | Figure 2 window | Both now exist: `results/m2/figure2_analogue.png` (paper-comparison, 0.01-100 min log window) and `results/m2/figure2_acquisition_window.png` (windowed to `config.frame_edges_minutes()`, 25 frame midpoints marked as dots). |
| Q5 | Grid validated only on frontal | Confirmed frontal already has the *largest* `k2+k3` (0.292) of the 4 regions — it was, without deliberate design, already the hardest case by the criterion named, and the existing three-way agreement table shows all 4 regions reach the same `~5.9e-10` floor. No new sweep needed. |

## 1. One-paragraph summary

M3 builds the analytic Jacobian of the forward operator (`src/jacobian.py`), a
conditioning analysis via our own power method and inverse power iteration
(`src/eigen.py`), the null-space experiment predicted by the paper's Proposition 12
made visible in linear algebra terms, and the IRGNM solver (`src/irgnm.py`) with
multi-parameter Tikhonov-style regularisation, two independent linear-solve paths, and
projection onto the feasible domain. The Jacobian matches finite differences to
`3.6e-6` (worst block); the conditioning analysis finds `cond(F') = 7.3e4` at the
ground truth (already quite ill-conditioned even before any noise is added); the
null-space experiment finds the predicted `K1`/`lambda` degeneracy **exactly**
(alignment cosine `1.00000000` with the analytically predicted direction, found via
inverse power iteration — the more commonly-used deflation-based full eigendecomposition
was measurably *wrong* on this specific near-degenerate matrix, a real finding
documented below); and noiseless recovery — the milestone's centrepiece test — reaches
median final relative error `4.3e-7` to `9.7e-7` across `delta_x in {0.1,...,0.4}`
using the paper's own published regularisation hyperparameters, unmodified.

## 2. Files added or changed

| Path | Lines | Purpose |
|---|---|---|
| `src/jacobian.py` | 230 | New. `phi2`, `_phi1_prime`, `unpack`, `forward_operator` (F), `analytic_jacobian` (F') |
| `src/eigen.py` | 87 | New. `power_method`, `deflate`, `symmetric_eigendecomposition`, `inverse_power_iteration` — course-syllabus power-method eigenvalue decomposition |
| `src/irgnm.py` | 209 | New. `project`, `RegularizationSchedule`, `DEFAULT_SCHEDULE`, `irgnm_step`, `run_irgnm` |
| `src/config.py` | 241 (+34) | `BLOOD_SAMPLE_FRAME_INDICES`/`Q_BLOOD_SAMPLES`/`blood_sample_times_minutes`, parameter-block slices, `PROJECTION_EPS` |
| `src/forward_model.py` | +9 | `arterial_input_integral` (Patlak x-axis) |
| `tests/test_jacobian.py` | 168 | New. phi2/phi1' correctness, Jacobian vs finite differences (ground truth + random feasible points), F build sanity |
| `tests/test_eigen.py` | 96 | New. power method / inverse power vs `numpy.linalg.eigvalsh`, deflation spectrum, ill-conditioned-matrix accuracy |
| `tests/test_null_space.py` | 107 | New. Function-level invariance (item 3b), Jacobian-level null-space alignment (item 3) |
| `tests/test_irgnm.py` | 119 | New. Projection, regularisation schedule, single-seed recovery, LU-vs-QR agreement, eq.(23)/(24) sampler statistics |
| `tests/test_forward_model.py` | 436 (+101) | Patlak-plot tests, finite-difference derivative cross-check |
| `experiments/m3_jacobian_and_solver.py` | 381 | New. Items 1-5: Jacobian verification, conditioning, null-space, invariance, scaling, LU-vs-QR sweep |
| `experiments/m3_irgnm_recovery.py` | 205 | New. Items 6-8: sampler verification, 80-run noiseless-recovery Monte Carlo, convergence figure |
| `experiments/m2_forward_model.py` | 523 (+152) | Patlak analysis + figure, acquisition-window Figure 2 |
| `DECISIONS.md` | 862 (+~500 this session) | M2-correction entries D-M2-7..10; M3 entries D-M3-1..9 |
| `logs/failures.md` | 134 (+~70) | M3 divergence table, LU-vs-QR outcome comparison, M3 mutation result |
| `PLAN.md` | 173 | Unchanged this session (M3 section was already corrected last session) |

## 3. What each component does, mathematically

- **`forward_operator` (F)**: paper eq. 18-20. `F^1` (100-dim) is `closed_form_C_T`
  (already-verified M2 code) looped over the 4 regions and 25 frames. `F^2` (4-dim,
  the new piece) is `C_WB_data(s_l)*f_m(s_l) - C_P(lambda,mu)(s_l)` at the `q=4` blood
  sample times — `C_WB_data` is fixed problem data (the ground-truth `C_WB`), not a
  function of the optimisation variable.
- **`analytic_jacobian` (F')**: derived by hand (full derivation in
  `src/jacobian.py`'s `_dCT_block` docstring and `DECISIONS.md` D-M3-3), reusing
  `phi1` and its derivative `phi1'` (built from the new `phi2` helper) for the
  `mu`-derivatives, which have their own removable singularity at `mu_j = -(k2+k3)`.
- **`phi2(x) = (e^x-1-x)/x^2`**: a second stability-treated helper (standard
  exponential-integrator "phi-function", `phi2 = (phi1-1)/x`), needed because
  `phi1'(x) = 1 + (x-1)*phi2(x)`.
- **`power_method`/`inverse_power_iteration`/`symmetric_eigendecomposition`**:
  course-syllabus power method for the largest eigenvalue of `F'^T F'`, our own LU
  (via `src.linalg`) driving inverse iteration for the smallest, and repeated
  Hotelling deflation for the full spectrum.
- **`irgnm_step`/`run_irgnm`**: eq. (26), the IRGNM update, with the multi-parameter
  regularisation generalisation of Remark 23 (`RegularizationSchedule`, ansatz eq. 27
  per block) and projection onto `D(F)` (eq. 18) applied after every step.
- **`arterial_input_integral`**: `int_0^t C_P(s) ds`, needed for the Patlak plot's
  x-axis (D-M2-7); reuses the `phi1` construction from `closed_form_C_T`.

## 4. Acceptance criteria — results

| Criterion (from PLAN.md) | Target | Measured | Pass? |
|---|---|---|---|
| Analytic Jacobian vs finite differences, max rel. error per block | ~1e-6 or better | lambda: **3.62e-6**, mu: **1.72e-6**, m: **1.27e-7**, K: **2.57e-8** (FD step tuned to `h=1.5e-5` via a documented sweep — D-M3 above) | Close to target; see §12.1 for discussion of why lambda/mu land slightly above 1e-6 |
| Conditioning: full singular value spectrum, condition number, numerical rank | reported | `cond(F') = 7.316e4`, `cond(F'^TF') = 5.353e9`; full 23-value spectrum in §5; numerical rank **23/23** (well-conditioned enough to be full rank despite the large condition number) | Yes |
| Null-space experiment: near-exact zero singular value, alignment with predicted direction, ratio to next smallest | alignment close to 1 | **exact structural match**: `F1 @ v_pred` norm `9.78e-17` (vs `F1` scale `166`); reduced smallest eigenvalue `-1.21e-16` (machine zero); alignment cosine **1.00000000**; ratio to next eigenvalue `1.23e-7` (singular-value ratio `3.51e-4`) | Yes |
| Confirm F^2 removes the near-null direction, report change | reported | smallest eigenvalue jumps from `~0` (F1 only) to **4.17e-6** (full F') — a `3.4e10`x increase | Yes |
| 3b: invariance at the function level, c=0.5 and c=2 | unchanged to machine precision | max abs diff in `C_T`: **exactly 0.0** for both `c` values, all 4 regions | Yes |
| Parameter scaling reported as a number | reported | see §5: `mu` block spans ratio **1269**, all-23 span ratio **2690** | Yes |
| Linear solver comparison: cond(normal eq.) vs cond(stacked), rel. diff, runtime, swept over alpha | reported | see §5 table; normal-eq. condition number reaches `1.43e12` by iteration 299 vs stacked's `1.20e6` (ratio matches `cond(stacked)^2` to 1%); LU-vs-QR step relative difference grows from `1.7e-16` to `1.28e-9` | Yes |
| Noiseless recovery: final rel. error, per delta_x, expect ~1e-7 or smaller | ~1e-7 | median final rel. error: `delta_x=0.1`: **4.26e-7**; `0.2`: **5.21e-7**; `0.3`: **5.78e-7**; `0.4`: **9.69e-7** (converged runs only; divergence counted separately, see §5) | Yes |
| Convergence plots, log axis, per-block traces | produced | `results/m3/noiseless_recovery_qr.png` | Yes |
| Log every divergent run | done | `logs/failures.md` M3 section: full table + specific seeds | Yes |
| No `scipy.optimize` anywhere | confirmed | guard test passes; `src/irgnm.py` uses only `src.linalg.solve` and `src.qr.lstsq` | Yes |

## 5. Key numerical results

**Item 1 — Jacobian vs finite differences** (ground truth, `h=1.5e-5` relative step,
chosen by sweeping `h in {1e-4,...,1e-8}` against the worst block — mu — and finding
the U-shaped truncation/round-off minimum): `lambda` 3.62e-6, `mu` 1.72e-6, `m` 1.27e-7,
`K` 2.57e-8. Also checked at 3 random feasible points (10% random perturbation,
clipped into `D(F)`): worst-case max relative error `1.2e-5` to `1.2e-5` (trial-dependent,
all `<2e-5`).

**Item 2 — conditioning at ground truth**: largest eigenvalue of `F'^TF'` (power
method, 7 iterations) = **22320.478**, matching `numpy.linalg.eigvalsh` to 8 significant
figures. Smallest (inverse power iteration, 9 iterations) = **4.169838e-6**, matching
numpy to 8 figures. `cond(F'^TF') = 5.353e9`, `cond(F') = 7.316e4`. Full spectrum (our
deflation-based decomposition vs numpy, top half max relative error `3.6e-13`):

```
22320.5, 2060.5, 1816.4, 1220.0, 559.0, 318.5, 89.0, 74.3, 53.8, 24.2,
5.005, 3.696, 2.972, 2.472, 1.774, 0.900, 0.169, 0.0764, 0.00341,
0.00221, 1.39e-4, 3.92e-5, 4.17e-6
```

Singular values of the raw `104x23` matrix `F'` (numpy, Track B reference) confirm
numerical rank **23/23** at tolerance `1e-8 * sigma_max`.

**Item 3 — null-space experiment** (see D-M3-4 for the full deflation-vs-inverse-power
story): `F1 @ v_pred` norm `9.775e-17` against `||F1|| = 166.1` — the predicted
direction is a null direction to essentially full float64 precision, confirming the
invariance is EXACT (not approximate) at the Jacobian level, matching item 3b's exact
function-level result. Restricting to the 20 non-trivial columns (dropping the 3
exactly-zero `m`-columns): smallest eigenvalue `-1.214e-16` (inverse power) vs numpy's
`5.65e-13` (both "computationally zero"); alignment with the predicted direction via
inverse power iteration: **cosine 1.00000000**. Via the deflation-based full spectrum
instead: cosine **0.0000137** — essentially orthogonal, the wrong answer, despite
reporting a plausible-looking near-zero eigenvalue. Ratio of smallest to next-smallest
eigenvalue: `1.229e-7` (singular value ratio `3.51e-4`). Adding `F2` back: smallest
eigenvalue of the full `23x23` matrix = `4.170e-6`, a `3.43e10`x increase.

**Item 4 — parameter scaling** (justifying multi-parameter regularisation):

| Block | min \|value\| | max \|value\| | ratio |
|---|---|---|---|
| lambda | 0.6355 | 10.9136 | 17.2 |
| mu | 0.0106 | 13.4522 | **1269** |
| m | 0.0050 | 0.2000 | 40.0 |
| K/k2/k3 | 0.0470 | 0.1790 | 3.8 |
| all 23 | 0.0050 | 13.4522 | **2690** |

**Item 5 — linear solver comparison** (`F'` at a real mid-optimisation point,
`delta_x=0.2` seed 2024, 50 QR-solver iterations in): see the table reproduced in §4;
figure `results/m3/solver_comparison.png`.

**Items 6-8 — noiseless recovery, 20 seeds per `delta_x`, QR solver (default),
`max_iter=300`**:

| delta_x | n seeds | strictly diverged | stalled (finished, err>1e-2) | median final err (converged) | mean iterations |
|---|---|---|---|---|---|
| 0.1 | 20 | 0 | 0 | 4.263e-07 | 300.0 |
| 0.2 | 20 | 3 | 0 | 5.207e-07 | 275.4 |
| 0.3 | 20 | 4 | 1 | 5.783e-07 | 273.9 |
| 0.4 | 20 | 9 | 1 | 9.694e-07 | 242.4 |

Sampler verification (eq. 23/24, `E[(sigma*gamma)^2] = delta_x^2 + delta_x/4`): measured
vs predicted relative error `3.5e-4` to `4.7e-3` across all 4 `delta_x` values (200,000
samples each) — confirms the `N(delta_x, delta_x/4)` VARIANCE interpretation, not std.

LU-vs-QR on the identical 80-seed set: 79/80 runs show identical converge/diverge
outcomes; the one exception (`delta_x=0.3`, seed `3870939591031771045`) has QR stall
at `0.033` while LU fully diverges to `4.37e11` — see `logs/failures.md`.

## 6. Figures produced

| File | What it shows | What the reader should look for |
|---|---|---|
| `results/m3/conditioning_spectrum.png` | Full 23-eigenvalue spectrum of `F'^TF'`, ours vs numpy, with the inverse-power smallest marked | Our deflation curve overlays numpy's almost exactly across the whole visible range; the red dotted line (inverse power result) sits at the true bottom of the spectrum |
| `results/m3/solver_comparison.png` | Left: cond(normal eq.) vs cond(stacked) across the IRGNM alpha schedule. Right: LU-vs-QR step relative disagreement | Left panel: the blue (normal eq.) curve visibly outpaces the orange (stacked) curve — by iteration 300 it's ~6 orders of magnitude higher. Right panel: monotonic growth from 1e-16 to ~1e-9 |
| `results/m2/patlak_plot.png` | `C_T/C_P` vs normalised time, 4 regions, fitted late-time line | Points should look linear past the marked cutoff; fitted slope printed in the legend should be close to the labelled `Ki` |
| `results/m2/figure2_acquisition_window.png` | Figure 2 analogue windowed to the actual 25-frame schedule, frame midpoints marked as dots | Dots should sit exactly on the continuous curves (they do, by construction — same closed-form evaluation) |
| `results/m3/noiseless_recovery_qr.png` | 4-panel convergence plot, one panel per `delta_x`, log-scale error vs iteration, per-block traces plus all-seed context in gray | Median-representative run in bold color; the classic IRGNM shape (long near-flat prefix, then rapid drop) should be visible in all 4 panels; panel titles show the diverged fraction |

## 7. Assumptions made this milestone

| Assumption | Why | Risk if wrong |
|---|---|---|
| `q=4` blood-sample times = 4 existing frame midtimes at indices (3,10,17,24) (D-M3-1) | Physically realistic, reuses the one time-grid concept already in the codebase, spread across the dynamic range | If M4 needs different/more blood samples for its identifiability experiments, this may need revisiting — flagged as open question |
| `PROJECTION_EPS=1e-3` (D-M3-2) | Given directly in the M3 spec; independently matches the paper's own choice | None — external agreement is reassuring |
| `phi2` series threshold `1e-4` (D-M3-3) | Measured: naive closed form's error is already `2.1e-10` at `x=1e-6`, three orders inside our target tolerance, well before `1e-4` | If the Jacobian is ever evaluated at points with much larger `\|mu*t\|` ranges, re-measure |
| Regularisation hyperparameters adopted from the paper's own tuning, not re-derived (D-M3-5) | Replicating the paper's full cross-validated tuning procedure (different ground-truth parameter set) was out of scope; verified to work before trusting | If M4's noisy-data setting needs different tuning (discrepancy principle now active), these values may need re-verification — flagged in §12 |
| QR is the default IRGNM solver (D-M3-7) | Provably better-conditioned (`cond(F')` vs `cond(F')^2`); empirically at least as good, strictly better on 1/80 seeds | Low risk; LU remains available and was directly compared |
| FD step `h=1.5e-5` for the Jacobian test (found via sweep, not the `1e-6` originally guessed) | The `mu` block's error is U-shaped in `h` (truncation above, round-off/cancellation below); `1.5e-5` is close to the measured minimum | None — measured, not guessed |

## 8. Failures, divergences, and things that did not work

Full detail in `logs/failures.md`, M3 section. Summary: 0/20, 3/20, 5/20 (4 diverged +
1 stalled), 10/20 (9 diverged + 1 stalled) failed-or-stalled runs at
`delta_x=0.1,0.2,0.3,0.4` respectively — not reseeded past, all recorded with their
exact seeds. Divergent runs blow up dramatically (final relative error up to `~5.8e78`)
because `mu` is unconstrained in `D(F)` and a bad step can push it positive, overflowing
`exp(mu*t)` — caught by an explicit finiteness check, not a crash. One concrete
LU-vs-QR outcome difference found and reported (§5, item 5).

No Jacobian-block or scaling-related failure needed diagnosing (PLAN.md item 8's
contingency) — noiseless recovery reached the ~1e-7 target directly with the paper's
own hyperparameters, so item 8's "if it does NOT reach ~1e-7" branch was not exercised
this milestone.

## 9. Track A / Track B audit

`tests/test_no_library_solvers.py::test_no_banned_library_solvers_under_src` passes
with `src/jacobian.py`, `src/eigen.py`, `src/irgnm.py` added — all three import only
`numpy` (elementwise/array ops), `src.linalg`, `src.qr`, `src.forward_model`,
`src.config`, and (in `jacobian.py`) `math`-free elementwise numpy. No `scipy` anywhere
under `src/`.

| File | Library call | Why allowed |
|---|---|---|
| `tests/test_eigen.py`, `experiments/m3_jacobian_and_solver.py` | `numpy.linalg.eigvalsh`, `numpy.linalg.svd`, `numpy.linalg.cond` | Independent reference for the conditioning analysis (Track B, tests/ + clearly-marked benchmark script) |
| `experiments/m3_jacobian_and_solver.py`, `experiments/m3_irgnm_recovery.py` | `numpy.random.default_rng` (test-fixture construction only, e.g. `np.linalg.qr` for a random orthogonal matrix in a synthetic eigen test) | Not part of the scientific pipeline; generates synthetic inputs only |

Track A routines written this milestone, and what each was checked against:

| Track A routine | File | Checked against |
|---|---|---|
| `phi2`, `_phi1_prime` | `src/jacobian.py` | 40-term Taylor series reference; central finite difference of `phi1` |
| `analytic_jacobian` | `src/jacobian.py` | Central finite differences of `forward_operator` (ground truth + 3 random feasible points) |
| `power_method`, `inverse_power_iteration`, `symmetric_eigendecomposition` | `src/eigen.py` | `numpy.linalg.eigvalsh` |
| `run_irgnm` (both `solver="lu"` and `solver="qr"`) | `src/irgnm.py` | Mutual agreement on well-behaved runs; the analytic Jacobian and forward operator it's built on (already 3-ways verified in M2/M3) |

## 10. Mutation check

Flipped the sign of the `dA1_da` term in `src/jacobian.py`'s `dCT_dk2` formula
(`- (k2/a)*dA1_da`, previously `+`) and re-ran the full suite before reverting.

| Mutation applied | Tests that failed | Verdict |
|---|---|---|
| `src/jacobian.py`, `_dCT_block`: sign flip in `dCT_dk2`'s `dA1_da` term | `test_jacobian_matches_finite_differences_at_ground_truth_per_block`, `test_jacobian_matches_finite_differences_at_random_feasible_points` (max relative error **2190**, not subtle), and — as a cascade confirming the solver actually depends on a correct Jacobian, not just its own unit test — `test_irgnm_recovers_ground_truth_from_small_perturbation_qr` (3 of 82) | **Caught immediately and hard.** No strengthening needed; reverted, full suite (82/82) re-run green. |

## 11. Reproduction commands

```
cd Boring-Project

# Full test suite (M0-M3), 82 tests
python3 -m pytest tests/ -v

# M3 experiments (writes results/m3/*.json, results/m3/*.png)
python3 experiments/m3_jacobian_and_solver.py
python3 experiments/m3_irgnm_recovery.py          # ~30s, 80 IRGNM runs

# M2-correction regeneration
python3 experiments/m2_forward_model.py           # now also writes Patlak + acquisition-window figures

# Mutation check (manual, as run this session):
#   1. edit src/jacobian.py: in _dCT_block, change
#      `dCT_dk2 = K1 * ((k3 / a**2) * (A1 - A2) + (k2 / a) * dA1_da)`
#      to use `-` instead of the second `+`
#   2. python3 -m pytest tests/ -q      # expect 3 failures out of 82
#   3. revert the edit
#   4. python3 -m pytest tests/ -q      # expect 82 passed

# LU-vs-QR full 80-seed comparison (used for logs/failures.md's specific-seed finding):
python3 -c "
from experiments.m3_irgnm_recovery import run_recovery_study
results_lu, failures_lu = run_recovery_study(solver='lu')
"
```

Environment: Python 3.12.2, numpy 2.2.6, scipy 1.13.1 (Track B, tests/experiments
only), matplotlib 3.9.2, pytest 9.1.1 (`requirements.txt`). Root seed for the Monte
Carlo study: `20240301` (all 80 per-run seeds derived via `src.rng.derive_seed`).

## 12. Uncertainties and open questions for the reviewer

1. **Jacobian FD agreement (`lambda`: 3.6e-6, `mu`: 1.7e-6) lands just above the "~1e-6
   or better" target, not below it.** Both blocks are limited by finite-difference
   precision itself (the `mu` block specifically shows the classic U-shaped
   truncation/round-off curve when `h` is swept — see D-M3 above), not by a Jacobian
   defect: the mutation check (§10) shows the test suite is sensitive enough to catch a
   real error 100x-1000x larger than this. Is "close to but not strictly under 1e-6,
   with the FD-step tradeoff documented" acceptable, or is a higher-precision reference
   (e.g. complex-step differentiation, which has no cancellation floor) worth adding for
   these two blocks specifically before M4?
2. **The deflation-vs-inverse-power finding (D-M3-4) was demonstrated on the real
   problem but not cleanly reproduced in a small synthetic unit test** (tried a
   cond~1e10 matrix with a well-separated geometric spectrum; the effect was much
   milder there than on the real matrix's clustered near-zero eigenspace). Is a
   real-problem-only demonstration (backed by the exact measured numbers in this
   report) sufficient evidence for the DECISIONS.md claim, or should more effort go
   into a clean synthetic reproduction (e.g. deliberately constructing a matrix with an
   exact rank-3 zero block plus one small eigenvalue) before this becomes a claim in the
   final M5 report?
3. **Regularisation hyperparameters are the paper's own, unmodified — is that
   sufficient justification, or does the assignment expect us to demonstrate our own
   tuning capability** (even a small, clearly-scoped grid search) somewhere before M5,
   given the project brief's general preference for measurement-based choices over adopted
   ones? We verified they work (median error targets met) rather than re-deriving them;
   flagging this explicitly since it's a real methodological choice, not an oversight.
4. **The `q=4` blood-sample times (D-M3-1) were chosen once, informally, for M3's
   Jacobian shape.** M4's identifiability experiments (per PLAN.md) specifically add a
   "single `C_P` measurement" to break the `K1`/`lambda` degeneracy — does that
   experiment use one of these same 4 points, a different one, or a fresh design? Not
   yet decided; flagged here so M4 doesn't have to reverse-engineer this milestone's
   choice.
5. **One LU-vs-QR outcome difference out of 80 seeds (§5, item 5) is a small sample to
   draw a general conclusion from**, even though it is directly explained by the
   conditioning analysis. Should M4's noisy-data Monte Carlo study (which will add its
   own regularisation dynamics via the discrepancy principle) re-run this LU-vs-QR
   comparison at larger scale, given noisy data may visit different regions of `D(F)`
   where the gap matters more?

## 13. Proposed next step

M4 (noise, regularisation, identifiability) should start with the noise model
(Poisson/Gaussian, `src/rng.py` already has both — D-M1-11's open decision) and the
three measurement setups (A/B/C per PLAN.md). Two things from this milestone should
inform M4's design directly: (a) the discrepancy-principle stopping rule
(`run_irgnm`'s `delta_y`/`tau` path) is implemented but untested against real noisy
data yet — its first real exercise should be an early, small-scale sanity check, not
the full 20-realisation study, in case the regularisation schedule (D-M3-5, tuned for
the noiseless case) needs adjustment for noisy data; (b) the identifiability signature
experiment (PLAN.md M4, "highest priority") is a direct continuation of this
milestone's null-space experiment (§5, item 3) — fitting with tissue data only should
reproduce the same `K1^i`-scaling degeneracy this milestone found in the Jacobian, now
showing up as a genuine parameter-recovery ambiguity rather than a linear-algebra
fact; the `q=4` blood-sample times defined here (open question 4 above) are the
natural mechanism for M4's "add a single `C_P` measurement" resolution step.
