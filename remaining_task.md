# remaining_task.md — everything still to be done, and everything you need to know to do it

This file is self-contained. If you are joining the project, read this and `README.md`
and you have the full picture; you do not need any other document to start working.

**One-sentence goal.** Show that the paper's exact-identifiability result is reproduced
by our own hand-written numerics — including the `K1`/`lambda` ambiguity that appears
when there is no arterial blood data, and its removal by a single blood measurement.

**Paper:** Holler, Morina & Schramm (2024), *Exact parameter identification in PET
pharmacokinetic modeling using the irreversible two tissue compartment model*, Phys.
Med. Biol. 69 165008 — in the repo as `paper-2.pdf`.

**Status:** milestones M0-M4.3 are done and verified. **M4.4, M4.5, and M5 remain.**
Everything below reflects the current state.

---

## Part 1 — Ground rules (read before writing any code)

These are the rules the project is graded on. Breaking them is worse than being slow.

1. **Hand-written numerics only under `src/`.** No `scipy.optimize`, `scipy.linalg`,
   `scipy.integrate`, `numpy.linalg.solve` / `lstsq` / `inv`, or `numpy.random` anywhere
   in `src/`. NumPy is allowed there only for array storage and elementwise arithmetic
   (`+ - * /`, `exp`, `sum`, slicing, `@` for plain matrix multiply).
   Library routines are permitted **only** inside `tests/` and clearly-marked benchmark
   scripts, and only as a reference to check our own code against. This is enforced
   automatically by `tests/test_no_library_solvers.py`, which walks the AST of every file
   under `src/` and resolves import aliases — so `import numpy as np; np.linalg.solve(...)`
   is caught too. If that test fails, your change is wrong, not the test.
   **The most likely way to break this in M4 is reaching for `scipy.optimize` to do the
   fitting. Use `src/irgnm.py` — that is the whole point of the assignment.**
2. **Every stochastic run takes an explicit integer seed.** No unseeded randomness
   anywhere. Use `src.rng.derive_seed(root_seed, "label=...", ...)` to make independent,
   reproducible sub-streams.
3. **Report numbers, not adjectives.** "Tests pass" is not a result. "Max relative error
   3.7e-13 over 200 random 50x50 systems" is a result. Every claim in a report should
   have a measured number attached.
4. **Failures are data, not embarrassments.** Divergent runs, non-convergence, and
   experiments that did not work go in `logs/failures.md` with the exact setting that
   produced them. The paper itself reports divergence counts (its Table 1). **A Monte
   Carlo study that reports zero failures everywhere will be treated as suspicious, not
   impressive** — check your divergence criterion before celebrating.
5. **Never loosen a tolerance to make a test pass.** If a test fails, either the code is
   wrong, or the tolerance was wrong *for a stated numerical reason* — in which case
   write that reason down. There is precedent for this in the project already (see
   "Numerical gotchas" below).
6. **Every modelling choice not dictated by the paper gets recorded** with a one-line
   justification: noise calibration, regularisation constants, stopping tolerances,
   thresholds, initial guesses.
7. **Keep `pytest` green.** It is **97 tests** right now. If your change breaks someone
   else's test, talk to them before editing their test.
8. **Each milestone ends with a report** in `handoffs/`, following `handoffs/TEMPLATE.md`:
   what was built, measured numbers against each acceptance criterion, assumptions,
   what failed, and open questions. `handoffs/RUN_M1.md`, `RUN_M2.md` and `RUN_M3.md` are
   in the repo as worked examples — copy their level of detail.

---

## Part 2 — What already exists (do not rebuild any of it)

Run `pip install -r requirements.txt`, then `pytest` (expect 82 passed).

| Module | What it provides |
|---|---|
| `src/config.py` | All ground-truth constants: arterial input `C_P`, plasma fraction `f`, the four regions' `(K1,k2,k3)`, the 25-frame schedule, blood-sample times, the 23-parameter layout, `PROJECTION_EPS` |
| `src/linalg.py` | LU with partial pivoting: `lu_factor`, `lu_solve`, `solve` |
| `src/qr.py` | Householder QR: `qr_factor`, `qr_solve`, `qr_lstsq`, `solve`, `lstsq` |
| `src/quadrature.py` | `trapezoid`, `simpson`, `simpson_diag` on non-uniform grids |
| `src/rng.py` | `LCG` uniform, `standard_normal` (Box-Muller), `poisson`, `normal`, `derive_seed` |
| `src/eigen.py` | `power_method`, `inverse_power_iteration`, `symmetric_eigendecomposition` |
| `src/forward_model.py` | `closed_form_C_T` (eq. 3), `quadrature_C_T` (eq. 1), `arterial_input`, `arterial_input_integral`, `parent_plasma_fraction`, `C_WB_from_C_P`, `C_PET` |
| `src/jacobian.py` | `forward_operator` (F, 104-dim), `analytic_jacobian` (F', 104x23), `unpack` |
| `src/irgnm.py` | `run_irgnm`, `irgnm_step`, `project`, `RegularizationSchedule` |

Verified results you can build on and cite:

- Forward model, three independent code paths (our closed form, our quadrature,
  `scipy.solve_ivp`) agree to **5.9e-10** and **1.7e-13** across all 4 regions and 25 frames.
- Analytic Jacobian verified against central finite differences, per parameter block.
- Noiseless IRGNM recovery: **median final relative error 4.3e-7** at `delta_x = 0.1`
  over 20 seeds, **zero divergences**; comparable at the other `delta_x` values.
- Null-space experiment: the predicted `K1`-vs-`lambda` non-identifiability direction is
  visible in the tissue-only Jacobian's singular vectors, and disappears once the
  blood-data block is added back.

Full milestone reports with every measured number are in `handoffs/RUN_M1.md`,
`RUN_M2.md`, `RUN_M3.md`. The failure log is `logs/failures.md`.

---

## Part 3 — Decisions already made that you must not silently contradict

These are the load-bearing decisions from the project's decision log. If you disagree
with one, raise it — do not just quietly do something different, because downstream
results depend on them.

**Forward model and numerics**

- `closed_form_C_T` implements the paper's eq. (3) through a single smooth helper
  `phi1(x) = expm1(x)/x` with `phi1(0) = 1`, instead of the paper's literal if/else on
  the two degenerate branches (`mu_j == 0`, `k2+k3+mu_j == 0`). It is algebraically
  identical — proven, and checked in tests against a literal transcription of the
  branched formula — but also numerically stable when a denominator is merely *near*
  zero, which the branched form is not. The Jacobian needs a second helper,
  `phi2(x) = (e^x - 1 - x)/x^2`, which **does** need a small-`x` Taylor series branch
  (unlike `phi1`, which needs none); the crossover was chosen by measurement.
- The quadrature path integrates on a dense **graded** grid over `[0, t]` for each
  requested `t` — `n=1601` points, `t_i = t_end * u_i^3` — **not** on the 25-frame
  measurement grid. The frames are where measurements live, not where integration has to
  happen. A uniform grid needs ~12801 points for comparable accuracy because `C_P`'s
  fastest component has a ~4.5 second time constant against a ~60 minute domain.
- Any ODE cross-check with `scipy.integrate.solve_ivp` uses `method="Radau"` (implicit).
  The system's rates span `-13.45` to `-0.0106` (ratio ~1270), so an explicit method has
  to resolve the fastest mode everywhere. Measured: Radau was ~2 orders of magnitude more
  accurate than RK45 at the same tolerance.
- **The late-time slope has two different correct answers**, and it matters which one you
  quote. The slope of `C_T/C_P` against normalised time `∫C_P ds / C_P(t)` is exactly the
  classical `Ki = K1*k3/(k2+k3)` — that is the **Patlak plot**, and it is a linear
  least-squares regression, i.e. **course outcome CO2**. The limit of `dC_T/dt / C_P` is a
  *different* quantity and carries a correction: `K1*(k3+mu_4)/(k2+k3+mu_4)`. Both are
  verified numerically. Do not "fix" one into the other.

**Solver**

- IRGNM uses **six** regularisation constants, not one: `(a,b)` for each of the metabolic,
  arterial-`C_P` and plasma-fraction blocks, with ansatz `x_i = a*exp(-b*i)`. A single
  scalar would over-penalise the small metabolic parameters relative to the large
  arterial ones, because the 23 parameters live on very different scales.
  Values are adopted from the paper's own tuned figures (Section 6), converted from
  base-2 to base-e: `a_alpha = 4000`, `a_beta = 100`, `a_gamma = 200`, `b = ln(2)/7 ≈ 0.099`,
  `tau = 6.8`. They were verified before being trusted — they give 4.3e-7 to 9.7e-7 final
  error with no adjustment.
- **QR (stacked least squares) is the default IRGNM solver**, not the normal equations.
  Forming `F'^T F' + Lambda` squares the condition number; measured across the schedule,
  `cond(normal eq.)` is `1.43e12` at iteration 299 versus `1.20e6` for the stacked matrix
  — the ratio is the square, to within 1%. On 79 of 80 Monte Carlo seeds both paths agree;
  on the one seed where they differ, QR stalls at 0.033 error while LU diverges to 4e11.
  `solver="lu"` is still available for comparisons.
- `max_iter = 300` for noiseless data, matching the paper's own choice, because the
  discrepancy-principle stopping rule cannot fire when `delta_y = 0`.
- The projection onto `D(F)` clips `A >= 0`, `xi1, xi2 <= 0`, and all twelve `K1,k2,k3`
  to `>= 1e-3`. **`mu` is deliberately unconstrained** (the paper's own domain), which
  means a bad step can send a `mu` positive and overflow `exp(mu*t)`. This is detected —
  every iterate is checked with `np.isfinite` and a non-finite iterate marks the run
  diverged — the overflow *warning* is suppressed for console readability, the *check* is not.

**Open, deliberately left for M4**

- Whether TAC-level noise should be Poisson-derived or Gaussian with time- and
  region-dependent sigma is **not yet decided**. Both generators exist in `src/rng.py`.
  Because we cut the sinogram/OSEM imaging chain, noise is added directly to the regional
  time-activity curves, and Gaussian may well be the more honest model. Build both, decide
  by measurement, and record the reasoning.

**Numerical gotchas already discovered the hard way**

- Simpson's error against `h` is U-shaped: fourth-order decay down to a round-off floor
  around 1e-10 to 1e-13, then *rising* again for finer grids. More points is not
  automatically safer. Measure accuracy at an `n` where truncation error still dominates.
- The Poisson sampler's Knuth branch silently returns wrong, `lambda`-independent values
  (~700-800) once `exp(-lambda)` underflows at `lambda >= 746` — not a hang, a wrong
  answer. Hence the normal-approximation branch above `lambda = 30`. If M4 needs large
  counts, this is already handled, but do not remove that branch.
- Our wall-clock timings in the tested size range are dominated by Python/NumPy dispatch
  overhead, not floating-point operations — the measured scaling exponent is ~1.1-1.2, not
  the ~3 that flop counts would predict. Any timing claim must be framed as operation
  counts against measured scaling, not as raw seconds.

---

## Part 4 — M4: noise, regularisation, and the identifiability experiments

### 4.1 Noise model — ✅ DONE

**Deliverables built:** `src/noise.py`, `src/montecarlo.py`, `tests/test_noise.py`,
`experiments/m4_noise_calibration.py`, `results/m4/noise_calibration.json`.

**Measured δ_y (Poisson / Gaussian):**
- high_count (target 0.003):  0.00296 ± 0.00026  /  0.00276 ± 0.00031
- normal_count (target 0.011): 0.01162 ± 0.00108  /  0.01087 ± 0.00114
- low_count (target 0.070):   0.07034 ± 0.00641  /  0.06719 ± 0.00535

**Decision recorded:** TAC noise = Poisson-derived; C_WB noise = Gaussian (D-M4-1).
**Tests:** 15 new tests, all passing. 97 total pass.

### 4.2 The three measurement setups — ✅ DONE

**Mechanism:** `active_mask` (boolean N_PARAMS array) and `include_blood` (bool) added
to `src/jacobian.py` and `src/irgnm.py`. Implemented once, no fork (D-M4-5).
- Setup A: `SETUP_A_MASK` (M_SLICE=False), `include_blood=False`
- Setup B: `active_mask=None`, `include_blood=True`, clean C_WB
- Setup C: `active_mask=None`, `include_blood=True`, Gaussian-noisy C_WB

All existing tests remain green.

### 4.3 The paper's tables and figures — ✅ DONE

**Grid:** 960 cells (3×4×4×20), runtime 256.5s. Results in `results/m4/`.
**498 / 960 divergences** logged in `logs/failures.md` (expected; mirrors paper's Table 1 trends).

Key trend check results:
- More divergences at lower count: ✅ AGREES (all 3 setups)
- Setup A diverges less than Setup C: ✅ AGREES (all noise levels)
- K1 recovered better than k3: ✅ AGREES for B and C; Setup A disagrees (expected — K1/λ ambiguity, Proposition 12)

**Figure 7 analogue:** `experiments/m4_plot_figure7.py` → `results/m4/figure7_analogue.png`
  (4 panels, one per δ_x, with K1/k2/k3/K curves and stopping-iteration annotation).

### 4.4 The identifiability signature experiment — the highest-value item in the project

This is the experiment that shows the paper's mathematics is genuinely encoded in our
code, rather than its formulas merely transcribed. Everything else in M4 is supporting
evidence for this.

Theory (the paper's Proposition 12): fit using **tissue TACs only**, with no arterial data
at all, and `k2` and `k3` are still recoverable — but every `K1^i` comes out wrong by the
**same** multiplicative constant `zeta`, with `C_P` scaled by `1/zeta` in compensation.
The ambiguity is one-dimensional and structural. Better optimisation cannot remove it.

Do this:

1. Drop the `F^2` block from the forward operator and the Jacobian, and run IRGNM.
2. Report the four ratios `K1_est^i / K1_true^i` for `i = 1..4` **and their spread**. The
   theory says the four must coincide — the spread (e.g. max relative deviation between
   them) is the headline number of the entire experiment.
3. Confirm `k2` and `k3` are recovered accurately in the same run while `K1` is not.
4. Confirm `C_P` comes out scaled by approximately `1/zeta`.
5. Add a **single** `C_P` measurement back and show `zeta -> 1`. Report how close.
6. Repeat at each noise level, so the reader can see whether the signature survives noise.

Start from `tests/test_null_space.py` and `experiments/m3_jacobian_and_solver.py` — M3
already found this null direction in the Jacobian's singular vectors. Showing it in
actual fitted parameters is the much stronger claim.

**If the four `K1` ratios do not coincide, say so loudly.** That means either our
Jacobian/solver has a bug or the theory is being misapplied — and either is far more
valuable surfaced than worked around.

### 4.5 Two supporting checks

- **Consistency (the paper's Theorem 21):** reconstruction error decreases as the noise
  level decreases. Plot it, report the numbers.
- **Regularisation on vs off:** show that regularisation reduces the variance of the
  recovered parameters across the 20 realisations. Report the variance both ways.

---

## Part 5 — M5: analysis, figures and report material

- **One-command regeneration.** A single entry point (`experiments/run_all.py` or a
  `Makefile`) that regenerates every figure and table in `results/` from fixed seeds, in
  order, on a clean checkout.
- **Timing and complexity.** Measure our LU, QR, Simpson and the IRGNM solve against
  problem size, reported as operation counts versus measured scaling — see the gotcha in
  Part 3 about why raw wall-clock alone is misleading here.
- **Track A vs Track B comparison.** For each hand-written routine, accuracy and runtime
  against its library equivalent (`numpy.linalg.solve`, `numpy.linalg.lstsq`,
  `scipy.integrate.solve_ivp`). Library calls stay in `tests/` and marked benchmarks.
- **Course-topic mapping.** A table mapping every topic in the CSE 402 outline — root
  finding, Gauss elimination / LU, QR, the power method, interpolation, least-squares
  regression, Newton-Cotes / trapezoid / Simpson / Romberg, random number generation,
  Monte Carlo — to the exact file and function where it is used and the result it
  produces. This is what a grader looks for first. Two already-finished items belong at
  the top of that table: the **Patlak plot is linear least-squares regression (CO2)**, and
  the **conditioning analysis in `src/eigen.py` is the power method (CO1)**.
- **Results narrative.** State *in advance* what counts as agreement with the paper:
  qualitative trend agreement, not digit matching — we deliberately cut the imaging chain,
  so absolute noise levels are not comparable. Then check the trends the paper reports:
  `K1` recovers better than `k2`/`k3`; known `C_P` beats `C_WB`-only, which beats noisy
  `C_WB`; the low-count setting fails, especially for the parameters of `f`.

Acceptance: `run_all.py` verified on a clean clone; a timing table with fitted scaling
exponents; a course-topic table with no gaps; and a trend checklist with a measured
verdict (agrees / disagrees / not testable) for each trend.

---

## Part 6 — Suggested order

The long pole is the 4.3 grid: 3 setups x 4 noise settings x 4 `delta_x` x 20 realisations
x up to 300 iterations. Everything else is small by comparison.

```
First    noise model (4.1) — everything in M4 waits on its interface
         in parallel: the parameter-mask mechanism (4.2), and the
         noiseless identifiability run (4.4 steps 1-5), which needs neither

Next     noise calibration finished -> launch the 4.3 grid early, it is the long job
         meanwhile: noisy identifiability variants (4.4 step 6), consistency and
         regularisation checks (4.5), and the M5 items that need no M4 output
         (run_all.py skeleton, timing harness, course-topic table)

Last     tables and figures from the completed grid, M5 narrative, full pytest green,
         one clean-clone run of run_all.py, and the M4/M5 handoff reports
```

If the grid is too slow, reduce the **iteration cap** and say so in the report — do not
quietly reduce the number of realisations, because the tables' statistics depend on it.

---

## Part 7 — Coordination notes

**Shared files, coordinate before editing:**

- `src/config.py` — everyone reads it; an incompatible edit breaks everything.
- `src/jacobian.py`, `src/irgnm.py` — 4.2 and 4.4 both need the parameter mask and the
  `F^2`-drop flag. Agree the mechanism once, implement it once, merge it, then fan out.
- `logs/failures.md` and the handoff reports — append, never rewrite someone else's entry.

**Git:** one branch per work item, run `pytest` before opening a PR, and do not commit
`__pycache__/` or large intermediate `.npy` dumps.

**Not in the repo:** `PLAN.md`, `DECISIONS.md` and `CLAUDE.md` are git-ignored. Everything
from them that you need is restated in this file; ask if you want the originals. The
milestone reports in `handoffs/` and the failure log in `logs/` **are** tracked, and they
are the best source of "what number did we actually get and why".
