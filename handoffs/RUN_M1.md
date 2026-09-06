# RUN REPORT — Milestone M1: Numerical primitives (Track A) — also covers M0

> This report covers M0 (Skeleton and scaffolding) and M1 (Numerical primitives)
> together, as instructed. Written for a reader with no access to the repository.
> All numbers below were measured by running the code in this session; the exact
> commands are in section 11.

---

## 1. One-paragraph summary

M0 built the repository skeleton: `src/config.py` holding every ground-truth constant
from the paper's Section 5.1 and Table 2 (arterial input, parent plasma fraction,
regional kinetics, the 25-frame schedule, the 23-parameter unknown-vector layout), a
deterministic config-hash function for artifact provenance, plotting helpers, and the
Track A/B guard test that statically scans `src/` for banned library calls via Python's
AST (not a plain grep, to avoid false positives on documentation that names the banned
routines). M1 built the four Track A numerical primitives named in `CLAUDE.md`: LU
factorisation with partial pivoting, Householder QR (factor, solve, and least squares),
a 64-bit LCG uniform generator with a Box-Muller normal sampler and a Knuth-algorithm
Poisson sampler on top of it, and trapezoid/Simpson integration that both support
non-uniform grids (needed later for the paper's irregular 25-frame PET schedule).
Every routine was checked against an independent NumPy/SciPy reference (Track B,
confined to `tests/` and `experiments/`) and against known closed-form answers, and one
routine was deliberately sabotaged to confirm the test suite actually catches a broken
implementation (section 10) — it did, in 5 different tests.

## 2. Files added or changed

| Path | Lines | Purpose |
|---|---|---|
| `conftest.py` | 12 | Puts repo root on `sys.path` so `import src...` works regardless of pytest invocation directory |
| `src/__init__.py` | 7 | Package marker + Track A/B rule reminder docstring |
| `src/config.py` | 205 | M0: every ground-truth constant, frame schedule, 23-param layout, config-hash provenance stamp |
| `src/plotting.py` | 47 | M0: shared matplotlib style + `results/<milestone>/<name>.png` save convention |
| `src/rng.py` | 185 | M1: `LCG` uniform generator, `standard_normal` (Box-Muller), `poisson`/`poisson_one` (Knuth), `derive_seed` |
| `src/linalg.py` | 111 | M1: `lu_factor`/`lu_solve`/`solve` (LU, partial pivoting), shared `forward_substitute`/`back_substitute` |
| `src/qr.py` | 136 | M1: `qr_factor`/`qr_solve`/`qr_lstsq`/`solve`/`lstsq` (Householder QR, no explicit Q formed) |
| `src/quadrature.py` | 99 | M1: `trapezoid`, `simpson`/`simpson_diag` on non-uniform grids |
| `tests/__init__.py` | 0 | Package marker |
| `tests/test_no_library_solvers.py` | 179 | M0: the Track A/B guard, AST-based; includes a meta-test proving the guard is not vacuous |
| `tests/test_config.py` | 64 | M0: config sanity + cross-process determinism of config-hash and seeded RNG |
| `tests/test_linalg.py` | 120 | M1: LU accuracy, Hilbert(8) stress test, singular-matrix detection, pivoting-necessity test |
| `tests/test_qr.py` | 126 | M1: QR accuracy, orthogonality/reconstruction check, least-squares vs `numpy.linalg.lstsq`, Hilbert(8) |
| `tests/test_quadrature.py` | 150 | M1: known-integral accuracy, measured convergence orders, exactness-for-quadratics checks |
| `tests/test_rng.py` | 153 | M1: uniform/normal/Poisson statistical validation |
| `experiments/_common.py` | 41 | Non-numerical: `results/` JSON-saving helper with config-hash/seed provenance |
| `experiments/m1_linalg_benchmark.py` | 164 | Produces `results/m1/linalg_benchmark.json` + `linalg_residuals.png` |
| `experiments/m1_rng_benchmark.py` | 171 | Produces `results/m1/rng_benchmark.json` + `rng_histograms.png` |
| `experiments/m1_quadrature_benchmark.py` | 120 | Produces `results/m1/quadrature_benchmark.json` + `quadrature_convergence.png` |
| `DECISIONS.md` | 118 | All M0/M1 modelling & engineering decisions, D-M0-1..4 and D-M1-1..7 |
| `logs/failures.md` | 37 | M0/M1 entries: a test-design bug found and fixed, the mutation-check result, an open risk for M4 |
| `requirements.txt` | 12 | Pinned dependency versions actually used this session |

## 3. What each component does, mathematically

- **`src/config.py`** — no algorithm; it is the single source of truth for every
  constant PLAN.md's M0 section names verbatim: the polyexponential arterial input
  `C_P(t) = sum lambda_j exp(mu_j t)` (Definition 2 of the paper, degree p=4), the
  biexponential parent plasma fraction `f(t) = A e^{xi1 t} + (1-A) e^{xi2 t}` (Remark
  18), `V_B` and the `C_PET` mixing equation, the four regions' `(K1, k2, k3)` from
  Table 2, and the 23-parameter layout matching the paper's own `x_dagger in R^23`
  (Section 5.1, page 17): `(lambda_1..4, mu_1..4, m_1..3, K1^i,k2^i,k3^i for i=1..4)`.
- **`src/linalg.py`** — Doolittle LU factorisation with partial (row) pivoting:
  `P A = L U`, implemented as the standard column-by-column elimination with the
  largest-magnitude pivot selection. `forward_substitute`/`back_substitute` solve the
  two triangular systems. This is the textbook algorithm used throughout numerical
  linear algebra (e.g. Chapra & Canale ch. 9-10, the course's own reference text).
- **`src/qr.py`** — Householder QR: for each column, a reflector `H_k = I - 2vv^T` is
  built to zero everything below the diagonal; `Q^T b` is computed by re-applying the
  stored reflectors (Q is never formed as a matrix). `qr_solve` back-substitutes `R x =
  Q^T b` for square systems; `qr_lstsq` uses the same machinery for the `m > n`
  least-squares case, with the exact residual norm read off the last `m-n` entries of
  `Q^T b`.
- **`src/rng.py`** — `LCG`: `state_{n+1} = (6364136223846793005 * state_n +
  1442695040888963407) mod 2^64` (Knuth/MMIX multiplier), uniform draws taken from the
  top 53 bits of the state. `standard_normal` implements the basic Box-Muller transform.
  `poisson_one`/`poisson` implement Knuth's multiplication algorithm (an exact simulation
  of the Poisson process via products of uniforms against `exp(-lambda)`), not an
  approximation. None of this corresponds to a specific paper equation; it is
  infrastructure the paper's Section 5.1 noise simulation (steps 5-7, adapted per
  `CLAUDE.md` section 6's scope cut) will need from M4 onward.
- **`src/quadrature.py`** — `trapezoid`: the standard composite trapezoid sum on an
  arbitrary grid. `simpson`: composite Simpson built from an exact closed-form integral
  of the unique quadratic interpolant through each consecutive point-triple (via
  Lagrange basis integration), which is algebraically identical to the textbook 1/3 rule
  on a uniform grid (verified in `test_quadratic_segment_integral_reduces_to_uniform_simpson_weights`)
  and generalises correctly to the paper's non-uniform 25-frame PET schedule that will be
  used from M2 onward. Nothing in the paper mandates this specific unequal-spacing
  formula; it exists because the pipeline needs it (see `DECISIONS.md` D-M1-4).

## 4. Acceptance criteria — results

### M0 (PLAN.md)

| Criterion (from PLAN.md) | Target | Measured | Pass? |
|---|---|---|---|
| Repo runs `pytest` with the guard test passing | guard passes | `tests/test_no_library_solvers.py`: 3/3 passed (incl. a meta-test proving the guard detects a synthetic violation) | Yes |
| Seeded RNG returns identical streams across processes | identical | Two separate `python3` subprocess invocations of `LCG(seed=12345).uniform_array(10)` produced byte-identical stdout | Yes |

### M1 (PLAN.md)

| Criterion (from PLAN.md) | Target | Measured | Pass? |
|---|---|---|---|
| LU/QR on 200 random systems, sizes 5..100: max rel. residual & max rel. error vs `numpy.linalg.solve` | small | LU: max rel. residual **5.39e-16**, max rel. error vs numpy **4.99e-16**. QR: max rel. residual **1.21e-15**, max rel. error vs numpy **1.35e-15** | Yes |
| LU/QR on Hilbert 8x8: report error and condition number, comment on stability | reported | cond₂(H₈) = **1.526e+10**. LU rel. error vs true x=1: **1.223e-07**. QR rel. error: **3.495e-07**. numpy reference: 8.31e-08. See discussion below — LU was *slightly* more accurate than QR here, not QR as textbook folklore might suggest | Yes (reported; see §5 discussion) |
| Trapezoid/Simpson on ≥3 known integrals: log-log error-vs-h plot, fitted convergence orders (expect ~2 and ~4) | ~2, ~4 | Trapezoid: **2.0008, 1.9999, 2.0000** (sin, exp, runge). Simpson: **4.013, 3.999, 5.999** (sin, exp, runge — runge shows apparent order 6, a real finding, discussed in §5) | Yes |
| RNG: 1e6 samples, mean/variance vs theory, chi-square or KS statistic, histogram figure. Poisson: mean==variance==lambda | reported | Uniform: mean 0.499721 (theory 0.5), var 0.083412 (theory 0.083333), chi²(df=9)=**12.684** (< crit@0.05=16.919). Normal: mean 0.000388, var 1.000037, max\|empirical CDF − Φ\|=**5.6e-4**. Poisson (n=200,000 each): λ=4 → mean 4.0013/var 4.0046; λ=10 → mean 9.998/var 9.942; λ=25 → mean 24.991/var 25.012 | Yes |

## 5. Key numerical results

**LU/QR, 200 random well-conditioned systems, sizes 5..100 (seed 2024):**

| | max rel. residual | max rel. error vs `numpy.linalg.solve` | mean rel. residual | total solve time (200 systems) |
|---|---|---|---|---|
| LU | 5.393e-16 | 4.993e-16 | 2.919e-16 | 0.0951 s |
| QR | 1.206e-15 | 1.349e-15 | 6.569e-16 | 0.1526 s |
| `numpy.linalg.solve` (reference) | — | — | — | 0.0049 s |

Our pure-Python-loop LU/QR are ~20-30x slower than NumPy's LAPACK-backed solver, as
expected for hand-written triangular loops vs compiled BLAS; both are still well under a
second for 200 systems up to 100x100.

**Hilbert 8x8 (deliberately ill-conditioned):**

- Condition number (2-norm, via `numpy.linalg.cond`, Track B diagnostic): **1.5258e10**.
- LU: relative error vs true `x = (1,...,1)` = **1.223e-07**; relative residual = 7.57e-17.
- QR: relative error vs true `x` = **3.495e-07**; relative residual = 2.69e-16.
- `numpy.linalg.solve` reference: relative error = 8.31e-08.
- LU vs QR relative difference in the solution vector: **2.272e-07**.
- **Discussion (as PLAN.md asks):** the classical guidance is that QR tends to be more
  stable than LU without pivoting, but our LU *does* use partial pivoting, which is the
  main stabiliser in practice; on this particular matrix LU came out ~2.9x more accurate
  than QR, not less. Both errors sit within about one order of magnitude of the
  round-off floor predicted by `cond(A) * machine_epsilon ≈ 1.526e10 * 2.22e-16 ≈
  3.4e-6`, i.e. both solvers are behaving exactly as ill-conditioning theory predicts —
  the specific ordering (LU slightly ahead of QR here) looks like round-off noise at
  this scale, not a genuine stability ranking, and we did not find a principled reason
  to claim one method is "more stable" on this specific example. We report this plainly
  rather than picking the answer that matches the textbook expectation.

**Quadrature convergence orders** (fit via our own `src.qr.lstsq` on `log(error)` vs
`log(h)`, over the h-range that stays clear of the round-off floor — see D-M1-6):

| Integral | Trapezoid order (fit n=9,17,33,65,129) | Simpson order (fit n=9,17,33) | Simpson error floor (min over sweep) | floor reached at n |
|---|---|---|---|---|
| ∫₀^π sin(x) dx = 2 | 2.0008 | 4.013 | 2.102e-10 | 513 |
| ∫₀¹ eˣ dx = e−1 | 1.9999 | 3.999 | 2.442e-14 | 1025 |
| ∫₀¹ 1/(1+x²) dx = π/4 | 2.0000 | **5.999** | 1.443e-14 | 1025 |

The Runge integrand's Simpson order came out at ~6, not ~4. We checked this is not a
bug: `test_simpson_exact_for_quadratics_on_nonuniform_grid` independently confirms exact
integration of quadratics on a non-uniform grid, and the ratio of successive errors for
this integrand is consistently ~64 (=2⁶) across three halvings, not a one-off — see
`results/m1/quadrature_benchmark.json`. We report it as an observed numerical curiosity
(likely a near-cancellation of the leading Simpson error term for this particular smooth,
even-symmetric-ish integrand over this particular interval) rather than claim to fully
explain it; PLAN.md's target was "expect ~4," which our asserted lower bound (`> 3.5`)
still honestly reflects — an upper bound would have wrongly failed on real behaviour.

All three Simpson error curves are visibly U-shaped in `results/m1/quadrature_convergence.png`:
4th-order decay down to a floor, then round-off-driven increase for finer grids — this
directly motivated D-M1-6 (see `DECISIONS.md`) and the fix logged in `logs/failures.md`.

**RNG (n=1,000,000 unless noted, `src.rng.LCG`, no `numpy.random` used anywhere):**

| Quantity | Measured | Theory |
|---|---|---|
| Uniform mean | 0.499721 | 0.5 |
| Uniform variance | 0.083412 | 0.083333 |
| Uniform min / max | 2.361e-06 / 0.999998 | (0, 1) |
| Uniform chi² (10 bins, df=9) | **12.684** | crit@0.05 = 16.919, crit@0.001 = 27.877 |
| Normal mean | 0.000388 | 0 |
| Normal variance | 1.000037 | 1 |
| Normal skewness | 0.00351 | 0 |
| Normal excess kurtosis | 0.00384 | 0 |
| Normal max\|empirical CDF − Φ\| over x∈[-3,3] | 5.64e-04 | 0 |

| Poisson λ | n | mean | variance | MC standard error (√(λ/n)) |
|---|---|---|---|---|
| 4 | 200,000 | 4.0013 | 4.0046 | 0.0045 |
| 10 | 200,000 | 9.9982 | 9.9419 | 0.0071 |
| 25 | 200,000 | 24.9913 | 25.0117 | 0.0112 |

PMF shape check at λ=1 (n=500,000): P(X=0) empirical 0.36742 vs theory `e⁻¹`=0.36788;
P(X=1) empirical 0.36737 vs theory 0.36788; P(X=2) empirical 0.18458 vs theory
`e⁻¹/2`=0.18394 — all within a few MC standard errors.

## 6. Figures produced

| File | What it shows | What the reader should look for |
|---|---|---|
| `results/m1/linalg_residuals.png` | LU and QR relative residual vs matrix size, 200 systems | Both curves stay flat around 1e-16 to 1e-15 (machine precision) across all sizes; no growth with n that would signal an unstable algorithm |
| `results/m1/quadrature_convergence.png` | log-log absolute error vs h, trapezoid and Simpson, for 3 integrands | Straight lines with the labelled measured slopes (~2 for trapezoid, ~4-6 for Simpson); Simpson's line visibly bends upward at small h — the round-off floor |
| `results/m1/rng_histograms.png` | Histogram of 1e6 LCG uniforms vs Unif(0,1) density; histogram of 1e6 Box-Muller normals vs N(0,1) density | Both histograms overlay their theoretical curves closely with no visible bias, gaps, or asymmetry |

## 7. Assumptions made this milestone

| Assumption | Why | Risk if wrong |
|---|---|---|
| `src/` is the importable package root (D-M0-1) | Matches CLAUDE.md's literal phrasing "anywhere under `src/`" | None; cosmetic |
| `config.py` lives at `src/config.py` (D-M0-2) | Keeps one import convention for all Track A modules and tests | None; trivial to add a root re-export |
| LCG with Knuth/MMIX multiplier, top-53-bits output (D-M1-1) | CLAUDE.md explicitly allows LCG; simpler/more auditable than a full Mersenne Twister for a teaching repo | LCGs have known lattice structure in high dimensions (Marsaglia); acceptable since downstream use is 1-D transforms and low-dim Monte Carlo, not quasi-random sampling — flagged as a standing risk, not resolved |
| Basic (non-polar) Box-Muller (D-M1-2) | Named explicitly in CLAUDE.md; easiest to verify against the closed-form target density | None significant; polar variant would be marginally faster |
| Knuth's multiplication algorithm for Poisson, no optimisation for large λ (D-M1-3) | Exact, easy to verify by hand; matches CLAUDE.md's preference for checkable code | **Carried into M4 as an open risk**: O(λ) cost per draw could be slow if noise calibration needs λ in the hundreds/thousands |
| Non-uniform Simpson via per-triple exact quadratic integration, trapezoid fallback on an odd leftover interval (D-M1-4) | The forward model (M2) needs non-uniform support for the paper's 25-frame schedule; a uniform-only Simpson would be dead code for the actual pipeline | If the fallback triggers often in M2 (odd numbers of remaining points), local accuracy drops to 2nd order for that one interval — worth checking once the real 25-point grid is used |
| Condition numbers computed via `numpy.linalg.cond`, confined to `experiments/` (D-M1-7) | Diagnostic-only use, explicitly permitted by CLAUDE.md for "clearly-marked benchmark scripts" | None; guard test would fail immediately if this leaked into `src/` |
| Quadrature test/benchmark `n` values chosen to stay clear of the float64 round-off floor (D-M1-6) | Discovered empirically this milestone; a naive fixed large `n` measures round-off noise, not algorithm correctness | If M2/M3 evaluate quadrature at very fine non-uniform grids, the same floor could appear there and should be re-checked, not assumed away |

## 8. Failures, divergences, and things that did not work

- No LU/QR solver divergence: partial pivoting handled every one of the 200 random
  systems plus the deliberately singular 3x3 test matrix (correctly raising
  `SingularMatrixError`) and the deliberately zero-pivot 2x2 test matrix (correctly
  solved via pivoting, confirming pivoting is actually wired in and not dead code).
- **Test-design bug found and fixed** (not a Track A bug): the first draft of the
  quadrature convergence-order test (a) had a sign error in converting a fitted slope to
  a convergence order, and (b) asserted Simpson's absolute error at n=2001 was below
  1e-9, which fails on a *correct* implementation because n=2001 is well past the
  round-off floor (measured floor: 2.1e-10 for sin, 2.4e-14 for exp, 1.4e-14 for runge,
  reached between n=257 and n=1025). Full account in `DECISIONS.md` D-M1-6 and
  `logs/failures.md`. This is exactly the kind of "tolerance was wrong for a stated
  numerical reason" case CLAUDE.md section 3 asks to document rather than silently patch.
- No Monte Carlo divergence to report yet — that class of failure starts at M3/M4
  (IRGNM iterations, noisy-data Monte Carlo).
- The Poisson sampler's O(λ) cost is not a failure today (verified fine at λ up to 25)
  but is an open risk explicitly logged for M4 (see section 7 and `logs/failures.md`).

## 9. Track A / Track B audit

`tests/test_no_library_solvers.py::test_no_banned_library_solvers_under_src` passes: an
AST walk of every `.py` file under `src/` (currently `__init__.py`, `config.py`,
`plotting.py`, `rng.py`, `linalg.py`, `qr.py`, `quadrature.py`) finds zero imports or
attribute accesses matching `scipy.*`, `numpy.linalg.solve`, `numpy.linalg.lstsq`,
`numpy.linalg.inv`, or `numpy.random.*` (with import-alias resolution, so `import numpy
as np; np.linalg.solve(...)` is caught, not just the literal `numpy.linalg.solve`
spelling). A companion meta-test (`test_guard_actually_detects_a_violation`) feeds the
guard's own detection logic a synthetic file containing `np.linalg.solve` and asserts it
is flagged, so the guard's silence is evidence of absence, not evidence that the guard is
inert.

Every place a banned symbol appears in this milestone's code, and why it is allowed:

| File | Library call | Why allowed |
|---|---|---|
| `tests/test_linalg.py`, `tests/test_qr.py` | `numpy.linalg.solve`, `numpy.linalg.lstsq`, `numpy.linalg.cond`, `numpy.random.default_rng` | Reference/diagnostic only, inside `tests/`, per CLAUDE.md section 1 |
| `experiments/m1_linalg_benchmark.py` | `numpy.linalg.solve`, `numpy.linalg.cond` | Clearly-marked benchmark script (module docstring states this); comparison against Track A is the entire point of the script |
| `tests/test_linalg.py`, `tests/test_qr.py`, `tests/test_quadrature.py`, `experiments/*` | `numpy.random.default_rng` / `numpy.random.Generator` | Generates synthetic test *inputs* (random matrices/vectors), not part of the scientific pipeline itself |

Track A routines written so far, and the Track B reference each was checked against:

| Track A routine | File | Track B reference used to verify it |
|---|---|---|
| `lu_factor`, `lu_solve`, `solve` | `src/linalg.py` | `numpy.linalg.solve` (200 random systems + Hilbert 8x8) |
| `qr_factor`, `qr_solve`, `qr_lstsq`, `solve`, `lstsq` | `src/qr.py` | `numpy.linalg.solve`, `numpy.linalg.lstsq` (200 random systems, overdetermined least squares, Hilbert 8x8); cross-checked against `src.linalg.solve` directly on Hilbert 8x8 |
| `LCG.uniform`/`uniform_array` | `src/rng.py` | Closed-form Unif(0,1) mean/variance, chi-square goodness of fit (hand-computed statistic, standard textbook critical values, not `scipy.stats`) |
| `standard_normal` (Box-Muller) | `src/rng.py` | Closed-form N(0,1) mean/variance/skewness/kurtosis; empirical CDF vs `math.erf`-based Φ(x) at several quantiles |
| `poisson`/`poisson_one` (Knuth) | `src/rng.py` | Closed-form Poisson mean=variance=λ; hand-derived PMF values at λ=1 (`e^-1`, `e^-1`, `e^-1/2` for k=0,1,2) |
| `trapezoid` | `src/quadrature.py` | 3 closed-form integrals; exact-for-linear-functions check on a non-uniform grid |
| `simpson`/`simpson_diag` | `src/quadrature.py` | 3 closed-form integrals; exact-for-quadratics check on a non-uniform grid; reduction to textbook uniform-grid Simpson weights |

## 10. Mutation check

We mutated `src/linalg.py`'s LU elimination update from
`U[k+1:, k+1:] -= np.outer(multipliers, U[k, k+1:])` to `+=` (flipping the sign of the
rank-1 elimination update — the mutation actually applied, not a hypothetical one) and
re-ran the full test suite (`python3 -m pytest tests/ -q`) before reverting.

| Mutation applied | Tests that failed | Verdict |
|---|---|---|
| `src/linalg.py`, LU elimination step: `-=` → `+=` (wrong-sign rank-1 update) | `test_lu_solves_200_random_systems_accurately`, `test_lu_matches_numpy_solve_on_200_random_systems`, `test_lu_factor_reproduces_A_up_to_permutation`, `test_lu_detects_exact_singular_matrix`, `test_qr_square_solve_matches_lu_on_hilbert_8x8` (5 of 40 tests) | **Caught.** Failures ranged from a direct residual blow-up (30.8% relative error vs numpy on the 200-system sweep) to a structural check (`P@A != L@U`, 7.6% relative residual) to a cross-solver disagreement (LU vs QR differ by 112% on Hilbert 8x8) and even flipped the singular-matrix test (the mutated elimination no longer reliably detects the deliberately singular 3x3 test matrix). No test strengthening was needed; the mutation was reverted immediately after confirming the failure and the full suite (40/40) was re-run green. |

## 11. Reproduction commands

```
cd Boring-Project

# Full test suite (M0 + M1), 40 tests
python3 -m pytest tests/ -v

# Track A/B guard alone
python3 -m pytest tests/test_no_library_solvers.py -v

# Benchmarks (each writes results/m1/*.json and results/m1/*.png)
python3 experiments/m1_linalg_benchmark.py       # seed 2024
python3 experiments/m1_rng_benchmark.py          # seed 123 (uniform/normal), 1000-1025 (poisson)
python3 experiments/m1_quadrature_benchmark.py   # deterministic, no seed needed

# Config sanity / provenance stamp
python3 src/config.py

# Mutation check (manual, as run this session):
#   1. edit src/linalg.py: change `U[k+1:, k+1:] -= np.outer(...)` to `+=`
#   2. python3 -m pytest tests/ -q      # expect 5 failures out of 40
#   3. revert the edit
#   4. python3 -m pytest tests/ -q      # expect 40 passed
```

Environment actually used this session: Python 3.12.2, numpy 2.2.6, scipy 1.13.1 (not
imported by any Track A code — reserved for M2's `solve_ivp` reference), matplotlib
3.9.2, pytest 9.1.1 (pinned in `requirements.txt`).

## 12. Uncertainties and open questions for the reviewer

1. **Hilbert(8) stability ordering.** LU (with partial pivoting) came out ~2.9x more
   accurate than QR on this specific ill-conditioned matrix (1.223e-07 vs 3.495e-07
   relative error). We did not find a principled reason to expect this direction rather
   than the reverse, and both errors are consistent with round-off amplified by the
   condition number (~3.4e-6 predicted ceiling, both well under it). Is a single-matrix
   comparison enough to "comment on which is more stable," as PLAN.md's M1 acceptance
   criterion asks, or should we run this over several ill-conditioned matrices (e.g.
   Hilbert 6/8/10/12, or random matrices constructed to a target condition number) before
   M5's final write-up claims anything about relative stability?
2. **Simpson's apparent 6th-order convergence on the Runge integrand.** We confirmed
   this isn't a bug (independent exactness check for quadratics passes, and the ~64x
   error ratio per halving is consistent across three refinements), but we don't have a
   from-first-principles explanation for *why* this particular integrand shows
   superconvergence over this particular interval. Worth a paragraph in the final report,
   or is "observed, not fully explained" acceptable to leave as-is?
3. **Poisson sampler performance ceiling (D-M1-3).** Knuth's algorithm is O(λ) per draw.
   We only validated it up to λ=25. If M4's noise calibration (approximating the paper's
   high/normal/low count settings, per `CLAUDE.md` section 6's scope note) needs λ in the
   hundreds or thousands per TAC point, this sampler needs revisiting before it goes
   inside a 20-realisation Monte Carlo loop — should that redesign happen now (small,
   contained change) or wait until M4 exposes the actual required λ range?
4. **LCG statistical quality at higher dimensions.** All M1 checks are 1-D (marginal
   uniform, marginal normal, marginal Poisson counts). We have not checked 2-D/3-D
   structure (e.g. a lattice-structure test, or correlation between consecutive draws),
   which is the classical LCG failure mode. Is this worth adding now, or only if a later
   milestone's Monte Carlo results look suspicious?
5. **Simpson's odd-interval trapezoid fallback (D-M1-4) is untested against the actual
   25-frame schedule.** `config.N_FRAMES == 25` (24 intervals, even — no fallback
   triggers for the full-schedule integral), but any sub-range of frames used inside M2's
   quadrature calls could have an odd interval count and silently drop to 2nd-order
   accuracy for one interval. Should M2 explicitly test this against `config.frame_edges_minutes()`
   subranges, or is the diagnostic flag returned by `simpson_diag` sufficient (i.e. M2's
   forward-model code should check the flag and log rather than us pre-emptively testing
   all sub-ranges now)?

## 13. Proposed next step

M2 (Forward model) should start with `closed_form` (eq. 3 of Lemma 6) and `quadrature`
(eq. 1 of Lemma 5), both evaluated on `config.frame_midtimes_minutes()`. Two things from
this milestone should be revisited first, per the open questions above: (a) confirm
whether `simpson_diag`'s fallback flag needs to be surfaced/logged when M2 integrates
over sub-ranges of the 25-frame grid rather than the full grid, and (b) decide now vs.
later whether the Poisson sampler needs a large-λ path before M4, since that is a larger
design decision than a normal M2 task and is cheaper to make with M1's code still fresh.
