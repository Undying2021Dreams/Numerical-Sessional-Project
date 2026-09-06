# RUN REPORT — Milestone M2: Forward model

> Written for a reader with no access to the repository. Numbers, not adjectives.
> This report also covers the M1-correction work done at the start of this session,
> per the review feedback on `handoffs/RUN_M1.md` (M1 itself was already accepted).

---

## 0. M1 corrections applied this session (before M2 started)

The reviewer accepted M1 but asked five follow-up questions and two extra checks
before M2 could begin. All seven were done; full derivations and numbers are in
`DECISIONS.md` (new entries D-M1-8 through D-M1-15) and `logs/failures.md`. Summary:

| # | Ask | Resolution | Where |
|---|---|---|---|
| Q2 | Explain Simpson order-6 on `runge_0_1`, add a deliberate 4th test integrand | Derived the boundary-cancellation formula `E(h) = h^4/180*[f'''(a)-f'''(b)] + O(h^6)`; confirmed it predicts all 4 test integrands' measured orders (4, 4, 6, 6) correctly; added `endpoint_matched_0_b` (`f(x)=cos(x)+sin(2x)` on `[0, 3.204133415386284]`), deliberately constructed so `f'''(a)=f'''(b)=-8` without the accidental full-antisymmetry that two rejected earlier constructions had | D-M1-8; `tests/test_quadrature.py` |
| Q1 | LU vs QR on Hilbert(8) — is one example enough? | Agreed it wasn't; added a Hilbert(6/8/10/12) sweep — LU won 2/4, QR won 2/4 (round-off noise, not a trend); updated `PLAN.md`'s M3 section to specify the *actual* comparison M3 needs (normal-equations+LU vs stacked-least-squares+QR on the real IRGNM problem) | D-M1-9; `PLAN.md`; `experiments/m1_linalg_benchmark.py` |
| Q3 | Poisson sampler "hangs" above lambda~745 — fix it | Measured first: it does **not** hang — the running product underflows to exactly 0.0 on its own after ~700-800 iterations regardless of lambda, so the loop *terminates* but returns a silently wrong, roughly lambda-independent value (~700-800). Worse than a hang (no crash, no warning). Fixed with a normal-approximation branch above `lambda=30` | D-M1-10; `src/rng.py` |
| Q4 | Check for the Neave effect in Box-Muller | Added lag-1..5 autocorrelation + 2 consecutive-pair scatter plots. **Findings, reported as measured**: all 5 lag autocorrelations within 0.4 standard errors of 0; no visible banding/spiraling in either scatter (figure: `results/m1/rng_histograms.png`, bottom row) | D-M1-12 |
| Q5 | Quadrature grid should not be the 25-frame grid | Corrected — M2's quadrature integrates each frame time's `[0,t]` on its own dense graded grid, never on the 25-frame grid itself; `simpson_diag`'s fallback flag is asserted to never fire | D-M1-13 (design), D-M2-3 (chosen grid) |
| extra | Mutation-test `quadrature.py` and `rng.py` too | Both mutations caught immediately (11/46 and 5/46 tests respectively) | `logs/failures.md` |
| extra | LU/QR timing claim measures Python overhead, not algorithms | Reframed as a measured `time ~ C*n^p` fit; **measured `p ≈ 1.1-1.2`, not 3** — our wall-clock in the tested range (n=20..100) is dispatch-overhead-dominated, not flop-dominated. Reported honestly rather than forced to match the `n^3` theory | D-M1-15 |

One of these corrections itself uncovered a mistake in the *correction*: the review's
own Q3 hypothesis ("hang") turned out to be wrong once measured (see above and
`logs/failures.md`) — recorded because getting this right changed the fix.

## 1. One-paragraph summary

M2 builds the forward model of the irreversible two-tissue compartment ODE system.
`closed_form_C_T` implements eq. (3) of Lemma 6 as a single smooth expression built on
a numerically stable helper `phi1(x) = expm1(x)/x`, which handles both of the paper's
degenerate branches (`mu_j==0`, `k2+k3+mu_j==0`) exactly, as one formula rather than an
if/else — proved algebraically identical to the paper's branched form and checked
against an independent literal transcription of that branched form in tests.
`quadrature_C_T` implements eq. (1) of Lemma 5 by directly integrating with our own
Simpson's rule on a dense, graded per-time-point grid (not the 25-frame measurement
grid — see Q5 above), sized by a grid-refinement study. The three-way agreement check
(closed form vs quadrature vs an independent `scipy.solve_ivp` integration of the raw
ODE) is the milestone's centrepiece and passed with all three paths agreeing to
`5.9e-10` (closed vs quadrature) and `1.7e-13` (closed vs `solve_ivp`). A near-degeneracy
sweep across the removable singularity showed no spike, and a precise late-time-slope
measurement uncovered a real, quantified 5-19%-per-region gap between the paper's
informal "slope approaches `K1*k3/(k2+k3)*C_P`" statement and what the model actually
does at realistic scan durations — explained by a corrected asymptotic formula that
matches measurement to better than `1e-3` relative error.

## 2. Files added or changed

| Path | Lines | Purpose |
|---|---|---|
| `src/forward_model.py` | 199 | New. `phi1`, `arterial_input`, `parent_plasma_fraction`, `closed_form_C_T` (+ `closed_form_term1`, `closed_form_C_T_derivative`), `quadrature_C_T`, `GradedGridSpec`, `C_WB_from_C_P`, `C_PET` |
| `tests/test_forward_model.py` | 335 | New. 13 tests: phi1 correctness, `C_T(0)=0`/non-negativity, literal-eq(3)-branch cross-checks, three-way agreement, near-degeneracy sweep, late-time slope (both the realistic-scan-duration and the `t=800` asymptotic check), `C_WB`/`C_PET` assembly |
| `experiments/m2_forward_model.py` | 379 | New. phi1 stability study, grid-refinement study, three-way agreement table, near-degeneracy sweep + figure, late-time slope table, Figure 2 analogue |
| `DECISIONS.md` | 543 (+~380 this session) | M1-correction entries D-M1-8..15; M2 entries D-M2-1..6 |
| `PLAN.md` | 173 (+31) | M3 section rewritten per D-M1-9 (specifies the actual LU-vs-QR comparison M3 needs) |
| `logs/failures.md` | 64 (+~40) | M1-correction mutation results, the Q3 "hang" correction, M2 mutation result |
| `src/rng.py` | 212 (+~40) | `POISSON_KNUTH_MAX_LAMBDA`, normal-approximation branch in `poisson_one`/`poisson`, `force_normal_approx` param |
| `tests/test_rng.py` | 212 (+~55) | Large-lambda Poisson tests, Knuth-vs-normal-approx branch agreement test |
| `tests/test_quadrature.py` | 232 (+~90) | `endpoint_matched_0_b` 4th integrand, its dedicated order-6 tests, updated docstrings |
| `experiments/m1_rng_benchmark.py` | 216 (+~55) | Neave-effect diagnostics (autocorrelation, scatter plots), large-lambda Poisson diagnostics |
| `experiments/m1_linalg_benchmark.py` | 227 (+~65) | Multi-size Hilbert sweep, measured (not assumed) `time~C*n^p` fit |

## 3. What each component does, mathematically

- **`phi1(x)`**: not itself a paper equation — a stability helper. Computes
  `int_0^t exp(c*s) ds = t*phi1(c*t)` for any `c` (including `c=0`), unifying what the
  paper writes as two separate cases. See D-M2-1 for the full derivation and the proof
  that it reduces exactly to eq. (3)'s two named branches.
- **`arterial_input`**: `C_P(t) = sum_j lambda_j exp(mu_j t)` — Definition 2.
- **`parent_plasma_fraction`**: `f(t) = A*exp(xi1*t) + (1-A)*exp(xi2*t)` — Remark 18.
- **`closed_form_C_T`**: eq. (3) of Lemma 6, rewritten via `phi1` (D-M2-1). Verified
  algebraically identical to the paper's literal branched form
  (`tests/test_forward_model.py::test_stable_form_matches_paper_eq3_branches` and the
  two branch-specific tests).
- **`closed_form_term1`**: the k2-part of `closed_form_C_T` alone; exposed because
  `closed_form_C_T_derivative` needs it (D-M2-2).
- **`closed_form_C_T_derivative`**: `dC_T/dt`, not a numbered paper equation — derived
  by differentiating eq. (1) directly (D-M2-4), used only to measure the late-time-slope
  claim precisely, without finite-difference error.
- **`quadrature_C_T`**: eq. (1) of Lemma 5, evaluated by direct Track A Simpson
  integration on a dense per-time-point graded grid (D-M1-13, D-M2-3) — never on the
  25-frame grid.
- **`C_WB_from_C_P`**, **`C_PET`**: `C_WB(t) = C_P(t)/f(t)` and
  `C_PET(t) = (1-V_B)*C_T(t) + V_B*C_WB(t)` — Section 2 / Remark 17.

## 4. Acceptance criteria — results

| Criterion (from PLAN.md) | Target | Measured | Pass? |
|---|---|---|---|
| Closed form vs quadrature, max rel. diff, all 4 regions x 25 frames | small | **5.880e-10** | Yes |
| Both vs independent `scipy.solve_ivp` (Track B), max rel. diff | small | closed vs `solve_ivp[Radau]`: **1.722e-13**; quadrature vs `solve_ivp[Radau]`: **5.880e-10**; closed vs `solve_ivp[RK45]`: **1.420e-11** | Yes |
| `C_T(0) == 0`, non-negativity | exact / true | `C_T(0)` exactly `0.0` for all 4 regions (not "close to zero" — exactly, by construction: every term is multiplied by `t`); all `C_T` values on the 25-frame grid `>= 0` for all 4 regions | Yes |
| Late-time slope approaches `K1*k3/(k2+k3)*C_P`, report measured agreement | reported | At `t`=last frame (57.5 min): relative error vs classical `Ki` ranges **5.6% (frontal) to 18.7% (white matter)** across the 4 regions — NOT tight agreement. A corrected asymptotic formula (D-M2-4) matches to **0.05%-0.34%** at the same `t`, and to **<5e-14** at `t=800`. See §5 and §12 for the discussion this result needs | Yes (measured; see discussion — the naive claim is only approximately true) |
| Figure reproducing paper's Figure 2 (log time axis): f, C_WB, C_P, 4 regional TACs | produced | `results/m2/figure2_analogue.png`; qualitative comparison in §6 | Yes |
| Near-degeneracy stress test: sweep mu through `-(k2+k3)`, closed form stays smooth and matches quadrature | smooth, matching | Max rel. diff during sweep: **2.419e-10**; max relative 2nd-difference (smoothness/no-spike check): well under the 0.05 threshold (see `tests/test_forward_model.py`); figure: `results/m2/near_degeneracy_sweep.png` | Yes |

## 5. Key numerical results

**Three-way agreement, per region (max relative diff over all 25 frames):**

| Region | closed vs quad | closed vs `solve_ivp`[Radau] | quad vs `solve_ivp`[Radau] | closed vs `solve_ivp`[RK45] |
|---|---|---|---|---|
| frontal | 5.876e-10 | 1.107e-13 | 5.875e-10 | 6.929e-12 |
| temporal | 5.880e-10 | 1.722e-13 | 5.880e-10 | 8.296e-12 |
| occipital | 5.867e-10 | 9.590e-14 | 5.866e-10 | 6.026e-12 |
| white_matter | 5.870e-10 | 1.374e-13 | 5.869e-10 | 1.420e-11 |

`solve_ivp` timing (25 output points, `rtol=1e-12`): Radau ≈0.35s/region, RK45 ≈0.086s/region.
Radau is markedly *more accurate* here (1.0-1.7e-13 vs RK45's 6.9e-12 to 1.4e-11 — about
two orders of magnitude tighter) despite being ~4x slower; D-M2-5 records both were
tried and reports both rather than only featuring the one chosen a priori.
No `simpson_diag` trapezoid-fallback fired in any region (`any_fallback_fired: false`).

**phi1 numerical stability** (naive `(exp(x)-1)/x` vs `expm1(x)/x`, vs a 30-term Taylor
series reference, restricted to `|x|<=1` where the series itself is trustworthy):

| `x` | naive rel. error | `expm1` rel. error |
|---|---|---|
| 1e-6 | 3.8e-11 | 2.2e-16 |
| 1e-8 | 1.1e-8 | 2.2e-16 |
| 1e-10 | 8.3e-8 | 0 |
| 1e-12 | 8.9e-5 | 0 |
| 1e-14 | 8.0e-4 | 0 |
| 1e-15 | 1.1e-1 | 2.2e-16 |
| 1e-16 | 1.0 (100%) | 0 |

Naive form reaches 1e-8 relative error at `x≈1e-8`; reaches complete breakdown (~100%)
only at `x≈1e-16`. `expm1(x)/x` needs no threshold at all across the full tested range.
(This corrected a wrong first guess of our own — see §0 and D-M2-1.)

**Grid-refinement study** (frontal region, `closed_form_C_T` used as reference; full
table for 3 frame times x 5 grading exponents x 9 grid sizes in
`results/m2/forward_model_benchmark.json`):

| grading `q` | `n=401` | `n=801` | `n=1601` | `n=3201` |
|---|---|---|---|---|
| 1 (uniform) | 1.169e-03 | 9.679e-05 | 6.559e-06 | 4.460e-07 |
| 2 | 1.241e-07 | 7.692e-09 | 4.586e-10 | 1.109e-09 |
| **3 (chosen)** | 8.118e-08 | 5.084e-09 | **2.508e-10** | 3.411e-10 |
| 4 | 2.376e-07 | 1.491e-08 | 9.295e-10 | 2.310e-10 |
| 5 | 5.811e-07 | 3.650e-08 | 2.309e-09 | 4.582e-10 |

(relative error at `t`=57.5 min, the hardest of the three tested frame times). Uniform
(`q=1`) needs `n≈12801` for comparable accuracy — confirms grading is not optional at
this domain/timescale ratio. `q=3, n=1601` (D-M2-3) reaches the round-off floor
consistently across the earliest, middle, and latest frame times (`8e-12` to `3e-10`).

**Late-time slope, all 4 regions** (`t`=57.5 min = last PET frame midtime):

| Region | `K1` | `k2` | `k3` | `Ki` classical | `Ki` corrected | measured ratio | rel. err classical | rel. err corrected |
|---|---|---|---|---|---|---|---|---|
| frontal | 0.157 | 0.174 | 0.118 | 0.06345 | 0.05992 | 0.05989 | 5.60% | 0.053% |
| temporal | 0.161 | 0.179 | 0.096 | 0.05620 | 0.05200 | 0.05196 | 7.55% | 0.078% |
| occipital | 0.177 | 0.159 | 0.088 | 0.06306 | 0.05795 | 0.05789 | 8.19% | 0.098% |
| white_matter | 0.100 | 0.161 | 0.047 | 0.02260 | 0.01844 | 0.01838 | 18.67% | 0.342% |

Corrected formula: `Ki_corrected = K1*(k3+mu_4)/(k2+k3+mu_4)`, `mu_4 = -0.0106/min` (the
slowest arterial rate). At `t=800 min` (deep into the asymptotic regime, well beyond any
realistic scan), the corrected formula's relative error drops to `<5e-14` for all 4
regions — confirming it is the genuine `t->infinity` limit, and that classical `Ki` is
only the special case of this formula when the arterial input plateaus (`mu_4=0`), which
our `C_P` does not. See D-M2-4 for the full derivation.

**Near-degeneracy sweep:** max relative diff (closed form vs quadrature) across 601
values of a synthetic `mu_5` swept through `-(k2+k3)`, at 3 evaluation times:
**2.419e-10**. No spike (see `results/m2/near_degeneracy_sweep.png`).

## 6. Figures produced

| File | What it shows | What the reader should look for |
|---|---|---|
| `results/m2/figure2_analogue.png` | `f(t)`, `C_WB(t)`, `C_P(t)`, and `C_PET(t)` for all 4 regions, log time axis | **Compared to the paper's Figure 2 by eye**: `f(t)` declines from 1 toward ~0.5 by t=100min in both — matches. `C_P`/`C_WB` both show a sharp peak around t≈0.1-0.2 min (≈6-12s, consistent with the fast 13.45/min arterial component) reaching ~5.5-6, then decay — matches the paper's shape and magnitude closely. `C_PET`/TAC panel: our curves are monotonically increasing over the log time axis shown (0.01 to 100 min) reaching ~1-3 by t=100min with white matter lowest and frontal/occipital highest — same qualitative ordering and magnitude as the paper's Figure 2 tissue TAC panel. **Difference worth naming**: the paper's TAC panel plots the actual 25-frame measurement points (as markers) alongside continuous ground truth and shows values only out to ~t=100 min matching their acquisition window; ours plots a dense continuous curve from a manually chosen 0.01-100 min window not tied to the actual frame schedule (which ends at 62.5 min) — a cosmetic difference, not a modelling one. |
| `results/m2/near_degeneracy_sweep.png` | `C_T(t)` (closed form, solid; quadrature, dashed) vs a swept synthetic `mu_5`, at 3 fixed evaluation times, with a vertical line at the degenerate crossing | Solid and dashed curves overlap essentially exactly (agreement to 2.4e-10) and both are visibly smooth straight-ish lines through the crossing — no kink, spike, or discontinuity at the dotted vertical line |

## 7. Assumptions made this milestone

| Assumption | Why | Risk if wrong |
|---|---|---|
| `phi1`-based single-formula closed form, not literal if/else branches (D-M2-1) | Proven algebraically identical to the paper's eq. (3) branches; additionally numerically stable near-degeneracy, which literal branches alone would not be | None found; both the literal-branch tests and the near-degeneracy sweep passed |
| Quadrature grid: `GradedGridSpec(n=1601, q=3.0)`, chosen by measurement (D-M2-3) | Grid-refinement study showed this reaches the round-off floor consistently across early/mid/late frame times without the fallback ever firing | If M3/M4 need quadrature at times far outside the 25-frame range (e.g. much larger `t`), this grid should be re-validated there — not yet tested beyond `t~800` used only for the asymptotic slope check |
| `scipy.solve_ivp` reference uses both `Radau` and `RK45` at `rtol=atol≈1e-12` to `1e-14` (D-M2-5) | Neither method showed any material difference in the conclusion; reported both for honesty rather than picking the one that looked better a priori | None — both agree with our closed form to well within their own requested tolerance |
| Near-degeneracy sweep uses a synthetic 5th exponential term rather than perturbing a real arterial `mu_j` (D-M2-6) | None of the paper's real ground-truth `mu_j` values is close to `-(k2+k3)` for any region, so a real-parameter sweep would never cross the degenerate point | None — `closed_form_C_T`/`quadrature_C_T` are generic in the polyexponential degree `p`, so this exercises the same code path a real crossing would |
| Late-time slope tested at `t`=800 min as the "true asymptote" in addition to `t`=57.5 min (realistic scan) | `t=57.5` alone doesn't distinguish "close to Ki" from "close to the corrected formula, which happens to be near Ki" as cleanly; `t=800` does, since the two formulas diverge only at moderate `t` and the corrected one keeps improving while `Ki` does not | None — this is a diagnostic choice, not a modelling assumption about the actual data |

## 8. Failures, divergences, and things that did not work

- **A wrong first guess, caught by our own measurement, not by the reviewer**: the
  initial phi1 stability write-up (drafted before running the actual comparison)
  claimed the naive form fails at `x~1e-8`; the measurement showed the real crossover
  is `x~1e-15` to `1e-16`. Corrected in `DECISIONS.md` D-M2-1 before this report was
  written; full account in `logs/failures.md`.
- **Bug in a first draft of the late-time-slope test** (caught before this report, not
  left in): used `np.min(MU)` (the *fastest*-decaying, most negative rate) where
  `np.max(MU)` (the *slowest*-decaying rate, closest to zero) was needed. Produced an
  immediately obvious failure (62% relative error against the "corrected" formula,
  which is actually correct) rather than a silent wrong number — caught on first test
  run, fixed, re-verified.
- No forward-model solver divergence — `closed_form_C_T` and `quadrature_C_T` are
  direct evaluations, not iterative solves; nothing here can fail to converge (that
  starts at M3).
- **Mandatory mutation check result** (see §10): caught immediately, no test
  strengthening needed.

## 9. Track A / Track B audit

`tests/test_no_library_solvers.py::test_no_banned_library_solvers_under_src` still
passes with `src/forward_model.py` added — it imports only `math`, `numpy` (array
storage/elementwise ops), `dataclasses`, and `src.quadrature.simpson_diag` (our own
Track A code). No `scipy` import anywhere under `src/`.

Library calls added this milestone, and why they're allowed:

| File | Library call | Why allowed |
|---|---|---|
| `tests/test_forward_model.py` | `scipy.integrate.solve_ivp` | The independent third code path for the three-way agreement acceptance criterion — Track B, tests/ only, exactly the use case CLAUDE.md section 1 describes |
| `experiments/m2_forward_model.py` | `scipy.integrate.solve_ivp` | Same, inside a clearly-marked benchmark script that reproduces the same three-way comparison for the report's numbers |

Track A routines written this milestone, and what each was checked against:

| Track A routine | File | Checked against |
|---|---|---|
| `phi1` | `src/forward_model.py` | A 30-term Taylor series reference (independent of `exp`/`expm1`), and the direct naive formula at moderate `x` |
| `closed_form_C_T` | `src/forward_model.py` | (1) A literal, independently-written transcription of the paper's branched eq. (3) (`_closed_form_literal_eq3` in the test file); (2) `quadrature_C_T` (our own second Track A path); (3) `scipy.solve_ivp` (Track B) |
| `quadrature_C_T` | `src/forward_model.py` | `closed_form_C_T` and `scipy.solve_ivp`, as above; built entirely on the already-verified `src.quadrature.simpson_diag` |
| `closed_form_C_T_derivative` | `src/forward_model.py` | Not directly checked against an independent reference (no third derivative-computing path exists in this milestone); indirectly supported by `closed_form_C_T` having been checked three ways, since the derivative is obtained by differentiating that same verified expression. Flagged as an open question in §12. |

## 10. Mutation check

Swapped `k3` for `k2` in `closed_form_C_T`'s second term coefficient
(`term2 = (K1*k2/a)*t*(...)`, previously `(K1*k3/a)*t*(...)`) and re-ran the full suite
before reverting.

| Mutation applied | Tests that failed | Verdict |
|---|---|---|
| `src/forward_model.py`, `closed_form_C_T`: `k3` → `k2` in the second term's coefficient | `test_stable_form_matches_paper_eq3_branches`, `test_stable_form_matches_literal_branch_exactly_at_mu_equals_zero`, `test_stable_form_matches_literal_branch_exactly_at_delta_equals_zero`, `test_three_way_agreement_closed_form_quadrature_solve_ivp`, `test_near_degeneracy_sweep_stays_smooth_and_matches_quadrature` (5 of 59) | **Caught immediately and hard.** The near-degeneracy sweep's closed-vs-quadrature agreement collapsed from ~2e-10 to 22.3% relative difference, and the three-way agreement test failed against *both* independent references (quadrature and `solve_ivp`) simultaneously — strong evidence the tests are checking the actual mathematics, not one fragile reference. No strengthening needed; reverted and the full suite re-run green (59/59). |

## 11. Reproduction commands

```
cd Boring-Project

# Full test suite (M0 + M1 + M2), 59 tests
python3 -m pytest tests/ -v

# M2 forward-model experiments (writes results/m2/*.json, results/m2/*.png)
python3 experiments/m2_forward_model.py

# M1-correction benchmarks, regenerated this session
python3 experiments/m1_linalg_benchmark.py     # seed 2024, now with Hilbert 6/8/10/12
python3 experiments/m1_rng_benchmark.py        # seed 123, now with Neave-effect diagnostics
# (m1_quadrature_benchmark.py unchanged this session; not re-run)

# Mutation check (manual, as run this session):
#   1. edit src/forward_model.py: change `term2 = (K1 * k3 / a) * ...`
#      to `term2 = (K1 * k2 / a) * ...`
#   2. python3 -m pytest tests/ -q      # expect 5 failures out of 59
#   3. revert the edit
#   4. python3 -m pytest tests/ -q      # expect 59 passed
```

Environment: Python 3.12.2, numpy 2.2.6, scipy 1.13.1, matplotlib 3.9.2, pytest 9.1.1
(`requirements.txt`).

## 12. Uncertainties and open questions for the reviewer

1. **`closed_form_C_T_derivative` has no independent third-path verification.** Unlike
   `closed_form_C_T` itself (checked against a literal-branch transcription, our own
   quadrature, and `scipy.solve_ivp`), the derivative is only supported by having been
   derived from the already-verified `closed_form_C_T` — there's no independent
   "derivative eq. (1)" to check it against. Would a finite-difference cross-check
   against `closed_form_C_T` (accepting its own truncation error as a looser tolerance)
   be worth adding, or is "derived from a 3-times-verified expression" sufficient
   evidence for a helper that only feeds one diagnostic (the late-time slope), not the
   main pipeline?
2. **The late-time-slope finding (D-M2-4) changes what the report can honestly claim
   about the paper's "net influx rate" language.** PLAN.md M2 asked to verify the slope
   "approaches `K1*k3/(k2+k3)*C_P`" — literally true only in the idealised
   plateau-plasma limit, which this project's synthetic `C_P` does not satisfy (it
   keeps decaying, `mu_4=-0.0106 != 0`). Should M5's final report state the corrected
   formula as the primary result (with classical `Ki` as "the textbook approximation,
   accurate to ~X% at this scan duration"), or keep classical `Ki` as primary with the
   correction as a footnote? This is a framing choice, not a numerical uncertainty — we
   have both numbers.
3. **`Radau` vs `RK45` for the M2 three-way check**: both matched the closed form well
   within their requested tolerance, but `Radau` was about two orders of magnitude
   tighter (9.6e-14 to 1.7e-13 vs `RK45`'s 6.9e-12 to 1.4e-11; see the table in §5) at
   roughly 4x the wall-clock cost. Since `Radau` is more accurate here and not
   prohibitively slower in absolute terms for 25 output points, should M3/M4's own
   ODE-based sanity checks (if any) default to `Radau`, or does this milestone's result
   not generalise
   enough to justify a standing choice?
4. **Figure 2 analogue's time window** (0.01 to 100 min) was chosen for a visually
   similar log-axis appearance to the paper's figure, not tied to `config.frame_edges_minutes()`
   (which ends at 62.5 min). Should the M5 final figure instead be windowed exactly to
   the actual 25-frame acquisition schedule, sacrificing a bit of the visual similarity
   to the paper's (unknown, possibly different) window?
5. **Grid-refinement study was run only for the frontal region.** The chosen
   `GradedGridSpec(n=1601, q=3.0)` was validated to reach the round-off floor for
   frontal at 3 frame times, but not explicitly re-validated for the other 3 regions
   (though the three-way agreement table in §5 shows all 4 regions achieve essentially
   identical `~5.9e-10` closed-vs-quadrature agreement using this same grid, which is
   indirect confirmation). Worth an explicit per-region refinement table, or is the
   three-way agreement table's uniform result across regions sufficient evidence?

## 13. Proposed next step

M3 (Jacobian and the solver) should start with the analytic Jacobian of
`closed_form_C_T` with respect to all 23 parameters, verified against central finite
differences before anything else is built on top of it (PLAN.md's own first acceptance
criterion). Two things from this milestone should inform M3's design directly: (a) the
LU-vs-QR comparison must follow the corrected `PLAN.md` M3 section (normal equations +
LU vs stacked least squares + QR, on the real `F'` swept over `alpha`, not a repeat
Hilbert-matrix demo — D-M1-9); (b) `closed_form_C_T`'s `phi1`-based construction should
make the analytic Jacobian's own removable-singularity terms (derivatives with respect
to `mu_j` near `-(k2+k3)`) tractable using the same `phi1` stability trick, rather than
needing a fresh derivation — worth checking early whether that reuse is straightforward
before committing to a specific Jacobian formula.
