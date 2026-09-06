# logs/failures.md — divergent runs, non-convergence, and things that did not work

Per CLAUDE.md section 4: these are DATA, not embarrassments. Every entry records the
exact setting that produced the failure. Ordered chronologically within each milestone.

---

## M0

(none — scaffolding only, nothing stochastic or iterative to diverge yet)

## M1

No solver divergence in this milestone (LU/QR either succeed or raise
`SingularMatrixError`; there is no iterative process yet to diverge — that starts at
M3's IRGNM). Two things worth recording as data rather than silently fixed away:

- **Test design bug (not a Track A code bug), found and fixed during this milestone:**
  the first draft of the quadrature convergence-order test fit `log(error)` vs `log(h)`
  with an incorrect sign, and separately checked Simpson's absolute error at n=2001 with
  a tolerance of 1e-9. Both failed on the correct implementation. Root cause: Simpson's
  true error vs h is U-shaped in float64 — 4th-order truncation decay down to a
  round-off floor around 1e-10 to 1e-13 (integrand-dependent) near n=257-513, then
  *increasing* error from round-off accumulation for finer grids. n=2001 was already
  past the floor. See `DECISIONS.md` D-M1-6 for the fix and reasoning, and
  `handoffs/RUN_M1.md` section 5 for the measured floor per integrand.
- **Mutation check (`handoffs/RUN_M1.md` section 10):** flipping the sign of the rank-1
  update in the LU elimination step (`-=` to `+=`) was caught immediately by 5 of the 40
  tests, including both Track B cross-checks and the LU-vs-QR agreement test. No
  strengthening was needed.

**Open risk carried forward (not a failure yet, flagged for M4):** the Poisson sampler
(`src/rng.py`, Knuth's algorithm) costs O(lambda) uniforms per draw. It is verified here
only at lambda in {1, 4, 10, 25}. If M4's noise calibration needs Poisson counts per TAC
point in the hundreds or thousands (plausible for a "high count" PET setting), this
sampler will be slow and should be revisited before being used inside a 20-realisation
Monte Carlo loop. See `DECISIONS.md` D-M1-3.

## M1 corrections (post-review)

- **Reviewer's Q3 hypothesis was itself wrong, and the measurement showed something
  worse than what was asked to be fixed.** The review request described the Poisson
  sampler's large-lambda failure as an infinite loop ("hang"). Direct measurement
  (`python3 -c` one-liner reproducing the unpatched Knuth loop at lambda=745/746/1000/
  5000, seed=1) showed it is not a hang: the running product `p` underflows to exactly
  `0.0` on its own after ~700-800 iterations regardless of `lambda` (once `exp(-lambda)`
  itself has also underflowed to 0.0, at `lambda >= 746`), so the loop *terminates* —
  but returns a value clustered around 700-800 that is essentially independent of the
  true `lambda`. At `lambda=1000` it returned `769` (true mean 1000, std≈31.6 — about
  7.3 standard deviations off); at `lambda=5000` it returned `726` (nonsense). This is a
  silent-wrong-answer bug, not a hang, and arguably more dangerous (no crash, no
  warning, a plausible-looking wrong integer). See `DECISIONS.md` D-M1-10 for the full
  measurement and the fix (normal-approximation branch above `lambda=30`).

- **Mutation check, extended per reviewer instruction (M1's original check only
  covered `src/linalg.py`):**

  | Mutation applied | File | Tests that failed | Verdict |
  |---|---|---|---|
  | Dropped the middle (`4x`-weight) term from the Simpson quadratic-segment integral (`return w0*y0 + w2*y2`, previously `w0*y0 + w1*y1 + w2*y2`) | `src/quadrature.py` | 11 of 46: both known-integral accuracy tests (all 4 integrands), both convergence-order tests (all 4 integrands), the dedicated order-6 test, the uniform-Simpson-weight-reduction test, and the non-uniform-quadratic-exactness test | **Caught immediately and hard** — the mutated result on the quadratic-exactness test was off by a factor of ~10x (got 7.52 vs exact 76.83), not a subtle drift. No strengthening needed. |
  | Dropped `sqrt()` in the Box-Muller radius (`r = -2.0*math.log(u1)`, previously `r = math.sqrt(-2.0*math.log(u1))`) | `src/rng.py` | 5 of 46: all three direct Box-Muller statistical tests (mean/variance, skew/kurtosis, empirical-CDF-vs-Phi), plus — as a nice confirmation that the dependency chain is actually tested, not just the top-level function — both large-lambda Poisson tests, since `poisson_one`'s normal-approximation branch (D-M1-10) is built on `standard_normal` | **Caught immediately.** No strengthening needed. The cascade into the Poisson tests was not designed in advance; it is a byproduct of D-M1-10's normal-approximation branch reusing `standard_normal`, and it is a genuinely useful signal (a bug in Box-Muller would have silently corrupted M4's noise model too, not just M1's own RNG report). |

  Both mutations were reverted immediately after confirming the failures; the full
  suite (`python3 -m pytest tests/ -q`) was re-run green (46/46) after each revert.

## M2

No solver divergence (nothing iterative yet). One design bug caught by our own
measurement before it reached the report — see `DECISIONS.md` D-M2-1's "corrected a
wrong first guess" note: the first-draft phi1 stability analysis claimed the naive
`(exp(x)-1)/x` form breaks down (50% relative error) around `x ~ 1e-8`, reasoning from
the `sqrt(machine_epsilon)` heuristic without having run the actual comparison yet.
Running `experiments/m2_forward_model.py::phi1_stability_study` showed the true
50%-relative-error crossover is at `x ~ 1e-15` to `1e-16`, three to four orders of
magnitude smaller; `x ~ 1e-8` is only where the naive form's error first reaches `1e-8`
itself (a real but much milder degradation). Fixed in `DECISIONS.md` before this report
was written, per CLAUDE.md section 3 (measure, don't guess, and if the guess was wrong,
say so).

**Mandatory mutation check (`handoffs/RUN_M2.md` section 10):**

| Mutation applied | File | Tests that failed | Verdict |
|---|---|---|---|
| Swapped `k3` for `k2` in `closed_form_C_T`'s second term coefficient (`term2 = (K1*k2/a)*...`, previously `(K1*k3/a)*...`) | `src/forward_model.py` | 5 of 59: both literal-eq(3)-branch tests, the general eq(3)-branch-agreement test, the three-way agreement centrepiece test (closed form vs quadrature vs `scipy.solve_ivp`), and the near-degeneracy sweep test | **Caught immediately and hard** — the near-degeneracy sweep's closed-vs-quadrature disagreement jumped to 22.3% (from a healthy ~2e-10), and the three-way agreement test failed simultaneously against BOTH independent references (quadrature and `solve_ivp`), which is a strong signal that a real bug was introduced (not a tolerance quirk affecting only one reference). No strengthening needed. |

Reverted immediately after confirming the failure; full suite (`python3 -m pytest
tests/ -q`) re-run green (59/59) after revert.

## M3

**Noiseless recovery divergence table** (PLAN.md M3 item 7: "log every divergent run
in `logs/failures.md` rather than reseeding past it" — no seed was excluded or
re-rolled to avoid a bad outcome; all 80 runs below are exactly the first 20 seeds
`derive_seed(20240301, "delta_x=<dx>", "seed_idx=<0..19>")` generates per `delta_x`,
solver = QR/stacked (the default — see `DECISIONS.md` D-M3-7)):

| delta_x | n seeds | strictly diverged (blew up / non-finite) | stalled (finished but final error > 1e-2) | converged to plan target |
|---|---|---|---|---|
| 0.1 | 20 | 0 | 0 | 20/20 |
| 0.2 | 20 | 3 | 0 | 17/20 |
| 0.3 | 20 | 4 | 1 | 15/20 |
| 0.4 | 20 | 9 | 1 | 10/20 |

Divergence count growing with `delta_x` matches the paper's own Table 1 qualitative
finding (their divergence counts also grow with the initial-guess perturbation level).
Divergent runs blow up dramatically once they leave the basin of attraction — measured
final relative errors for the 18 strictly-diverged runs at `delta_x` in {0.2,0.3,0.4}
range from `0.82` up to `~5.8e78` (`results/m3/irgnm_recovery.json` has the full
per-seed list) — i.e. `mu` (unconstrained in `D(F)`) runs away and `exp(mu*t)`
overflows; caught by the solver's own finiteness check (`src/irgnm.py`), not a crash.

**LU-vs-QR solver comparison on the full 80-seed sweep** (re-run with `solver="lu"`,
identical seeds): 79 of 80 runs show IDENTICAL convergence/divergence outcomes between
the two solvers (same seeds diverge, same seeds converge, final errors matching to
3-4 significant digits) — the measured conditioning gap between the two solve paths
(`DECISIONS.md` D-M3-6) is real but too small at this problem's scale to flip most
outcomes. **One seed differs concretely**: `delta_x=0.3`, seed
`3870939591031771045` — QR *stalls* at a moderate final error (`0.0329`, not
converged but not blown up) while LU *fully diverges* on the exact same initial guess
(final error `4.37e11`). This is the one direct, reproducible piece of evidence in
this Monte Carlo study that QR's better conditioning changes an actual outcome, not
just a diagnostic number — reported as the concrete example, not just the conditioning
sweep's abstract prediction.

**Mutation check** (`handoffs/RUN_M3.md` section 10): flipped the sign of the
`dA1_da` term in `src/jacobian.py`'s `dCT_dk2` formula (`- (k2/a)*dA1_da`, previously
`+`). Caught immediately by 3 of 82 tests: both Jacobian-vs-finite-difference tests
(`test_jacobian_matches_finite_differences_at_ground_truth_per_block`,
`..._at_random_feasible_points` — the latter with a max relative error of `2190`, not
a subtle drift) and, as a useful cascade, the noiseless-recovery test
(`test_irgnm_recovers_ground_truth_from_small_perturbation_qr`), confirming a broken
Jacobian breaks the solver built on top of it, not just its own unit test. No
strengthening needed; reverted and the full suite (`python3 -m pytest tests/ -q`)
re-run green (82/82) after revert.
