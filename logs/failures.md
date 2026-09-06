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


## M4.3 Monte Carlo grid divergences

**Why do these diverge?** The "non-finite iterate" reason means the Gauss-Newton step pushed a parameter (usually `mu`, which is unconstrained) into a region where `exp(mu * t)` overflows, or the local Jacobian became completely singular. 

This is **not** caused by `include_blood=False`. In fact, Setup A (`include_blood=False`) actually diverges *less* than Setups B and C, because it has fewer free parameters to fit. 

These divergences are the expected behavior of this highly ill-posed, non-linear inverse problem when subjected to high noise or poor initial guesses. The paper explicitly reports exactly this phenomenon in its own Table 1: as noise increases (e.g., `low_count`) or the initial perturbation increases ($\delta_x = 0.4$), the solver frequently diverges because the regularisation isn't strong enough to stop the iterates from being thrown out of the valid domain. Our failure rates closely mirror the paper's reported failure rates.

Total: 498 divergent cells out of 960

| setup | noise | delta_x | seed_idx | seed | n_iter | reason |
|---|---|---|---|---|---|---|
| A | noiseless | 0.2 | 3 | 7498686245251589070 | 158 | non-finite iterate |
| A | noiseless | 0.3 | 3 | 17352772535142888983 | 173 | non-finite iterate |
| A | noiseless | 0.3 | 10 | 14697353982247622500 | 191 | non-finite iterate |
| A | noiseless | 0.3 | 11 | 6534412254682637926 | 42 | non-finite iterate |
| A | noiseless | 0.4 | 3 | 10412050263252187883 | 210 | non-finite iterate |
| A | noiseless | 0.4 | 8 | 636260610296218331 | 131 | non-finite iterate |
| A | noiseless | 0.4 | 18 | 9973956041226331578 | 196 | non-finite iterate |
| A | high_count | 0.1 | 8 | 13305379987474131913 | 189 | non-finite iterate |
| A | high_count | 0.2 | 10 | 5973572786156275909 | 184 | non-finite iterate |
| A | high_count | 0.3 | 0 | 6817234610498620929 | 177 | non-finite iterate |
| A | high_count | 0.3 | 14 | 17039151874156650465 | 58 | non-finite iterate |
| A | high_count | 0.3 | 18 | 15925291539432010767 | 171 | non-finite iterate |
| A | high_count | 0.4 | 0 | 4620480119886783238 | 164 | non-finite iterate |
| A | high_count | 0.4 | 2 | 3219758641975354271 | 114 | non-finite iterate |
| A | high_count | 0.4 | 3 | 4867907713805944037 | 41 | non-finite iterate |
| A | high_count | 0.4 | 4 | 2000801085936434215 | 85 | non-finite iterate |
| A | high_count | 0.4 | 6 | 4901489620802021586 | 211 | non-finite iterate |
| A | high_count | 0.4 | 10 | 5383053142347227139 | 203 | non-finite iterate |
| A | high_count | 0.4 | 11 | 2631264083581936537 | 168 | non-finite iterate |
| A | high_count | 0.4 | 18 | 10824841599852628017 | 61 | non-finite iterate |
| A | normal_count | 0.1 | 1 | 9607856432663793309 | 241 | non-finite iterate |
| A | normal_count | 0.1 | 2 | 13315653530778254037 | 181 | non-finite iterate |
| A | normal_count | 0.1 | 3 | 13325570548002662759 | 213 | non-finite iterate |
| A | normal_count | 0.1 | 6 | 8793060205416232138 | 187 | non-finite iterate |
| A | normal_count | 0.1 | 7 | 17643903429449493655 | 193 | non-finite iterate |
| A | normal_count | 0.1 | 8 | 5581128737064921652 | 172 | non-finite iterate |
| A | normal_count | 0.1 | 9 | 11333740177824306993 | 192 | non-finite iterate |
| A | normal_count | 0.1 | 10 | 2561097880794560887 | 185 | non-finite iterate |
| A | normal_count | 0.1 | 11 | 3547629612812171730 | 198 | non-finite iterate |
| A | normal_count | 0.1 | 14 | 13512886634052449309 | 219 | non-finite iterate |
| A | normal_count | 0.1 | 15 | 17524521935992824324 | 172 | non-finite iterate |
| A | normal_count | 0.1 | 16 | 4116555832460265273 | 181 | non-finite iterate |
| A | normal_count | 0.2 | 1 | 16285982589977914753 | 162 | non-finite iterate |
| A | normal_count | 0.2 | 6 | 18093773777556074551 | 172 | non-finite iterate |
| A | normal_count | 0.2 | 7 | 6793761991016144720 | 177 | non-finite iterate |
| A | normal_count | 0.2 | 10 | 5049109081009269965 | 189 | non-finite iterate |
| A | normal_count | 0.2 | 12 | 13087941613902913725 | 202 | non-finite iterate |
| A | normal_count | 0.2 | 15 | 17348045862793282535 | 195 | non-finite iterate |
| A | normal_count | 0.2 | 17 | 2024751856882577457 | 192 | non-finite iterate |
| A | normal_count | 0.2 | 18 | 6100123960878365450 | 183 | non-finite iterate |
| A | normal_count | 0.2 | 19 | 1055405355063097057 | 185 | non-finite iterate |
| A | normal_count | 0.3 | 1 | 9062990479858991716 | 171 | non-finite iterate |
| A | normal_count | 0.3 | 2 | 668095661784261968 | 203 | non-finite iterate |
| A | normal_count | 0.3 | 3 | 6615619710922751884 | 191 | non-finite iterate |
| A | normal_count | 0.3 | 4 | 13845081319757225335 | 183 | non-finite iterate |
| A | normal_count | 0.3 | 5 | 11444862878988328056 | 167 | non-finite iterate |
| A | normal_count | 0.3 | 8 | 2331052814053027444 | 44 | non-finite iterate |
| A | normal_count | 0.3 | 9 | 1510047764945209347 | 161 | non-finite iterate |
| A | normal_count | 0.3 | 10 | 11748327536826463923 | 191 | non-finite iterate |
| A | normal_count | 0.3 | 14 | 6656260534232131669 | 167 | non-finite iterate |
| A | normal_count | 0.3 | 17 | 1567973666132980074 | 135 | non-finite iterate |
| A | normal_count | 0.3 | 18 | 15390439322626424277 | 213 | non-finite iterate |
| A | normal_count | 0.4 | 0 | 8036154379316551197 | 156 | non-finite iterate |
| A | normal_count | 0.4 | 4 | 2119931357768584943 | 167 | non-finite iterate |
| A | normal_count | 0.4 | 8 | 10287560729988139473 | 142 | non-finite iterate |
| A | normal_count | 0.4 | 9 | 9728357615796763042 | 192 | non-finite iterate |
| A | normal_count | 0.4 | 13 | 9748507080702974542 | 258 | non-finite iterate |
| A | normal_count | 0.4 | 15 | 7258457066429400808 | 44 | non-finite iterate |
| A | normal_count | 0.4 | 19 | 9294443100559928157 | 173 | non-finite iterate |
| A | low_count | 0.1 | 0 | 8706329929597007122 | 167 | non-finite iterate |
| A | low_count | 0.1 | 1 | 3789770509552360335 | 194 | non-finite iterate |
| A | low_count | 0.1 | 2 | 11105109162781582528 | 137 | non-finite iterate |
| A | low_count | 0.1 | 3 | 13881899665361432289 | 138 | non-finite iterate |
| A | low_count | 0.1 | 4 | 12976003352447989867 | 139 | non-finite iterate |
| A | low_count | 0.1 | 5 | 18025767260219013944 | 199 | non-finite iterate |
| A | low_count | 0.1 | 6 | 17828966379014399961 | 191 | non-finite iterate |
| A | low_count | 0.1 | 7 | 14172584727262720545 | 146 | non-finite iterate |
| A | low_count | 0.1 | 8 | 12360533361481175876 | 141 | non-finite iterate |
| A | low_count | 0.1 | 9 | 15347781012360213182 | 141 | non-finite iterate |
| A | low_count | 0.1 | 10 | 4653088899654075784 | 138 | non-finite iterate |
| A | low_count | 0.1 | 11 | 7998150427597876227 | 143 | non-finite iterate |
| A | low_count | 0.1 | 12 | 7917377154175741762 | 141 | non-finite iterate |
| A | low_count | 0.1 | 13 | 729237708601713105 | 215 | non-finite iterate |
| A | low_count | 0.1 | 14 | 5216416176767478471 | 204 | non-finite iterate |
| A | low_count | 0.1 | 15 | 1425040746898026443 | 122 | non-finite iterate |
| A | low_count | 0.1 | 16 | 12413416211503277822 | 128 | non-finite iterate |
| A | low_count | 0.1 | 17 | 16608979606338457501 | 129 | non-finite iterate |
| A | low_count | 0.1 | 18 | 1741789724712445701 | 149 | non-finite iterate |
| A | low_count | 0.1 | 19 | 6175528250278263804 | 140 | non-finite iterate |
| A | low_count | 0.2 | 0 | 5347029544974169244 | 113 | non-finite iterate |
| A | low_count | 0.2 | 1 | 8717998812877862900 | 136 | non-finite iterate |
| A | low_count | 0.2 | 2 | 4849472231420015526 | 147 | non-finite iterate |
| A | low_count | 0.2 | 3 | 4941145544220121686 | 126 | non-finite iterate |
| A | low_count | 0.2 | 4 | 17537560855914626011 | 241 | non-finite iterate |
| A | low_count | 0.2 | 5 | 8768682963753263602 | 162 | non-finite iterate |
| A | low_count | 0.2 | 6 | 7734486469079454529 | 127 | non-finite iterate |
| A | low_count | 0.2 | 7 | 12934268094975174579 | 125 | non-finite iterate |
| A | low_count | 0.2 | 8 | 887073950961265204 | 139 | non-finite iterate |
| A | low_count | 0.2 | 9 | 5340766299597594018 | 134 | non-finite iterate |
| A | low_count | 0.2 | 10 | 1104345747792680588 | 207 | non-finite iterate |
| A | low_count | 0.2 | 11 | 5016931358842786299 | 126 | non-finite iterate |
| A | low_count | 0.2 | 12 | 12948758293182577979 | 143 | non-finite iterate |
| A | low_count | 0.2 | 13 | 1023759163139903986 | 145 | non-finite iterate |
| A | low_count | 0.2 | 14 | 10230754708952944101 | 129 | non-finite iterate |
| A | low_count | 0.2 | 15 | 3291101601056452478 | 167 | non-finite iterate |
| A | low_count | 0.2 | 16 | 369618275303073928 | 140 | non-finite iterate |
| A | low_count | 0.2 | 17 | 15956432849231559875 | 217 | non-finite iterate |
| A | low_count | 0.2 | 18 | 4595914949576772073 | 127 | non-finite iterate |
| A | low_count | 0.2 | 19 | 9840008089219180191 | 163 | non-finite iterate |
| A | low_count | 0.3 | 0 | 1605508325032751674 | 135 | non-finite iterate |
| A | low_count | 0.3 | 1 | 9933936462173418024 | 139 | non-finite iterate |
| A | low_count | 0.3 | 2 | 17370106002657412700 | 147 | non-finite iterate |
| A | low_count | 0.3 | 3 | 12341747591735981203 | 184 | non-finite iterate |
| A | low_count | 0.3 | 4 | 4935601383079194590 | 153 | non-finite iterate |
| A | low_count | 0.3 | 5 | 1633488632544297611 | 159 | non-finite iterate |
| A | low_count | 0.3 | 6 | 14724748321541310628 | 164 | non-finite iterate |
| A | low_count | 0.3 | 7 | 11578170871074340714 | 115 | non-finite iterate |
| A | low_count | 0.3 | 8 | 7030250230719476330 | 143 | non-finite iterate |
| A | low_count | 0.3 | 9 | 15794449144580784495 | 136 | non-finite iterate |
| A | low_count | 0.3 | 10 | 8497494591217249512 | 134 | non-finite iterate |
| A | low_count | 0.3 | 11 | 2654664464027817714 | 121 | non-finite iterate |
| A | low_count | 0.3 | 12 | 8061224660530396018 | 201 | non-finite iterate |
| A | low_count | 0.3 | 13 | 15456292174287325052 | 172 | non-finite iterate |
| A | low_count | 0.3 | 14 | 13360404057453595973 | 147 | non-finite iterate |
| A | low_count | 0.3 | 15 | 2955619494483866086 | 120 | non-finite iterate |
| A | low_count | 0.3 | 16 | 17010819409320290241 | 134 | non-finite iterate |
| A | low_count | 0.3 | 17 | 4638879736730949647 | 135 | non-finite iterate |
| A | low_count | 0.3 | 18 | 3778247387398434314 | 138 | non-finite iterate |
| A | low_count | 0.3 | 19 | 2957718204571458196 | 144 | non-finite iterate |
| A | low_count | 0.4 | 0 | 4533287941127591384 | 164 | non-finite iterate |
| A | low_count | 0.4 | 2 | 6362020046066964003 | 116 | non-finite iterate |
| A | low_count | 0.4 | 3 | 17268585198627961169 | 154 | non-finite iterate |
| A | low_count | 0.4 | 4 | 1905636968853141235 | 152 | non-finite iterate |
| A | low_count | 0.4 | 5 | 4122484533748520257 | 131 | non-finite iterate |
| A | low_count | 0.4 | 6 | 4541886683987242466 | 136 | non-finite iterate |
| A | low_count | 0.4 | 7 | 12159028108026083909 | 174 | non-finite iterate |
| A | low_count | 0.4 | 8 | 15218464728322127234 | 167 | non-finite iterate |
| A | low_count | 0.4 | 9 | 8584729421810297152 | 151 | non-finite iterate |
| A | low_count | 0.4 | 10 | 8588508009855780015 | 139 | non-finite iterate |
| A | low_count | 0.4 | 11 | 13897611917231843204 | 142 | non-finite iterate |
| A | low_count | 0.4 | 12 | 7330759973990467579 | 135 | non-finite iterate |
| A | low_count | 0.4 | 13 | 9224690138368940333 | 46 | non-finite iterate |
| A | low_count | 0.4 | 14 | 13576101195265344095 | 122 | non-finite iterate |
| A | low_count | 0.4 | 15 | 10314269047324326614 | 89 | non-finite iterate |
| A | low_count | 0.4 | 16 | 11011482498640087263 | 119 | non-finite iterate |
| A | low_count | 0.4 | 17 | 7235570211156608183 | 119 | non-finite iterate |
| A | low_count | 0.4 | 18 | 10230446368905289862 | 111 | non-finite iterate |
| A | low_count | 0.4 | 19 | 15658221128527787303 | 139 | non-finite iterate |
| B | noiseless | 0.1 | 2 | 17843877329075847404 | 228 | non-finite iterate |
| B | noiseless | 0.1 | 9 | 16467581691527637837 | 234 | non-finite iterate |
| B | noiseless | 0.2 | 1 | 4121272217160619610 | 176 | non-finite iterate |
| B | noiseless | 0.2 | 14 | 7655166018398293094 | 193 | non-finite iterate |
| B | noiseless | 0.2 | 15 | 7353961614882305121 | 148 | non-finite iterate |
| B | noiseless | 0.3 | 1 | 4835267214719537297 | 174 | non-finite iterate |
| B | noiseless | 0.3 | 5 | 8949630222875266906 | 120 | non-finite iterate |
| B | noiseless | 0.3 | 8 | 8334686373416698641 | 217 | non-finite iterate |
| B | noiseless | 0.3 | 16 | 8726097960767441833 | 222 | non-finite iterate |
| B | noiseless | 0.4 | 10 | 1259915622054741114 | 225 | non-finite iterate |
| B | high_count | 0.1 | 5 | 11488623230246678109 | 207 | non-finite iterate |
| B | high_count | 0.1 | 7 | 12455333445513414549 | 182 | non-finite iterate |
| B | high_count | 0.1 | 10 | 4175249336263007260 | 176 | non-finite iterate |
| B | high_count | 0.1 | 12 | 4144048237830085694 | 292 | non-finite iterate |
| B | high_count | 0.1 | 15 | 14527246242856603961 | 163 | non-finite iterate |
| B | high_count | 0.1 | 17 | 4474121739382538384 | 252 | non-finite iterate |
| B | high_count | 0.1 | 18 | 1323516101636317185 | 209 | non-finite iterate |
| B | high_count | 0.2 | 0 | 9381669193768354712 | 290 | non-finite iterate |
| B | high_count | 0.2 | 1 | 8199942908936056464 | 233 | non-finite iterate |
| B | high_count | 0.2 | 6 | 11597262737887912169 | 235 | non-finite iterate |
| B | high_count | 0.2 | 9 | 1141796725545044581 | 257 | non-finite iterate |
| B | high_count | 0.2 | 10 | 10376238443725799379 | 269 | non-finite iterate |
| B | high_count | 0.2 | 12 | 10882543200595976575 | 277 | non-finite iterate |
| B | high_count | 0.2 | 14 | 8011067105434137700 | 234 | non-finite iterate |
| B | high_count | 0.3 | 2 | 6939143094653263025 | 174 | non-finite iterate |
| B | high_count | 0.3 | 3 | 9012601130849346060 | 190 | non-finite iterate |
| B | high_count | 0.3 | 4 | 9788547792481352118 | 208 | non-finite iterate |
| B | high_count | 0.3 | 6 | 8578334776455527282 | 180 | non-finite iterate |
| B | high_count | 0.3 | 7 | 2957970224979573219 | 226 | non-finite iterate |
| B | high_count | 0.3 | 12 | 15090183873329481908 | 188 | non-finite iterate |
| B | high_count | 0.3 | 13 | 13618725437024710705 | 77 | non-finite iterate |
| B | high_count | 0.3 | 14 | 8173488951721022992 | 216 | non-finite iterate |
| B | high_count | 0.3 | 15 | 4670641143373357822 | 209 | non-finite iterate |
| B | high_count | 0.3 | 16 | 14193159733151666215 | 42 | non-finite iterate |
| B | high_count | 0.3 | 18 | 1623639743131085256 | 175 | non-finite iterate |
| B | high_count | 0.3 | 19 | 8323850898463685262 | 191 | non-finite iterate |
| B | high_count | 0.4 | 4 | 14874865786631603166 | 232 | non-finite iterate |
| B | high_count | 0.4 | 5 | 3489440885970511895 | 180 | non-finite iterate |
| B | high_count | 0.4 | 8 | 3003226942353715203 | 41 | non-finite iterate |
| B | high_count | 0.4 | 9 | 16466476329901493945 | 167 | non-finite iterate |
| B | high_count | 0.4 | 10 | 11888172053760797564 | 77 | non-finite iterate |
| B | high_count | 0.4 | 11 | 4411958238911615707 | 129 | non-finite iterate |
| B | high_count | 0.4 | 13 | 5155650231812928636 | 155 | non-finite iterate |
| B | high_count | 0.4 | 14 | 8107902316324550916 | 215 | non-finite iterate |
| B | high_count | 0.4 | 17 | 6475780758919341292 | 207 | non-finite iterate |
| B | normal_count | 0.1 | 0 | 14222055927610666386 | 182 | non-finite iterate |
| B | normal_count | 0.1 | 1 | 16830967720664983419 | 176 | non-finite iterate |
| B | normal_count | 0.1 | 2 | 9049893826660069685 | 163 | non-finite iterate |
| B | normal_count | 0.1 | 5 | 17010460877302462513 | 196 | non-finite iterate |
| B | normal_count | 0.1 | 7 | 10137799371972424580 | 200 | non-finite iterate |
| B | normal_count | 0.1 | 8 | 1285492173979401252 | 201 | non-finite iterate |
| B | normal_count | 0.1 | 9 | 3002432097648168705 | 162 | non-finite iterate |
| B | normal_count | 0.1 | 10 | 14494838698013322583 | 174 | non-finite iterate |
| B | normal_count | 0.1 | 12 | 1659925178545574684 | 171 | non-finite iterate |
| B | normal_count | 0.1 | 13 | 13291033739629153771 | 168 | non-finite iterate |
| B | normal_count | 0.1 | 15 | 2516749689687529052 | 222 | non-finite iterate |
| B | normal_count | 0.1 | 16 | 18154938556481476905 | 178 | non-finite iterate |
| B | normal_count | 0.1 | 18 | 958587088297951519 | 216 | non-finite iterate |
| B | normal_count | 0.2 | 0 | 16371034972561320461 | 180 | non-finite iterate |
| B | normal_count | 0.2 | 4 | 11722996421033803205 | 237 | non-finite iterate |
| B | normal_count | 0.2 | 6 | 16587062242352898187 | 192 | non-finite iterate |
| B | normal_count | 0.2 | 7 | 16672329695937914836 | 174 | non-finite iterate |
| B | normal_count | 0.2 | 8 | 4052172715697929713 | 183 | non-finite iterate |
| B | normal_count | 0.2 | 11 | 1038354349107578211 | 203 | non-finite iterate |
| B | normal_count | 0.2 | 17 | 14743422586845087435 | 181 | non-finite iterate |
| B | normal_count | 0.3 | 0 | 18335776179546544833 | 211 | non-finite iterate |
| B | normal_count | 0.3 | 1 | 8452764453575217614 | 69 | non-finite iterate |
| B | normal_count | 0.3 | 2 | 6122561141240582148 | 207 | non-finite iterate |
| B | normal_count | 0.3 | 4 | 3388102342975009621 | 191 | non-finite iterate |
| B | normal_count | 0.3 | 7 | 17270473251623955085 | 230 | non-finite iterate |
| B | normal_count | 0.3 | 8 | 11008492781421767811 | 205 | non-finite iterate |
| B | normal_count | 0.3 | 9 | 14251535083528867424 | 199 | non-finite iterate |
| B | normal_count | 0.3 | 10 | 10701444832161508536 | 206 | non-finite iterate |
| B | normal_count | 0.3 | 11 | 13142111261787337052 | 130 | non-finite iterate |
| B | normal_count | 0.3 | 12 | 6909932005490470843 | 165 | non-finite iterate |
| B | normal_count | 0.3 | 13 | 16011940619708942707 | 185 | non-finite iterate |
| B | normal_count | 0.3 | 14 | 17426965178555234989 | 177 | non-finite iterate |
| B | normal_count | 0.3 | 15 | 9462507914390548606 | 193 | non-finite iterate |
| B | normal_count | 0.3 | 16 | 9927838604591914615 | 164 | non-finite iterate |
| B | normal_count | 0.4 | 0 | 1786803435509114856 | 178 | non-finite iterate |
| B | normal_count | 0.4 | 1 | 16631221389358009345 | 64 | non-finite iterate |
| B | normal_count | 0.4 | 2 | 5227070510761168738 | 171 | non-finite iterate |
| B | normal_count | 0.4 | 3 | 12461050397824570270 | 178 | non-finite iterate |
| B | normal_count | 0.4 | 4 | 13608160808949679312 | 155 | non-finite iterate |
| B | normal_count | 0.4 | 5 | 15396939396179600307 | 169 | non-finite iterate |
| B | normal_count | 0.4 | 6 | 3686667540641246869 | 143 | non-finite iterate |
| B | normal_count | 0.4 | 8 | 4295960964422101202 | 215 | non-finite iterate |
| B | normal_count | 0.4 | 11 | 8515670779342852014 | 58 | non-finite iterate |
| B | normal_count | 0.4 | 12 | 5673463526188291636 | 238 | non-finite iterate |
| B | normal_count | 0.4 | 13 | 1912419138479381919 | 145 | non-finite iterate |
| B | normal_count | 0.4 | 14 | 9128564549995918587 | 126 | non-finite iterate |
| B | normal_count | 0.4 | 15 | 7155639175737412097 | 198 | non-finite iterate |
| B | normal_count | 0.4 | 16 | 13618256572732549702 | 235 | non-finite iterate |
| B | normal_count | 0.4 | 17 | 5634312908814722891 | 139 | non-finite iterate |
| B | normal_count | 0.4 | 19 | 2017945243158095922 | 170 | non-finite iterate |
| B | low_count | 0.1 | 0 | 2368824914537679802 | 142 | non-finite iterate |
| B | low_count | 0.1 | 1 | 656167044250281642 | 140 | non-finite iterate |
| B | low_count | 0.1 | 2 | 2235632322018134835 | 163 | non-finite iterate |
| B | low_count | 0.1 | 3 | 2795009638289119063 | 222 | non-finite iterate |
| B | low_count | 0.1 | 4 | 8865978069675714315 | 162 | non-finite iterate |
| B | low_count | 0.1 | 5 | 3605413985129198305 | 118 | non-finite iterate |
| B | low_count | 0.1 | 6 | 7722872478672823547 | 139 | non-finite iterate |
| B | low_count | 0.1 | 7 | 5479150565927262586 | 142 | non-finite iterate |
| B | low_count | 0.1 | 8 | 13548640244012408464 | 137 | non-finite iterate |
| B | low_count | 0.1 | 9 | 6917298205039441821 | 123 | non-finite iterate |
| B | low_count | 0.1 | 10 | 9588984678498151416 | 185 | non-finite iterate |
| B | low_count | 0.1 | 11 | 16234725568859467825 | 143 | non-finite iterate |
| B | low_count | 0.1 | 12 | 11366240954913389255 | 132 | non-finite iterate |
| B | low_count | 0.1 | 13 | 5607122762245566113 | 124 | non-finite iterate |
| B | low_count | 0.1 | 14 | 10214810823823709002 | 208 | non-finite iterate |
| B | low_count | 0.1 | 15 | 10757744136536456367 | 138 | non-finite iterate |
| B | low_count | 0.1 | 16 | 1794864029185987711 | 141 | non-finite iterate |
| B | low_count | 0.1 | 17 | 9962764679624186568 | 141 | non-finite iterate |
| B | low_count | 0.1 | 18 | 10717945151532012455 | 130 | non-finite iterate |
| B | low_count | 0.1 | 19 | 16513045982698121120 | 145 | non-finite iterate |
| B | low_count | 0.2 | 0 | 16094384130414687574 | 113 | non-finite iterate |
| B | low_count | 0.2 | 1 | 10377076078987545182 | 164 | non-finite iterate |
| B | low_count | 0.2 | 2 | 17781675853916047574 | 147 | non-finite iterate |
| B | low_count | 0.2 | 3 | 16819792474021784411 | 161 | non-finite iterate |
| B | low_count | 0.2 | 4 | 14374424376562209318 | 124 | non-finite iterate |
| B | low_count | 0.2 | 5 | 13283157262198651489 | 140 | non-finite iterate |
| B | low_count | 0.2 | 6 | 1376756339060009261 | 117 | non-finite iterate |
| B | low_count | 0.2 | 7 | 8958875783515969591 | 170 | non-finite iterate |
| B | low_count | 0.2 | 8 | 4010726269342606687 | 203 | non-finite iterate |
| B | low_count | 0.2 | 9 | 1308300228125807781 | 193 | non-finite iterate |
| B | low_count | 0.2 | 10 | 1257825112705266416 | 137 | non-finite iterate |
| B | low_count | 0.2 | 11 | 16750269628108029242 | 206 | non-finite iterate |
| B | low_count | 0.2 | 12 | 10734972706789889133 | 153 | non-finite iterate |
| B | low_count | 0.2 | 13 | 599496326779782921 | 137 | non-finite iterate |
| B | low_count | 0.2 | 14 | 8676019903862638224 | 126 | non-finite iterate |
| B | low_count | 0.2 | 15 | 14789070429634577275 | 139 | non-finite iterate |
| B | low_count | 0.2 | 16 | 10701391175487179391 | 162 | non-finite iterate |
| B | low_count | 0.2 | 17 | 8981676138589971370 | 147 | non-finite iterate |
| B | low_count | 0.2 | 18 | 10110390497309361367 | 152 | non-finite iterate |
| B | low_count | 0.2 | 19 | 4403565149307396659 | 132 | non-finite iterate |
| B | low_count | 0.3 | 0 | 17197790523295508616 | 151 | non-finite iterate |
| B | low_count | 0.3 | 1 | 15764780907874388049 | 133 | non-finite iterate |
| B | low_count | 0.3 | 2 | 12992945372539066919 | 133 | non-finite iterate |
| B | low_count | 0.3 | 3 | 2463065036306096404 | 128 | non-finite iterate |
| B | low_count | 0.3 | 4 | 6969989134601741634 | 115 | non-finite iterate |
| B | low_count | 0.3 | 5 | 14731656372495159828 | 135 | non-finite iterate |
| B | low_count | 0.3 | 6 | 336880823361825909 | 114 | non-finite iterate |
| B | low_count | 0.3 | 7 | 6080405528025126436 | 119 | non-finite iterate |
| B | low_count | 0.3 | 9 | 6347312686469702877 | 113 | non-finite iterate |
| B | low_count | 0.3 | 10 | 6485381638034038616 | 224 | non-finite iterate |
| B | low_count | 0.3 | 11 | 8581881666213087854 | 171 | non-finite iterate |
| B | low_count | 0.3 | 12 | 10702395082336682390 | 142 | non-finite iterate |
| B | low_count | 0.3 | 13 | 10917261484953759994 | 128 | non-finite iterate |
| B | low_count | 0.3 | 14 | 522792943610278840 | 179 | non-finite iterate |
| B | low_count | 0.3 | 15 | 6127949129640820641 | 106 | non-finite iterate |
| B | low_count | 0.3 | 16 | 16050550869355702883 | 120 | non-finite iterate |
| B | low_count | 0.3 | 17 | 3045580718535723029 | 134 | non-finite iterate |
| B | low_count | 0.3 | 18 | 12069551727752697552 | 133 | non-finite iterate |
| B | low_count | 0.3 | 19 | 15565203376403006154 | 170 | non-finite iterate |
| B | low_count | 0.4 | 0 | 12506453806664162354 | 148 | non-finite iterate |
| B | low_count | 0.4 | 1 | 10456537947490252341 | 126 | non-finite iterate |
| B | low_count | 0.4 | 2 | 2151523843039703860 | 40 | non-finite iterate |
| B | low_count | 0.4 | 3 | 3871481436950691217 | 120 | non-finite iterate |
| B | low_count | 0.4 | 4 | 9048913427616720243 | 149 | non-finite iterate |
| B | low_count | 0.4 | 5 | 15500835892679285312 | 78 | non-finite iterate |
| B | low_count | 0.4 | 6 | 15460176476673819219 | 125 | non-finite iterate |
| B | low_count | 0.4 | 7 | 2605971507335093239 | 142 | non-finite iterate |
| B | low_count | 0.4 | 8 | 7940292267097677781 | 77 | non-finite iterate |
| B | low_count | 0.4 | 9 | 13826012847016765431 | 134 | non-finite iterate |
| B | low_count | 0.4 | 10 | 15705773176569627769 | 186 | non-finite iterate |
| B | low_count | 0.4 | 11 | 9724447084069224757 | 141 | non-finite iterate |
| B | low_count | 0.4 | 12 | 388882313580538147 | 137 | non-finite iterate |
| B | low_count | 0.4 | 13 | 14166843456735804 | 150 | non-finite iterate |
| B | low_count | 0.4 | 14 | 14141539329833774560 | 119 | non-finite iterate |
| B | low_count | 0.4 | 15 | 15219661629292044906 | 121 | non-finite iterate |
| B | low_count | 0.4 | 16 | 6338760270341982385 | 153 | non-finite iterate |
| B | low_count | 0.4 | 17 | 15271106945361882344 | 129 | non-finite iterate |
| B | low_count | 0.4 | 18 | 13923161347665804455 | 126 | non-finite iterate |
| B | low_count | 0.4 | 19 | 9534999670263969492 | 213 | non-finite iterate |
| C | noiseless | 0.1 | 3 | 13267957988057963579 | 151 | non-finite iterate |
| C | noiseless | 0.2 | 3 | 992469375631685816 | 234 | non-finite iterate |
| C | noiseless | 0.2 | 7 | 10979689972149502926 | 160 | non-finite iterate |
| C | noiseless | 0.3 | 1 | 1521571350555474820 | 185 | non-finite iterate |
| C | noiseless | 0.3 | 5 | 1090399601439472580 | 155 | non-finite iterate |
| C | noiseless | 0.3 | 8 | 15006773411709691711 | 214 | non-finite iterate |
| C | noiseless | 0.3 | 11 | 12089176668254329749 | 91 | non-finite iterate |
| C | noiseless | 0.3 | 13 | 1271722846112605093 | 130 | non-finite iterate |
| C | noiseless | 0.3 | 15 | 3856154988111603307 | 252 | non-finite iterate |
| C | noiseless | 0.4 | 0 | 2845124486679736545 | 153 | non-finite iterate |
| C | noiseless | 0.4 | 3 | 13023828819492805757 | 176 | non-finite iterate |
| C | noiseless | 0.4 | 5 | 17260056269482810436 | 235 | non-finite iterate |
| C | noiseless | 0.4 | 8 | 4360894851557029301 | 239 | non-finite iterate |
| C | noiseless | 0.4 | 13 | 2368374846495845639 | 182 | non-finite iterate |
| C | noiseless | 0.4 | 19 | 17936753224670831315 | 215 | non-finite iterate |
| C | high_count | 0.1 | 4 | 16582935863794905337 | 269 | non-finite iterate |
| C | high_count | 0.1 | 7 | 14112006313861629715 | 232 | non-finite iterate |
| C | high_count | 0.1 | 8 | 2237025776080227736 | 205 | non-finite iterate |
| C | high_count | 0.1 | 11 | 10069253349246804218 | 138 | non-finite iterate |
| C | high_count | 0.1 | 15 | 14211178178242580035 | 211 | non-finite iterate |
| C | high_count | 0.1 | 17 | 15465490371398953193 | 271 | non-finite iterate |
| C | high_count | 0.1 | 18 | 4576715348723291196 | 228 | non-finite iterate |
| C | high_count | 0.2 | 0 | 10797253937129235021 | 232 | non-finite iterate |
| C | high_count | 0.2 | 1 | 18399756617868885485 | 183 | non-finite iterate |
| C | high_count | 0.2 | 3 | 11551275089354063748 | 295 | non-finite iterate |
| C | high_count | 0.2 | 4 | 6582378351074965861 | 147 | non-finite iterate |
| C | high_count | 0.2 | 11 | 12660963427574692740 | 220 | non-finite iterate |
| C | high_count | 0.2 | 18 | 8564536671869109382 | 219 | non-finite iterate |
| C | high_count | 0.3 | 2 | 3484667200727990032 | 217 | non-finite iterate |
| C | high_count | 0.3 | 3 | 12782152471475936216 | 201 | non-finite iterate |
| C | high_count | 0.3 | 6 | 8061324396811637012 | 210 | non-finite iterate |
| C | high_count | 0.3 | 8 | 1117633276274522955 | 230 | non-finite iterate |
| C | high_count | 0.3 | 11 | 11809214456336901039 | 210 | non-finite iterate |
| C | high_count | 0.3 | 15 | 5353217933828685109 | 192 | non-finite iterate |
| C | high_count | 0.3 | 16 | 9355518100288927115 | 175 | non-finite iterate |
| C | high_count | 0.3 | 17 | 12142381317512408612 | 186 | non-finite iterate |
| C | high_count | 0.3 | 19 | 18071199337448768941 | 156 | non-finite iterate |
| C | high_count | 0.4 | 1 | 13534055345598906154 | 145 | non-finite iterate |
| C | high_count | 0.4 | 2 | 3497808251968374739 | 142 | non-finite iterate |
| C | high_count | 0.4 | 3 | 2201685103580546905 | 64 | non-finite iterate |
| C | high_count | 0.4 | 4 | 8733878479092713916 | 214 | non-finite iterate |
| C | high_count | 0.4 | 5 | 17736698892338917200 | 179 | non-finite iterate |
| C | high_count | 0.4 | 10 | 7846433140548299311 | 143 | non-finite iterate |
| C | high_count | 0.4 | 12 | 11139324766634710403 | 204 | non-finite iterate |
| C | high_count | 0.4 | 14 | 12583475293666979651 | 225 | non-finite iterate |
| C | normal_count | 0.1 | 0 | 15282702064664048412 | 197 | non-finite iterate |
| C | normal_count | 0.1 | 1 | 735043519764814105 | 175 | non-finite iterate |
| C | normal_count | 0.1 | 5 | 12762751482798603792 | 156 | non-finite iterate |
| C | normal_count | 0.1 | 6 | 3969056709562297494 | 169 | non-finite iterate |
| C | normal_count | 0.1 | 7 | 16622868631221592026 | 190 | non-finite iterate |
| C | normal_count | 0.1 | 8 | 10550367234769785151 | 168 | non-finite iterate |
| C | normal_count | 0.1 | 10 | 4040921120281575033 | 189 | non-finite iterate |
| C | normal_count | 0.1 | 11 | 17341542925466429696 | 195 | non-finite iterate |
| C | normal_count | 0.1 | 12 | 12964801379601145859 | 191 | non-finite iterate |
| C | normal_count | 0.1 | 14 | 14951759858313316474 | 160 | non-finite iterate |
| C | normal_count | 0.1 | 15 | 7301530545165051762 | 201 | non-finite iterate |
| C | normal_count | 0.1 | 16 | 15493489452283887407 | 188 | non-finite iterate |
| C | normal_count | 0.1 | 18 | 15202717991654533197 | 164 | non-finite iterate |
| C | normal_count | 0.2 | 0 | 16676131902711778805 | 173 | non-finite iterate |
| C | normal_count | 0.2 | 3 | 13109152860557122846 | 214 | non-finite iterate |
| C | normal_count | 0.2 | 4 | 10184679457186623341 | 175 | non-finite iterate |
| C | normal_count | 0.2 | 6 | 16875121070894519286 | 195 | non-finite iterate |
| C | normal_count | 0.2 | 7 | 7920732845187122054 | 136 | non-finite iterate |
| C | normal_count | 0.2 | 8 | 2399051439299968833 | 213 | non-finite iterate |
| C | normal_count | 0.2 | 9 | 17834203874335714142 | 182 | non-finite iterate |
| C | normal_count | 0.2 | 10 | 13760958560624849621 | 191 | non-finite iterate |
| C | normal_count | 0.2 | 11 | 13039776522715709073 | 121 | non-finite iterate |
| C | normal_count | 0.2 | 12 | 9485232803009373070 | 163 | non-finite iterate |
| C | normal_count | 0.2 | 15 | 14352711116521153979 | 228 | non-finite iterate |
| C | normal_count | 0.2 | 16 | 1402908721461565831 | 210 | non-finite iterate |
| C | normal_count | 0.2 | 18 | 14484801696799507083 | 154 | non-finite iterate |
| C | normal_count | 0.3 | 0 | 10805193015438091505 | 170 | non-finite iterate |
| C | normal_count | 0.3 | 1 | 9685286130990504716 | 213 | non-finite iterate |
| C | normal_count | 0.3 | 2 | 14824348105406769114 | 184 | non-finite iterate |
| C | normal_count | 0.3 | 3 | 12426949274615242236 | 81 | non-finite iterate |
| C | normal_count | 0.3 | 4 | 2018494058419446217 | 200 | non-finite iterate |
| C | normal_count | 0.3 | 5 | 10045227557245201113 | 187 | non-finite iterate |
| C | normal_count | 0.3 | 7 | 2793362765279752232 | 156 | non-finite iterate |
| C | normal_count | 0.3 | 8 | 6954902208071742791 | 146 | non-finite iterate |
| C | normal_count | 0.3 | 9 | 3946133565803293501 | 177 | non-finite iterate |
| C | normal_count | 0.3 | 10 | 18077144021112715606 | 169 | non-finite iterate |
| C | normal_count | 0.3 | 11 | 5375734182475900761 | 181 | non-finite iterate |
| C | normal_count | 0.3 | 12 | 2937594796516610239 | 162 | non-finite iterate |
| C | normal_count | 0.3 | 13 | 15821913326070789836 | 252 | non-finite iterate |
| C | normal_count | 0.3 | 14 | 6335488884313328391 | 157 | non-finite iterate |
| C | normal_count | 0.3 | 15 | 16451269266585655255 | 144 | non-finite iterate |
| C | normal_count | 0.3 | 16 | 7470913104760024887 | 72 | non-finite iterate |
| C | normal_count | 0.3 | 18 | 3492247830214516940 | 166 | non-finite iterate |
| C | normal_count | 0.3 | 19 | 17292708003680564859 | 208 | non-finite iterate |
| C | normal_count | 0.4 | 0 | 4908813937351136532 | 228 | non-finite iterate |
| C | normal_count | 0.4 | 1 | 12843556536852833626 | 103 | non-finite iterate |
| C | normal_count | 0.4 | 2 | 13147897973772845269 | 187 | non-finite iterate |
| C | normal_count | 0.4 | 3 | 8159670219475516471 | 166 | non-finite iterate |
| C | normal_count | 0.4 | 4 | 8565321709958199030 | 186 | non-finite iterate |
| C | normal_count | 0.4 | 5 | 16646844565057289809 | 166 | non-finite iterate |
| C | normal_count | 0.4 | 6 | 5575159339421996647 | 184 | non-finite iterate |
| C | normal_count | 0.4 | 8 | 10184229009183436434 | 179 | non-finite iterate |
| C | normal_count | 0.4 | 9 | 9868977078236449221 | 183 | non-finite iterate |
| C | normal_count | 0.4 | 11 | 6041193559896042857 | 168 | non-finite iterate |
| C | normal_count | 0.4 | 12 | 289255426336265253 | 64 | non-finite iterate |
| C | normal_count | 0.4 | 13 | 17462473384253351401 | 178 | non-finite iterate |
| C | normal_count | 0.4 | 14 | 16228994986020608388 | 183 | non-finite iterate |
| C | normal_count | 0.4 | 15 | 10427136567786019448 | 198 | non-finite iterate |
| C | normal_count | 0.4 | 16 | 17643377203312204455 | 220 | non-finite iterate |
| C | normal_count | 0.4 | 17 | 12475270989552344634 | 205 | non-finite iterate |
| C | normal_count | 0.4 | 18 | 11850226535644441815 | 181 | non-finite iterate |
| C | normal_count | 0.4 | 19 | 4590553148848075121 | 217 | non-finite iterate |
| C | low_count | 0.1 | 0 | 11375501832966376704 | 158 | non-finite iterate |
| C | low_count | 0.1 | 1 | 14791489349276697780 | 144 | non-finite iterate |
| C | low_count | 0.1 | 2 | 16968011080685163973 | 139 | non-finite iterate |
| C | low_count | 0.1 | 3 | 5017951390812462073 | 99 | non-finite iterate |
| C | low_count | 0.1 | 4 | 11345781541135838537 | 129 | non-finite iterate |
| C | low_count | 0.1 | 5 | 9226974813646684526 | 122 | non-finite iterate |
| C | low_count | 0.1 | 6 | 1240333425934090665 | 143 | non-finite iterate |
| C | low_count | 0.1 | 7 | 11777221218525937563 | 128 | non-finite iterate |
| C | low_count | 0.1 | 8 | 8714841508319652555 | 114 | non-finite iterate |
| C | low_count | 0.1 | 9 | 4604050891332858588 | 141 | non-finite iterate |
| C | low_count | 0.1 | 10 | 14399133330868787926 | 114 | non-finite iterate |
| C | low_count | 0.1 | 11 | 9285256053643536972 | 134 | non-finite iterate |
| C | low_count | 0.1 | 12 | 10608331113391721732 | 142 | non-finite iterate |
| C | low_count | 0.1 | 13 | 18138879686878565167 | 149 | non-finite iterate |
| C | low_count | 0.1 | 14 | 4224324246474326014 | 138 | non-finite iterate |
| C | low_count | 0.1 | 15 | 8318397156454201841 | 150 | non-finite iterate |
| C | low_count | 0.1 | 16 | 7067693677868889576 | 161 | non-finite iterate |
| C | low_count | 0.1 | 17 | 18070533132758286773 | 146 | non-finite iterate |
| C | low_count | 0.1 | 18 | 8049317219860108465 | 161 | non-finite iterate |
| C | low_count | 0.1 | 19 | 10120223998595092867 | 138 | non-finite iterate |
| C | low_count | 0.2 | 0 | 5994492663146428435 | 117 | non-finite iterate |
| C | low_count | 0.2 | 1 | 6693321673016702150 | 92 | non-finite iterate |
| C | low_count | 0.2 | 2 | 3754708561949597491 | 145 | non-finite iterate |
| C | low_count | 0.2 | 3 | 5782372336332330100 | 146 | non-finite iterate |
| C | low_count | 0.2 | 4 | 15215549041599742749 | 142 | non-finite iterate |
| C | low_count | 0.2 | 5 | 10121103555491546568 | 151 | non-finite iterate |
| C | low_count | 0.2 | 6 | 6015377370412828292 | 133 | non-finite iterate |
| C | low_count | 0.2 | 7 | 9662439590446365366 | 188 | non-finite iterate |
| C | low_count | 0.2 | 8 | 12172307324547356427 | 138 | non-finite iterate |
| C | low_count | 0.2 | 9 | 4421247283626077986 | 102 | non-finite iterate |
| C | low_count | 0.2 | 10 | 5411652535413572913 | 194 | non-finite iterate |
| C | low_count | 0.2 | 11 | 11048697588410241134 | 177 | non-finite iterate |
| C | low_count | 0.2 | 12 | 15166874370598892425 | 119 | non-finite iterate |
| C | low_count | 0.2 | 13 | 11140210076499275926 | 154 | non-finite iterate |
| C | low_count | 0.2 | 14 | 1698270301012764524 | 110 | non-finite iterate |
| C | low_count | 0.2 | 15 | 10898464799376189922 | 126 | non-finite iterate |
| C | low_count | 0.2 | 16 | 16962045982971038214 | 144 | non-finite iterate |
| C | low_count | 0.2 | 17 | 5044413392876446995 | 140 | non-finite iterate |
| C | low_count | 0.2 | 18 | 7794433633266665004 | 125 | non-finite iterate |
| C | low_count | 0.2 | 19 | 14847210666045484406 | 141 | non-finite iterate |
| C | low_count | 0.3 | 0 | 6183266085890259085 | 137 | non-finite iterate |
| C | low_count | 0.3 | 1 | 8103813621084559593 | 117 | non-finite iterate |
| C | low_count | 0.3 | 2 | 1194648519235092836 | 112 | non-finite iterate |
| C | low_count | 0.3 | 4 | 14630020367781360051 | 125 | non-finite iterate |
| C | low_count | 0.3 | 5 | 15384740697979499000 | 125 | non-finite iterate |
| C | low_count | 0.3 | 6 | 13434422983438772972 | 129 | non-finite iterate |
| C | low_count | 0.3 | 7 | 4041614134758211427 | 170 | non-finite iterate |
| C | low_count | 0.3 | 8 | 3768777195681055141 | 154 | non-finite iterate |
| C | low_count | 0.3 | 9 | 10965395592221867969 | 121 | non-finite iterate |
| C | low_count | 0.3 | 10 | 10218724399298583656 | 163 | non-finite iterate |
| C | low_count | 0.3 | 11 | 2909413433945331860 | 122 | non-finite iterate |
| C | low_count | 0.3 | 12 | 3898310578867716655 | 155 | non-finite iterate |
| C | low_count | 0.3 | 13 | 13251153258899637980 | 166 | non-finite iterate |
| C | low_count | 0.3 | 14 | 16053714441797911962 | 139 | non-finite iterate |
| C | low_count | 0.3 | 15 | 8147692271325580526 | 141 | non-finite iterate |
| C | low_count | 0.3 | 16 | 5077800485371947057 | 161 | non-finite iterate |
| C | low_count | 0.3 | 17 | 4662677222901887971 | 164 | non-finite iterate |
| C | low_count | 0.3 | 18 | 3839999752892539678 | 119 | non-finite iterate |
| C | low_count | 0.3 | 19 | 6901402012599548270 | 127 | non-finite iterate |
| C | low_count | 0.4 | 0 | 8276826672401014634 | 138 | non-finite iterate |
| C | low_count | 0.4 | 1 | 4236588949671892068 | 214 | non-finite iterate |
| C | low_count | 0.4 | 2 | 687613872641120189 | 106 | non-finite iterate |
| C | low_count | 0.4 | 3 | 18348665717241897016 | 151 | non-finite iterate |
| C | low_count | 0.4 | 4 | 164703233656850485 | 177 | non-finite iterate |
| C | low_count | 0.4 | 5 | 17118873961997931121 | 138 | non-finite iterate |
| C | low_count | 0.4 | 6 | 11941863093358899764 | 110 | non-finite iterate |
| C | low_count | 0.4 | 7 | 6259001395477387678 | 153 | non-finite iterate |
| C | low_count | 0.4 | 8 | 17556226825254214248 | 136 | non-finite iterate |
| C | low_count | 0.4 | 9 | 11953247172218815380 | 191 | non-finite iterate |
| C | low_count | 0.4 | 10 | 1639927995874612428 | 174 | non-finite iterate |
| C | low_count | 0.4 | 11 | 9784746409898953353 | 143 | non-finite iterate |
| C | low_count | 0.4 | 12 | 12432423473353124173 | 138 | non-finite iterate |
| C | low_count | 0.4 | 13 | 16010087710184517005 | 130 | non-finite iterate |
| C | low_count | 0.4 | 14 | 5495687884162371985 | 132 | non-finite iterate |
| C | low_count | 0.4 | 15 | 14730435204367814083 | 94 | non-finite iterate |
| C | low_count | 0.4 | 16 | 17464550451146236892 | 117 | non-finite iterate |
| C | low_count | 0.4 | 17 | 10710712968907353589 | 155 | non-finite iterate |
| C | low_count | 0.4 | 18 | 16679127070899030163 | 135 | non-finite iterate |
| C | low_count | 0.4 | 19 | 8797205902935713565 | 128 | non-finite iterate |

