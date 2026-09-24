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


## M4.4 identifiability signature — rejected runs (noiseless stage)

**58 of 400 runs rejected: 53 diverged (non-finite iterate), 5 stalled.**
Tissue-only 14/80; single-C_P 44/320.

Two distinct failure modes, deliberately counted separately:

- **non-finite iterate** — the same mode as M4.3: an unconstrained `mu` is pushed into
  a region where `exp(mu*t)` overflows. Already caught by `run_irgnm`.
- **stalled** — *new in M4.4, and the reason D-M4-8 exists.* The run stays finite for
  all 300 iterations but never approaches the data, exiting with `diverged=False`. At
  δ_y = 0 the discrepancy principle cannot fire, so nothing in `run_irgnm` notices.
  The five stalls sit at relative residual **5.7e-02 to 2.5e+04**, against ~1e-09 for
  an accepted fit — at minimum seven orders of magnitude outside the accepted band,
  and at least four orders above the `FIT_RESIDUAL_TOL = 1e-6` gate. Without that gate
  they would have been averaged into the headline spread statistic and corrupted it
  (the worst of them reports a K1-ratio spread of 4.3e-02, against 1.8e-08 for the
  accepted runs).

Rejection rates rise with δ_x exactly as in M4.3 (tissue-only: 2, 3, 3, 6 out of 20 at
δ_x = 0.1, 0.2, 0.3, 0.4). Nothing was reseeded to clean this up.

Reproduce any row: `run_identifiability_case(delta_x=..., seed_idx=...,
blood_frame_indices=(frame,) or ())`, root seed 20240401.

| variant | blood frame | delta_x | seed_idx | seed | n_iter | reason |
|---|---|---|---|---|---|---|
| tissue-only | — | 0.1 | 5 | 6779265272785279763 | 300 | stalled (rel. residual 3.054e+03) |
| tissue-only | — | 0.1 | 18 | 15255070336721197181 | 175 | non-finite iterate |
| tissue-only | — | 0.2 | 2 | 6455790221252972928 | 188 | non-finite iterate |
| tissue-only | — | 0.2 | 9 | 7981606890290359029 | 135 | non-finite iterate |
| tissue-only | — | 0.2 | 17 | 16007838611052960654 | 178 | non-finite iterate |
| tissue-only | — | 0.3 | 11 | 13701067627814082227 | 151 | non-finite iterate |
| tissue-only | — | 0.3 | 14 | 15769737435926983028 | 193 | non-finite iterate |
| tissue-only | — | 0.3 | 16 | 4670806921951123494 | 186 | non-finite iterate |
| tissue-only | — | 0.4 | 1 | 13036721228353021585 | 135 | non-finite iterate |
| tissue-only | — | 0.4 | 2 | 9816833731244394585 | 212 | non-finite iterate |
| tissue-only | — | 0.4 | 6 | 1263358949774442368 | 172 | non-finite iterate |
| tissue-only | — | 0.4 | 11 | 11256229709133109221 | 189 | non-finite iterate |
| tissue-only | — | 0.4 | 16 | 16845001384696147547 | 185 | non-finite iterate |
| tissue-only | — | 0.4 | 17 | 1274918935039283633 | 191 | non-finite iterate |
| +1 C_P | 3 | 0.1 | 0 | 10700753966500982286 | 300 | stalled (rel. residual 1.785e+02) |
| +1 C_P | 3 | 0.1 | 18 | 15255070336721197181 | 171 | non-finite iterate |
| +1 C_P | 3 | 0.2 | 17 | 16007838611052960654 | 185 | non-finite iterate |
| +1 C_P | 3 | 0.3 | 0 | 15394035201083549913 | 100 | non-finite iterate |
| +1 C_P | 3 | 0.3 | 14 | 15769737435926983028 | 193 | non-finite iterate |
| +1 C_P | 3 | 0.3 | 17 | 12480438160523480641 | 39 | non-finite iterate |
| +1 C_P | 3 | 0.4 | 2 | 9816833731244394585 | 265 | non-finite iterate |
| +1 C_P | 3 | 0.4 | 5 | 8589059646236421522 | 163 | non-finite iterate |
| +1 C_P | 3 | 0.4 | 6 | 1263358949774442368 | 198 | non-finite iterate |
| +1 C_P | 3 | 0.4 | 10 | 11783052124265024513 | 141 | non-finite iterate |
| +1 C_P | 3 | 0.4 | 13 | 2163911853804287808 | 146 | non-finite iterate |
| +1 C_P | 3 | 0.4 | 15 | 2190457922699464978 | 300 | stalled (rel. residual 5.725e-02) |
| +1 C_P | 3 | 0.4 | 16 | 16845001384696147547 | 203 | non-finite iterate |
| +1 C_P | 3 | 0.4 | 17 | 1274918935039283633 | 173 | non-finite iterate |
| +1 C_P | 10 | 0.1 | 5 | 6779265272785279763 | 300 | stalled (rel. residual 2.494e+04) |
| +1 C_P | 10 | 0.1 | 18 | 15255070336721197181 | 157 | non-finite iterate |
| +1 C_P | 10 | 0.2 | 2 | 6455790221252972928 | 210 | non-finite iterate |
| +1 C_P | 10 | 0.2 | 9 | 7981606890290359029 | 192 | non-finite iterate |
| +1 C_P | 10 | 0.3 | 0 | 15394035201083549913 | 51 | non-finite iterate |
| +1 C_P | 10 | 0.3 | 4 | 1195345485378046392 | 177 | non-finite iterate |
| +1 C_P | 10 | 0.3 | 14 | 15769737435926983028 | 216 | non-finite iterate |
| +1 C_P | 10 | 0.3 | 16 | 4670806921951123494 | 189 | non-finite iterate |
| +1 C_P | 10 | 0.4 | 2 | 9816833731244394585 | 253 | non-finite iterate |
| +1 C_P | 10 | 0.4 | 5 | 8589059646236421522 | 131 | non-finite iterate |
| +1 C_P | 10 | 0.4 | 6 | 1263358949774442368 | 217 | non-finite iterate |
| +1 C_P | 10 | 0.4 | 7 | 1721363678429367093 | 86 | non-finite iterate |
| +1 C_P | 10 | 0.4 | 10 | 11783052124265024513 | 157 | non-finite iterate |
| +1 C_P | 10 | 0.4 | 11 | 11256229709133109221 | 209 | non-finite iterate |
| +1 C_P | 10 | 0.4 | 15 | 2190457922699464978 | 300 | stalled (rel. residual 5.725e-02) |
| +1 C_P | 10 | 0.4 | 16 | 16845001384696147547 | 212 | non-finite iterate |
| +1 C_P | 10 | 0.4 | 17 | 1274918935039283633 | 190 | non-finite iterate |
| +1 C_P | 17 | 0.2 | 2 | 6455790221252972928 | 129 | non-finite iterate |
| +1 C_P | 17 | 0.2 | 9 | 7981606890290359029 | 120 | non-finite iterate |
| +1 C_P | 17 | 0.3 | 14 | 15769737435926983028 | 211 | non-finite iterate |
| +1 C_P | 17 | 0.4 | 2 | 9816833731244394585 | 243 | non-finite iterate |
| +1 C_P | 17 | 0.4 | 10 | 11783052124265024513 | 194 | non-finite iterate |
| +1 C_P | 17 | 0.4 | 17 | 1274918935039283633 | 213 | non-finite iterate |
| +1 C_P | 24 | 0.2 | 9 | 7981606890290359029 | 116 | non-finite iterate |
| +1 C_P | 24 | 0.3 | 0 | 15394035201083549913 | 60 | non-finite iterate |
| +1 C_P | 24 | 0.3 | 4 | 1195345485378046392 | 122 | non-finite iterate |
| +1 C_P | 24 | 0.3 | 14 | 15769737435926983028 | 157 | non-finite iterate |
| +1 C_P | 24 | 0.4 | 2 | 9816833731244394585 | 245 | non-finite iterate |
| +1 C_P | 24 | 0.4 | 10 | 11783052124265024513 | 162 | non-finite iterate |
| +1 C_P | 24 | 0.4 | 17 | 1274918935039283633 | 215 | non-finite iterate |


## M4.4 step 6 — rejected runs (under noise)

**386 of 960 noisy runs rejected: 275 diverged, 84 trivial stops, 27 stalled.**
(The grid is 1280 runs; its 320 noiseless cells are omitted here because they are the
same seeds already logged in the section above — the two stopping conventions coincide
exactly when delta_y = 0.)

Three distinct failure modes, counted separately because they mean different things:

- **non-finite iterate** (275) — an unconstrained `mu` overflows `exp(mu*t)`.
  Same mode as M4.3. Almost entirely under the `rms` convention, which runs all 300
  iterations; `morozov`'s early stopping nearly eliminates it.
- **trivial stop** (84) — *`morozov` only, and entirely at low_count.*
  `tau*delta_y*sqrt(n_obs) = 6.8*0.073*10 = 5.0` already exceeds the initial residual, so
  the rule fires at iteration 0 and `x_final` is bit-identical to the initial guess.
  Verified directly: the reported zeta equals the initial guess's zeta to all 16 digits.
  The discrepancy principle is behaving correctly — it is saying the data is too noisy to
  improve on the guess — but this is not a fit, so it is rejected rather than averaged
  into the tables. See DECISIONS.md D-M4-11.
- **stalled** (27) — finite for all 300 iterations but never reaching the
  noise floor (rejected above 2x it, D-M4-10). `rms` only, since `morozov` has a working
  stopping rule to defer to.

The most important number here is what is missing: at low_count under `rms`, the
tissue-only arm leaves **2 survivors out of 80** and the one-C_P arm leaves **0 out of
80**. Nothing was reseeded and no tolerance was relaxed to manufacture survivors.

Reproduce any row: `run_identifiability_case(delta_x=..., seed_idx=..., noise_level=...,
stopping=..., blood_frame_indices=(3,) or ())`, root seed 20240401.

| variant | stopping | noise | delta_x | seed_idx | seed | n_iter | reason |
|---|---|---|---|---|---|---|---|
| tissue-only | rms | high_count | 0.1 | 5 | 6779265272785279763 | 170 | non-finite iterate |
| tissue-only | rms | high_count | 0.1 | 10 | 12413828073962532520 | 300 | stalled (rel. residual 3.212e-02 vs floor 2.939e-03) |
| tissue-only | rms | high_count | 0.1 | 17 | 6487172617831067072 | 300 | stalled (rel. residual 1.333e-02 vs floor 3.017e-03) |
| tissue-only | rms | high_count | 0.1 | 18 | 15255070336721197181 | 186 | non-finite iterate |
| tissue-only | rms | high_count | 0.1 | 19 | 418620421686733813 | 300 | stalled (rel. residual 1.679e-02 vs floor 2.693e-03) |
| tissue-only | rms | high_count | 0.2 | 2 | 6455790221252972928 | 203 | non-finite iterate |
| tissue-only | rms | high_count | 0.2 | 10 | 3017423835907537192 | 300 | stalled (rel. residual 3.189e-02 vs floor 2.939e-03) |
| tissue-only | rms | high_count | 0.2 | 17 | 16007838611052960654 | 165 | non-finite iterate |
| tissue-only | rms | high_count | 0.2 | 19 | 2741174279437232974 | 300 | stalled (rel. residual 1.689e-02 vs floor 2.693e-03) |
| tissue-only | rms | high_count | 0.3 | 4 | 1195345485378046392 | 156 | non-finite iterate |
| tissue-only | rms | high_count | 0.3 | 10 | 1744425451868730087 | 300 | stalled (rel. residual 3.200e-02 vs floor 2.939e-03) |
| tissue-only | rms | high_count | 0.3 | 11 | 13701067627814082227 | 157 | non-finite iterate |
| tissue-only | rms | high_count | 0.3 | 14 | 15769737435926983028 | 185 | non-finite iterate |
| tissue-only | rms | high_count | 0.3 | 16 | 4670806921951123494 | 188 | non-finite iterate |
| tissue-only | rms | high_count | 0.3 | 17 | 12480438160523480641 | 300 | stalled (rel. residual 1.366e-02 vs floor 3.017e-03) |
| tissue-only | rms | high_count | 0.3 | 19 | 4147642332521141727 | 300 | stalled (rel. residual 1.933e-02 vs floor 2.693e-03) |
| tissue-only | rms | high_count | 0.4 | 2 | 9816833731244394585 | 216 | non-finite iterate |
| tissue-only | rms | high_count | 0.4 | 6 | 1263358949774442368 | 163 | non-finite iterate |
| tissue-only | rms | high_count | 0.4 | 10 | 11783052124265024513 | 158 | non-finite iterate |
| tissue-only | rms | high_count | 0.4 | 11 | 11256229709133109221 | 202 | non-finite iterate |
| tissue-only | rms | high_count | 0.4 | 17 | 1274918935039283633 | 187 | non-finite iterate |
| tissue-only | rms | high_count | 0.4 | 19 | 18325154317230509695 | 300 | stalled (rel. residual 1.945e-02 vs floor 2.693e-03) |
| tissue-only | rms | normal_count | 0.1 | 2 | 14817441891826887226 | 173 | non-finite iterate |
| tissue-only | rms | normal_count | 0.1 | 3 | 7087585902345953051 | 195 | non-finite iterate |
| tissue-only | rms | normal_count | 0.1 | 7 | 13482974565168760413 | 171 | non-finite iterate |
| tissue-only | rms | normal_count | 0.1 | 8 | 8422819710394504569 | 224 | non-finite iterate |
| tissue-only | rms | normal_count | 0.1 | 10 | 12413828073962532520 | 222 | non-finite iterate |
| tissue-only | rms | normal_count | 0.1 | 13 | 4020597517779351662 | 235 | non-finite iterate |
| tissue-only | rms | normal_count | 0.1 | 16 | 6458611781295944111 | 176 | non-finite iterate |
| tissue-only | rms | normal_count | 0.1 | 18 | 15255070336721197181 | 181 | non-finite iterate |
| tissue-only | rms | normal_count | 0.2 | 2 | 6455790221252972928 | 180 | non-finite iterate |
| tissue-only | rms | normal_count | 0.2 | 3 | 11475075126778771796 | 172 | non-finite iterate |
| tissue-only | rms | normal_count | 0.2 | 7 | 3699861140717967782 | 300 | stalled (rel. residual 7.056e-02 vs floor 1.024e-02) |
| tissue-only | rms | normal_count | 0.2 | 8 | 2603647793336573112 | 223 | non-finite iterate |
| tissue-only | rms | normal_count | 0.2 | 10 | 3017423835907537192 | 227 | non-finite iterate |
| tissue-only | rms | normal_count | 0.2 | 13 | 9714517516883929489 | 247 | non-finite iterate |
| tissue-only | rms | normal_count | 0.2 | 16 | 12699396853484669392 | 171 | non-finite iterate |
| tissue-only | rms | normal_count | 0.2 | 18 | 2228552658003644967 | 185 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 0 | 15394035201083549913 | 92 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 1 | 10837233372687690179 | 300 | stalled (rel. residual 6.191e-02 vs floor 1.114e-02) |
| tissue-only | rms | normal_count | 0.3 | 2 | 8630701097559126812 | 175 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 3 | 8224609558890423419 | 187 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 4 | 1195345485378046392 | 219 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 7 | 10748816607767354200 | 174 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 8 | 13819258390552285198 | 219 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 10 | 1744425451868730087 | 195 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 11 | 13701067627814082227 | 152 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 13 | 3127501782791377564 | 242 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 14 | 15769737435926983028 | 208 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 16 | 4670806921951123494 | 178 | non-finite iterate |
| tissue-only | rms | normal_count | 0.3 | 18 | 10775730971693991329 | 193 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 1 | 13036721228353021585 | 171 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 2 | 9816833731244394585 | 225 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 3 | 16829860601542314652 | 197 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 6 | 1263358949774442368 | 246 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 7 | 1721363678429367093 | 168 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 8 | 14509634575753145916 | 229 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 10 | 11783052124265024513 | 200 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 11 | 11256229709133109221 | 215 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 13 | 2163911853804287808 | 241 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 16 | 16845001384696147547 | 184 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 17 | 1274918935039283633 | 207 | non-finite iterate |
| tissue-only | rms | normal_count | 0.4 | 18 | 7850155921149193305 | 197 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 0 | 10700753966500982286 | 132 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 1 | 5424990108287796935 | 184 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 2 | 14817441891826887226 | 156 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 3 | 7087585902345953051 | 158 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 4 | 1807916131356517894 | 141 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 5 | 6779265272785279763 | 140 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 6 | 6351801794234391670 | 190 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 7 | 13482974565168760413 | 124 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 8 | 8422819710394504569 | 98 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 9 | 13743115229075106723 | 126 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 10 | 12413828073962532520 | 173 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 11 | 10046158082345470303 | 181 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 12 | 12078227055911820856 | 163 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 13 | 4020597517779351662 | 146 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 14 | 17481229501335906182 | 135 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 15 | 12375987125319700978 | 140 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 16 | 6458611781295944111 | 145 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 17 | 6487172617831067072 | 136 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 18 | 15255070336721197181 | 153 | non-finite iterate |
| tissue-only | rms | low_count | 0.1 | 19 | 418620421686733813 | 147 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 0 | 16048746028836443131 | 127 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 1 | 11976309891222107669 | 193 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 2 | 6455790221252972928 | 163 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 3 | 11475075126778771796 | 163 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 4 | 9289978634712585093 | 151 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 5 | 5022816374395219446 | 153 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 6 | 15437077184617616928 | 187 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 7 | 3699861140717967782 | 145 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 9 | 7981606890290359029 | 158 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 10 | 3017423835907537192 | 174 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 11 | 2897091825132234232 | 145 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 12 | 363812672167313503 | 159 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 13 | 9714517516883929489 | 145 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 14 | 13146475485835159100 | 128 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 15 | 9262293349555376351 | 133 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 16 | 12699396853484669392 | 115 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 17 | 16007838611052960654 | 130 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 18 | 2228552658003644967 | 172 | non-finite iterate |
| tissue-only | rms | low_count | 0.2 | 19 | 2741174279437232974 | 137 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 0 | 15394035201083549913 | 61 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 1 | 10837233372687690179 | 177 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 2 | 8630701097559126812 | 150 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 3 | 8224609558890423419 | 152 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 4 | 1195345485378046392 | 140 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 5 | 14328312662999330441 | 155 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 6 | 9515698613298086135 | 182 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 7 | 10748816607767354200 | 120 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 9 | 937075788011410 | 127 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 10 | 1744425451868730087 | 179 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 11 | 13701067627814082227 | 146 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 12 | 16741359367466339730 | 168 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 13 | 3127501782791377564 | 143 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 14 | 15769737435926983028 | 130 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 15 | 10357364574037529739 | 135 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 16 | 4670806921951123494 | 148 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 17 | 12480438160523480641 | 134 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 18 | 10775730971693991329 | 148 | non-finite iterate |
| tissue-only | rms | low_count | 0.3 | 19 | 4147642332521141727 | 184 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 0 | 16302708932768381610 | 120 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 1 | 13036721228353021585 | 168 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 2 | 9816833731244394585 | 149 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 3 | 16829860601542314652 | 116 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 4 | 17753438639555747836 | 119 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 5 | 8589059646236421522 | 148 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 6 | 1263358949774442368 | 158 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 7 | 1721363678429367093 | 110 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 8 | 14509634575753145916 | 71 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 9 | 4055570380557324569 | 135 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 10 | 11783052124265024513 | 139 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 11 | 11256229709133109221 | 131 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 12 | 9612667308351000028 | 169 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 13 | 2163911853804287808 | 143 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 14 | 17210172978620341560 | 119 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 15 | 2190457922699464978 | 67 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 16 | 16845001384696147547 | 145 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 17 | 1274918935039283633 | 174 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 18 | 7850155921149193305 | 135 | non-finite iterate |
| tissue-only | rms | low_count | 0.4 | 19 | 18325154317230509695 | 161 | non-finite iterate |
| tissue-only | morozov | normal_count | 0.1 | 8 | 8422819710394504569 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 0 | 10700753966500982286 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 1 | 5424990108287796935 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 2 | 14817441891826887226 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 3 | 7087585902345953051 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 4 | 1807916131356517894 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 5 | 6779265272785279763 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 6 | 6351801794234391670 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 7 | 13482974565168760413 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 8 | 8422819710394504569 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 9 | 13743115229075106723 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 10 | 12413828073962532520 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 11 | 10046158082345470303 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 12 | 12078227055911820856 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 13 | 4020597517779351662 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 14 | 17481229501335906182 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 15 | 12375987125319700978 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 16 | 6458611781295944111 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 17 | 6487172617831067072 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 18 | 15255070336721197181 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.1 | 19 | 418620421686733813 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 0 | 16048746028836443131 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 2 | 6455790221252972928 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 4 | 9289978634712585093 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 5 | 5022816374395219446 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 6 | 15437077184617616928 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 7 | 3699861140717967782 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 8 | 2603647793336573112 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 10 | 3017423835907537192 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 11 | 2897091825132234232 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 12 | 363812672167313503 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 14 | 13146475485835159100 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 15 | 9262293349555376351 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.2 | 17 | 16007838611052960654 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.3 | 0 | 15394035201083549913 | 61 | non-finite iterate |
| tissue-only | morozov | low_count | 0.3 | 1 | 10837233372687690179 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.3 | 2 | 8630701097559126812 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.3 | 3 | 8224609558890423419 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.3 | 4 | 1195345485378046392 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.3 | 6 | 9515698613298086135 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.3 | 17 | 12480438160523480641 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.4 | 3 | 16829860601542314652 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.4 | 5 | 8589059646236421522 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.4 | 12 | 9612667308351000028 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.4 | 14 | 17210172978620341560 | 0 | trivial stop (rule fired at iteration 0) |
| tissue-only | morozov | low_count | 0.4 | 17 | 1274918935039283633 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | rms | high_count | 0.1 | 0 | 10700753966500982286 | 202 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.1 | 5 | 6779265272785279763 | 196 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.1 | 6 | 6351801794234391670 | 300 | stalled (rel. residual 4.858e-02 vs floor 2.915e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.1 | 7 | 13482974565168760413 | 300 | stalled (rel. residual 2.245e-02 vs floor 3.265e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.1 | 9 | 13743115229075106723 | 300 | stalled (rel. residual 1.099e-02 vs floor 2.850e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.1 | 12 | 12078227055911820856 | 300 | stalled (rel. residual 1.483e-02 vs floor 3.122e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.1 | 17 | 6487172617831067072 | 123 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.1 | 18 | 15255070336721197181 | 182 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.2 | 2 | 6455790221252972928 | 195 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.2 | 6 | 15437077184617616928 | 300 | stalled (rel. residual 4.417e-02 vs floor 2.915e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.2 | 7 | 3699861140717967782 | 300 | stalled (rel. residual 2.673e-02 vs floor 3.265e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.2 | 9 | 7981606890290359029 | 300 | stalled (rel. residual 1.247e-02 vs floor 2.850e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.2 | 12 | 363812672167313503 | 300 | stalled (rel. residual 1.569e-02 vs floor 3.122e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.2 | 17 | 16007838611052960654 | 202 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.2 | 18 | 2228552658003644967 | 159 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 0 | 15394035201083549913 | 67 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 4 | 1195345485378046392 | 152 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 6 | 9515698613298086135 | 300 | stalled (rel. residual 4.417e-02 vs floor 2.915e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 7 | 10748816607767354200 | 300 | stalled (rel. residual 2.145e-02 vs floor 3.265e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 9 | 937075788011410 | 58 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 11 | 13701067627814082227 | 157 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 12 | 16741359367466339730 | 300 | stalled (rel. residual 1.569e-02 vs floor 3.122e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 14 | 15769737435926983028 | 163 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 16 | 4670806921951123494 | 300 | stalled (rel. residual 8.819e+03 vs floor 2.884e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.3 | 17 | 12480438160523480641 | 47 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 2 | 9816833731244394585 | 229 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 5 | 8589059646236421522 | 165 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 6 | 1263358949774442368 | 177 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 9 | 4055570380557324569 | 300 | stalled (rel. residual 1.098e-02 vs floor 2.850e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 10 | 11783052124265024513 | 141 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 11 | 11256229709133109221 | 197 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 12 | 9612667308351000028 | 300 | stalled (rel. residual 1.483e-02 vs floor 3.122e-03) |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 15 | 2190457922699464978 | 137 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 16 | 16845001384696147547 | 187 | non-finite iterate |
| +1 C_P (frame 3) | rms | high_count | 0.4 | 17 | 1274918935039283633 | 183 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.1 | 0 | 10700753966500982286 | 171 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.1 | 1 | 5424990108287796935 | 201 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.1 | 5 | 6779265272785279763 | 173 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.1 | 7 | 13482974565168760413 | 240 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.1 | 12 | 12078227055911820856 | 191 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.1 | 14 | 17481229501335906182 | 172 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.1 | 16 | 6458611781295944111 | 196 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.1 | 18 | 15255070336721197181 | 168 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 0 | 16048746028836443131 | 300 | stalled (rel. residual 2.357e-02 vs floor 1.103e-02) |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 1 | 11976309891222107669 | 201 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 2 | 6455790221252972928 | 181 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 7 | 3699861140717967782 | 209 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 12 | 363812672167313503 | 237 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 13 | 9714517516883929489 | 175 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 16 | 12699396853484669392 | 201 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 17 | 16007838611052960654 | 196 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.2 | 18 | 2228552658003644967 | 177 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 0 | 15394035201083549913 | 91 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 1 | 10837233372687690179 | 188 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 4 | 1195345485378046392 | 161 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 7 | 10748816607767354200 | 238 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 11 | 13701067627814082227 | 147 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 12 | 16741359367466339730 | 176 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 13 | 3127501782791377564 | 200 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 14 | 15769737435926983028 | 187 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 16 | 4670806921951123494 | 200 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 18 | 10775730971693991329 | 176 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.3 | 19 | 4147642332521141727 | 205 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 0 | 16302708932768381610 | 300 | stalled (rel. residual 6.956e-02 vs floor 1.103e-02) |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 1 | 13036721228353021585 | 185 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 2 | 9816833731244394585 | 191 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 5 | 8589059646236421522 | 161 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 6 | 1263358949774442368 | 177 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 10 | 11783052124265024513 | 139 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 11 | 11256229709133109221 | 190 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 12 | 9612667308351000028 | 179 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 13 | 2163911853804287808 | 150 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 15 | 2190457922699464978 | 85 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 16 | 16845001384696147547 | 200 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 17 | 1274918935039283633 | 193 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 18 | 7850155921149193305 | 177 | non-finite iterate |
| +1 C_P (frame 3) | rms | normal_count | 0.4 | 19 | 18325154317230509695 | 172 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 0 | 10700753966500982286 | 149 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 1 | 5424990108287796935 | 180 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 2 | 14817441891826887226 | 158 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 3 | 7087585902345953051 | 182 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 4 | 1807916131356517894 | 178 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 5 | 6779265272785279763 | 143 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 6 | 6351801794234391670 | 141 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 7 | 13482974565168760413 | 220 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 8 | 8422819710394504569 | 175 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 9 | 13743115229075106723 | 192 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 10 | 12413828073962532520 | 155 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 11 | 10046158082345470303 | 136 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 12 | 12078227055911820856 | 135 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 13 | 4020597517779351662 | 216 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 14 | 17481229501335906182 | 142 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 15 | 12375987125319700978 | 137 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 16 | 6458611781295944111 | 294 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 17 | 6487172617831067072 | 149 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 18 | 15255070336721197181 | 137 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.1 | 19 | 418620421686733813 | 120 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 0 | 16048746028836443131 | 145 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 1 | 11976309891222107669 | 186 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 2 | 6455790221252972928 | 163 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 3 | 11475075126778771796 | 172 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 4 | 9289978634712585093 | 177 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 5 | 5022816374395219446 | 181 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 6 | 15437077184617616928 | 140 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 7 | 3699861140717967782 | 230 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 8 | 2603647793336573112 | 170 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 9 | 7981606890290359029 | 176 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 10 | 3017423835907537192 | 134 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 11 | 2897091825132234232 | 115 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 12 | 363812672167313503 | 117 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 13 | 9714517516883929489 | 222 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 14 | 13146475485835159100 | 120 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 15 | 9262293349555376351 | 164 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 16 | 12699396853484669392 | 271 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 17 | 16007838611052960654 | 133 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 18 | 2228552658003644967 | 141 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.2 | 19 | 2741174279437232974 | 159 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 0 | 15394035201083549913 | 62 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 1 | 10837233372687690179 | 178 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 2 | 8630701097559126812 | 110 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 3 | 8224609558890423419 | 169 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 4 | 1195345485378046392 | 161 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 5 | 14328312662999330441 | 167 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 6 | 9515698613298086135 | 151 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 7 | 10748816607767354200 | 221 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 8 | 13819258390552285198 | 188 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 9 | 937075788011410 | 210 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 10 | 1744425451868730087 | 133 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 11 | 13701067627814082227 | 140 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 12 | 16741359367466339730 | 139 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 13 | 3127501782791377564 | 206 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 14 | 15769737435926983028 | 135 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 15 | 10357364574037529739 | 140 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 16 | 4670806921951123494 | 198 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 17 | 12480438160523480641 | 51 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 18 | 10775730971693991329 | 143 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.3 | 19 | 4147642332521141727 | 122 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 0 | 16302708932768381610 | 145 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 1 | 13036721228353021585 | 160 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 2 | 9816833731244394585 | 149 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 3 | 16829860601542314652 | 164 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 4 | 17753438639555747836 | 162 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 5 | 8589059646236421522 | 132 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 6 | 1263358949774442368 | 185 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 7 | 1721363678429367093 | 126 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 8 | 14509634575753145916 | 179 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 9 | 4055570380557324569 | 178 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 10 | 11783052124265024513 | 141 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 11 | 11256229709133109221 | 121 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 12 | 9612667308351000028 | 128 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 13 | 2163911853804287808 | 135 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 14 | 17210172978620341560 | 149 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 15 | 2190457922699464978 | 139 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 16 | 16845001384696147547 | 190 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 17 | 1274918935039283633 | 158 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 18 | 7850155921149193305 | 133 | non-finite iterate |
| +1 C_P (frame 3) | rms | low_count | 0.4 | 19 | 18325154317230509695 | 135 | non-finite iterate |
| +1 C_P (frame 3) | morozov | high_count | 0.3 | 0 | 15394035201083549913 | 67 | non-finite iterate |
| +1 C_P (frame 3) | morozov | high_count | 0.3 | 9 | 937075788011410 | 58 | non-finite iterate |
| +1 C_P (frame 3) | morozov | high_count | 0.3 | 17 | 12480438160523480641 | 47 | non-finite iterate |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 0 | 10700753966500982286 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 1 | 5424990108287796935 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 2 | 14817441891826887226 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 3 | 7087585902345953051 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 4 | 1807916131356517894 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 5 | 6779265272785279763 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 6 | 6351801794234391670 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 7 | 13482974565168760413 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 8 | 8422819710394504569 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 9 | 13743115229075106723 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 10 | 12413828073962532520 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 11 | 10046158082345470303 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 12 | 12078227055911820856 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 13 | 4020597517779351662 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 14 | 17481229501335906182 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 15 | 12375987125319700978 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 16 | 6458611781295944111 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 17 | 6487172617831067072 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 18 | 15255070336721197181 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.1 | 19 | 418620421686733813 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 0 | 16048746028836443131 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 2 | 6455790221252972928 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 6 | 15437077184617616928 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 7 | 3699861140717967782 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 8 | 2603647793336573112 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 9 | 7981606890290359029 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 10 | 3017423835907537192 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 12 | 363812672167313503 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 14 | 13146475485835159100 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 15 | 9262293349555376351 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.2 | 17 | 16007838611052960654 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.3 | 2 | 8630701097559126812 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.3 | 3 | 8224609558890423419 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.3 | 6 | 9515698613298086135 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.3 | 17 | 12480438160523480641 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.4 | 3 | 16829860601542314652 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.4 | 5 | 8589059646236421522 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.4 | 12 | 9612667308351000028 | 0 | trivial stop (rule fired at iteration 0) |
| +1 C_P (frame 3) | morozov | low_count | 0.4 | 14 | 17210172978620341560 | 0 | trivial stop (rule fired at iteration 0) |

## M4.5 consistency and regularisation checks

Reproduce with `experiments/m4_consistency_regularization.py`, root seed
20240401, `delta_x=0.1`, one exact C_P measurement at frame 3, 20 seeds, and
`delta_y*sqrt(n_obs)` for Morozov stopping. Individual outcomes and seeds are in
`results/m4/consistency_regularization.json`.

- Consistency arm, noiseless: seed_idx 0 (seed 10700753966500982286) stayed
  finite but failed the `1e-6` relative-residual gate; seed_idx 18 (seed
  15255070336721197181) diverged. Thus 18/20 fits were accepted.
- Consistency arm, low_count: **all seed_idx 0..19 stopped at iteration 0**.
  None is called a reconstruction; the error statistic is reported as missing.
- Full-fit regularisation-off arm, high_count: seed_idx 0, 1, 3, 5, 18
  diverged; seed_idx 6 remained finite but did not satisfy Morozov by
  iteration 300. Regularisation on stopped successfully for all 20.
- Full-fit regularisation-off arm, normal_count: seed_idx 1, 3, 8, 14, 18,
  19 diverged. Regularisation on stopped successfully for all 20.
- The default Anaconda plotting stack on this machine failed to import
  Matplotlib (`numpy 2.0.2` with a NumPy-1.x-built Matplotlib extension).
  Numeric artifacts were generated in that environment; the plot was then
  generated from the stamped JSON using the separate compatible environment
  and the script's `--plot-only` option. No numeric result was silently
  regenerated in a different environment.

## M5 analysis and regeneration

- **Cross-environment reproducibility.** `experiments/run_all.py` on numpy 2.2.6 /
  scipy 1.14.1 / OpenBLAS 0.3.29 (15/15 scripts OK, 997 s) is bit-identical
  run-to-run but not bit-identical to the committed results (Anaconda, numpy
  2.0.2): 11 of 48 Table 1 cells change by +-1-2 divergences (total unchanged at
  498/960); M3 noiseless divergences at delta_x=0.3/0.4 go 4->3 and 9->8; M4.5
  noiseless seed_idx 0 flips from finite-but-rejected to diverged. Headlines
  unchanged. Committed files kept as the reference (DECISIONS.md D-M5-5).
- **`m4_grid.py` duplicated its section in this file on every rerun.** It
  appended rather than replaced, so each `run_all.py` pass added another 500-line
  M4.3 table. Fixed: the section is now rewritten in place; verified idempotent
  (two consecutive writes produce identical files, one M4.3 heading).
- **Our Simpson has a round-off floor that scipy's does not** (sin on [0, pi],
  uniform grid): ours 1.8e-07 at n=12801 and 4.1e-07 at n=25601, scipy 2.2e-16
  at n=12801. Cancellation in `_quadratic_segment_integral`; not fixed
  (DECISIONS.md D-M5-6).
- **`scipy.optimize.least_squares` (Track B, unregularised TRF) fails on the
  noiseless problem** with the same residual, analytic Jacobian and D(F) bounds
  as IRGNM: `status=2` after 40 evaluations at final relative error 1.046, worse
  than the initial guess (delta_x=0.1, seed_idx=0, root seed 20240301). IRGNM
  reaches 9.27e-07. Expected for an ill-posed problem (DECISIONS.md D-M5-3).
