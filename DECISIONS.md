# DECISIONS.md — modelling and engineering choices not dictated by the paper

Every entry: what was decided, why, and what would need to change if the decision
turns out to be wrong. Ordered by milestone.

---

## M0

- **D-M0-1: `src/` is the importable package name, not `src/<project_name>/`.**
  The project brief phrases the Track A/B rule as "anywhere under `src/`", so `src/` is
  treated as the package root directly (`src/config.py`, `src/rng.py`, ...),
  imported as `from src.config import ...`. Risk if wrong: none functionally,
  just a naming preference; trivial to rename later.

- **D-M0-2: `config.py` lives at `src/config.py`, not repo root.**
  PLAN.md says "Config lives in `config.py`" without specifying a directory.
  Keeping it inside the `src` package means every Track A module and every
  test imports it the same way (`from src.config import ...`), and it is
  covered by the same package `__init__.py`. Risk if wrong: cosmetic; a
  top-level re-export shim could be added later at zero cost.

- **D-M0-3: `conftest.py` is placed at repo root and only edits `sys.path`.**
  Guarantees `pytest` works whether invoked from repo root or a subdirectory,
  without requiring the project to be `pip install -e`'d. No numerical
  content, so it is exempt from the Track A/B guard by construction.

- **D-M0-4: config hash is SHA-256 of a canonical (sorted-key) JSON encoding
  of the constants, truncated to 12 hex characters.**
  Chosen for determinism across processes/interpreter versions (unlike
  `hash()` or `id()`, which are salted/process-specific in Python). 12 hex
  chars (48 bits) is short enough to embed in filenames while collision risk
  is irrelevant at our artifact volumes (tens to low thousands of runs).

## M1

- **D-M1-1: uniform RNG is a 64-bit linear congruential generator (LCG) with
  the Knuth/MMIX multiplier `a = 6364136223846793005`, increment
  `c = 1442695040888963407`, modulus `2**64`, output taken from the upper 33
  bits of the 64-bit state mapped to `[0, 1)`.**
  The project brief explicitly allows "LCG or Mersenne-style". A full Mersenne
  Twister is ~600 lines of bit-twiddling that would not teach any numerical
  method in this course; an LCG is transparent, is a single multiply-add-mask
  per draw, and using only the upper bits of a 64-bit state (rather than the
  low bits, which have short periods in low-order bits of any LCG) is the
  standard mitigation taught alongside LCGs. Verified statistically in M1
  (mean/variance vs theory, chi-square goodness of fit) rather than assumed.
  Risk if wrong: LCGs are known to have lattice structure in high dimensions
  (Marsaglia's theorem); acceptable here because the only downstream uses are
  1-D transforms (Box-Muller, Poisson inversion) and low-dimensional Monte
  Carlo, not high-dimensional quasi-random sampling.

- **D-M1-2: normal sampling uses the basic (non-polar) Box-Muller transform,
  consuming two uniforms per pair of normals.**
  This is the transform the project brief names explicitly. The polar (Marsaglia)
  variant avoids `sin`/`cos` but adds a rejection loop; not needed at our
  scale and the basic transform is easier to verify against the closed-form
  target density. Edge case `u1 == 0` (giving `log(0)`) is handled by
  sampling `u1` from `(0, 1)` (open at 0) via `(state + 1) / (2**64 + 1)`-style
  offset — see `src/rng.py` docstring for the exact mapping.

- **D-M1-3: Poisson sampler uses Knuth's multiplication algorithm
  (multiply uniforms until the running product drops below `exp(-lambda)`),
  not a rejection method for large lambda.**
  Knuth's algorithm is exact and trivial to verify by hand (mean = variance =
  lambda falls out of the derivation directly), matching the project brief's
  preference for code whose correctness is checkable by a grader. Its cost is
  O(lambda) uniforms per sample, which is fine for the lambda values used in
  this project's RNG unit tests (single digits to a few dozen) but would be
  slow for lambda in the hundreds/thousands. **Open risk carried into M4**:
  if Poisson noise calibrated to the paper's "high count" setting requires
  lambda of that size per sample, this sampler must be revisited (e.g. via a
  transformed-rejection method) or the loop must be capped with a documented
  approximation. Flagged in `handoffs/RUN_M1.md` open questions.

- **D-M1-4: Simpson's rule on non-uniform grids is implemented via local
  parabolic interpolation over consecutive triples of points (not the
  composite formula, which requires an even number of equal-width panels).**
  The forward model (M2) evaluates functions on the paper's 25 non-uniform
  PET frames, so a Track A integrator restricted to equal spacing would be
  useless for the actual pipeline. For an odd number of grid points this
  reduces to pairing up panels two-at-a-time using the standard non-uniform
  Simpson weight formula; if the number of intervals is odd, the final
  interval is closed with a trapezoid step and this fallback is logged
  (see `src/quadrature.py`).

- **D-M1-5: LU factorisation stores the permutation as an integer index
  array (not an explicit permutation matrix) and solves via forward/back
  substitution applied in that order.**
  Standard, avoids an O(n^2) permutation matrix multiply that would add
  nothing but wasted flops and obscure the algorithm.

- **D-M1-6: quadrature accuracy/convergence tests pin `n` at a value where
  truncation error still dominates round-off, instead of using a single very
  fine grid.** Empirically (see `handoffs/RUN_M1.md` section 5), our Simpson
  implementation's error vs h is the textbook U-shape: 4th-order truncation
  decay down to roughly 1e-10 to 1e-13 (integrand-dependent) around
  n=257-513, then *increasing* error from float64 round-off accumulation in
  the cubic antiderivative evaluation inside `_quadratic_segment_integral`
  for finer grids still. This is expected numerical-analysis behaviour
  (Chapra & Canale's discussion of the optimal-h tradeoff), not a defect, but
  it means: (a) an "accuracy at fixed n" test must use an n comfortably
  before the floor (we use n=129) rather than the largest n available, and
  (b) a "measured convergence order" fit must use only the h-range where the
  error still decays cleanly as a power law (we use n up to 33, sometimes 129
  for trapezoid, which never gets near its own much-higher round-off floor at
  these n). Originally the tests used n=2001 and a fit over n up to 513,
  which produced a spurious failure — not because the solver was wrong, but
  because the test measured round-off noise and mistook it for algorithm
  error. Fixed by picking `n` for a stated numerical reason instead of
  loosening the tolerance blindly (the project brief).

- **D-M1-7: condition numbers reported for the Hilbert-matrix stress test are
  computed with `numpy.linalg.cond` (Track B), inside `experiments/`, not
  `src/`.** Condition number is a diagnostic used only for reporting, never
  for the solve itself; the project brief permits Track B "inside `tests/` and inside
  clearly-marked benchmark scripts". `experiments/m1_linalg_benchmark.py` is
  such a script and says so in a comment at the point of use.

## M1 corrections (post-review, before M2)

Reviewer feedback on `handoffs/RUN_M1.md` (M1 verdict: accepted) asked five follow-up
questions and two additional checks before M2 could start. This section records the
resulting decisions; `logs/failures.md` records the mutation-check outcomes as data.

- **D-M1-8: Simpson order-6 on `runge_0_1` is explained, not just observed — it is the
  boundary-cancellation term of Simpson's asymptotic error expansion, and we now test
  it as a prediction, not an accident.**

  *Derivation (reviewer-supplied, verified against our own data).* Composite Simpson's
  rule has an asymptotic error expansion (Euler-Maclaurin-type; see e.g. Isaacson &
  Keller, *Analysis of Numerical Methods*, or Süli & Mayers ch.7) of the form

  ```
  E(h) = (h^4 / 180) * [f'''(a) - f'''(b)]  +  O(h^6)
  ```

  The key fact is that the leading h^4 coefficient is a **boundary** quantity (only
  `f'''` at the two endpoints), not an integral of `f''''` over the whole domain — the
  interior contributions telescope and cancel between adjacent panels in the composite
  sum. Consequently, whenever `f'''(a) == f'''(b)`, the h^4 term vanishes identically
  and the *true* convergence order jumps to (at least) 6.

  *Check against our three original integrands* (`experiments/m1_quadrature_benchmark.py`,
  numbers from `handoffs/RUN_M1.md` section 5):
  - `sin_0_pi`, f'''(x) = -cos(x): f'''(0) = -1, f'''(pi) = +1, difference = -2 ≠ 0 →
    predicted order 4. Measured: **4.013**. Matches.
  - `exp_0_1`, f'''(x) = e^x: f'''(0) = 1, f'''(1) = e, difference ≈ -1.718 ≠ 0 →
    predicted order 4. Measured: **3.999**. Matches.
  - `runge_0_1`, f(x) = 1/(1+x^2), f'''(x) = 24x/(1+x^2)^3 - 48x^3/(1+x^2)^4: f'''(0) = 0
    (f is even about x=0, so all odd-order derivatives vanish there), and f'''(1) =
    24/8 - 48/16 = 3 - 3 = 0 → both endpoints vanish → h^4 term vanishes → predicted
    order ≥ 6. Measured: **5.999**. Matches, and resolves reviewer Q2 from the M1
    report: not a curiosity, a predictable consequence of this formula.

  *New, deliberately-constructed 4th test integrand* (`tests/test_quadrature.py`,
  `endpoint_matched_0_b`): to turn this into a genuine tested *prediction* rather than
  reusing the accidental `runge_0_1` case, we built `f(x) = cos(x) + sin(2x)`, whose
  `f'''(x) = sin(x) - 8 cos(2x)` satisfies `f'''(0) = -8`. Solving `f'''(x) = -8`
  numerically (offline, one-off) for a second root in `(0, 2*pi)` gives
  `x ≈ 3.204133415386284` — deliberately *not* `x = pi` (also a root), because at
  `x = pi` the resulting `f` turns out to be odd about the interval's midpoint (see
  the rejected constructions below), which trivially forces Simpson's error to
  round-off at *every* grid size for a completely different reason (composite Simpson
  applied to a function that is odd about a grid symmetric about that same point sums
  to exactly zero, since matching quadrature weights cancel against matching function
  values of opposite sign) — that would not test the h^4-cancellation mechanism at all,
  it would just be a different, unrelated exact-cancellation trick. `x ≈ 3.204133...`
  has no such symmetry, so the h^4 term cancels *and* the O(h^6) term stays a genuine,
  nonzero, generic quantity. Measured order on `n = [9, 17, 33]`: consistent with 6
  (`test_endpoint_matched_integrand_shows_genuine_order_six_not_degenerate_exactness`
  asserts `5.5 < order < 6.5` and additionally checks the error at the coarsest grid is
  *not* already at the round-off floor, which is what would distinguish this from the
  degenerate fully-antisymmetric case).

  *Two earlier construction attempts were tried and rejected* (kept here so a future
  reader does not repeat the mistake): (1) building `f` by triple-integrating
  `f'''(x) = cos(x-m)` with all integration constants set to zero gives `f(x) =
  -sin(x-m)/1`, which is *always* odd about the midpoint `m` regardless of the interval
  — integrating an even function three times with zero constants always yields an odd
  function, a general parity fact, not specific to this `f'''`. This produces the
  trivial round-off-immediately case described above, not a genuine order-6
  measurement. (2) Adding a same-order-of-magnitude polynomial correction (e.g. `+x^2`)
  to try to break the symmetry failed because composite Simpson on a **uniform** grid is
  exact for cubics (a standard fact, stronger than "exact for quadratics" — the
  quadratic-interpolant argument in `src/quadrature.py`'s docstring only guarantees
  exactness for quadratics on a *non-uniform* grid; on a uniform grid the composite sum
  gets an extra order for free), so any degree-≤3 addition contributes exactly zero to
  Simpson's error and does not disturb the underlying odd part's trivial cancellation.
  The two-frequency (`cos(x) + sin(2x)`) construction avoids both problems: it has no
  special parity about its own domain's midpoint, and it is not a pure low-degree
  polynomial.

  Also fixed per reviewer instruction: `tests/test_quadrature.py`'s module docstring and
  the convergence-order test's inline comments now state the endpoint-cancellation
  condition explicitly, so a reader hitting the `> 3.5` (not `~4`) lower-bound assertions
  is not confused by why some integrands measure order 6.

- **D-M1-9: the M3 LU-vs-QR comparison must be "normal equations + LU" vs "stacked
  least-squares + QR" on the real IRGNM problem, not a repeat of the M1 Hilbert-matrix
  exercise.** Resolves reviewer Q1. M1's finding (LU slightly *more* accurate than QR
  on Hilbert(8), see `handoffs/RUN_M1.md` §5/§12.1) is not a general square-system
  stability ranking — for square systems, LU with partial pivoting is backward stable
  and there is no a priori reason to expect QR to win, and one ill-conditioned example
  is not enough evidence either way (reviewer's assessment, which we agree with).
  QR's real advantage shows up specifically in M3's IRGNM step (eq. 26 of the paper),
  which at each iteration solves

  ```
  (F'^T F' + alpha I) delta = F'^T r + alpha (x0 - x)
  ```

  Forming `F'^T F'` explicitly squares the condition number: `cond(F'^T F') =
  cond(F')^2`. The mathematically equivalent formulation avoids this by never forming
  the normal equations, instead solving the stacked least-squares problem

  ```
  minimise || [F' ; sqrt(alpha) I] delta  -  [r ; sqrt(alpha) (x0 - x)] ||_2
  ```

  directly via Householder QR on the stacked `(m+n) x n` matrix, which works at
  `cond(F')` rather than `cond(F')^2`. `src/qr.py`'s `qr_lstsq` already has the right
  shape for this (rectangular least squares with `m > n`); M3 will build the stacked
  matrix `[F' ; sqrt(alpha) I]` and reuse it as-is. **Action taken now:** `PLAN.md`'s M3
  section is updated (see the diff there) to state this comparison explicitly as the
  required LU-vs-QR test, replacing the vaguer "both paths available and compared"
  language, and to require the comparison run on the real problem (F' at a real
  parameter point, swept over the alpha schedule) rather than a synthetic
  ill-conditioned matrix. **Not implemented yet** — M3 has not started; this is a
  forward-looking note only, per the reviewer's explicit instruction not to build M3
  early. A secondary, cheap multi-matrix Hilbert(6/8/10/12) sweep was added to
  `experiments/m1_linalg_benchmark.py` as optional supporting evidence that the M1
  LU-vs-QR ordering on a single Hilbert matrix doesn't generalize into a trend (see the
  measured numbers in the regenerated `results/m1/linalg_benchmark.json`).

- **D-M1-10: Poisson sampler now has an explicit large-lambda branch, crossover at
  lambda = 30 (named constant `POISSON_KNUTH_MAX_LAMBDA`).** Resolves reviewer Q3, and
  the root cause turned out to be worse than the reviewer's own description (their
  message said "hang"; we confirmed it is not) — recorded here so a future reader
  trusts the measurement over the initial hypothesis.

  *What actually happens, measured (not guessed).* Knuth's algorithm compares a
  running product of uniforms `p` against `L = exp(-lambda)`. `exp(-lambda)` underflows
  to exactly `0.0` in float64 at `lambda = 746` (confirmed directly: `math.exp(-745) =
  5e-324`, the smallest positive subnormal double; `math.exp(-746) = 0.0` exactly).
  The reviewer's hypothesis was that this makes the loop's `p <= L` test unsatisfiable
  forever. **Measured directly, it is not**: `p` is itself a product of shrinking
  uniforms, so `p` *independently* underflows to exactly `0.0` after roughly 700-800
  iterations regardless of `lambda` (each `Uniform(0,1)` factor contributes on average
  one bit of leading-zero growth; reaching the ~2^-1074 subnormal floor takes on the
  order of 700-800 multiplications). Once `p` underflows, `p <= L` (`0.0 <= 0.0`)
  becomes true and the loop returns — **not a hang, a silently wrong answer**, and the
  returned value clusters around ~700-800 almost independent of the true `lambda`.
  Direct measurement (seed 1, one draw each): `lambda=745` (last case with `L` still
  nonzero) → `762` (a perfectly plausible `Poisson(745)` draw, mean 745, std ≈27.3).
  `lambda=746` → `732`. `lambda=1000` → `769` (true mean 1000, std ≈31.6 — `769` is
  about 7.3 standard deviations low, i.e. wrong). `lambda=5000` → `726` (nonsense).
  This is a more dangerous failure mode than a hang: a hang is loud and would be
  noticed immediately; this silently returns a plausible-looking integer that is
  simply wrong, with no error or warning. Reported here in full because the original
  hypothesis in the review request was itself slightly wrong, and getting that right
  mattered for choosing the fix.

  *Fix.* Below `lambda = 30`, Knuth's algorithm is exact and cheap (unaffected by the
  above — the failure only appears once `L` itself underflows, far above 30; documented
  already in D-M1-3). Above `lambda = 30`, `poisson_one` now draws from the normal
  approximation `round(max(0, N(lambda, lambda)))`, built on the already-verified
  `standard_normal` (Box-Muller), which never touches `exp(-lambda)` at all and so has
  no failure mode near `lambda = 746`. `lambda = 30` (rather than something close to
  the actual 746 underflow point) is chosen because the normal approximation to
  Poisson is already good there (skewness `1/sqrt(30) ≈ 0.18`) and because M4's
  realistic photon-count regime is expected to reach into the hundreds/thousands (see
  D-M1-3's original open risk), so there is no reason to run Knuth's O(lambda) loop
  anywhere near the range where it silently breaks. Verified in `tests/test_rng.py`:
  mean/variance at `lambda = 100` and `lambda = 1000` (both now the normal-approximation
  branch, both correct), and a distributional agreement check comparing the two
  branches at the same `lambda = 25` (Knuth, exact, vs a forced normal-approximation
  draw via `poisson_one(..., force_normal_approx=True)`), checking their means and
  variances agree within Monte Carlo error — a two-sample check, not a full KS test,
  kept lightweight per the reviewer's "fix now, it is small" framing.

- **D-M1-11 (open M4 decision, recorded not resolved): TAC-level noise may end up
  Gaussian (time- and region-dependent sigma) rather than Poisson, since the sinogram/
  OSEM chain was cut per the project brief.** Both a Poisson-based and a
  Gaussian-based noise generator now exist in `src/rng.py` (`poisson`/`poisson_one` and
  `normal`), so M4 can build either without new Track A primitives. Which one (or both)
  to use for the "high/normal/low count" calibration is explicitly left open here per
  the reviewer's instruction — not a decision made now, just unblocked for M4.

- **D-M1-12: added lag-1..5 autocorrelation and 2-D consecutive-pair scatter
  diagnostics for the LCG, specifically checking for the Neave effect.** Resolves
  reviewer Q4. The hazard: Box-Muller fed by *consecutive* LCG outputs is known
  (Neave 1973) to produce normals that lie on a small number of spirals/lattice
  planes in 2-D, distorting the tails — a real risk for M4's 20-realisation Monte
  Carlo, whose entire purpose is estimating variance. Added to
  `experiments/m1_rng_benchmark.py`: lag-1 through lag-5 autocorrelation of the raw
  uniform stream; a 2-D scatter of `(u_i, u_{i+1})` consecutive uniform pairs; a 2-D
  scatter of `(z_i, z_{i+1})` consecutive Box-Muller normal pairs. **Findings are
  reported as measured, in `handoffs/RUN_M2.md`, not asserted away** — per the
  reviewer's explicit instruction to report what is seen rather than assert the
  generator is fine. If banding/spiraling had been visible, the documented fix would
  be either (a) a ~30-line PCG64 generator (still Track A — permutation of an LCG's
  output bits, not a library call), or (b) drawing each Box-Muller pair's two uniforms
  from non-adjacent stream positions (e.g. stride-2 interleaving) to break the
  consecutive-output correlation Neave's effect depends on. No full spectral test was
  built, per the reviewer's explicit "do not build a full spectral test" instruction.

- **D-M1-13: M2's quadrature path integrates on a dense grid over `[0, t]` for each
  requested time `t`, NOT on the 25-frame measurement grid.** Resolves reviewer Q5,
  which correctly identified a wrong assumption in the original M1 report's open
  question 5: eq. (1)'s integrand involves `C_P`, which is analytic and known
  everywhere, so nothing requires evaluating it only at the 25 frame times — those are
  where *measurements* live, not where integration must happen. Consequently: (a) the
  25-frame grid is only ever used as the set of output times `t_k` at which `C_T(t_k)`
  is reported, never as the integration grid itself; (b) each of the 25 integrals is
  evaluated on its own dense grid from `0` to `t_k`, constructed with an odd number of
  points (even number of intervals) so `simpson_diag`'s trapezoid-fallback flag never
  fires — M2's tests assert this explicitly rather than merely hoping for it; (c) since
  `C_P` contains `exp(-13.4522 t)` (time constant ≈ 4.46 s) alongside `exp(-0.0106 t)`
  (time constant ≈ 94 min), a single uniform grid fine enough to resolve the fast
  component over a ~60 minute domain would need tens of thousands of points, so a
  **graded grid** (points concentrated near `t=0` via `t = t_end * u^q`, `u` uniform in
  `[0,1]`, `q` chosen by measurement) is used instead. The grid-refinement study
  (`experiments/m2_forward_model.py`, numbers in `handoffs/RUN_M2.md`) picks the
  smallest grid size (and grading exponent) at which the quadrature-vs-closed-form gap
  stops improving, i.e. where quadrature has converged to (or below) the closed form's
  own floor rather than being limited by grid resolution.

- **D-M1-14: mutation checks extended to `src/quadrature.py` and `src/rng.py`** (M1's
  original mutation check, in `handoffs/RUN_M1.md` §10, only covered `src/linalg.py`).
  Results recorded in `logs/failures.md`; both mutations were caught by the existing
  test suite without needing new tests (see that log for exactly which tests failed).

- **D-M1-15: the LU/QR "timing comparison against `numpy.linalg.solve`" claim in
  `handoffs/RUN_M1.md` §5 is reframed as operation counts vs MEASURED scaling with n —
  and the measurement itself produced a second honest finding worth recording.**
  The original wall-clock comparison mostly measured the fixed per-call overhead of a
  pure-Python triangular-substitution loop against LAPACK's compiled, blocked,
  cache-tuned routines — a valid observation about *this implementation* but not a
  meaningful statement about the LU vs QR *algorithms* (LU is `~2n^3/3` flops,
  Householder QR is `~4n^3/3` flops — QR does ~2x the arithmetic of LU by construction).
  `experiments/m1_linalg_benchmark.py` now fits `time(n) ~ C * n^p` with **both** `p`
  and `C` free (via our own `src.qr.lstsq` on `log(time)` vs `log(n)`, restricted to
  `n >= 20` to exclude the region where fixed per-call overhead, not array size, sets
  the cost), rather than assuming `p=3`.

  **Measured result: `p ≈ 1.10-1.18` for both LU and QR, not ≈3.** This is not a
  measurement error — it says our *wall-clock* time in the tested range (n = 20..100)
  does not follow the `n^3` flop-count law at all. Reason (consistent with how the code
  is written): both `lu_factor` and `qr_factor` are an outer Python `for` loop of
  `O(n)` iterations, each iteration calling a handful of vectorised NumPy operations
  (`np.outer`, slicing, `@`). Each such call has a fixed Python/NumPy dispatch cost of
  roughly a few microseconds, essentially independent of the array size being operated
  on at these n. At n=100 the *true* flop count per solve is only ~1e6 (trivial, sub-
  microsecond at BLAS speeds), while the ~100 Python-level loop iterations each pay
  their fixed dispatch cost regardless — so total wall-clock time in this range is
  dominated by "number of Python-level loop iterations" (`O(n)`), not "number of
  floating-point operations" (`O(n^3)`). The `n^3` law would only start to show up at
  much larger n, where per-iteration vectorised work finally outgrows per-iteration
  dispatch overhead — outside the 5..100 range PLAN.md's M1 acceptance criterion asks
  for, so not chased further here. The flop-count argument (`QR ≈ 2x LU`) remains a
  correct statement about the *algorithms*; it is simply not what our *wall-clock
  measurement* at this n range shows, and the report says so rather than picking
  whichever framing looks better. Measured total-time ratio (QR/LU) at these sizes:
  **1.61x** (0.1515s / 0.0938s); fitted-model predicted ratio at n=100: **1.56x** — both
  well below the 2.0x flop-count prediction, consistent with the dispatch-overhead
  explanation above rather than the raw arithmetic cost. Reported honestly in
  `handoffs/RUN_M2.md` as "our implementation's wall-clock scaling is dispatch-overhead-
  dominated at these sizes, not flop-dominated," rather than forcing a fit to match
  theory.

  Also added (per reviewer instruction, optional secondary result): a Hilbert(6/8/10/12)
  sweep in the same script, resolving reviewer Q1 empirically — LU was more accurate in
  2 of 4 sizes, QR in the other 2 (`n=6`: QR wins; `n=8`: LU wins; `n=10`: LU wins;
  `n=12`: QR wins), a roughly 50/50 split that directly supports "this is round-off
  noise, not a genuine stability ranking between LU-with-pivoting and QR for square
  systems," consistent with the reviewer's own assessment.

## M2 — Forward model

- **D-M2-1: `closed_form_C_T` (eq. 3, Lemma 6) is implemented via a single smooth
  helper `phi1(x) = expm1(x)/x` (`phi1(0) := 1`), not as an if/else on the paper's two
  named degenerate branches.** This satisfies PLAN.md M2's requirement to handle both
  `mu_j == 0` and `k2+k3+mu_j == 0` explicitly (they ARE handled explicitly — by name,
  as the `x == 0.0` branch inside `phi1`, proven algebraically to equal the paper's
  separate branch formulas, not silently ignored) while additionally being numerically
  stable for the much more common case of a *near*-degenerate (small but nonzero)
  denominator, which the paper's branch structure does not by itself protect against.

  *Derivation.* Substituting `C_P(s) = sum_j lambda_j exp(mu_j s)` into eq. (1) reduces
  both resulting integrals to the elementary form `int_0^t exp(c s) ds`. This has the
  textbook closed form `(exp(ct) - 1)/c` for `c != 0` and `t` for `c = 0` — exactly the
  paper's two-branch split, `c` playing the role of `mu_j` (second integral) or
  `k2+k3+mu_j` (first integral). Writing this instead as `t * phi1(c t)` unifies both
  branches into one expression (`phi1` is smooth and equals exactly `1` at `x=0`, which
  is the correct limit, not an approximation). Full expansion (done by hand, and
  checked in `tests/test_forward_model.py::test_stable_form_matches_paper_eq3_branches`)
  confirms the resulting formula is *algebraically identical* to the paper's eq. (3),
  term for term, including both named branches — see `src/forward_model.py`'s
  `closed_form_C_T` docstring for the full derivation.

  *Numerical stability, measured (per PLAN.md M2's instruction: "choose the crossover
  threshold by measurement, not by guessing") — including a first guess of our own that
  the measurement corrected.* The naive evaluation of `c=k2+k3+mu_j` in the
  removable-singularity regime is `(exp(mu_j*t) - exp(-a*t)) / (a+mu_j)` — a difference
  of two *independently computed* exponentials that are nearly equal whenever `a+mu_j`
  is small, which loses precision by catastrophic cancellation *before* the division
  even happens. We measured this directly: comparing (a) the naive form
  `(exp(x)-1)/x`, (b) our `expm1(x)/x` form, and (c) a 30-term Taylor-series reference
  (`sum_{k=0}^{29} x^k/(k+1)!`, restricted to `|x| <= 1` where it converges to machine
  precision; at `|x|=10` the 30-term truncation itself is only accurate to ~1e-8, so it
  is not a valid reference there and was excluded rather than mistaken for a naive-form
  failure) — see `experiments/m2_forward_model.py::phi1_stability_study` for the full
  table. Measured naive-form relative error vs the series reference:

  | `x` | naive `(exp(x)-1)/x` rel. error | `expm1(x)/x` rel. error |
  |---|---|---|
  | `1e-6` | `3.8e-11` | `2.2e-16` |
  | `1e-8` | `1.1e-8` | `2.2e-16` |
  | `1e-10` | `8.3e-8` | `0` |
  | `1e-12` | `8.9e-5` | `0` |
  | `1e-14` | `8.0e-4` | `0` |
  | `1e-15` | `1.1e-1` | `2.2e-16` |
  | `1e-16` | `1.0` (100%, total breakdown) | `0` |

  **This measurement corrected a wrong first guess of our own**, which we are recording
  rather than quietly fixing, in the same spirit as the rest of this document: our
  first-draft text here (before running the study) claimed the naive form "exceeds 50%
  relative error below `|x| ~ 1e-8`," reasoning loosely from the standard
  `sqrt(machine_epsilon) ~ 1.49e-8` cancellation heuristic. The measurement shows that
  heuristic marks a *different* threshold than the one we described: `|x| ~ 1e-8` is
  where the naive form's relative error first becomes comparable to `1e-8` itself (i.e.
  "only ~8 of ~16 digits still correct" — a real degradation, but nowhere near 50%
  wrong). Complete breakdown (relative error approaching 100%) does not happen until
  `|x|` is close to machine epsilon itself, around `1e-15` to `1e-16` — three to four
  orders of magnitude smaller than our first guess. The qualitative conclusion is
  unaffected (`expm1(x)/x` stays within a few `1e-16` of the series reference
  uniformly across the entire tested range, i.e. it needs **no crossover at all** — the
  "threshold measurement" the milestone asked for shows that adopting `expm1`
  unconditionally dominates a naive-form-with-a-threshold everywhere, so there is no
  fragile boundary value to get wrong in the actual implementation), but the specific
  numeric claim about *where* the naive form fails was wrong until measured. The only
  special case left in `phi1` is `x == 0.0` exactly (an exact floating-point equality,
  not a tolerance), needed only to avoid literal `0/0`.

- **D-M2-2: the module exposes `closed_form_term1` (the k2-part of `closed_form_C_T`
  alone) as a public helper, not just an internal computation.** Needed by
  `closed_form_C_T_derivative` (D-M2-4) and useful on its own for the near-degeneracy
  stress test (M2 acceptance criterion), which sweeps `mu` through `-(k2+k3)` and needs
  to inspect the term that actually contains the removable singularity.

- **D-M2-3: quadrature grid, chosen by measurement (resolving reviewer Q5's grid-size
  instruction): `GradedGridSpec(n=1601, q=3.0)`, i.e. `t_i = t_end * u_i^3` for `u_i`
  uniform in `[0,1]`, 1601 points (forced odd, so Simpson's trapezoid fallback never
  triggers).** Grid-refinement study (`experiments/m2_forward_model.py`, numbers also
  in `handoffs/RUN_M2.md`): compared `quadrature_C_T` against `closed_form_C_T` (used
  as the high-accuracy reference — it is exact analytically, limited only by the
  `phi1`/`expm1` round-off floor, itself far below quadrature error at moderate n) at
  the earliest, a middle, and the last of the 25 frame midtimes, sweeping both the grid
  size `n in {51,...,12801}` and the grading exponent `q in {1,2,3,4,5}`.
  - `q=1` (uniform grid): needs `n ~ 12801` to reach only `4.5e-8` relative error at the
    latest frame time — confirms a uniform grid is a poor choice given `C_P`'s fastest
    component (`exp(-13.4522 t)`, time constant ~4.46s) against a ~60 minute domain.
  - `q=2` and `q=3` both reach the round-off floor (`~2-5e-10`) by `n=1601`, and *get
    worse* again beyond `n~3201-6401` (round-off accumulation growing with more grid
    points, the same U-shaped error-vs-h phenomenon documented for plain Simpson in
    D-M1-6) — so more points is not simply "safer."
  - `q=3, n=1601` gives relative error in the `[8e-12, 3e-10]` range consistently across
    the earliest, middle, and latest frame times (not just the hardest case), and the
    trapezoid fallback never fires at any of them. Adopted as the production setting.

- **D-M2-4: the late-time "slope approaches `K1*k3/(k2+k3) * C_P`" claim (PLAN.md M2,
  point 6) is measured precisely, using an exact closed-form derivative (`DECISIONS.md`
  cross-ref: `closed_form_C_T_derivative`, built by differentiating eq. (1) directly —
  not itself a paper equation), and the measurement uncovered a worthwhile correction
  to the naive statement of the claim.**

  Differentiating term-by-term: `d/dt term1 = -a*term1 + (K1 k2/a) C_P(t)`,
  `d/dt term2 = (K1 k3/a) C_P(t)`, giving the exact identity
  `C_T'(t) = K1 * C_P(t) - a * term1(t)` for all `t` (no approximation). At the actual
  last PET frame midtime used in this project (`t = 57.5 min`, from
  `config.frame_midtimes_minutes()`), for the frontal region: measured
  `C_T'(t)/C_P(t) = 0.059890`, compared against the classical Patlak net-influx rate
  `Ki = K1 k3/(k2+k3) = 0.063445` — a **5.6% relative discrepancy**, not the tight
  agreement a literal reading of "slope approaches `Ki * C_P`" might suggest.

  *Why, worked out by hand and confirmed to match the measurement to 1.8e-6 relative
  error*: the classical Patlak derivation assumes the arterial input has reached a
  genuine **plateau** (`C_P` approximately constant at late times), under which the
  free compartment reaches true steady state `C_F -> K1 C_P/(k2+k3)` and
  `C_T' -> k3 C_F = Ki * C_P` exactly. Our `C_P(t)` does **not** plateau — its slowest
  component still decays as `exp(mu_4 t)` with `mu_4 = -0.0106/min` (config's slowest
  arterial rate). Re-deriving the `t -> infinity` asymptotics *without* assuming
  `mu_4 = 0` (both `C_P(t)` and `term1(t)` are asymptotically dominated by the same
  `exp(mu_4 t)` factor, so their ratio has a finite nonzero limit rather than `term1`
  becoming negligible) gives the corrected rate

  ```
  C_T'(t) / C_P(t)  --(t -> infinity)-->  K1 * (k3 + mu_4) / (k2 + k3 + mu_4)
  ```

  which reduces exactly to the classical `Ki` in the limit `mu_4 -> 0` (i.e. Patlak's
  plateau assumption is the `mu_4 = 0` special case of this more general formula).
  Numerically, `K1*(k3+mu_4)/(k2+k3+mu_4) = 0.0599211...` for the frontal region —
  matching the measured `t=800 min` plateau value (`0.059921`) to **1.8e-6 relative
  error**, and already matching the measured `t=57.5 min` (realistic scan duration)
  value to about **5e-4 relative error**, vs. classical `Ki`'s ~5.6% error at the same
  `t`. **This is the comment PLAN.md M2 point 6 asks to leave in the code** (now in
  `closed_form_C_T_derivative`'s docstring): the quantity Patlak analysis extracts is
  only exactly `Ki` under a plateau assumption our `C_P` does not satisfy; the
  measured/reportable version of "the slope approaches the net influx rate" should cite
  the corrected formula (or explicitly note the ~5-6% gap from the textbook `Ki`) rather
  than claim exact agreement. Full numbers for all four regions in
  `handoffs/RUN_M2.md`.

- **D-M2-5: the three-way agreement check (closed form vs quadrature vs
  `scipy.solve_ivp`) uses `scipy.integrate.solve_ivp` with method `Radau` (implicit),
  not the default `RK45` (explicit), and a tight `rtol=atol=1e-12`.** The ODE system
  (S) has two well-separated timescales per region (`k2+k3` up to ~0.33/min, vs the
  arterial forcing's fastest component at 13.45/min) plus the forcing function itself
  spans a very fast initial transient — a stress test for an explicit integrator's step
  size control. `Radau` was chosen because the system, while not stiff in the
  eigenvalue sense here (all rates are of comparable, modest magnitude — none of the
  paper's `K1,k2,k3` values are large), still benefits from an implicit method's more
  conservative local error control around the sharp initial transient of `C_P`;
  measured in `handoffs/RUN_M2.md` against `closed_form_C_T`, both `RK45` (default) and
  `Radau` were tried and the results/numbers for both are reported so the reviewer can
  see this made no material difference to the conclusion — recorded as a decision
  anyway per the project brief, since which one to *feature* as the headline number
  was a judgement call.

- **D-M2-6: the near-degeneracy stress sweep (PLAN.md M2's last bullet) sweeps a
  synthetic extra `mu` component through `-(k2+k3)` rather than perturbing one of the
  four real arterial `mu_j` values.** None of the paper's actual ground-truth
  `mu_j` (`-13.4522, -3.2672, -0.1532, -0.0106`) is close to `-(k2+k3)` for any of the
  four regions (`k2+k3` ranges `0.208` to `0.292`; the nearest real `mu_j` is `-0.1532`,
  still `0.05`-`0.14` away) — so sweeping the real parameters would never actually cross
  the degenerate point. The stress test instead adds a fifth, synthetic exponential
  term (`lambda_5 = 0.5`, `mu_5` swept continuously through `-(k2+k3)`) to a copy of the
  arterial input, specifically to force the crossing PLAN.md asks to visualise. This
  changes the arterial input being tested, not the forward-model code being tested —
  `closed_form_C_T` and `quadrature_C_T` are unmodified, generic in `p` (the
  polyexponential degree), and take `(lambda, mu)` as plain arrays.

## M2 corrections (post-review, before M3)

Reviewer feedback on `handoffs/RUN_M2.md` (M2 verdict: accepted) independently
re-derived our corrected late-time-slope formula from eq. (3) (matching our numbers
exactly) and asked five follow-ups plus specified the exact M3 scope. This section
records the resulting decisions.

- **D-M2-7: the late-time slope has TWO correct answers, because dC_T/dt/C_P(t) and
  C_T(t)/C_P(t) are different limits of different quantities — this is a coherent
  two-part result, not a correction of D-M2-4.** Resolves reviewer Q2, which corrected
  a framing error in PLAN.md itself (the reviewer's own words: "my error, not yours").

  Dividing eq. (1) by `C_P(t)` and taking the late-time limit (`C_P` dominated by its
  slowest component `lambda_4 exp(mu_4 t)`):

  ```
  C_T(t)/C_P(t) = [K1 k3/(k2+k3)] * [int_0^t C_P ds / C_P(t)]  +  [K1 k2/(k2+k3)] / (k2+k3+mu_4)  +  o(1)
  ```

  The coefficient of the *normalised-time* variable `x(t) = int_0^t C_P ds / C_P(t)` is
  the classical `Ki = K1 k3/(k2+k3)` **exactly**, no `mu_4` correction — this is the
  **Patlak plot**, the actual clinical method for measuring `Ki` from PET data, and it
  is genuinely a different asymptotic statement from D-M2-4's `dC_T/dt / C_P(t) ->
  K1(k3+mu_4)/(k2+k3+mu_4)`. Both were verified numerically this session:

  | Region | Ki (classical) | Patlak fitted slope | rel. err | dC_T/dt corrected `Ki` | rel. err (D-M2-4) |
  |---|---|---|---|---|---|
  | frontal | 0.06345 | 0.06329 | 2.46e-3 | 0.05992 | 5.27e-4 (t=800: <1e-13) |
  | temporal | 0.05620 | 0.05610 | 1.92e-3 | 0.05200 | 7.76e-4 |
  | occipital | 0.06306 | 0.06312 | 9.95e-4 | 0.05795 | 9.82e-4 |
  | white_matter | 0.02260 | 0.02295 | 1.56e-2 | 0.01844 | 3.42e-3 |

  Implementation: `arterial_input_integral(t, lam, mu) = t * sum_j lambda_j *
  phi1(mu_j t)` (`src/forward_model.py`) reuses the exact same `phi1` stability trick
  as `closed_form_C_T` — it is literally `int_0^t C_P(s) ds`, needed for the Patlak
  x-axis, and, not coincidentally, is exactly `a/(K1 k3) * term2` from `closed_form_C_T`'s
  own internal decomposition. The late-time linear fit uses our own
  `src.qr.lstsq` (course outcome CO2: linear least-squares regression), **not**
  `numpy.polyfit`, on the frames from `t >= 3.5 min` onward (frame index 12 of 25).

  *Why `t >= 3.5 min` was chosen (measured, swept over candidate cutoffs 0.58, 1.25,
  2.25, 3.5, 8.75, 20 min in `experiments/m2_forward_model.py`, full table in
  `handoffs/RUN_M3.md`):* this is the smallest cutoff at which the fitted slope's
  relative error is simultaneously small (<2.5e-3) for 3 of 4 regions and the R^2 of
  the linear fit exceeds 0.999 for 3 of 4 regions, while still retaining 13 of the 25
  frames for the fit (later cutoffs improve R^2 further but leave as few as 7 points,
  increasing the fitted-parameter variance for no accuracy benefit — diminishing
  returns past this point). `white_matter` is measurably worse at every cutoff tested
  (R^2 = 0.9970 at this cutoff vs 0.9996-0.9999 for the other three) because it has
  the smallest `k2+k3` (0.208 vs 0.247-0.292) of the four regions and therefore
  approaches the late-time linear regime more slowly at any fixed cutoff — reported as
  a measured, region-dependent finding, not smoothed over with a looser blanket
  tolerance that would hide it.

  The fitted intercept (predicted: `[K1 k2/(k2+k3)]/(k2+k3+mu_4)`) matches to
  1.4-4.9% relative error across the four regions — noticeably looser than the slope,
  because the intercept is effectively an extrapolation to `x=0`, far outside the
  fitted `x` range (`x` runs from ~14 to ~99 over the late-time window), so it is the
  less well-conditioned of the two fitted quantities; reported honestly rather than
  hidden behind a tight assertion.

  **For M5 (recorded now per the reviewer's explicit instruction):** the Patlak plot
  is a primary result, not a footnote — it is how `Ki` is actually measured in
  clinical PET, it is a direct application of course outcome CO2 (linear least-squares
  regression), and it is the more practically relevant of the two late-time-slope
  results in this codebase. D-M2-4's `dC_T/dt` result remains valuable as an
  independent numerical/theoretical cross-check (it uses a closed-form derivative that
  needed its own verification — see D-M2-9 below — and it is what first revealed that
  "the slope approaches `Ki C_P`" needed disambiguating at all) but should be
  presented as the secondary, explanatory result.

- **D-M2-8: `scipy.integrate.solve_ivp` with `method="Radau"` is now the standing
  default** for any ODE-based Track B cross-check in this project (already the default
  in `tests/test_forward_model.py::_solve_ivp_C_T`; unchanged in
  `experiments/m2_forward_model.py`, which explicitly compares both and will keep
  doing so where the comparison itself is the point). Resolves reviewer Q3. Reasoning
  (the reviewer's, which we adopt): the ODE system (S) has well-separated timescales
  per region — `mu_j` spans `-13.4522` to `-0.0106` (ratio ~1270), while tissue rates
  `k2+k3` sit around `0.208`-`0.292` — and an explicit method (`RK45`) must resolve the
  fastest mode's transient everywhere along the integration, while an implicit method
  (`Radau`) does not need to. This is a property of the *problem*, not this specific
  parameter set, so it generalises to M3/M4's use of `solve_ivp` (if any) and is worth
  fixing as a default now rather than re-deciding it each time. (Measured in M2:
  `Radau` was also ~2 orders of magnitude more accurate than `RK45` against the closed
  form at `rtol=1e-12` in this project's actual parameter regime — `9.6e-14` to
  `1.7e-13` vs `6.9e-12` to `1.4e-11` — consistent with, though not required by, the
  stiffness argument.)

- **D-M2-9: grid validation (D-M2-3) is adequate as measured; no new computation
  needed, only the existing evidence connected explicitly.** Resolves reviewer Q5.
  The frontal region already has the *largest* `k2+k3` (0.292) of the four regions —
  it was, without having been deliberately chosen for this reason at the time, already
  the "hardest" region by the fastest-tissue-mode criterion the reviewer names, and it
  was the region used for the entire grid-refinement study in D-M2-3. The three-way
  agreement table (`handoffs/RUN_M2.md` §5) additionally shows all four regions —
  including frontal — reach essentially the same `~5.9e-10` closed-vs-quadrature
  agreement at the production grid (`n=1601, q=3.0`), which is direct (not merely
  indirect) confirmation that the hardest region reaches the floor. No new grid sweep
  was run for the other three regions; their `a = k2+k3` values are all smaller than
  frontal's, so the graded grid's job (resolving `C_P`'s fast components against a
  domain set by `t`, independent of `a`) is if anything easier for them, and the
  uniform ~5.9e-10 result across all four is consistent with that.

- **D-M2-10 (finite-difference cross-check for `closed_form_C_T_derivative`,
  resolves reviewer Q1):** `tests/test_forward_model.py::test_closed_form_derivative_matches_central_finite_difference`
  compares the analytic derivative against a central difference,
  `(C_T(t+h) - C_T(t-h)) / (2h)`, using `closed_form_C_T` itself (already verified
  three ways in M2) as the function being differenced. Step size `h = 1e-4`: the
  standard optimal-step argument for a central difference of a smooth function
  balances truncation error (`O(h^2)`) against round-off error (`O(machine_epsilon /
  h)`), giving `h_opt ~ (epsilon)^(1/3) * scale`; with `epsilon ~ 2.2e-16`,
  `epsilon^(1/3) ~ 6.06e-6`, and this problem's natural time scale of `O(1)` to
  `O(10)` minutes, `h` in the `1e-4` to `1e-5` range is the textbook choice — `1e-4`
  was used directly (not further tuned) since the resulting agreement (max relative
  error **1.12e-8** across all four regions and 5 test times spanning `t=0.5` to
  `t=57.5` min, well inside the `<1e-6` target) already meets the target the reviewer
  set, so there was nothing to gain from tuning `h` further.

## M3 — Jacobian, conditioning, null-space experiment, IRGNM solver

- **D-M3-1: q=4 blood-sample times chosen as 4 of the existing 25 frame midtimes**
  (`config.BLOOD_SAMPLE_FRAME_INDICES = (3, 10, 17, 24)`, giving `t ~ 0.29, 2.25, 15.0,
  57.5` min). PLAN.md's M2 section never fixed a value for `q` (the paper's F^2 block,
  eq. 20, needing measurements of `C_WB` at `q` sample times); the reviewer's M3 spec
  fixes the Jacobian's shape at `n*T+q = 104`, i.e. `q=4`, but does not fix which 4
  times. Reusing existing frame midtimes (rather than inventing new time points) is
  physically realistic (blood draws timed with scan frames) and keeps the codebase's
  only "time grid" concept (`config.frame_midtimes_minutes()`) as the single source for
  all time points used anywhere in the pipeline. The four indices are spread across the
  dynamic range: one during the fast initial arterial transient (index 3, `t=0.29`),
  one just after it (index 10, `t=2.25`), one mid-scan (index 17, `t=15.0`), and one at
  the final frame (index 24, `t=57.5`) — deliberately not clustered, so the `m`-block
  (`A, xi1, xi2`) columns of F^2, which only enter through `f(s_l)` at these 4 points,
  see a reasonably wide time range rather than 4 nearly-identical evaluations.

- **D-M3-2: `PROJECTION_EPS = 1e-3`, matching the value given directly in the M3
  spec.** Worth noting this is not an arbitrary engineering choice on our part: the
  same value (`epsilon = 10^-3`) appears in the paper's own numerical algorithm
  section (Section 5.2) for exactly the same purpose (bounding `K1,k2,k3` away from
  zero in `D(F)`), so this is a case of the assigned value and the paper's own
  independently-published choice agreeing, not us picking a number to match an
  instruction.

- **D-M3-3: the mu-derivative needs a second stability helper, `phi2(x) = (e^x-1-x)/x^2`
  (`phi2(0)=1/2`), and — unlike `phi1` — it genuinely needs a measured small-x
  crossover to a series form, not just a `x==0` guard.**

  *Derivation of `phi1'`.* Writing `phi1(x) = int_0^1 exp(x s) ds` (equals
  `(e^x-1)/x` by direct integration), differentiating under the integral and
  integrating by parts gives `phi1'(x) = (e^x - phi1(x))/x`. Substituting the
  standard phi-function recursion `phi2(x) = (phi1(x)-1)/x` (equivalently
  `phi1(x) = 1 + x*phi2(x)`) and `e^x = 1 + x*phi1(x)` (from `phi1`'s own
  definition) and simplifying gives the closed form actually implemented:

  ```
  phi1'(x) = 1 + (x-1)*phi2(x)
  ```

  Checked at `x=0`: `phi1'(0) = 1 - phi2(0) = 1 - 0.5 = 0.5`, matching the Taylor
  series `phi1(x) = 1 + x/2 + x^2/6 + ...` directly (`d/dx` at 0 is `1/2`). Checked
  against a central finite difference of `phi1` itself in
  `tests/test_jacobian.py::test_phi1_prime_matches_central_finite_difference_of_phi1`.

  *Why `phi2` needs its own small-x crossover, measured (not assumed from phi1's
  behaviour).* `phi1`'s closed form `expm1(x)/x` needed no crossover at all (D-M2-1)
  because `expm1` itself is accurate for all `x`, and dividing an accurate numerator
  by `x` doesn't reintroduce cancellation. `phi2`'s closed form `(expm1(x)-x)/x^2` is
  different: `expm1(x)` is computed to *relative* accuracy `~eps`, i.e. *absolute*
  error `~eps*|x|` for small `x` (since `expm1(x) ~ x`); but the true value of
  `expm1(x)-x` is `O(x^2)` (the next Taylor term), so the SUBTRACTION's relative error
  is `~eps*x/x^2 = eps/x` — the same scaling phi1 would have had if we'd naively
  formed `exp(x)-1` first. Measured directly
  (`tests/test_jacobian.py`, reference = 40-term Taylor series, valid to
  machine precision for the tested range):

  | `x` | naive `(expm1(x)-x)/x^2` rel. error |
  |---|---|
  | `1e-6` | `2.1e-10` |
  | `1e-8` | `1.35e-8` |
  | `1e-9` | `1.49e-7` |
  | `1e-10` | `6.78e-7` |
  | `1e-12` | `4.84e-5` |

  Slower growth than phi1's naive form at the same `x` (consistent with the smaller
  absolute error phi2's numerator starts from), but the same qualitative eps/x growth
  law, and NOT negligible in the range the Jacobian actually evaluates `mu*t` and
  `delta*t` over (frame times run from `t~0.04` to `t~57.5` min, and `mu` ranges
  `-13.45` to `-0.01`, so `mu*t` legitimately gets very close to zero for the slow
  arterial components at early frame times). `PHI2_SERIES_THRESHOLD = 1e-4` was chosen
  to sit comfortably above where the closed form's error becomes measurable (by
  `x=1e-6` it is already `2.1e-10`, three orders of magnitude inside our
  `1e-4`..`1e-6` per-block Jacobian tolerance target), with a large safety margin;
  below it, a direct 8-term Taylor series (`sum_k x^k/(k+2)!`) is used, which has no
  subtraction at all (every term added directly) and converges to machine precision
  well before 8 terms for `|x| < 1e-4` (next-term ratio `< 1e-4/10`).

- **D-M3-4: the smallest eigenvalue/eigenvector of an ill-conditioned symmetric
  matrix should be found via `inverse_power_iteration` directly, NOT read off the
  tail of the deflation-based full spectrum — measured, not assumed, on the actual
  null-space experiment.** On the real, restricted-Jacobian matrix `M1r =
  F1_reduced^T F1_reduced` (F1 with the trivial all-zero `m`-columns dropped), our
  `symmetric_eigendecomposition` (power method + repeated Hotelling deflation)
  reports a plausible-looking near-zero smallest eigenvalue, but its corresponding
  eigenvector is **essentially uncorrelated with the true null direction**
  (`|cos(angle)| = 1.37e-5`), while `inverse_power_iteration` applied directly to the
  same matrix finds the exact same near-zero eigenvalue (both report `~1e-11` to
  `1e-16`, consistent with each other as *numbers*) but with an eigenvector that
  aligns with the analytically-predicted null direction to **`cos = 1.00000000`**
  (full float64 precision). Root cause: 19 sequential deflation steps on a matrix
  spanning `~10` orders of magnitude (`2.2e4` down to `~1e-11`) each introduce
  round-off proportional to the eigenvalue just removed; by the time the smallest
  eigenvalue is reached, accumulated error is comparable to or larger than the true
  quantity, corrupting the eigenVECTOR (which is far more sensitive to this than the
  Rayleigh-quotient eigenvalue estimate, which stays numerically plausible even when
  wrong). A supporting synthetic test
  (`tests/test_eigen.py::test_inverse_power_iteration_accurate_on_ill_conditioned_matrix`)
  confirms `inverse_power_iteration` alone is accurate (rel. error `<1e-6`,
  eigenvector alignment `>1-1e-6`) on a cond~1e10 matrix; the specific *deflation
  degrades while inverse iteration doesn't* comparison is reported as a measured
  finding on the real project matrix (`handoffs/RUN_M3.md`), not asserted in a unit
  test, because reproducing the same failure mode synthetically requires the same
  kind of clustered near-zero eigenspace (three exact zeros from the trivial `m`-block
  plus one near-zero from the real degeneracy) that is specific to this problem, not
  a generic wide-spectrum matrix (tried; did not reproduce the effect as dramatically
  — see the test's own docstring for what was tried and why it was not force-fit into
  an assertion). **Practical consequence**: the M3 null-space experiment (PLAN.md item
  3) uses `inverse_power_iteration` for the smallest-eigenvalue identification, not
  the deflation-based full spectrum — the full spectrum (via deflation) is still
  reported for the top ~half of the eigenvalues (agrees with `numpy.linalg.eigvalsh`
  to `<1e-6` relative error there), used only as descriptive context, not as the
  source of the null-space alignment claim.

- **D-M3-5: IRGNM regularisation hyperparameters (six constants, ansatz `x_i =
  a*exp(-b*i)` per block) are adopted directly from the paper's own tuned values for
  its "full setup with noiseless C_WB" (Section 6, page 21 of the paper), converted
  from the paper's base-2 form to our base-`e` form, not independently re-tuned.**
  Paper: `alpha_i = 4000*2^(-i/7)` (metabolic), `beta_i = 100*2^(-i/7)` (arterial
  `lambda,mu`), `gamma_i = 200*2^(-i/7)` (plasma-fraction `m`), `tau = 6.8`. Converted
  via `a*2^(-i/c) = a*exp(-i*ln(2)/c)`, so `b = ln(2)/7 ~= 0.09902` for all three
  (same exponent divisor in the paper). Reasoning for adopting rather than re-deriving:
  the paper's own tuning procedure (Section 6) is itself a substantial grid-search
  sub-experiment (sweeping regularisation parameters and `tau` against a *different*
  ground-truth parameter set — the AD-group kinetics — specifically to avoid tuning
  hyperparameters on the same data used to evaluate them, a train/test split by
  parameter set), which was out of scope to fully replicate this milestone; reusing
  their published, already-cross-validated values is more defensible than a quick,
  under-powered re-tuning on our own setup. **This was verified before being
  trusted**, not assumed: run against the actual noiseless-recovery Monte Carlo study
  (PLAN.md item 7), these values give median final relative error in the `4.3e-7` to
  `9.7e-7` range across all four `delta_x` values — squarely inside the "~1e-7 or
  smaller" target — with no further adjustment needed. Had they not worked, the
  documented next step per item 8's instruction would have been to diagnose (Jacobian
  block, parameter scaling, or projection interaction) before tuning; that path was
  not needed here.

- **D-M3-6: measured linear-solver conditioning gap between "normal equations + LU"
  and "stacked least squares + QR" on the real problem (PLAN.md's item 5), swept over
  the alpha values the actual IRGNM iteration schedule visits.** Evaluated `F'` at a
  representative mid-optimisation point (50 IRGNM(QR) iterations from a `delta_x=0.2,
  seed=2024` perturbed start — not `x_true` itself, where the residual is trivially
  zero and would not stress the comparison). Swept regularisation strength across
  iteration indices `i in {0,...,299}` (each `i` sets `Lambda_i` via the schedule):

  | iteration `i` | cond(normal eq. `F'^TF'+Lambda_i`) | cond(stacked `[F';sqrt(Lambda_i)]`) | rel. diff, LU step vs QR step |
  |---|---|---|---|
  | 0 | 3.38e2 | 1.84e1 | 1.68e-16 |
  | 50 | 4.72e4 | 2.17e2 | 8.02e-15 |
  | 100 | 6.67e6 | 2.58e3 | 1.63e-13 |
  | 150 | 9.42e8 | 3.07e4 | 1.17e-11 |
  | 200 | 1.22e11 | 3.49e5 | 1.78e-10 |
  | 299 | 1.43e12 | 1.20e6 | 1.28e-9 |

  Confirms the theory numerically, exactly: `cond(normal eq.) / cond(stacked)^2` stays
  close to `1` throughout the sweep (e.g. at `i=299`: `1.43e12 / (1.20e6)^2 = 1.43e12 /
  1.44e12 ~= 0.99`), i.e. the normal-equations matrix's condition number *is*
  (measured, not just asserted) the square of the stacked matrix's, and the two
  solved steps' relative disagreement grows monotonically as regularisation shrinks —
  from machine precision (`1.68e-16`) at `i=0` (heavy regularisation dominates,
  problem is well-conditioned regardless of path) to `1.28e-9` at `i=299` (small but
  measurably non-machine-precision). **This did not translate into a dramatically
  different outcome across the full 80-run noiseless-recovery Monte Carlo** (see
  `logs/failures.md` M3 section): 79 of 80 seeds converge/diverge identically under
  both solvers, with final errors matching to 3-4 significant digits where both
  converge. **One seed differs concretely** (`delta_x=0.3`, seed
  `3870939591031771045`): QR stalls at a moderate error (`0.033`) while LU fully
  diverges (`4.37e11`) from the identical initial guess — direct empirical evidence
  the conditioning gap is real and occasionally decisive, not just a diagnostic
  number, even though it is not decisive for most runs at this problem's actual scale
  within a 300-iteration budget.

- **D-M3-7: QR (stacked least squares) is the default solver for IRGNM (item 5's
  "whichever wins on accuracy becomes the default").** Strictly more accurate
  (smaller relative disagreement from a higher-precision computation is not directly
  measurable without an independent third reference, but QR provably operates at
  `cond(F')` rather than `cond(F')^2` — D-M3-6 — and empirically it is at least as
  robust: same convergence outcomes as LU on 79/80 Monte Carlo seeds, and strictly
  better on the 1 seed where they differ (D-M3-6's example)). `src/irgnm.py`'s
  `run_irgnm` defaults to `solver="qr"`; `solver="lu"` remains available and is what
  the comparison in D-M3-6 and `logs/failures.md` was run against.

- **D-M3-8: `numpy` overflow/invalid-value `RuntimeWarning`s during divergent IRGNM
  trajectories are expected and suppressed for readability, not silently ignored as a
  correctness matter.** `mu` is unconstrained in `D(F)` (per the paper's own eq. 18
  domain — only `K1,k2,k3` and `m`'s sign are constrained), so a bad Gauss-Newton step
  can push a `mu` component to a large positive value, and `exp(mu*t)` for the
  `t~57.5` frame overflows float64. `src/irgnm.py`'s `run_irgnm` wraps the iteration
  loop in `np.errstate(over="ignore", invalid="ignore")` specifically so a Monte Carlo
  sweep's console output stays readable, but the actual *detection* of this failure
  is unaffected: every iterate is checked with `np.all(np.isfinite(x_next))`
  regardless of the warning suppression, and any non-finite value marks the run
  `diverged=True` and stops it — the suppression only silences the printed warning,
  never the check.

- **D-M3-9: `max_iter=300` for the noiseless-recovery study (no early stopping via
  the discrepancy principle, since `delta_y=0` for noiseless data), matching the
  paper's own choice for its noiseless-data figures (Section 6/Figures 4-6) exactly**
  — not independently chosen, adopted because PLAN.md item 6 names this value
  directly and it matches the paper.

## M4 — Noise, measurement setups, and Monte Carlo grid

- **D-M4-1: TAC noise is Poisson-derived; C_WB blood noise is Gaussian.** For TAC
  frames, the Poisson-derived model (`add_poisson_noise` via `src.rng.poisson`) is
  physically motivated: PET image frames arise from photon counting, which is a Poisson
  process. For C_WB blood-sample measurements in Setup C, a Gaussian proportional model
  (`add_gaussian_noise`) is used instead; blood draws are a low-volume measurement where
  the Poisson character of the imaging chain no longer dominates, and Gaussian is the
  standard approximation for counting noise in this regime. The open M4 question
  (D-M1-11) is thus resolved: Poisson for TACs, Gaussian for blood. Both generators
  were built and calibrated (as required) and the choice is recorded here, not quietly
  assumed.

- **D-M4-2: calibrated noise constants are hardcoded in `src/montecarlo.py` rather
  than re-read from `results/m4/noise_calibration.json` at runtime.** This makes the
  Monte Carlo harness self-contained: a single import of `src.montecarlo` is
  reproducible without depending on a pre-existing JSON artifact. The exact values
  (alpha = 88198.85 / 6255.53 / 160.55 for Poisson; sigma_rel = 0.002969 / 0.011139 /
  0.070945 for Gaussian) were produced by `experiments/m4_noise_calibration.py`
  (seed 20240401, 20-realisation bisection) and pasted in with their source noted in a
  comment. If the calibration is rerun with a different seed or sample count, the
  constants in `src/montecarlo.py` must be updated to match.

- **D-M4-3: `ROOT_SEED = 20240401` for the M4 Monte Carlo grid.** Chosen to be
  distinct from M3's `ROOT_SEED = 20240301` so that M4 seed streams are independent
  of M3's, even if the `derive_seed` label strings happen to collide. The year-month-day
  naming convention (YYYYMMDD) is a project-wide convention from M3.

- **D-M4-4: Figure 7 analogue uses `MAX_ITER = 200` (not 300).** The paper's own
  Figure 7 shows iterations up to 200 on its x-axis, and the figure's purpose is to
  illustrate the trajectory shape and the stopping iteration, not the full 300-step
  noiseless-recovery budget. Using 200 here matches the paper's visual and keeps the
  representative runs comparable to the paper's figure. The grid itself (M4.3) still
  uses 300 for all cells (matching the noiseless setting's budget), with early stopping
  via the discrepancy principle for noisy runs.

- **D-M4-5: the `active_mask` and `include_blood` mechanism (M4.2) is a single
  implementation in `src/jacobian.py` and `src/irgnm.py`, not separate forks.**
  `remaining_task.md` §4.2 explicitly requires this. The mask is a boolean array of
  length N_PARAMS=23; when provided, only the True columns of the Jacobian are used in
  the linear sub-solve, and the result is scattered back into a full delta before the
  projection step. The `include_blood` flag drops F² rows from the stacked system
  entirely. Both M4.2 and M4.4 use the same parameters; no forking was needed.

- **D-M4-6: "a single C_P measurement" (M4.4 step 5) is implemented as the existing
  F² block with `s_blood` truncated to one time point — no new forward operator.**
  F² is defined as `C_WB_data(s)·f_m(s) − C_P(λ,μ)(s)` (eq. 20). In the M4.4 runs the
  plasma-fraction parameters m are frozen at ground truth, so `f_m` is the true f and
  `C_WB_data·f_true = C_P_true` exactly; F² therefore reduces to
  `C_P_true(s) − C_P(λ,μ)(s)`, which is *literally* a measurement of C_P at s. This is
  asserted as an identity in
  `tests/test_identifiability.py::test_blood_block_vanishes_at_ground_truth`
  (|F²(x_true)| < 1e-12 at all four candidate times). Writing a separate direct-C_P
  operator would have duplicated code to compute the same residual.
  Which of the four candidate blood times to use was *measured*, not assumed, per
  project rule 6 — all four are swept in `experiments/m4_identifiability.py`.

- **D-M4-7: the blood measurement in the M4.4 step-5 runs is left noiseless even when
  the TACs are noisy.** Step 5 asks what an *exact* C_P value does to ζ; adding blood
  noise at the same time would confound "the ambiguity is removed" with "the
  measurement that removes it is itself uncertain". Noisy blood data is already
  covered by M4.3 Setup C, so nothing is lost. `build_observations` therefore applies
  Poisson TAC noise only.

- **D-M4-8: M4.4 adds a relative-residual acceptance gate, `FIT_RESIDUAL_TOL = 1e-6`,
  local to `src/identifiability.py`.** `run_irgnm`'s `diverged` flag only checks that
  the iterates stayed finite. That is necessary but not sufficient for *noiseless*
  data: with δ_y = 0 the discrepancy principle cannot fire, so a run that stalls far
  from the data still exits after `max_iter` with `diverged=False`, and its ζ is
  meaningless. Measured over 20 seeds at δ_x = 0.1, tissue-only: 18 runs land at
  relative residual 1.9e-10 to 2.3e-9, one returns NaN (correctly flagged by the
  existing check), and one (seed_idx=5) stalls at **3.05e+03** while reporting
  `diverged=False` — with K1-ratio spread 4.3e-02 and k3 error 5.6e-01, i.e. visibly
  garbage. The good/bad gap spans twelve orders of magnitude, so the threshold is not
  delicate; 1e-6 sits well inside it and means "the fit reproduces the data to six
  significant figures". For *noisy* data the gate defers to the paper's own criterion
  (the discrepancy principle fired), since the residual floor is then set by δ_y.
  Note this **tightens** acceptance — a stalled fit is counted and reported as a
  failure rather than averaged into the statistics — so it is not a rule-4 tolerance
  loosening. It is deliberately scoped to this module: `src/irgnm.py` is untouched and
  M4.3's published divergence counts are unchanged.
  **Open item this raises for M4.5/M5:** M4.3's noiseless cells used the same
  finiteness-only criterion, so some of its "converged" runs may be stalls of this
  kind. Re-auditing the M4.3 tables against this gate was left out of M4.4 as
  out-of-scope; it is logged here so the decision is visible rather than forgotten.

- **D-M4-9: M4.4 step 6 runs BOTH discrepancy-principle conventions side by side,
  because the existing one never fires.** `src.noise.compute_delta_y` returns an RMS
  (`||C_noisy − C_clean|| / sqrt(n_obs)`), but `run_irgnm` stops when `||r|| ≤ τ·δ_y`
  where `||r||` is a plain 2-norm. The comparison is therefore `sqrt(n_obs) = 10x` too
  strict, and **measured: the rule fires in zero of 240 noisy tissue-only runs** — every
  one reaches `max_iter = 300`. This was inherited from M4.3 (its "converged" counts are
  really "did not go non-finite" counts). Two conventions are now selectable via
  `run_identifiability_case(stopping=...)`:
    - `"rms"` — δ_y passed through unchanged. M4.3's convention; kept so M4.4's counts
      stay comparable with M4.3's Table 1.
    - `"morozov"` — δ_y passed as `δ_y·sqrt(n_obs)`, i.e. the actual noise *norm*, which
      is what the Morozov discrepancy principle compares against. Dimensionally
      consistent; the rule then fires at iteration ~24–71.
  Neither was adopted as "the" answer because they answer different questions, and the
  difference is itself a result: measured at high_count, δ_x = 0.1, seed 0 — `rms` runs
  to 300 iterations and gives K1-ratio spread 8.5e-03 with k3 error 5.7e-01, while
  `morozov` stops at iteration 71 and gives spread 2.7e-02 with k3 error 4.4e-02. Running
  to convergence settles the iterates onto the null manifold (tight spread) at the cost
  of fitting noise; stopping early regularises (better k3) but truncates the signature
  before it has fully formed. The M4.4 report gives both tables.
  `src/irgnm.py` is **not** modified — the scaling is applied by the caller — so M4.3's
  published numbers are untouched.

- **D-M4-10: noisy acceptance gate, `NOISY_RESIDUAL_FACTOR = 2.0` × the noise floor,
  for the `"rms"` convention only.** Under `"rms"` the discrepancy rule never fires, so
  there is no stopping-rule criterion to defer to and D-M4-8's noiseless threshold
  (1e-6) is meaningless — the residual cannot go below the noise. The natural floor is
  `||noise|| / ||y|| = δ_y·sqrt(n_obs) / ||y||`. Measured over the 240 noisy tissue-only
  cells: non-diverged fits cluster at **0.78–0.97×** that floor (just below it, as
  expected once the iterates begin fitting noise), while failures sit at **5.6–10.9×**.
  2.0 lies inside that gap with roughly 2x headroom on each side. Under `"morozov"` the
  gate defers to the discrepancy principle instead, since there it genuinely fires.

- **D-M4-11: a `"morozov"` run whose discrepancy rule fires at iteration 0 is rejected,
  not counted as a fit.** Such a run has taken no IRGNM step at all — verified:
  `x_final` is bit-identical to the projected initial guess, and the ζ it reports is
  exactly the initial guess's ζ (1.0624496260234935 in both, at low_count, δ_x = 0.1,
  seed_idx 0). It says nothing whatever about identifiability. **Measured: this happens
  in 20 of 20 runs at low_count, δ_x = 0.1**, because `τ·δ_y·sqrt(n_obs) = 6.8·0.073·10
  = 5.0` already exceeds the initial residual. That is the discrepancy principle working
  correctly — it is telling us the data is too noisy to improve on the guess — but
  reporting its ζ as evidence would have been badly misleading, since the tables would
  have shown a *tighter*-looking ζ range at low_count than at high_count purely because
  no fitting occurred. `fit_accepted` therefore requires `converged_at >= 1` under
  `"morozov"`, and `summarise` reports `n_trivial` as its own column, distinct from
  divergences and stalls.

- **D-M4-12: `zeta_from_lambda` returns NaN for a λ component that is exactly zero**
  rather than raising or emitting a divide-by-zero warning. Divergent iterates can drive
  a λ component to 0; `spread` already maps any non-finite ratio set to NaN, so the
  failure propagates visibly into the statistics instead of being hidden by a warning
  printed once and then suppressed.

- **D-M4-13: M4.5 uses the corrected Morozov norm and one exact early C_P sample.**
  `compute_delta_y` is an RMS, while IRGNM compares a 2-norm; pass
  `delta_y*sqrt(n_obs)` as in D-M4-9. The single sample at frame 3 removes the
  structural K1/lambda ambiguity measured in M4.4. Freeze the unobservable
  plasma-fraction block at truth and use delta_x=0.1, 20 fixed seeds; this is a
  controlled consistency check of identifiable kinetic parameters, not a repeat
  of the full 23-parameter M4.3 grid. Reject iteration-0 stops and report them.

- **D-M4-14: variance is compared on paired observations in two ways.** The
  primary full-fit comparison uses the same observations and initial guess for
  regularisation on/off at each seed; variance is the sum of sample variances
  of the 12 truth-normalised kinetic parameters, computed only on seeds where
  *both* fits stopped after at least one step. Report failures separately because
  conditioning on survivors can bias a variance. A fixed-Jacobian one-step
  check at the same x0 and schedule iteration 70 isolates observation-noise
  variance with all 20 seeds retained. Iteration 70 lies in the high-count
  stopping range measured in M4.4; it is a diagnostic, not a claim that a
  one-step iterate is a complete reconstruction. Zero diagonal means all six
  regularisation strengths are exactly off; no alternative fitter is used.

- **D-M5-1: `run_all.py` invokes every experiment script as a fresh subprocess**
  (same interpreter, `sys.executable`), rather than importing and calling each
  script's functions in one process. Reason: several scripts set module-level
  matplotlib state, and a couple (`m4_grid.py`, `m4_identifiability_noisy.py`)
  take minutes and print progress meant to be read live — subprocess isolation
  keeps one script's state from leaking into the next and keeps stdout
  attributable to the script that produced it. Default behaviour is fail-fast
  (a later script that reads an earlier script's JSON would fail anyway with a
  more confusing error); `--continue` overrides this for a full pass/fail
  report.

- **D-M5-2: the IRGNM solver's "timing vs problem size" is measured on
  synthetic stacked systems, not the real PET problem.** The real problem is
  fixed at 104 residuals x 23 parameters — there is no meaningful way to "make
  it bigger" without changing the model. What actually scales with problem
  size is the stacked least-squares solve `src.qr.lstsq` performs once per
  IRGNM iteration (`irgnm_step`'s `"qr"` path); that exact call is benchmarked
  on synthetic matrices that preserve the real problem's rows:cols aspect
  ratio (104:23 ~= 4.52:1), so the *shape* of the linear system is faithful
  even though the entries are synthetic. A single real `run_irgnm` call at the
  actual fixed size is also timed (ms/iteration) to ground the synthetic curve
  in one concrete number. `numpy.linalg.lstsq` is the Track B reference at
  the same synthetic sizes.

- **D-M5-3: the IRGNM-vs-`scipy.optimize.least_squares` Track B comparison
  gives scipy the same analytic Jacobian and the same D(F) box bounds** (via
  `jac=analytic_jacobian`, `bounds=`), so the comparison is between two
  optimisation *strategies* on identical residual/derivative code, not between
  two different Jacobians. **Measured result, delta_x=0.1, seed_idx=0,
  noiseless, tissue+blood:** our regularised IRGNM reaches final relative
  error 9.27e-07 in 300 iterations (0.49s); `scipy.optimize.least_squares`
  (method="trf", unregularised) reports `status=2` ("xtol satisfied", i.e. it
  believes it converged) after 40 function evaluations (0.05s) at final
  relative error 1.05 — **worse than the initial guess**. This is not a bug in
  either solver: it is the expected behaviour of an unregularised
  Gauss-Newton-family method on the paper's genuinely ill-posed inverse
  problem (the whole reason Tikhonov regularisation via IRGNM's six-block
  schedule is the paper's method, not incidental). Reported as a positive
  result for the project's central claim, not filed as a scipy bug.

- **D-M5-4: the M5 trend checklist's "known C_P vs clean/noisy C_WB"
  comparison uses `rel_error_K_final`** (the 12 kinetic parameters K1/k2/k3
  only) from the raw M4.3 grid records for Setup B/C, not
  `table2_parameters.json`'s `mean_rel_error_total` (all 23 parameters,
  including arterial lambda/mu and plasma-fraction m). The M4.5 known-C_P arm's
  `metabolic_error` function measures the same 12-parameter subset; comparing
  it against a 23-parameter total would have been apples-to-oranges and had
  understated Setup B/C's kinetic accuracy in an earlier draft of this
  analysis. Even with the metric fixed, the comparison remains only
  *informative*, not fully controlled: the known-C_P arm freezes m at truth
  and uses the corrected Morozov stopping rule (D-M4-9), while Setup B/C fit
  all 23 parameters under the uncorrected `rms` convention, and only 7-13 of
  20 seeds per cell survive (non-diverged) at normal_count/high_count.
  Similarly, the "low-count fails, especially for f" trend's per-noise-level
  mean m-error on *survivors* is reported alongside the divergence rate, not
  in place of it: at low_count only 3 of 60 runs converge at all, so the
  survivors' mean error is a 3-sample statistic dominated by survivorship
  bias and must not be read as "f recovers fine at low_count".

- **D-M5-5: committed M1-M4 results are kept as the reference; they are
  reproducible bit-for-bit only on the numerical stack that produced them.**
  A full `run_all.py` pass (15/15 scripts OK, 997 s) on numpy 2.2.6 / scipy
  1.14.1 / OpenBLAS 0.3.29 is bit-identical run-to-run (M4.5 rerun compared with
  `cmp`), but differs from the committed files (Anaconda, numpy 2.0.2) by last-bit
  round-off (~1e-15 relative), which near-divergent IRGNM trajectories amplify
  into different discrete outcomes: 11 of 48 Table 1 cells change by +-1-2
  divergences (**total unchanged, 498/960**); M3 noiseless divergences at
  delta_x=0.3/0.4 go 4->3 and 9->8; M4.4 accepted fits 66->68 with the **worst
  K1-ratio spread unchanged at 1.83e-08**; M4.5 noiseless accepted 18->17. Every
  headline conclusion is unchanged. The committed files are restored rather than
  overwritten, because RUN_M4.md cites them and they are a teammate's published
  numbers; the environment sensitivity is itself the result.

- **D-M5-6: the Simpson round-off floor is a property of our formulation, not of
  Simpson's rule — this revises D-M1-6.** D-M1-6 described the rising error at
  fine grids as "expected ... not a defect". The M5 Track B comparison on the same
  uniform grids (sin on [0, pi]) shows `scipy.integrate.simpson` decaying to
  2.2e-16 at n=12801 while ours *rises* to 1.8e-07 (4.1e-07 at n=25601); the two
  agree to within 2x up to n=401. The cause is `_quadratic_segment_integral`
  evaluating cubic antiderivatives in absolute coordinates and differencing them
  over a segment of width 2h, divided by a denominator of order h^2 — catastrophic
  cancellation that grows as h shrinks. Not fixed in M5: changing
  `src/quadrature.py` would shift every M1/M2 number, and at the pipeline's grid
  (`n=1601`, graded) the error is already bounded by the measured 5.9e-10
  closed-form-vs-quadrature agreement. Evaluating the basis integrals in local
  coordinates (shift by x0) is the likely fix, left as an open item.
