# PLAN.md — milestones and acceptance criteria

Work strictly in order. A milestone is DONE only when every acceptance criterion is met
with a recorded number, and `handoffs/RUN_M<n>.md` has been written. Do not begin the
next milestone until the student returns with review feedback.

---

## M0 — Skeleton and scaffolding

Build: repo layout, `config.py` holding all ground-truth constants, seeded RNG plumbing,
plotting helpers, `DECISIONS.md`, `logs/failures.md`, `handoffs/TEMPLATE.md`, and the
`tests/test_no_library_solvers.py` guard.

Ground-truth constants to hard-code in `config.py` (from the paper, Section 5.1 and
Table 2):

- Arterial plasma input, degree p = 4:
  `C_P(t) = -10.9136*exp(-13.4522t) + 9.545*exp(-3.2672t) + 0.7331*exp(-0.1532t) + 0.6355*exp(-0.0106t)`
  with t in minutes.
- Parent plasma fraction (biexponential, Remark 18):
  `f(t) = 0.2*exp(-0.2t) + 0.8*exp(-0.005t)`, and `C_WB(t) = C_P(t)/f(t)`.
- Fractional blood volume `V_B = 0.05`, with `C_PET = (1-V_B)*C_T + V_B*C_WB`.
- Four regions, control-group kinetics (K1, k2, k3):
  - frontal      0.1570, 0.1740, 0.1180
  - temporal     0.1610, 0.1790, 0.0960
  - occipital    0.1770, 0.1590, 0.0880
  - white matter 0.1000, 0.1610, 0.0470
- Frame schedule, 25 frames: 4x5s, 4x10s, 4x30s, 2x60s, 3x150s, 6x300s, 2x600s.
  Store both frame edges and frame mid-times.
- Unknown vector layout, 23 parameters:
  `(lambda_1..4, mu_1..4, m_1..3, K1^1,k2^1,k3^1, ..., K1^4,k2^4,k3^4)` where m are the
  three parameters of the biexponential f.

Acceptance:
- Repo runs `pytest` with the guard test passing.
- Seeded RNG returns identical streams across processes.

---

## M1 — Numerical primitives (Track A)

Build: LU with partial pivoting (factor + solve), Householder QR (factor + solve +
least squares), our own uniform RNG, Box-Muller normal, Poisson sampler, trapezoid and
Simpson integration on non-uniform grids.

Acceptance (report the actual numbers):
- LU and QR solve 200 random well-conditioned systems of sizes 5..100; max relative
  residual and max relative error vs `numpy.linalg.solve` (Track B, in tests only).
- LU and QR on a deliberately ill-conditioned system (e.g. Hilbert 8x8): report the error
  and the condition number, and comment on which is more stable and why.
- Trapezoid and Simpson on at least three integrals with known closed form; a log-log
  error-vs-h plot; fitted convergence orders reported numerically (expect ~2 and ~4).
- RNG: 1e6 samples, report mean/variance vs theory, a chi-square or KS statistic, and a
  histogram figure. Poisson sampler checked for mean == variance == lambda.

---

## M2 — Forward model

Build:
- `closed_form`: eq. (3) of Lemma 6, including the degenerate branches when
  `mu_j == 0` or `k2 + k3 + mu_j == 0`, with numerically safe handling of
  near-degenerate denominators (use `expm1`-style stable forms; document this).
- `quadrature`: eq. (1) of Lemma 5 evaluated with our own Track A integrator.
- `C_PET` assembly including `V_B`.

Acceptance:
- Closed form vs our quadrature: max relative difference across all 4 regions and all 25
  frames — report the number.
- Both vs an independent ODE integration of system (S) using `scipy.solve_ivp`
  (Track B, tests only) — report the number.
- Sanity: `C_T(0) == 0`, non-negativity, and late-time slope approaches
  `K1*k3/(k2+k3) * C_P` — report the measured agreement.
- A figure reproducing the qualitative content of the paper's Figure 2 (log time axis):
  f, C_WB, C_P, and the four regional TACs.
- Near-degeneracy stress test: sweep `mu` through `-(k2+k3)` and show the closed form
  stays smooth and matches quadrature (this is the round-off / cancellation experiment).

---

## M3 — Jacobian and the solver

Build: analytic Jacobian of the forward operator with respect to all 23 parameters;
Gauss-Newton; then IRGNM per eq. (26) with regularisation ansatz `alpha_i = a*exp(-b*i)`
(eq. 27), projection onto the feasible set `D(F)`, and the discrepancy-principle stopping
rule. Each IRGNM step's linear system must be solved with our own LU and, separately,
our own QR — both paths available and compared.

**LU-vs-QR comparison, specified precisely (post-M1-review correction, see
DECISIONS.md D-M1-9 — this replaces the vaguer original wording).** Do NOT repeat
M1's Hilbert-matrix exercise for this comparison; a single ill-conditioned matrix
already showed (M1 report + the M1-correction multi-size Hilbert sweep) that LU-vs-QR
accuracy ordering on a *generic square system* is round-off noise, not a real
stability difference — LU with partial pivoting is backward stable and there is no a
priori reason to expect QR to win there. QR's genuine advantage in this milestone is
specific to the IRGNM normal equations: forming `F'^T F' + alpha I` explicitly (the
literal reading of eq. 26) squares the condition number, `cond(F'^T F') = cond(F')^2`.
The two paths to compare are therefore:

1. **Normal equations + LU**: form `F'^T F' + alpha I` and `F'^T r + alpha(x0-x)`
   explicitly, solve with `src.linalg.solve`.
2. **Stacked least squares + QR**: solve
   `minimise || [F' ; sqrt(alpha) I] delta - [r ; sqrt(alpha)(x0-x)] ||_2` directly via
   `src.qr.qr_lstsq` on the stacked `(m+n) x n` matrix — never forming `F'^T F'`, so
   this path works at `cond(F')`, not `cond(F')^2`.

Run both on the real problem: `F'` evaluated at a real (not synthetic) parameter point
from the forward model, swept over the `alpha_i` regularisation schedule (eq. 27),
across at least one noiseless and one noisy setting. Report, per `alpha`: `cond(F')`,
`cond(F'^T F' + alpha I)` (should be visibly larger, ideally close to the squared
value), the two paths' solutions' relative disagreement, and — if it can be
constructed — a case where the normal-equations path visibly degrades (loses digits,
or fails to reduce the residual as much as the stacked-QR path) while the stacked path
does not. This is the sharpest, most on-topic test of "QR earns its keep here" that
this milestone can produce; a repeat Hilbert-matrix demo would not test the actual
mechanism at play (normal-equations squaring) and should not be used as the headline
result for this criterion.

Acceptance:
- Analytic Jacobian vs central finite differences: max relative error over random
  parameter points — report the number (expect ~1e-6 or better).
- **Noiseless recovery.** From perturbed initial guesses at `delta_x` in
  {0.1, 0.2, 0.3, 0.4}, recover the ground truth from noise-free data. Report final
  relative error `||x_k - x_dagger|| / ||x_dagger||` for each — expect ~1e-7 or smaller.
  Plot the convergence curves (this is the paper's Figures 4-6, top row).
- LU path vs QR path (normal equations vs stacked least squares, per the comparison
  specified above): agreement of the iterates, plus `cond(F')` vs `cond(F'^T F' +
  alpha I)` reported across the alpha sweep, plus a runtime comparison reported as
  operation counts vs measured scaling (see DECISIONS.md D-M1-15 for why a bare
  wall-clock number is not sufficient on its own).
- If a run diverges, log it rather than tuning it away.

---

## M4 — Noise, regularisation, and the identifiability experiments

Build: noise model at the TAC level (Poisson-derived and Gaussian variants) with three
levels approximating the paper's high / normal / low count settings; the three setups
(A: f known and C_WB noiseless; B: f unknown, C_WB noiseless; C: f unknown, C_WB noisy);
Monte Carlo harness of 20 realisations per cell.

Acceptance:
- Divergence table across (setup x noise level x delta_x), in the shape of the paper's
  Table 1.
- Reconstructed parameter table (mean and std over non-divergent runs) in the shape of
  the paper's Table 2.
- Relative error per parameter type (K1 vs k2 vs k3), matching the paper's Figure 7.
- **The identifiability signature experiment (highest priority in this milestone).**
  Fit using tissue TACs only, with no arterial data at all. Theory (Proposition 12)
  predicts: `k2` and `k3` recover correctly, while every `K1^i` is wrong by the SAME
  multiplicative constant `zeta`, with `C_P` scaled by `1/zeta`. Report the four ratios
  `K1_est^i / K1_true^i` and their spread — the theory says they must coincide. Then add
  a single `C_P` measurement and show `zeta -> 1`. This is the experiment that
  demonstrates the paper's mathematics is actually encoded in the code.
- Consistency check: error decreases as noise decreases (Theorem 21).
- Regularisation on vs off: show it reduces variance across realisations.

---

## M5 — Analysis, figures, and report material

Build: final figure and table generation, timing/complexity measurements, a
Track A vs Track B accuracy-and-runtime comparison, and a written summary mapping every
course topic to where it appears in the code.

Acceptance:
- All figures regenerate from a single command with fixed seeds.
- A results narrative stating, explicitly and in advance, what counts as agreement with
  the paper: qualitative trend agreement, not digit matching (the imaging chain was cut,
  so absolute noise levels are not comparable).
- Trends to check against the paper: `K1` recovers better than `k2`/`k3`; known `C_P`
  beats `C_WB`-only, which beats noisy `C_WB`; the low-count setting fails, especially
  for the parameters of `f`.