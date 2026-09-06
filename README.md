# PET Pharmacokinetic Parameter Identification — CSE 402 Project

A BUET CSE 402 (Numerical Analysis, Simulation and Modeling) sessional project that
reimplements, by hand, the numerical core of:

> Holler, Morina, Schramm (2024). *Exact parameter identification in PET
> pharmacokinetic modeling using the irreversible two tissue compartment model.*
> Phys. Med. Biol. 69 165008. (`paper-2.pdf`)

The course grades on the student implementing standard numerical methods from
scratch, so every solver, integrator, and random number generator here is our own
code — no `scipy.optimize`, `scipy.linalg`, `scipy.integrate`, `numpy.linalg.solve`,
or `numpy.random` anywhere under `src/`. Library routines are only used in `tests/`
and clearly-marked benchmark scripts under `experiments/`, purely to check our own
implementations against an independent reference.

`402.pdf` is the project proposal this work is graded against.

---

## Start here — read these before doing anything

> **AI coding agents: read [`AGENTS.md`](AGENTS.md) first.** It is the session-start brief
> — the hard constraints, which project function to use instead of each library call, and
> what to pick up next. Claude Code users: `cp AGENTS.md CLAUDE.md` once after cloning so
> it auto-loads (`CLAUDE.md` is git-ignored, so this stays local to you).

**If you are picking up a task (human or AI agent), read in this order. Do not start
writing code until you have read at least the first two.**

| Order | File | Why |
|---|---|---|
| 1 | **`remaining_task.md`** | The single self-contained brief: the non-negotiable ground rules, what already exists and is verified, the decisions you must not silently contradict, and the full remaining task list. **If you read only one file, read this one.** |
| 2 | **`PLAN.md`** | The milestone specification: what each milestone must build and the acceptance criteria that define "done". Your work is graded against these, so check them before you start, not after. |
| 3 | **`DECISIONS.md`** | Every judgement call made so far and why (numbered `D-M<milestone>-<n>`). Check here before changing an existing approach — most surprising choices are deliberate and measured. |
| 4 | **`handoffs/RUN_M1.md`, `RUN_M2.md`, `RUN_M3.md`** | The milestone reports, with every measured number. Also the standard your own report is held to; `handoffs/TEMPLATE.md` is the format to follow. |
| 5 | **`logs/failures.md`** | Divergences, non-convergence, and things that did not work — including bugs already found and fixed. Read it so you do not rediscover them. |
| 6 | **`paper-2.pdf`** | The source paper. Go to it for the specific equation, lemma or proposition you are implementing; the code comments name them explicitly (e.g. "eq. (3) of Lemma 6"). |

**Five rules that will get your work rejected if broken** (full versions in
`remaining_task.md` Part 1):

1. **No library solvers under `src/`** — no `scipy.optimize`, `scipy.linalg`,
   `scipy.integrate`, `numpy.linalg.solve`/`lstsq`/`inv`, or `numpy.random`. The whole
   assignment is implementing these by hand. `tests/test_no_library_solvers.py` enforces
   it automatically, alias-resolution included. Library calls belong in `tests/` and
   marked benchmark scripts only, as references to check our own code against.
2. **Everything stochastic takes an explicit seed** (`src.rng.derive_seed` for sub-streams).
3. **Report measured numbers, not adjectives.** "Tests pass" is not a result.
4. **Failures are data** — log divergences in `logs/failures.md` rather than reseeding
   past them. A Monte Carlo study reporting zero failures is suspicious, not impressive.
5. **Never loosen a tolerance to go green.** If a test fails, the code is wrong, or the
   tolerance was wrong for a stated numerical reason that you write down.

> **Note for AI agents:** `CLAUDE.md` is intentionally not tracked in this repository, so
> it will not be auto-loaded from a fresh clone. `AGENTS.md` carries the same standing
> rules and is tracked — treat it, together with `remaining_task.md` Part 1, as the
> authoritative rule set.

---

## Where the project stands right now

**Milestones M0-M4.3 are complete and verified. M4.4, M4.5, and M5 remain.**
The full task list is in `remaining_task.md`. Highlights of what is finished:

- **Numerical primitives** (`src/linalg.py`, `src/qr.py`, `src/rng.py`,
  `src/quadrature.py`): LU factorisation with partial pivoting, Householder QR
  (solve + least squares), a 64-bit LCG with a Box-Muller normal sampler and a
  Knuth-algorithm Poisson sampler, and trapezoid/Simpson integration on non-uniform
  grids. All four are checked against independent references (NumPy/SciPy in
  `tests/`, closed-form values, or a second Track A code path) with measured errors
  in the `1e-9` to machine-precision range.

- **Forward model** (`src/forward_model.py`): the irreversible two-tissue
  compartment ODE system's closed-form solution (paper eq. 3) and an independent
  direct-quadrature evaluation (paper eq. 1), both built on the primitives above.
  The two paths, plus a third independent ODE integration, agree to `5.9e-10` and
  `1.7e-13` relative difference across all four brain regions and all 25 PET frame
  times used in the paper's synthetic experiment.

- **Jacobian, conditioning and solver** (`src/jacobian.py`, `src/eigen.py`,
  `src/irgnm.py`): the analytic Jacobian of the forward operator with respect to all
  23 parameters, verified against central finite differences; a conditioning analysis
  using our own power method and inverse power iteration; the null-space experiment
  showing the paper's predicted `K1`/`lambda` non-identifiability direction; and an
  IRGNM solver with multi-parameter regularisation that recovers the ground truth from
  noise-free data to a **median relative error of 4.3e-7** over 20 seeds, with zero
  divergences.

- **Noise modeling and measurement setups** (`src/noise.py`, `src/montecarlo.py`):
  Poisson-derived TAC noise and Gaussian blood noise calibrated to the paper's
  high/normal/low count levels. Three distinct measurement setups (A: fixed plasma
  fraction, clean blood; B: full setup, clean blood; C: full setup, noisy blood)
  implemented cleanly without solver duplication. A 960-cell Monte Carlo grid
  successfully reproducing the divergence trends (Table 1), parameter recovery
  accuracies (Table 2), and error trajectories (Figure 7) reported in the paper.

Config (`src/config.py`) hard-codes every ground-truth constant from the paper's
Section 5.1 and Table 2 (arterial input, parent plasma fraction, four regional
kinetic parameter sets, the 25-frame acquisition schedule), so every later milestone
works against one shared, checkable source of truth. Everything is unit-tested
(`tests/`, run with `pytest` — currently **97 tests**) and every reported number is
regenerated by a script under `experiments/`, writing figures and JSON summaries to
`results/`.

## What comes next

Two milestones remain — full detail, with acceptance criteria, in `remaining_task.md`:

1. **M4.4/4.5 — Identifiability and consistency.** The highest-value item in
   the project — the identifiability signature experiment: recovering kinetic parameters
   from tissue curves alone, with no arterial blood sampling, and showing that every
   `K1` comes out wrong by the *same* multiplicative constant, exactly as the paper's
   Proposition 12 predicts, then removing that ambiguity with a single blood measurement.
2. **M5 — Analysis and report.** One-command regeneration of every figure and table with
   fixed seeds, timing and complexity measurements, a Track A vs Track B accuracy and
   runtime comparison, and a table mapping every course topic to where it appears in the
   code.

## Repository layout

```
src/            Track A implementation (the only code graded on originality)
tests/          pytest suite, including the guard that forbids library solvers under src/
experiments/    Scripts that regenerate every figure/table in results/, with fixed seeds
results/        Generated figures and JSON summaries (reproducible from experiments/)
```

## Running it

```
pip install -r requirements.txt
pytest                                    # full test suite
python3 experiments/m1_linalg_benchmark.py
python3 experiments/m1_rng_benchmark.py
python3 experiments/m1_quadrature_benchmark.py
python3 experiments/m2_forward_model.py
```
