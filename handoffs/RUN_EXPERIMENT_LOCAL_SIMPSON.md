# Run report: experimental local-coordinate Simpson weights

## 1. Summary

Implemented the D-M5-6 suggestion as an optional change to the evaluation of
nonuniform Simpson weights. This is the same mathematical integration rule,
computed without subtracting cubic antiderivatives in absolute coordinates.
It improves the independent PET quadrature calculation; the IRGNM call path
does not use it. The original default and historical results remain available.

## 2. Files

`src/quadrature.py`: new weight helper and option.
`src/forward_model.py`: pass that option to the two integrals.
`experiments/local_simpson.py`: deterministic paired comparison and figure.
`tests/test_local_simpson.py`: 11 numerical checks.
`EXPERIMENT_README.md`: derivation, rationale, measurements and presentation claim.
`README.md`, `DECISIONS.md`, `logs/failures.md`: link and experiment records.
`.gitignore`: exclude the local validation environment.

## 3. Mathematics

Numerically evaluate the integrals in Lemma 5, equation (1), of `paper-2.pdf`
(printed page 5). The contribution is our evaluation of the quadratic Lagrange
weights in local coordinates, not a new equation from the paper. The full
derivation is in `EXPERIMENT_README.md`.

## 4. Acceptance checks

This is an optional follow-up, not a new PLAN.md milestone.

| Check | Measured result |
|---|---|
| Full existing and new suite | 151 passed in 131.53 seconds |
| Hand-written-solver guard | 3 passed in 0.11 seconds |
| Translated nonuniform quadratic | Local absolute error 0 against float64 reference at offsets 0, 1000, 1e6 |
| Fourth-order uniform-grid check | Passes 3.9 < measured pairwise order < 4.1 |
| PET, 6401 nodes, all 100 outputs | Maximum relative error 1.285976e-12, below 1e-11 target |
| Historical data preserved | No existing results files modified |

## 5. Numerical results

| PET nodes | Baseline max relative error | Local max relative error |
|---:|---:|---:|
| 401 | 8.415091e-8 | 8.415774e-8 |
| 1601 | 5.880318e-10 | 3.293577e-10 |
| 6401 | 2.270355e-8 | 1.285976e-12 |
| 12801 | 1.269894e-7 | 7.988898e-14 |

At 12801 nodes, sine absolute error is 1.785855e-7 baseline versus 1.332268e-15
local. Exp and Runge comparisons are also in the generated JSON and README.
No randomness, timing advantage or IRGNM convergence advantage is claimed.

## 6. Figure

`results/experimental_local_simpson/convergence.png`: refinement curves for the
sine integral and maximum error across the four PET regions and 25 frame times.

## 7. Assumptions

D-EXP-1 records the optional default, unchanged grid and summation, deterministic
study and use of the closed form as a finite-precision PET reference. Source
hashes, environment and config hash `2f38ba062972` are saved with the results.

## 8. Failures and limitations

At 401 nodes the local PET error is slightly larger; this is retained in the
report. Local coordinates do not fix already-rounded node locations, strongly
unequal-grid conditioning, exponential overflow or the paper's failure-table gap.
See the dedicated section in `logs/failures.md`.

## 9. Track A / Track B audit

The new integrator uses arithmetic only. No SciPy or library solver is used by
the experiment. References are elementary exact integrals and the existing
closed-form PET calculation. Matplotlib is used only to plot results. The full
test suite retains its pre-existing Track B reference comparisons.

## 10. Mutation check

In a separate Python process, assigned the baseline helper to
`src.quadrature._quadratic_segment_integral_local`, then ran the tests matching
`shifted_nonuniform or fine_grid_sine`. Result: 3 failed, 1 passed, 7 deselected.
The offset-1000, offset-1e6 and fine-grid sine cases detect reintroduction of the
cancellation problem. The mutation existed only in that process's memory.

## 11. Reproduction

```powershell
.venv/Scripts/python.exe experiments/local_simpson.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest tests/test_no_library_solvers.py -q
```

Python 3.12.1, NumPy 2.2.6, Windows 11; project dependencies installed from
`requirements.txt`. The experiment uses no random seed (`seed: null`).

## 12. Open questions

Should the opt-in formula become the default with regenerated M1/M2 reports?
That is a separate adoption decision. The remaining Table 1 differences need
their own diagnosis because Simpson is absent from the fitting call path.

## 13. Proposed next step

Use the experiment and README as the small course contribution. Present the
calculation accuracy and refinement behavior, with the explicit scope above.
