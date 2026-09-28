# Mainline numerical changes: calculate the same PET model more reliably

## What was adopted

The stable local-coordinate Simpson formula is now the default. Three closely
related improvements also run by default: factor exponential convolution terms
before evaluating them, calculate their derivatives without subtracting nearly
equal quantities, and keep exponential decay inside the numerical integral.

These are **changes to numerical evaluation**, not changes to the PET equations,
the IRGNM algorithm, its regularisation schedule, parameter constraints or stopping
rule. They address identifiable arithmetic defects with algebraically equivalent
expressions. The implementation remains hand-written; no library solver replaces
the course work.

| Change | Where | Why it should help |
|---|---|---|
| Local Simpson weights by default | `src/quadrature.py` | Avoid subtracting cubic antiderivatives at nearby absolute coordinates |
| Factored exponential convolution | `src/forward_model.py` | Avoid an overflowing integral multiplied by an underflowed decay |
| Stable derivative and convolution moments | `src/forward_model.py`, `src/jacobian.py` | Preserve small derivatives and keep the Jacobian consistent with the forward model |
| Decay inside the quadrature integrand | `src/forward_model.py` | Avoid the same intermediate overflow in the independent numerical path |

The first change improves integration. The second and third also enter the
actual nonlinear fit. The fourth keeps the independent integration path useful
over a wider range of parameters.

## 1. Local-coordinate Simpson weights

### Theory

For nodes `x0 < x1 < x2`, let `h0=x1-x0`, `h1=x2-x1`, `H=h0+h1`. Integrate the
quadratic Lagrange interpolant after shifting the variable to `u=x-x0`. Its
weights simplify to

```text
w0 = H/6 * (2-h1/h0)
w1 = H/6 * (H/h0) * (H/h1)
w2 = H/6 * (2-h0/h1)
I  = w0*y0 + w1*y1 + w2*y2.
```

The old implementation expanded the basis in absolute coordinates and evaluated
differences of cubic antiderivatives. Both formulas are exact for the same
quadratic in exact arithmetic. The new one avoids that subtraction before the
small interval-width denominator magnifies its error.

When `h0=h1=h`, the weights reduce to `h/3,4h/3,h/3`. On an arbitrary irregular
grid, quadratic exactness is guaranteed; cubic exactness requires equal spacing.
This distinction is also documented in the
[SciPy Simpson reference](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.integrate.simpson.html).
No SciPy integration call is used in our implementation.

### Intuition and example

An area should not change merely because we label the left endpoint 1000 instead
of 0. For the same samples of `2+3u+4u^2` at
`u=[0,1/8,1/2,3/4,1]`, the exact integral is `29/6`.
The earlier controlled experiment measured a baseline error of `6.36e-7` after
shifting the nodes by 1000, while the local formula matched the float64 reference.
This example isolates coordinate-related cancellation from truncation error.

The production choice is now `simpson(x,y)` or `simpson_diag(x,y)` without any
extra argument. `local_weights=False` remains available strictly as a comparison
control. Pair ordering, sequential summation and the final trapezoid policy are
unchanged. The improvement is in arithmetic accuracy, not polynomial degree or
theoretical convergence order.

## 2. Factor the exponential convolution before computing it

### Theory

With `a=k2+k3`, the tissue model contains one kernel for each arterial rate:

```text
E(t,a,mu) = integral_0^t exp(-a*(t-s))*exp(mu*s) ds
         = t*exp(-a*t)*phi1((a+mu)*t),

phi1(z) = expm1(z)/z,  phi1(0)=1.
```

`expm1` protects against cancellation near zero. It does not protect against
overflow when `(a+mu)*t` is large and positive. Let `d=a+mu`. Reflect the
integration variable about the endpoint when `d>=0`, or equivalently factor the
larger endpoint exponential. For `t>=0`, this gives the identity

```text
E(t,a,mu) = t * exp(max(mu,-a)*t) * phi1(-abs(d)*t).
```

When `a>=0` and `mu<=0`, both exponential arguments are nonpositive. No enormous
exponential is created merely to be multiplied by a tiny one. At `d=0`, the
formula has the correct value `t*exp(-a*t)` directly, with no division by `d`.
Positive `mu` remains permitted, as in the existing parameter domain; genuinely
unrepresentable exponential growth may still overflow.

### Intuition and example

Consider `t=60`, `a=20`, `mu=-0.01`. The old expression tries to combine
`exp(-1200)` with `phi1(1199.4)`: in float64 that becomes `0*infinity`, or `NaN`.
But the integral itself is small and finite:

```text
E = (exp(-0.6)-exp(-1200))/19.99
  approximately 0.0274543089591809.
```

The new formula returns `0.027454308959180922`; an 80-digit Decimal calculation,
rounded to float64, gives `0.027454308959180912`. Relative error is `3.79e-16`.
This is a representable answer rescued by better evaluation, not by clipping a
parameter or replacing a failed value with a convenient constant.

The stress example deliberately uses a much larger decay than the ground-truth
parameters. Its role is to expose the mechanism. Whether iteration trajectories
benefit is measured separately below.

## 3. Compute derivatives as stable moments

Changing the forward calculation without attending to its derivatives would
leave the fitting process exposed to the old arithmetic problems. There are two
parts to the derivative improvement.

### 3a. Avoid cancellation in phi1 prime

The original derivative was evaluated as `1+(z-1)*phi2(z)`, where
`phi2(z)=(exp(z)-1-z)/z^2`. For large negative `z`, the second term approaches
`-1`; adding it to 1 loses the small answer, which approaches `1/z^2`.

Away from zero, use the equivalent formula

```text
phi1'(z) = ((z-1)*exp(z)+1)/z^2.
```

Near zero that numerator would itself cancel, so on `abs(z)<=1` use

```text
phi1'(z) = sum from k=0 to infinity of (k+1)*z^k/(k+2)!
         = 1/2 + z/3 + z^2/8 + z^3/30 + ... .
```

Twenty terms, evaluated by Horner's rule, suffice on that interval: the absolute
tail is bounded by `2e-20`, below double-precision rounding at values of this
size. The threshold and term count follow this truncation bound, not tuning to
make a PET result look better. Only the selected branch is evaluated.

**Example:** at `z=-1000`, the old relative error against an 80-digit reference is
`8.23e-11`, while the new value equals the float64-rounded reference `1e-6`.
At the deliberately extreme `z=-1e8`, the old relative error is about 11%; the
new result is `1e-16`. This does not imply normal PET fits have 11% derivative
error; it demonstrates the limiting cancellation mechanism.

### 3b. Avoid cancellation in the convolution derivatives

Differentiate the integral itself:

```text
dE/dmu = integral_0^t s     * exp(-a*(t-s))*exp(mu*s) ds
dE/da  = -integral_0^t (t-s)* exp(-a*(t-s))*exp(mu*s) ds.
```

Set `z=-abs(a+mu)*t`, `S=exp(max(mu,-a)*t)`, `p=phi1(z)`, `q=phi1'(z)`.
Rescaling time to `[0,1]`, and reflecting it for `a+mu>=0`, yields

| Case | `dE/dmu` | `dE/da` |
|---|---|---|
| `a+mu >= 0` | `t^2*S*(p-q)` | `-t^2*S*q` |
| `a+mu < 0` | `t^2*S*q` | `-t^2*S*(p-q)` |

The old calculation for `dE/da` used `-t*E + B`, where `B` could nearly equal
`t*E`. The moment formula calculates the smaller quantity directly. For
nonpositive `z`, `p-q` is the integral of `(1-u)*exp(z*u)` on `[0,1]`; it is
not a subtraction of nearly equal terms in the large-negative-argument limit.

At the degenerate point `a+mu=0`, `p=1`, `q=1/2`, so both branches give
`dE/dmu=t^2*exp(-a*t)/2` and `dE/da=-t^2*exp(-a*t)/2`. The forward calculation
and derivatives therefore agree across the branch boundary.

These moments are substituted into the existing analytic Jacobian. All
regularisation, projection and step calculations remain the same.

## 4. Keep exponential decay inside numerical quadrature

The independent quadrature path previously computed

```text
exp(-a*t) * integral_0^t exp(a*s)*C_P(s) ds.
```

It now computes the identical quantity as

```text
integral_0^t exp(-a*(t-s))*C_P(s) ds.
```

For `0<=s<=t` and `a>=0`, the extra exponential factor lies in `[0,1]`. This
avoids asking the integrator to sum enormous numbers only to shrink them later.
It also provides a numerically independent check of the factored closed form.

For the `a=20,t=60` example, the old integrand overflows near the endpoint;
the new one stays finite. The endpoint layer is sharp, so a sufficiently fine
grid is still needed for accuracy. Stable evaluation does not eliminate
discretization error or make every grid suitable for every parameter.

## How the contribution is tested

The tests use exact polynomial integrals, known elementary integrals, 80-digit
Decimal scalar references, finite-difference Jacobian checks, and the existing
independent ODE reference. The full suite passed **176 tests** after adoption.

`experiments/local_simpson.py` compares the two Simpson weight formulas while
holding the current forward-model evaluation and samples fixed.
`experiments/numerical_stability.py` retains explicit historical forward and
derivative formulas as controls and runs **240 noiseless fits per arm**:
3 setups, 4 perturbation levels and all 20 original seeds.

The fit comparison freezes observations before switching formulas, uses exactly
the same initial guesses, and keeps QR, IRGNM, projection, regularisation and the
300-step cap fixed. It reports both nonfinite/solver failures and the paper-style
“final error did not improve” criterion, together with rescued and regressed
cases. Every failure remains in the saved per-run data.

The old screenshot is historical context, not the control arm for a causal
claim: environment changes and slightly different observations can change
borderline trajectories. Use the paired experiment to assess the effect of
evaluation arithmetic on noiseless failures, and the regenerated full grid to
report current results.

An additional test-quality check temporarily restored the historical derivative
and convolution formulas in a separate Python process. Four targeted checks
failed: the derivative at `-100`, `-1000` and `-1e8`, and the large-decay forward
calculation. This also exposed an initially too-permissive default absolute
tolerance in the small-derivative test. The assertion now explicitly uses a
relative tolerance with zero absolute allowance; it passes the stable formula
and detects the historical loss of digits. No production source was mutated.

The comparison controls were checked against the pre-adoption repository
formulas: forward values and derivatives matched bit-for-bit at all 100
ground-truth region/frame combinations.

## Reproduction and provenance

From the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 -u experiments/run_all.py
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest tests/test_no_library_solvers.py -q
```

For just the contribution's comparisons:

```powershell
.venv/Scripts/python.exe experiments/local_simpson.py
.venv/Scripts/python.exe experiments/numerical_stability.py
```

The machine-readable contribution results are
[`comparison.json`](results/experimental_local_simpson/comparison.json) and
[`stability.json`](results/numerical_changes/stability.json). The complete run is
recorded in [`run_all_manifest.json`](results/m5/run_all_manifest.json).

The runner now includes both numerical-change experiments: **18 scripts** in
dependency order. It sets UTF-8 for subprocess output so the reports' mathematical
symbols work on Windows. Generated JSON records the config hash, seed, Python
and NumPy versions, platform and a hash of the `src/` files. Timing measurements
are machine-dependent and do not have to reproduce bit-for-bit.

The regenerated M1–M5 files are the current results. Earlier milestone handoffs
and the first Simpson experiment report document earlier runs; use the current
adoption report for updated numbers. The noise model and the original versus
paper-settings experiment variants remain unchanged.

## Limits on the claim

These changes improve arithmetic accuracy and representable range. They do not
prove global convergence of a nonlinear inverse problem, remove structural
non-identifiability, supply missing measurements or eliminate noise. A genuinely
positive arterial exponent can still overflow. Very uneven quadrature nodes can
still amplify errors; node-spacing information already lost in float64 cannot
be recovered. Sequential summation still has round-off.

The stable derivative evaluates a polynomial and additional factors, so better
accuracy is not a promise of faster fits. The paired experiment records elapsed
time for each arm as an implementation-cost diagnostic. It includes differences
in iteration counts and machine load; it is not an isolated operation-count
comparison or a guaranteed speed ratio.

The low-count iteration-zero stops from the screenshot cannot be fixed by a
better derivative calculated later. Their stopping/noise-model explanation must
remain distinct from numerical failures during an iteration.

This is a standard numerical-stability contribution to this implementation,
not a claim to have invented Simpson integration, Taylor evaluation or
exponential factorisation. A suitable presentation theme is: **algebraically
equivalent formulas can behave very differently in finite precision, and the
difference should be verified both on a controlled example and in the actual
application.**

## Connection to the numerical-analysis course

| Course idea | Concrete calculation in this contribution |
|---|---|
| Polynomial interpolation | Integrate the same three-point Lagrange polynomial in local coordinates |
| Newton–Cotes quadrature | Show equal-spacing Simpson weights and measure refinement error |
| Floating-point cancellation | Translate identical polynomial samples; compare two algebraically equal evaluations |
| Overflow and underflow | Explain why `0*infinity` can hide a finite convolution integral |
| Taylor approximation and error bounds | Bound the omitted terms of the 20-term derivative polynomial |
| Horner evaluation | Evaluate that polynomial with a short multiplication/addition recurrence |
| Numerical differentiation | Check the analytic Jacobian against central differences and high-precision scalar derivatives |
| Controlled simulation | Compare the same seeded initial guesses and observations in two evaluation paths |

For a short presentation, lead with Simpson's translated-quadratic example,
then the `t=60,a=20,mu=-0.01` convolution example. Finish with the regenerated
application measurements and clearly separate calculation accuracy from fit
convergence. That keeps the contribution focused on numerical analysis.
