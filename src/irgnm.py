"""Track A: IRGNM (iteratively regularized Gauss-Newton method), paper eq. (26)/(27).

Two solver paths for each step, per PLAN.md's corrected M3 section
(DECISIONS.md D-M1-9):
  - "lu": form the normal equations (F'^T F' + Lambda_i) and solve with our
    own `src.linalg.solve`.
  - "qr": solve the equivalent stacked least-squares problem
    min || [F' ; sqrt(Lambda_i)] delta - [r ; sqrt(Lambda_i)(x0-x_i)] ||_2
    with our own Householder QR (`src.qr.lstsq`), never forming F'^T F'.

Multi-parameter regularisation (Remark 23 / reviewer's M3 item 4): Lambda_i
is a diagonal matrix, not a scalar, with three blocks (metabolic, arterial
C_P, plasma-fraction f), each following the paper's ansatz (27),
`x_i = a * exp(-b * i)`, with its own (a, b) — see DECISIONS.md D-M3-5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import LAMBDA_SLICE, METABOLIC_START, MU_SLICE, M_SLICE, N_PARAMS, PROJECTION_EPS
from src.jacobian import analytic_jacobian, forward_operator
from src.linalg import solve as lu_solve
from src.qr import lstsq as qr_lstsq


def project(x: np.ndarray) -> np.ndarray:
    """Projection onto D(F) = R^4 x R^4 x [0,inf) x (-inf,0]^2 x [eps,inf)^{3x4}
    (paper eq. 18). Explicit clipping, applied after every IRGNM step."""
    x = x.copy()
    x[M_SLICE.start] = max(x[M_SLICE.start], 0.0)  # A >= 0
    x[M_SLICE.start + 1] = min(x[M_SLICE.start + 1], 0.0)  # xi1 <= 0
    x[M_SLICE.start + 2] = min(x[M_SLICE.start + 2], 0.0)  # xi2 <= 0
    x[METABOLIC_START:] = np.maximum(x[METABOLIC_START:], PROJECTION_EPS)  # K1,k2,k3 >= eps
    return x


@dataclass(frozen=True)
class RegularizationSchedule:
    """Six regularisation constants: (a,b) for each of the metabolic
    (alpha), arterial C_P (beta), and plasma-fraction f (gamma) blocks,
    following ansatz (27): x_i = a * exp(-b * i)."""

    a_alpha: float
    b_alpha: float
    a_beta: float
    b_beta: float
    a_gamma: float
    b_gamma: float

    def diag(self, i: int) -> np.ndarray:
        alpha_i = self.a_alpha * np.exp(-self.b_alpha * i)
        beta_i = self.a_beta * np.exp(-self.b_beta * i)
        gamma_i = self.a_gamma * np.exp(-self.b_gamma * i)
        d = np.empty(N_PARAMS)
        d[LAMBDA_SLICE] = beta_i
        d[MU_SLICE] = beta_i
        d[M_SLICE] = gamma_i
        d[METABOLIC_START:] = alpha_i
        return d


# Adopted directly from the paper's own tuned "full setup, noiseless C_WB"
# hyperparameters (Section 6, page 21): alpha_i=4000*2^(-i/7),
# beta_i=100*2^(-i/7), gamma_i=200*2^(-i/7), tau=6.8. Converted from base-2
# to the a*exp(-b*i) form used here: b = ln(2)/7 for all three (same
# exponent divisor in the paper). See DECISIONS.md D-M3-5 for why these were
# adopted rather than independently re-tuned, and what was checked before
# trusting them.
_LN2_OVER_7 = float(np.log(2.0) / 7.0)
DEFAULT_SCHEDULE = RegularizationSchedule(
    a_alpha=4000.0, b_alpha=_LN2_OVER_7,
    a_beta=100.0, b_beta=_LN2_OVER_7,
    a_gamma=200.0, b_gamma=_LN2_OVER_7,
)
DEFAULT_TAU = 6.8


def irgnm_step(
    x_i: np.ndarray,
    x0: np.ndarray,
    y_delta: np.ndarray,
    t_frames: np.ndarray,
    s_blood: np.ndarray,
    C_WB_data: np.ndarray,
    reg_diag: np.ndarray,
    solver: str = "qr",
    *,
    active_mask: np.ndarray | None = None,
    include_blood: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One IRGNM step (eq. 26). Returns (x_{i+1} (projected), F(x_i), F'[x_i]).

    Parameters
    ----------
    active_mask : boolean array of length N_PARAMS (23), or None.
        When provided, only the True entries are treated as free parameters
        in this step.  The linear sub-problem is solved for the active
        sub-vector; the inactive parameters are held fixed at x_i.
        This is the mechanism for M4.2 Setup A (fix the 3 plasma-fraction
        parameters m) without forking the solver.
    include_blood : passed to forward_operator / analytic_jacobian.
        When False, the F^2 blood-data block is dropped from the residual
        and Jacobian (M4.4 identifiability experiment, M4.2 Setup A).
    """
    Fx = forward_operator(x_i, t_frames, s_blood, C_WB_data, include_blood=include_blood)
    r = y_delta - Fx
    Fp = analytic_jacobian(x_i, t_frames, s_blood, C_WB_data, include_blood=include_blood)

    # --- Restrict to active columns if a mask is given -------------------
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        Fp_active = Fp[:, active_mask]
        reg_active = reg_diag[active_mask]
        x0_active  = x0[active_mask]
        x_i_active = x_i[active_mask]
    else:
        Fp_active  = Fp
        reg_active = reg_diag
        x0_active  = x0
        x_i_active = x_i

    if solver == "lu":
        M = Fp_active.T @ Fp_active + np.diag(reg_active)
        rhs = Fp_active.T @ r + reg_active * (x0_active - x_i_active)
        delta_active = lu_solve(M, rhs)
    elif solver == "qr":
        sq = np.sqrt(reg_active)
        A_stack = np.vstack([Fp_active, np.diag(sq)])
        b_stack = np.concatenate([r, sq * (x0_active - x_i_active)])
        delta_active, _resid = qr_lstsq(A_stack, b_stack)
    else:
        raise ValueError(f"unknown solver {solver!r}, expected 'lu' or 'qr'")

    # Scatter active delta back into a full N_PARAMS vector -----------------
    if active_mask is not None:
        delta = np.zeros(len(x_i))
        delta[active_mask] = delta_active
    else:
        delta = delta_active

    x_next = project(x_i + delta)
    return x_next, Fx, Fp


def _armijo_line_search(
    x_i: np.ndarray,
    x_candidate: np.ndarray,
    x0: np.ndarray,
    y_delta: np.ndarray,
    Fx: np.ndarray,
    Fp: np.ndarray,
    reg_diag: np.ndarray,
    t_frames: np.ndarray,
    s_blood: np.ndarray,
    C_WB_data: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
    include_blood: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    """Backtrack on the current iteration's regularized objective.

    Phi_i(x) = (||F(x)-y_delta||^2 + sum(reg_diag*(x-x0)^2))/2.
    The regularization weights stay fixed during this search. Use the
    feasible segment toward the projected IRGNM endpoint; if projection
    destroys descent, use a diagonally scaled projected-gradient direction.
    Accept Phi_i(x_i + alpha*p) <= Phi_i(x_i) + 1e-4*alpha*grad(Phi_i)@p.
    Try alpha=1, 1/2, ..., 2^-30; non-finite trials are rejected. Return
    (accepted iterate, forward value, alpha, number of halvings), or None
    on a stationary point / failed search. A failed search is not convergence.
    """
    r = Fx - y_delta
    offset = x_i - x0
    merit = float(0.5 * (r @ r + offset @ (reg_diag * offset)))
    gradient = Fp.T @ r + reg_diag * offset
    if active_mask is not None:
        gradient = np.where(np.asarray(active_mask, dtype=bool), gradient, 0.0)
    if not np.isfinite(merit) or not np.all(np.isfinite(gradient)):
        raise ValueError("non-finite Armijo objective or gradient")

    direction = x_candidate - x_i
    slope = float(gradient @ direction)
    if not np.all(np.isfinite(direction)) or not np.isfinite(slope) or slope >= 0.0:
        # Diagonal Gauss-Newton scaling respects the different parameter
        # scales. For positive scaling, box projection guarantees descent
        # unless this projected-gradient step is zero (up to rounding).
        diagonal = np.sum(Fp * Fp, axis=0) + reg_diag
        diagonal = np.where(diagonal > 0.0, diagonal, 1.0)
        direction = project(x_i - gradient / diagonal) - x_i
        slope = float(gradient @ direction)
    if not np.all(np.isfinite(direction)) or not np.isfinite(slope) or slope >= 0.0:
        return None

    alpha = 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for backtracks in range(31):
            x_trial = project(x_i + alpha * direction)
            if np.array_equal(x_trial, x_i):
                break  # further halvings cannot produce a representable step
            if np.all(np.isfinite(x_trial)):
                Fx_trial = forward_operator(
                    x_trial, t_frames, s_blood, C_WB_data,
                    include_blood=include_blood,
                )
                r_trial = Fx_trial - y_delta
                offset_trial = x_trial - x0
                merit_trial = float(0.5 * (
                    r_trial @ r_trial + offset_trial @ (reg_diag * offset_trial)
                ))
                # Strict decrease also prevents accepting a rounded no-op.
                if (np.isfinite(merit_trial) and merit_trial < merit
                        and merit_trial <= merit + 1e-4 * alpha * slope):
                    return x_trial, Fx_trial, alpha, backtracks
            alpha *= 0.5
    return None


def run_irgnm(
    x0: np.ndarray,
    y_delta: np.ndarray,
    t_frames: np.ndarray,
    s_blood: np.ndarray,
    C_WB_data: np.ndarray,
    schedule: RegularizationSchedule = DEFAULT_SCHEDULE,
    tau: float = DEFAULT_TAU,
    delta_y: float = 0.0,
    max_iter: int = 300,
    solver: str = "qr",
    x_true: np.ndarray | None = None,
    *,
    active_mask: np.ndarray | None = None,
    include_blood: bool = True,
    line_search: bool = False,
) -> dict:
    """Run IRGNM from initial guess `x0` until the discrepancy principle fires
    or `max_iter` is reached.  For noiseless data (`delta_y=0`), the rule
    cannot fire, so the loop runs the full `max_iter` unless it fails or the
    optional line search stalls.

    Parameters
    ----------
    active_mask : optional boolean array of length N_PARAMS (23).
        When provided, only the True parameters are updated at each step.
        Inactive parameters are frozen at their x0 values.  Used for
        M4.2 Setup A (fix the 3 plasma-fraction parameters m_1..3).
    include_blood : when False, drop F^2 from the forward operator and
        Jacobian throughout the run.  Used for M4.2 Setup A and the
        M4.4 identifiability experiment.
    line_search : opt-in Armijo backtracking on the regularized objective
        with the current schedule weights held fixed. Adds accepted
        ``step_sizes``, ``line_search_backtracks`` and
        ``line_search_stalled_at`` to the result. A stall retains the last
        accepted iterate and does not set ``converged_at`` or ``diverged``.
        False preserves the original full-step behavior and result keys.

    If `x_true` is given, the relative error trajectory is recorded.
    """
    x_i = project(x0.copy())
    history: dict[str, list] = {
        "residual_norm": [],
        "rel_error_total": [],
        "rel_error_lambda": [],
        "rel_error_mu": [],
        "rel_error_m": [],
        "rel_error_K": [],
    }
    if line_search:
        history["step_sizes"] = []
        history["line_search_backtracks"] = []
    line_search_stalled_at = None
    x_true_norm = float(np.sqrt(x_true @ x_true)) if x_true is not None else None

    def _record(x_cur, r_norm):
        history["residual_norm"].append(r_norm)
        if x_true is not None:
            diff = x_cur - x_true
            history["rel_error_total"].append(float(np.sqrt(diff @ diff)) / x_true_norm)
            for name, sl in (
                ("lambda", LAMBDA_SLICE),
                ("mu", MU_SLICE),
                ("m", M_SLICE),
                ("K", slice(METABOLIC_START, N_PARAMS)),
            ):
                d = diff[sl]
                t = x_true[sl]
                denom = float(np.sqrt(t @ t))
                history[f"rel_error_{name}"].append(float(np.sqrt(d @ d)) / denom if denom > 0 else float("nan"))

    Fx0 = forward_operator(x_i, t_frames, s_blood, C_WB_data, include_blood=include_blood)
    r0 = y_delta - Fx0
    r_norm = float(np.sqrt(r0 @ r0))
    _record(x_i, r_norm)

    converged_at = None
    diverged = False
    i = 0
    # Divergent trajectories can transiently push an unconstrained parameter
    # (mu has no upper bound in D(F)) into a region where exp() overflows —
    # a real, correctly-detected failure (caught below via the finiteness
    # check and `SingularMatrixError`), not a bug; suppressed here only to
    # keep a Monte Carlo sweep's output readable, per DECISIONS.md D-M3-8.
    with np.errstate(over="ignore", invalid="ignore"):
        while i < max_iter:
            if r_norm <= tau * delta_y and delta_y > 0.0:
                converged_at = i
                break
            reg_diag = schedule.diag(i)
            try:
                x_next, _Fx, _Fp = irgnm_step(
                    x_i, x0, y_delta, t_frames, s_blood, C_WB_data,
                    reg_diag, solver,
                    active_mask=active_mask,
                    include_blood=include_blood,
                )
                if line_search:
                    accepted = _armijo_line_search(
                        x_i, x_next, x0, y_delta, _Fx, _Fp, reg_diag,
                        t_frames, s_blood, C_WB_data,
                        active_mask=active_mask, include_blood=include_blood,
                    )
                    if accepted is None:
                        line_search_stalled_at = i
                        history["failure_reason"] = "Armijo line search stalled: no sufficient decrease"
                        break
                    x_next, Fx_next, step_size, backtracks = accepted
                    history["step_sizes"].append(step_size)
                    history["line_search_backtracks"].append(backtracks)
            except Exception as exc:  # noqa: BLE001 — log as a divergence, don't crash the sweep
                diverged = True
                history["failure_reason"] = str(exc)
                break

            if not np.all(np.isfinite(x_next)):
                diverged = True
                break

            x_i = x_next
            Fx_i = (Fx_next if line_search else forward_operator(
                x_i, t_frames, s_blood, C_WB_data, include_blood=include_blood,
            ))
            r_i = y_delta - Fx_i
            r_norm = float(np.sqrt(r_i @ r_i))
            if not np.isfinite(r_norm):
                diverged = True
                _record(x_i, float("nan"))
                break
            _record(x_i, r_norm)
            i += 1

    return {
        "x_final": x_i,
        "n_iterations": i,
        "converged_at": converged_at,
        "diverged": diverged,
        **({"line_search_stalled_at": line_search_stalled_at} if line_search else {}),
        **history,
    }
