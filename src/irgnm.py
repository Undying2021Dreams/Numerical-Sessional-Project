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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One IRGNM step (eq. 26). Returns (x_{i+1} (projected), F(x_i), F'[x_i])."""
    Fx = forward_operator(x_i, t_frames, s_blood, C_WB_data)
    r = y_delta - Fx
    Fp = analytic_jacobian(x_i, t_frames, s_blood, C_WB_data)

    if solver == "lu":
        M = Fp.T @ Fp + np.diag(reg_diag)
        rhs = Fp.T @ r + reg_diag * (x0 - x_i)
        delta = lu_solve(M, rhs)
    elif solver == "qr":
        sq = np.sqrt(reg_diag)
        A_stack = np.vstack([Fp, np.diag(sq)])
        b_stack = np.concatenate([r, sq * (x0 - x_i)])
        delta, _resid = qr_lstsq(A_stack, b_stack)
    else:
        raise ValueError(f"unknown solver {solver!r}, expected 'lu' or 'qr'")

    x_next = project(x_i + delta)
    return x_next, Fx, Fp


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
) -> dict:
    """Run IRGNM from initial guess `x0` (also used as the fixed
    regularisation reference point, per eq. 26) until the discrepancy
    principle fires (`||F(x_i)-y_delta|| <= tau*delta_y`) or `max_iter` is
    reached. For noiseless data (`delta_y=0`), the discrepancy rule
    essentially never fires, so the loop runs the full `max_iter` — per
    PLAN.md M3 item 6/7.

    If `x_true` is given, the relative error trajectory `||x_i - x_true|| /
    ||x_true||` (total and per parameter block) is recorded at every
    iteration, for the noiseless-recovery convergence plots.
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

    Fx0 = forward_operator(x_i, t_frames, s_blood, C_WB_data)
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
                x_next, _Fx, _Fp = irgnm_step(x_i, x0, y_delta, t_frames, s_blood, C_WB_data, reg_diag, solver)
            except Exception as exc:  # noqa: BLE001 — log as a divergence, don't crash the sweep
                diverged = True
                history["failure_reason"] = str(exc)
                break

            if not np.all(np.isfinite(x_next)):
                diverged = True
                break

            x_i = x_next
            Fx_i = forward_operator(x_i, t_frames, s_blood, C_WB_data)
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
        **history,
    }
