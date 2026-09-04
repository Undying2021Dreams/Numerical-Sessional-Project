"""Ground-truth constants and configuration for the PET pharmacokinetics project.

Every numerical constant that is dictated by the paper (Holler, Morina, Schramm
2024, "Exact parameter identification in PET pharmacokinetic modeling using the
irreversible two tissue compartment model", Phys. Med. Biol. 69 165008) lives here
and nowhere else, so that a reader can check every figure/table against a single
source of truth. See PLAN.md, milestone M0, for the list this file must satisfy.

Nothing in this module performs numerical linear algebra, optimisation, or random
sampling; it only defines data. That keeps it exempt from the Track A / Track B
guard in tests/test_no_library_solvers.py by construction (no banned symbols are
even relevant here), and it is `import`-safe from src/, tests/, and experiments/.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 1. Arterial plasma input C_P(t), polyexponential of degree p = 4.
#    Paper, Section 5.1, item 1 (page 16):
#    C_P(t) = -10.9136 e^{-13.4522 t} + 9.545 e^{-3.2672 t}
#              + 0.7331 e^{-0.1532 t} + 0.6355 e^{-0.0106 t},  t in minutes.
#    This is C_P(t) = sum_j lambda_j * exp(mu_j * t) per Definition 2.
# ---------------------------------------------------------------------------
ARTERIAL_DEGREE_P: int = 4

ARTERIAL_LAMBDA: tuple[float, ...] = (-10.9136, 9.545, 0.7331, 0.6355)
ARTERIAL_MU: tuple[float, ...] = (-13.4522, -3.2672, -0.1532, -0.0106)

# ---------------------------------------------------------------------------
# 2. Parent plasma fraction f(t), biexponential model (Remark 18):
#    f(t) = A e^{xi1 t} + (1 - A) e^{xi2 t}.
#    Paper's synthetic ground truth (Section 5.1, item 2, page 16):
#    f(t) = 0.2 e^{-0.2 t} + 0.8 e^{-0.005 t}.
#    So (A, xi1, xi2) = (0.2, -0.2, -0.005); these are the "m_1, m_2, m_3"
#    entries of the 23-parameter unknown vector x (see UNKNOWN_LAYOUT below).
#    C_WB(t) = C_P(t) / f(t).
# ---------------------------------------------------------------------------
F_A: float = 0.2
F_XI1: float = -0.2
F_XI2: float = -0.005

# ---------------------------------------------------------------------------
# 3. Fractional blood volume and the PET measurement equation:
#    C_PET(t) = (1 - V_B) * C_T(t) + V_B * C_WB(t).
#    Paper, Section 5.1, item 3 (page 16): V_B = 0.05.
# ---------------------------------------------------------------------------
V_B: float = 0.05

# ---------------------------------------------------------------------------
# 4. Regional kinetic parameters (K1, k2, k3), control group.
#    Paper, Table 2, "Ground truth parameters" block (page 24), consistent
#    with the values reported from Jagust et al 1991 used throughout the
#    paper's numerical experiments.
# ---------------------------------------------------------------------------
REGION_NAMES: tuple[str, ...] = ("frontal", "temporal", "occipital", "white_matter")

# name -> (K1, k2, k3)
REGION_KINETICS: dict[str, tuple[float, float, float]] = {
    "frontal": (0.1570, 0.1740, 0.1180),
    "temporal": (0.1610, 0.1790, 0.0960),
    "occipital": (0.1770, 0.1590, 0.0880),
    "white_matter": (0.1000, 0.1610, 0.0470),
}

N_REGIONS: int = len(REGION_NAMES)

# ---------------------------------------------------------------------------
# 5. Frame schedule: 25 dynamic frames.
#    Paper, Section 5.1, item 4 (page 16):
#    4x5s, 4x10s, 4x30s, 2x60s, 3x150s, 6x300s, 2x600s, post injection.
#    Model equations use t in minutes, so we store both seconds and minutes.
# ---------------------------------------------------------------------------
FRAME_SCHEDULE_SECONDS: tuple[tuple[int, float], ...] = (
    (4, 5.0),
    (4, 10.0),
    (4, 30.0),
    (2, 60.0),
    (3, 150.0),
    (6, 300.0),
    (2, 600.0),
)

N_FRAMES: int = sum(count for count, _ in FRAME_SCHEDULE_SECONDS)  # == 25


def _frame_durations_seconds() -> np.ndarray:
    durations = []
    for count, duration in FRAME_SCHEDULE_SECONDS:
        durations.extend([duration] * count)
    return np.asarray(durations, dtype=np.float64)


def frame_edges_seconds() -> np.ndarray:
    """Return the 26 frame boundary times in seconds, edges[0] == 0."""
    durations = _frame_durations_seconds()
    edges = np.concatenate(([0.0], np.cumsum(durations)))
    return edges


def frame_edges_minutes() -> np.ndarray:
    return frame_edges_seconds() / 60.0


def frame_midtimes_minutes() -> np.ndarray:
    """Midpoint of each of the 25 frames, in minutes (t argument of C_P etc.)."""
    edges = frame_edges_minutes()
    return 0.5 * (edges[:-1] + edges[1:])


def frame_widths_minutes() -> np.ndarray:
    edges = frame_edges_minutes()
    return edges[1:] - edges[:-1]


# ---------------------------------------------------------------------------
# 5b. Blood-sample ("q") time points for the F^2 block of the forward
#     operator (paper eq. 18-20: F^2 compares q measurements of C_WB*f
#     against the model's C_P). q = 4, per M3's Jacobian shape spec
#     (n*T + q = 4*25 + 4 = 104). Chosen as 4 of the existing frame
#     midtimes (physically: blood draws timed with scan frames), spread
#     across the dynamic range: one during the fast initial decay, one just
#     after it, one mid-scan, one at the last frame. See DECISIONS.md D-M3-1.
# ---------------------------------------------------------------------------
BLOOD_SAMPLE_FRAME_INDICES: tuple[int, ...] = (3, 10, 17, 24)
Q_BLOOD_SAMPLES: int = len(BLOOD_SAMPLE_FRAME_INDICES)


def blood_sample_times_minutes() -> np.ndarray:
    mids = frame_midtimes_minutes()
    return mids[np.array(BLOOD_SAMPLE_FRAME_INDICES)]


# ---------------------------------------------------------------------------
# 6. Unknown parameter vector layout, 23 parameters total:
#    (lambda_1..4, mu_1..4, m_1..3, K1^1,k2^1,k3^1, ..., K1^4,k2^4,k3^4)
#    m_1..3 are (A, xi1, xi2) of the biexponential f. Order of regions follows
#    REGION_NAMES. This layout is the contract used by the forward model and
#    Jacobian (built in M2/M3); defined here so every later milestone agrees
#    on indexing.
# ---------------------------------------------------------------------------
UNKNOWN_LAYOUT: tuple[str, ...] = (
    "lambda_1", "lambda_2", "lambda_3", "lambda_4",
    "mu_1", "mu_2", "mu_3", "mu_4",
    "m_1", "m_2", "m_3",
    *[f"{p}^{r}" for r in REGION_NAMES for p in ("K1", "k2", "k3")],
)

N_PARAMS: int = len(UNKNOWN_LAYOUT)  # == 23
assert N_PARAMS == 4 + 4 + 3 + 3 * N_REGIONS == 23

# Index slices into the 23-vector, per block (M3: Jacobian, projection, regularisation).
LAMBDA_SLICE = slice(0, 4)
MU_SLICE = slice(4, 8)
M_SLICE = slice(8, 11)
METABOLIC_START = 11  # first index of the 12 K1/k2/k3 entries


def region_slice(region_index: int) -> slice:
    """Slice into the 23-vector for region `region_index`'s (K1, k2, k3)."""
    start = METABOLIC_START + 3 * region_index
    return slice(start, start + 3)


# D(F) domain, paper eq. (18): lambda, mu unconstrained; m in [0,inf) x
# (-inf,0]^2; each of the 12 metabolic parameters in [eps, inf). eps = 1e-3
# per the reviewer's M3 spec — see DECISIONS.md D-M3-2 for why this value.
PROJECTION_EPS: float = 1e-3


def ground_truth_vector() -> np.ndarray:
    """Assemble x_dagger in R^23 per UNKNOWN_LAYOUT, from the constants above."""
    x = list(ARTERIAL_LAMBDA) + list(ARTERIAL_MU) + [F_A, F_XI1, F_XI2]
    for name in REGION_NAMES:
        x.extend(REGION_KINETICS[name])
    x = np.asarray(x, dtype=np.float64)
    assert x.shape == (N_PARAMS,)
    return x


# ---------------------------------------------------------------------------
# 7. Config hash, for provenance stamping of everything written to results/.
# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def config_snapshot() -> dict:
    """A plain-data dict of every constant in this module, for hashing/dumping."""
    return {
        "arterial_degree_p": ARTERIAL_DEGREE_P,
        "arterial_lambda": ARTERIAL_LAMBDA,
        "arterial_mu": ARTERIAL_MU,
        "f_A": F_A,
        "f_xi1": F_XI1,
        "f_xi2": F_XI2,
        "V_B": V_B,
        "region_names": REGION_NAMES,
        "region_kinetics": REGION_KINETICS,
        "frame_schedule_seconds": FRAME_SCHEDULE_SECONDS,
        "unknown_layout": UNKNOWN_LAYOUT,
    }


def config_hash() -> str:
    """Short, stable hash of the ground-truth config for artifact provenance.

    CLAUDE.md section 2: "Every artifact written to results/ records the seed
    and config hash that produced it." Deterministic across processes because
    it hashes a canonical (sorted-keys) JSON encoding, not Python repr/id.
    """
    canonical = json.dumps(config_snapshot(), sort_keys=True, default=_json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RunProvenance:
    """Stamp attached to every artifact written under results/."""

    seed: int
    config_hash: str = field(default_factory=config_hash)
    milestone: str = ""
    description: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


if __name__ == "__main__":
    print("config_hash:", config_hash())
    print("N_PARAMS:", N_PARAMS)
    print("N_FRAMES:", N_FRAMES)
    print("ground_truth_vector:", ground_truth_vector())
    print("frame_midtimes_minutes[:5]:", frame_midtimes_minutes()[:5])
