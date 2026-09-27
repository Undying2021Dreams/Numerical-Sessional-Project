"""M5: one-command regeneration of every figure and table under `results/`.

Runs every experiment script in dependency order, in a fresh subprocess per
script (DECISIONS.md D-M5-1: subprocess isolation avoids one script's
module-level state — matplotlib figures, cached constants — leaking into the
next), with the same interpreter that ran this script. All seeds are fixed
inside each script (root seeds are printed as they run), so this is
deterministic on a clean checkout: `pip install -r requirements.txt` then
`python3 experiments/run_all.py`.

The M5 scripts (`m5_timing_complexity.py`, `m5_track_ab_comparison.py`,
`m5_trend_checklist.py`) read JSON already written by the M1-M4 scripts, so
those must run first; order among the M1-M4 scripts themselves does not
matter (verified: none of them import another experiment script's output),
but they are kept in milestone order for readability of the log.

Default behaviour is fail-fast: if a script exits non-zero, later scripts
that depend on its output would fail anyway with a confusing error, so the
run stops immediately and prints the failing script's stderr. Pass
`--continue` to run everything regardless and get a full pass/fail report.

Usage:
    python3 experiments/run_all.py
    python3 experiments/run_all.py --continue
    python3 experiments/run_all.py --skip m1_rng_benchmark m4_grid   # dev convenience
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# (label, script filename) in milestone order.
SCRIPTS = [
    ("M1 linalg", "m1_linalg_benchmark.py"),
    ("M1 quadrature", "m1_quadrature_benchmark.py"),
    ("M1 rng", "m1_rng_benchmark.py"),
    ("M2 forward model", "m2_forward_model.py"),
    ("M3 jacobian and solver", "m3_jacobian_and_solver.py"),
    ("M3 IRGNM recovery", "m3_irgnm_recovery.py"),
    ("M4.1 noise calibration", "m4_noise_calibration.py"),
    ("M4.3 grid (long pole, ~250-300s)", "m4_grid.py"),
    ("M4.3 grid under the paper's settings (~14 min)", "m4_grid_paper_settings.py"),
    ("M4.3 Figure 7 analogue", "m4_plot_figure7.py"),
    ("M4.4 identifiability (noiseless)", "m4_identifiability.py"),
    ("M4.4 identifiability (noisy, ~270s)", "m4_identifiability_noisy.py"),
    ("M4.5 consistency & regularisation", "m4_consistency_regularization.py"),
    ("M5 timing & complexity", "m5_timing_complexity.py"),
    ("M5 Track A vs Track B", "m5_track_ab_comparison.py"),
    ("M5 trend checklist", "m5_trend_checklist.py"),
]


def run_one(label: str, filename: str) -> dict:
    script_path = EXPERIMENTS_DIR / filename
    print(f"\n{'=' * 70}\n{label}  ({filename})\n{'=' * 70}")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    print(proc.stdout[-4000:])  # tail only: some scripts print large sweeps
    ok = proc.returncode == 0
    if not ok:
        print(f"--- FAILED (exit {proc.returncode}), stderr: ---\n{proc.stderr}")
    print(f"[{label}] {'OK' if ok else 'FAILED'} in {dt:.1f}s")
    return {"label": label, "script": filename, "ok": ok, "time_s": dt,
             "returncode": proc.returncode,
             "stderr_tail": proc.stderr[-2000:] if not ok else ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue", dest="keep_going", action="store_true",
                         help="run every script even after a failure")
    parser.add_argument("--skip", nargs="*", default=[],
                         help="substrings of script filenames to skip (dev convenience)")
    args = parser.parse_args()

    records = []
    t_start = time.perf_counter()
    for label, filename in SCRIPTS:
        if any(s in filename for s in args.skip):
            print(f"\n[skipping] {label} ({filename})")
            records.append({"label": label, "script": filename, "ok": None,
                             "time_s": 0.0, "returncode": None, "skipped": True})
            continue
        record = run_one(label, filename)
        records.append(record)
        if not record["ok"] and not args.keep_going:
            print(f"\nStopping after failure in {label} (pass --continue to run the rest anyway).")
            break
    total_time = time.perf_counter() - t_start

    n_ok = sum(1 for r in records if r["ok"])
    n_failed = sum(1 for r in records if r["ok"] is False)
    n_skipped = sum(1 for r in records if r.get("skipped"))

    print(f"\n{'=' * 70}\nrun_all.py summary: {n_ok} ok, {n_failed} failed, {n_skipped} skipped, "
          f"total {total_time:.1f}s\n{'=' * 70}")
    for r in records:
        status = "SKIPPED" if r.get("skipped") else ("OK" if r["ok"] else "FAILED")
        print(f"  [{status:7s}] {r['label']:40s} {r['time_s']:8.1f}s")

    save_json("m5", "run_all_manifest", {
        "total_time_s": total_time,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "records": records,
    })

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
