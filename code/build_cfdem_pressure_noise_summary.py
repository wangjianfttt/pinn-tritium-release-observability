#!/usr/bin/env python3
"""Rebuild the CFD-DEM pressure-noise robustness summary.

The input table contains pressure-anchor source inversions for 9 CFD-DEM
conditions, 400 random seeds and five pressure-noise levels. The output JSON
summarizes the noise level at which pressure/RTD anchoring remains inside the
5% source-error target used by the manuscript.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cfdem_pressure_anchor_noise_robustness"
TABLE = OUT / "cfdem_pressure_anchor_noise_robustness_summary.csv"
DETAIL = OUT / "cfdem_pressure_anchor_noise_robustness_rows.csv"
CHECKS = OUT / "cfdem_pressure_anchor_noise_robustness_checks.csv"
SUMMARY = OUT / "cfdem_pressure_anchor_noise_robustness_summary.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def row_for(rows: list[dict[str, str]], noise_pct: float) -> dict[str, str]:
    for row in rows:
        if abs(f(row, "pressure_noise_pct") - noise_pct) < 1e-12:
            return row
    raise KeyError(f"missing pressure noise {noise_pct}")


def main() -> int:
    rows = read_rows(TABLE)
    detail_rows = read_rows(DETAIL)
    pass_rows = [row for row in rows if f(row, "p95_source_error_pct") < 5.0]
    fail_rows = [row for row in rows if f(row, "p95_source_error_pct") >= 5.0]
    two = row_for(rows, 2.0)
    five = row_for(rows, 5.0)

    summary = {
        "n_conditions": len({row["condition"] for row in detail_rows}),
        "n_seeds": len({row["seed"] for row in detail_rows}),
        "n_trials": len(detail_rows),
        "noise_levels_pct": [f(row, "pressure_noise_pct") for row in rows],
        "max_pressure_noise_with_p95_source_below_5pct": max(f(row, "pressure_noise_pct") for row in pass_rows),
        "first_pressure_noise_failing_p95_source_gate_pct": min(f(row, "pressure_noise_pct") for row in fail_rows),
        "two_pct_pressure_noise_p95_source_error_pct": f(two, "p95_source_error_pct"),
        "two_pct_pressure_noise_pass_rate_pct": 100.0 * f(two, "pass_rate_5pct"),
        "two_pct_pressure_noise_p95_velocity_error_pct": f(two, "p95_velocity_error_pct"),
        "five_pct_pressure_noise_p95_source_error_pct": f(five, "p95_source_error_pct"),
        "five_pct_pressure_noise_pass_rate_pct": 100.0 * f(five, "pass_rate_5pct"),
        "five_pct_pressure_noise_rtd_flag_rate_pct": f(five, "rtd_gate_flag_rate_pct"),
        "claim": (
            "Pressure/RTD anchoring remains source-useful under moderate pressure "
            "noise in the archived CFD-DEM matrix, and the 5% pressure-noise case "
            "marks a measurement-quality boundary."
        ),
        "boundary": (
            "Noise is applied to archived pressure calibration and test probes; "
            "this is not a new CFD-DEM mesh or packing-seed validation."
        ),
    }

    checks = [
        {
            "check_id": "two_pct_pressure_noise_passes_p95_source_target",
            "status": "PASS" if summary["two_pct_pressure_noise_p95_source_error_pct"] < 5.0 else "FAIL",
            "detail": f"{summary['two_pct_pressure_noise_p95_source_error_pct']:.2f}% < 5%",
        },
        {
            "check_id": "two_pct_pressure_noise_has_full_pass_rate",
            "status": "PASS" if summary["two_pct_pressure_noise_pass_rate_pct"] == 100.0 else "FAIL",
            "detail": f"pass={summary['two_pct_pressure_noise_pass_rate_pct']:.1f}%",
        },
        {
            "check_id": "five_pct_pressure_noise_exposes_boundary",
            "status": "PASS" if summary["five_pct_pressure_noise_p95_source_error_pct"] > 5.0 else "FAIL",
            "detail": f"{summary['five_pct_pressure_noise_p95_source_error_pct']:.2f}% > 5%",
        },
        {
            "check_id": "all_pressure_noise_trials_counted",
            "status": "PASS" if summary["n_trials"] == 18000 else "FAIL",
            "detail": f"trials={summary['n_trials']}",
        },
    ]
    summary["n_checks"] = len(checks)
    summary["n_pass_checks"] = sum(item["status"] == "PASS" for item in checks)
    summary["all_checks_pass"] = summary["n_checks"] == summary["n_pass_checks"]

    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with CHECKS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    print(
        "CFD-DEM pressure noise: "
        f"2% p95={summary['two_pct_pressure_noise_p95_source_error_pct']:.2f}%, "
        f"5% p95={summary['five_pct_pressure_noise_p95_source_error_pct']:.2f}%"
    )
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
