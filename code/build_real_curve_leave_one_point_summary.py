#!/usr/bin/env python3
"""Rebuild the sparse Park/FEC leave-one-point predictive summary.

The digitized Park/FEC curve contains four HT and four HTO points. The
manuscript uses it only as a weak time-scale constraint. This script aggregates
the leave-one-point detail table and bootstrap predictions into the summary
numbers used in the SI, so the external-curve claim is tied to point-level
evidence instead of to a hand-kept JSON file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real_curve_leave_one_point_predictive"
DETAIL = OUT / "real_curve_leave_one_point_detail.csv"
BOOT = OUT / "real_curve_leave_one_point_bootstrap_predictions.csv"
CHECKS = OUT / "real_curve_leave_one_point_checks.csv"
SUMMARY = OUT / "real_curve_leave_one_point_summary.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def truth(row: dict[str, str], key: str) -> bool:
    return row[key].strip().lower() == "true"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    detail = rows(DETAIL)
    boot = rows(BOOT)
    if not detail or not boot:
        raise ValueError("leave-one-point detail and bootstrap tables must be non-empty")

    species = sorted({row["series"] for row in detail})
    n_holdout = len(detail)
    n_boot_by_point: dict[tuple[str, str], int] = {}
    for row in boot:
        key = (row["series"], row["holdout_time_min"])
        n_boot_by_point[key] = n_boot_by_point.get(key, 0) + 1
    n_boot_values = sorted(set(n_boot_by_point.values()))
    if len(n_boot_values) != 1:
        raise ValueError(f"inconsistent bootstrap counts: {n_boot_values}")

    point_errors = [f(row, "point_abs_error_norm") for row in detail]
    ht_errors = [f(row, "point_abs_error_norm") for row in detail if row["series"] == "HT"]
    hto_errors = [f(row, "point_abs_error_norm") for row in detail if row["series"] == "HTO"]
    bootstrap_covered = [truth(row, "covered_by_bootstrap_interval") for row in detail]
    expanded_covered = [
        truth(row, "covered_by_digitization_expanded_interval") for row in detail
    ]
    residual_pass = [truth(row, "within_25pct_residual_gate") for row in detail]
    training_counts = sorted({int(row["n_training_points"]) for row in detail})

    summary = {
        "n_species": len(species),
        "n_holdout_points": n_holdout,
        "n_bootstrap_per_holdout": n_boot_values[0],
        "max_point_abs_error_norm": max(point_errors),
        "median_point_abs_error_norm": median(point_errors),
        "residual_gate_pass_rate": sum(residual_pass) / n_holdout,
        "expanded_interval_coverage": sum(expanded_covered) / n_holdout,
        "bootstrap_interval_coverage": sum(bootstrap_covered) / n_holdout,
        "ht_max_abs_error_norm": max(ht_errors),
        "hto_max_abs_error_norm": max(hto_errors),
    }
    checks = [
        {
            "check_id": "all_holdouts_use_three_training_points",
            "status": "PASS" if training_counts == [3] else "FAIL",
            "detail": "anchor plus two non-anchor points fit each held-out prediction",
        },
        {
            "check_id": "predictive_boundary_classified",
            "status": "PASS"
            if summary["max_point_abs_error_norm"] > 0.25
            and summary["residual_gate_pass_rate"] < 1.0
            else "FAIL",
            "detail": (
                f"residual_pass={summary['residual_gate_pass_rate']:.3f}; "
                f"max_abs_error={summary['max_point_abs_error_norm']:.3f}"
            ),
        },
        {
            "check_id": "expanded_predictive_coverage_boundary_classified",
            "status": "PASS" if summary["expanded_interval_coverage"] < 0.5 else "FAIL",
            "detail": f"coverage={summary['expanded_interval_coverage']:.3f}",
        },
        {
            "check_id": "bootstrap_only_interval_boundary_visible",
            "status": "PASS" if summary["bootstrap_interval_coverage"] == 0.0 else "FAIL",
            "detail": f"bootstrap_only_coverage={summary['bootstrap_interval_coverage']:.3f}",
        },
        {
            "check_id": "late_ht_tail_blocks_high_weight_training",
            "status": "PASS" if summary["ht_max_abs_error_norm"] > 0.45 else "FAIL",
            "detail": f"HT max_abs_error={summary['ht_max_abs_error_norm']:.3f}",
        },
        {
            "check_id": "not_two_timescale_calibration",
            "status": "PASS",
            "detail": "single effective time-scale predictive check only; open/closed calibration remains blocked",
        },
    ]

    with CHECKS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    summary["n_checks"] = len(checks)
    summary["n_pass_checks"] = sum(row["status"] == "PASS" for row in checks)
    summary["claim_boundary"] = (
        "Sparse Park/FEC leave-one-point predictive check supports a weak real "
        "time-scale prior; it does not identify separate open/closed material constants."
    )
    SUMMARY.write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )

    n_fail = sum(row["status"] != "PASS" for row in checks)
    print(f"wrote {CHECKS.relative_to(ROOT)}")
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    print(
        "real-curve leave-one: "
        f"median={summary['median_point_abs_error_norm']:.3f}, "
        f"max={summary['max_point_abs_error_norm']:.3f}, "
        f"expanded coverage={summary['expanded_interval_coverage']:.3f}"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
