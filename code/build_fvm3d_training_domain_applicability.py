#!/usr/bin/env python3
"""Rebuild the 3D training-domain applicability summary.

This script aggregates completed source/field family distance tests. It does
not train a new model. The input detail table contains one row per validation
or out-of-family source/field cell; the summary reports which predictions
remain in the calibrated 3D training domain before source estimates are passed
to module-scale calculations.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "fvm3d_training_domain_applicability_gate"
DETAIL = RESULT_DIR / "fvm3d_training_domain_applicability_gate_detail.csv"
CHECKS = RESULT_DIR / "fvm3d_training_domain_applicability_gate_checks.csv"
SUMMARY = RESULT_DIR / "fvm3d_training_domain_applicability_gate_summary.json"


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value == "" else float(value)


def i(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def finite(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def by_severity(rows: list[dict[str, str]], severity: str) -> list[dict[str, str]]:
    chunk = [row for row in rows if row["severity_id"] == severity]
    if not chunk:
        raise ValueError(f"missing severity_id={severity}")
    return chunk


def weighted_capture(rows: list[dict[str, str]]) -> float:
    n_fail = sum(i(row, "n_fail") for row in rows)
    n_fail_accepted = sum(i(row, "n_fail_accepted") for row in rows)
    if n_fail == 0:
        return 0.0
    return (n_fail - n_fail_accepted) / n_fail


def max_accepted(rows: list[dict[str, str]], key: str) -> float:
    values = finite([f(row, key) for row in rows])
    if not values:
        return math.nan
    return max(values)


def main() -> int:
    if not DETAIL.exists():
        raise FileNotFoundError(f"missing detail table: {DETAIL}")

    with DETAIL.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty detail table: {DETAIL}")

    validation = [row for row in rows if row["group_id"] == "in_envelope_validation"]
    ood = [row for row in rows if row["group_id"] == "ood_source_family"]
    if not validation or not ood:
        raise ValueError("detail table must contain validation and out-of-family rows")

    nominal = by_severity(rows, "nominal")
    strong = by_severity(rows, "strong")
    extreme = by_severity(rows, "extreme")
    pathological = by_severity(rows, "pathological")

    summary = {
        "n_train": 1600,
        "n_validation": sum(i(row, "n_samples") for row in validation),
        "n_ood_cells": len(ood),
        "n_test_per_cell": i(ood[0], "n_samples"),
        "k_neighbours": 12,
        "in_domain_quantile": 0.99,
        "distance_threshold": 1.4630225884957184,
        "validation_accept_rate": mean([f(row, "accept_rate") for row in validation]),
        "validation_closed_p95_pct": max(f(row, "closed_error_p95_pct") for row in validation),
        "validation_zone_l1_p95_pct": max(f(row, "zone_l1_p95_pct") for row in validation),
        "nominal_mean_accept_rate": mean([f(row, "accept_rate") for row in nominal]),
        "nominal_worst_accepted_closed_p95_pct": max_accepted(
            nominal, "accepted_closed_error_p95_pct"
        ),
        "nominal_worst_accepted_zone_l1_p95_pct": max_accepted(
            nominal, "accepted_zone_l1_p95_pct"
        ),
        "strong_mean_accept_rate": mean([f(row, "accept_rate") for row in strong]),
        "strong_failed_sample_capture_rate": weighted_capture(strong),
        "extreme_mean_accept_rate": mean([f(row, "accept_rate") for row in extreme]),
        "extreme_failed_sample_capture_rate": weighted_capture(extreme),
        "pathological_mean_accept_rate": mean([f(row, "accept_rate") for row in pathological]),
        "pathological_failed_sample_capture_rate": weighted_capture(pathological),
        "extreme_worst_accepted_closed_p95_pct": max_accepted(
            extreme, "accepted_closed_error_p95_pct"
        ),
        "extreme_worst_accepted_zone_l1_p95_pct": max_accepted(
            extreme, "accepted_zone_l1_p95_pct"
        ),
        "pathological_worst_accepted_closed_p95_pct": max_accepted(
            pathological, "accepted_closed_error_p95_pct"
        ),
        "pathological_worst_accepted_zone_l1_p95_pct": max_accepted(
            pathological, "accepted_zone_l1_p95_pct"
        ),
        "claim_boundary": (
            "Synthetic chemistry-resolved 3D FVM training-domain applicability "
            "test; blocks out-of-family predictions before module propagation "
            "but is not reactor-specific validation."
        ),
    }

    checks = [
        {
            "check_id": "validation_mostly_accepted",
            "status": "PASS" if summary["validation_accept_rate"] > 0.98 else "FAIL",
            "detail": f"validation accept={summary['validation_accept_rate']:.3f}",
        },
        {
            "check_id": "nominal_accepted_predictions_remain_inside_5pct",
            "status": "PASS"
            if (
                summary["nominal_worst_accepted_closed_p95_pct"] < 5.0
                and summary["nominal_worst_accepted_zone_l1_p95_pct"] < 5.0
            )
            else "FAIL",
            "detail": (
                f"closed={summary['nominal_worst_accepted_closed_p95_pct']:.2f}%; "
                f"zone={summary['nominal_worst_accepted_zone_l1_p95_pct']:.2f}%"
            ),
        },
        {
            "check_id": "extreme_failures_are_captured",
            "status": "PASS"
            if summary["extreme_failed_sample_capture_rate"] > 0.95
            else "FAIL",
            "detail": f"capture={summary['extreme_failed_sample_capture_rate']:.3f}",
        },
        {
            "check_id": "most_severe_failures_are_captured",
            "status": "PASS"
            if summary["pathological_failed_sample_capture_rate"] > 0.95
            else "FAIL",
            "detail": f"capture={summary['pathological_failed_sample_capture_rate']:.3f}",
        },
        {
            "check_id": "acceptance_tightens_with_severity",
            "status": "PASS"
            if (
                summary["nominal_mean_accept_rate"]
                > summary["strong_mean_accept_rate"]
                > summary["extreme_mean_accept_rate"]
                > 0.0
                and summary["pathological_mean_accept_rate"]
                < summary["strong_mean_accept_rate"]
            )
            else "FAIL",
            "detail": (
                f"nom={summary['nominal_mean_accept_rate']:.3f}; "
                f"strong={summary['strong_mean_accept_rate']:.3f}; "
                f"extreme={summary['extreme_mean_accept_rate']:.3f}; "
                f"severe={summary['pathological_mean_accept_rate']:.3f}"
            ),
        },
    ]

    with CHECKS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    SUMMARY.write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )

    n_fail = sum(check["status"] != "PASS" for check in checks)
    print(f"wrote {CHECKS.relative_to(ROOT)}")
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    print(
        "training-domain applicability: "
        f"validation_accept={100.0 * summary['validation_accept_rate']:.2f}%, "
        f"severe_capture={100.0 * summary['pathological_failed_sample_capture_rate']:.2f}%, "
        f"accepted_closed={summary['pathological_worst_accepted_closed_p95_pct']:.2f}%"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
