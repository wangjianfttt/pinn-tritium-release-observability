#!/usr/bin/env python3
"""Rebuild the explicit 3D source-head recovery summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fvm3d_source_operator_head_recovery"
ROWS = OUT / "fvm3d_source_operator_head_recovery_summary.csv"
CHECKS = OUT / "fvm3d_source_operator_head_recovery_checks.csv"
SUMMARY = OUT / "fvm3d_source_operator_head_recovery_summary.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def write_checks(rows: list[dict[str, object]]) -> None:
    with CHECKS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_rows(ROWS)
    holdout_cases = sorted({row["holdout_case"] for row in rows})
    train_rows = sorted({int(row["n_train_rows"]) for row in rows})
    eval_rows = sorted({int(row["n_eval_rows"]) for row in rows})

    source_ratios = [f(row, "source_head_total_rms_ratio") for row in rows]
    multiplier_errors = [f(row, "source_head_total_multiplier_error_pct") for row in rows]
    source_reductions = [f(row, "head_vs_pde_source_rms_reduction") for row in rows]
    multiplier_reductions = [
        f(row, "head_vs_pde_multiplier_error_reduction") for row in rows
    ]

    checks: list[dict[str, object]] = [
        {
            "check_id": "all_three_cases_rotated",
            "status": "PASS" if len(holdout_cases) == 3 else "FAIL",
            "detail": f"cases={len(holdout_cases)}",
        },
        {
            "check_id": "source_head_better_than_pde_derived_all_splits",
            "status": "PASS" if min(source_reductions) > 1.0 else "FAIL",
            "detail": f"min_reduction={min(source_reductions):.2f}",
        },
        {
            "check_id": "source_head_multiplier_error_bounded",
            "status": "PASS" if max(multiplier_errors) < 15.0 else "FAIL",
            "detail": f"max_error={max(multiplier_errors):.2f}%",
        },
        {
            "check_id": "source_head_rms_ratio_bounded",
            "status": "PASS" if max(source_ratios) < 0.40 else "FAIL",
            "detail": f"max_rms_ratio={max(source_ratios):.3f}",
        },
        {
            "check_id": "training_split_has_no_holdout_case",
            "status": "PASS"
            if all(row["holdout_case"] not in row["train_cases"] for row in rows)
            else "FAIL",
            "detail": "leave-one-case source-head training",
        },
    ]
    write_checks(checks)

    summary = {
        "n_holdout_cases": len(holdout_cases),
        "n_train_rows_per_split": train_rows[0] if len(train_rows) == 1 else train_rows,
        "n_eval_rows_per_split": eval_rows[0] if len(eval_rows) == 1 else eval_rows,
        "n_source_basis_terms": 8,
        "median_head_source_rms_ratio": median(source_ratios),
        "max_head_source_rms_ratio": max(source_ratios),
        "median_head_multiplier_error_pct": median(multiplier_errors),
        "max_head_multiplier_error_pct": max(multiplier_errors),
        "min_head_vs_pde_source_rms_reduction": min(source_reductions),
        "median_head_vs_pde_source_rms_reduction": median(source_reductions),
        "min_head_vs_pde_multiplier_error_reduction": min(multiplier_reductions),
        "claim_boundary": (
            "Synthetic 3D FVM source-operator head trained on held-in 3D cases "
            "and evaluated on held-out 3D cases; a constructive training-data "
            "rule for simulation-informed PINNs, not experimental validation."
        ),
        "n_checks": len(checks),
        "n_pass_checks": sum(row["status"] == "PASS" for row in checks),
    }

    output = {
        "summary": summary,
        "rows": [
            {
                key: (
                    int(value)
                    if key in {"n_train_rows", "n_eval_rows"}
                    else float(value)
                    if key
                    not in {
                        "holdout_case",
                        "train_cases",
                    }
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        ],
        "checks": checks,
    }
    SUMMARY.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY}")
    print(
        "explicit source-head minimum RMS improvement "
        f"{summary['min_head_vs_pde_source_rms_reduction']:.2f}x"
    )


if __name__ == "__main__":
    main()
