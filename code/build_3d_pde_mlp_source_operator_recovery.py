#!/usr/bin/env python3
"""Rebuild the 3D PDE-MLP source-operator recovery summary.

The heavy field fits are stored as CSV artifacts. This script recomputes the
published summary and checks from those artifacts so the manuscript number is
traceable to executable code rather than to a hand-kept JSON file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fvm3d_pde_mlp_source_operator_recovery"
PAIRS = OUT / "fvm3d_pde_mlp_source_operator_recovery_pairs.csv"
ROWS = OUT / "fvm3d_pde_mlp_source_operator_recovery_rows.csv"
CHECKS = OUT / "fvm3d_pde_mlp_source_operator_recovery_checks.csv"
SUMMARY = OUT / "fvm3d_pde_mlp_source_operator_recovery_summary.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def write_checks(rows: list[dict[str, object]]) -> None:
    with CHECKS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    pair_rows = read_rows(PAIRS)
    model_rows = read_rows(ROWS)

    reductions = [as_float(row, "source_rms_reduction_factor") for row in pair_rows]
    pde_errors = [as_float(row, "pde_total_multiplier_error_pct") for row in pair_rows]
    data_errors = [as_float(row, "data_total_multiplier_error_pct") for row in pair_rows]
    field_ratios = [as_float(row, "field_rmse_ratio_pde_to_data") for row in pair_rows]
    pde_gains = [as_float(row, "pde_residual_gain") for row in pair_rows]

    holdout_cases = sorted({row["holdout_case"] for row in pair_rows})
    residual_weights = sorted({float(row["residual_weight"]) for row in model_rows})

    checks: list[dict[str, object]] = [
        {
            "check_id": "all_three_cases_rotated",
            "status": "PASS" if len(holdout_cases) == 3 else "FAIL",
            "detail": f"cases={len(holdout_cases)}",
        },
        {
            "check_id": "source_operator_improved_all_splits",
            "status": "PASS" if min(reductions) > 1.0 else "FAIL",
            "detail": f"min_reduction={min(reductions):.3f}",
        },
        {
            "check_id": "source_operator_median_gain_substantial",
            "status": "PASS" if median(reductions) > 1.5 else "FAIL",
            "detail": f"median_reduction={median(reductions):.3f}",
        },
        {
            "check_id": "field_fit_not_destroyed",
            "status": "PASS" if max(field_ratios) < 1.25 else "FAIL",
            "detail": f"max_field_ratio={max(field_ratios):.3f}",
        },
        {
            "check_id": "pde_residual_gain_positive",
            "status": "PASS" if min(pde_gains) > 1.0 else "FAIL",
            "detail": f"min_pde_gain={min(pde_gains):.3f}",
        },
    ]
    write_checks(checks)

    summary = {
        "n_holdout_cases": len(holdout_cases),
        "n_models": len(model_rows),
        "n_train_data_rows": 220,
        "n_train_residual_rows": 260,
        "n_eval_rows": 520,
        "pde_weight": max(residual_weights),
        "min_source_rms_reduction_factor": min(reductions),
        "median_source_rms_reduction_factor": median(reductions),
        "max_source_rms_reduction_factor": max(reductions),
        "median_pde_total_multiplier_error_pct": median(pde_errors),
        "max_pde_total_multiplier_error_pct": max(pde_errors),
        "median_data_total_multiplier_error_pct": median(data_errors),
        "max_field_rmse_ratio": max(field_ratios),
        "min_pde_residual_gain": min(pde_gains),
        "claim_boundary": (
            "Synthetic chemistry-resolved 3D FVM source-operator recovery from "
            "trainable MLP fields; connects analytic PDE residual training to "
            "local source-operator reliability, not reactor-specific validation."
        ),
        "n_checks": len(checks),
        "n_pass_checks": sum(row["status"] == "PASS" for row in checks),
    }

    output = {
        "summary": summary,
        "pairs": [
            {
                key: (float(value) if key != "holdout_case" else value)
                for key, value in row.items()
            }
            for row in pair_rows
        ],
        "checks": checks,
    }
    SUMMARY.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY}")
    print(
        "PDE-derived median source multiplier error "
        f"{summary['median_pde_total_multiplier_error_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
