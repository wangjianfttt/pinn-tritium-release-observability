#!/usr/bin/env python3
"""Check the stored leave-one-3D-case PDE-MLP training outputs.

This script verifies the split-level optimizer outputs before the manuscript
summary is rebuilt. It is intentionally lightweight: the expensive optimizer
run is represented by archived model and metric CSV files, while this check
confirms that those files contain the physical leave-one-case structure used
by the manuscript.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fvm3d_trainable_pde_mlp_leave_one_case"
METRICS = OUT / "fvm3d_trainable_pde_mlp_leave_one_case_metrics.csv"
MODELS = OUT / "fvm3d_trainable_pde_mlp_leave_one_case_models.csv"
CHECKS = OUT / "fvm3d_trainable_pde_mlp_leave_one_case_input_checks.csv"
SUMMARY = OUT / "fvm3d_trainable_pde_mlp_leave_one_case_input_check_summary.json"

EXPECTED_CASES = {
    "uniform_dry",
    "heterogeneous_wet_wall",
    "heterogeneous_wet_bypass",
}
EXPECTED_MODELS = {
    "data_only",
    "pde_constrained_w4",
    "pde_constrained_w8",
}
EXPECTED_SPLITS = {"train_cases", "holdout_case"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def passfail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def selected_rows(metrics: list[dict[str, str]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for holdout in sorted(EXPECTED_CASES):
        holdout_rows = [
            row
            for row in metrics
            if row["holdout_case"] == holdout and row["split_id"] == "holdout_case"
        ]
        data_rows = [row for row in holdout_rows if row["model_kind"] == "data_only"]
        pde_rows = [row for row in holdout_rows if row["model_kind"].startswith("pde_constrained")]
        if len(data_rows) != 1 or not pde_rows:
            continue
        data = data_rows[0]
        best = min(pde_rows, key=lambda row: as_float(row, "pde_residual_rms_scaled"))
        selected.append(
            {
                "holdout_case": holdout,
                "model_kind": best["model_kind"],
                "pde_residual_gain": as_float(data, "pde_residual_rms_scaled")
                / as_float(best, "pde_residual_rms_scaled"),
                "field_rmse_ratio": as_float(best, "total_field_rmse_scaled")
                / as_float(data, "total_field_rmse_scaled"),
            }
        )
    return selected


def main() -> int:
    metrics = read_rows(METRICS)
    models = read_rows(MODELS)
    checks: list[dict[str, str]] = []

    metric_cases = {row["holdout_case"] for row in metrics}
    model_cases = {row["holdout_case"] for row in models}
    checks.append(
        {
            "check_id": "three_holdout_cases_present",
            "status": passfail(metric_cases == EXPECTED_CASES and model_cases == EXPECTED_CASES),
            "detail": f"metric_cases={sorted(metric_cases)} model_cases={sorted(model_cases)}",
        }
    )

    metric_splits = {row["split_id"] for row in metrics}
    checks.append(
        {
            "check_id": "train_and_holdout_splits_present",
            "status": passfail(metric_splits == EXPECTED_SPLITS),
            "detail": f"splits={sorted(metric_splits)}",
        }
    )

    split_complete = True
    split_details: list[str] = []
    leakage_free = True
    for holdout in sorted(EXPECTED_CASES):
        for split in sorted(EXPECTED_SPLITS):
            kinds = {
                row["model_kind"]
                for row in metrics
                if row["holdout_case"] == holdout and row["split_id"] == split
            }
            if kinds != EXPECTED_MODELS:
                split_complete = False
            split_details.append(f"{holdout}:{split}:{sorted(kinds)}")
        train_case_fields = {
            row["train_cases"]
            for row in metrics + models
            if row["holdout_case"] == holdout
        }
        for train_case_field in train_case_fields:
            train_cases = set(train_case_field.split("+"))
            if holdout in train_cases or train_cases != (EXPECTED_CASES - {holdout}):
                leakage_free = False

    checks.append(
        {
            "check_id": "each_case_has_data_and_pde_models",
            "status": passfail(split_complete),
            "detail": "; ".join(split_details),
        }
    )
    checks.append(
        {
            "check_id": "holdout_case_not_in_training_cases",
            "status": passfail(leakage_free),
            "detail": "training cases are the other two 3D fields for each rotation",
        }
    )

    model_kinds_by_case = {
        holdout: {
            row["model_kind"]
            for row in models
            if row["holdout_case"] == holdout
        }
        for holdout in EXPECTED_CASES
    }
    checks.append(
        {
            "check_id": "model_table_has_three_models_per_rotation",
            "status": passfail(
                len(models) == 9
                and all(kinds == EXPECTED_MODELS for kinds in model_kinds_by_case.values())
            ),
            "detail": f"models={len(models)}",
        }
    )

    nfev_values = [as_int(row, "nfev") for row in models]
    checks.append(
        {
            "check_id": "iterative_optimizer_records_present",
            "status": passfail(bool(nfev_values) and min(nfev_values) >= 70),
            "detail": f"nfev={min(nfev_values)}-{max(nfev_values)}",
        }
    )

    selected = selected_rows(metrics)
    gains = [float(row["pde_residual_gain"]) for row in selected]
    ratios = [float(row["field_rmse_ratio"]) for row in selected]
    checks.append(
        {
            "check_id": "selected_pde_model_reduces_residual",
            "status": passfail(len(gains) == 3 and min(gains) > 1.0),
            "detail": f"gain={min(gains):.3f}-{max(gains):.3f}",
        }
    )
    checks.append(
        {
            "check_id": "selected_pde_model_preserves_field_fit",
            "status": passfail(len(ratios) == 3 and max(ratios) < 1.0),
            "detail": f"max_field_rmse_ratio={max(ratios):.3f}",
        }
    )

    with CHECKS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    summary = {
        "n_metrics_rows": len(metrics),
        "n_model_rows": len(models),
        "n_holdout_cases": len(metric_cases),
        "min_pde_residual_gain": min(gains),
        "median_pde_residual_gain": median(gains),
        "max_field_rmse_ratio": max(ratios),
        "n_pass_checks": sum(row["status"] == "PASS" for row in checks),
        "n_checks": len(checks),
        "claim_boundary": (
            "Checks the stored leave-one-3D-case training outputs used by the "
            "manuscript; it verifies split structure and metrics, and does not "
            "create external experimental validation."
        ),
    }
    SUMMARY.write_text(json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {CHECKS.relative_to(ROOT)}")
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    print(
        "checked leave-one PDE-MLP inputs: "
        f"checks={summary['n_pass_checks']}/{summary['n_checks']} "
        f"gain={summary['min_pde_residual_gain']:.3f}--{max(gains):.3f} "
        f"max_field_rmse_ratio={summary['max_field_rmse_ratio']:.3f}"
    )
    return 0 if summary["n_pass_checks"] == summary["n_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
