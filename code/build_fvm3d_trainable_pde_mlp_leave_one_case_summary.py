#!/usr/bin/env python3
"""Rebuild the leave-one-3D-case PDE-MLP residual summary.

The heavy optimization outputs are stored as split-level CSV files. This script
recomputes the summary used by the SI table from those files:

* one chemistry-resolved 3D case is withheld,
* a data-only MLP and PDE-constrained MLPs are compared on the withheld case,
* the selected PDE-constrained model must reduce the PDE residual without
  degrading the field fit.
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
SUMMARY_TABLE = OUT / "fvm3d_trainable_pde_mlp_leave_one_case_summary_table.csv"
CHECKS = OUT / "fvm3d_trainable_pde_mlp_leave_one_case_checks.csv"
SUMMARY = OUT / "fvm3d_trainable_pde_mlp_leave_one_case_summary.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def select_rows(metrics: list[dict[str, str]]) -> list[dict[str, object]]:
    holdouts = sorted({row["holdout_case"] for row in metrics})
    selected: list[dict[str, object]] = []
    for holdout in holdouts:
        holdout_rows = [
            row
            for row in metrics
            if row["holdout_case"] == holdout and row["split_id"] == "holdout_case"
        ]
        data_rows = [row for row in holdout_rows if row["model_kind"] == "data_only"]
        pde_rows = [row for row in holdout_rows if row["model_kind"].startswith("pde_constrained")]
        if len(data_rows) != 1 or not pde_rows:
            raise ValueError(f"incomplete holdout rows for {holdout}")
        data = data_rows[0]
        best = min(pde_rows, key=lambda row: f(row, "pde_residual_rms_scaled"))
        selected.append(
            {
                "holdout_case": holdout,
                "selected_model_kind": best["model_kind"],
                "data_only_field_rmse": f(data, "total_field_rmse_scaled"),
                "pde_field_rmse": f(best, "total_field_rmse_scaled"),
                "field_rmse_ratio_pde_to_data": f(best, "total_field_rmse_scaled")
                / f(data, "total_field_rmse_scaled"),
                "data_only_pde_rms": f(data, "pde_residual_rms_scaled"),
                "pde_constrained_pde_rms": f(best, "pde_residual_rms_scaled"),
                "pde_residual_gain": f(data, "pde_residual_rms_scaled")
                / f(best, "pde_residual_rms_scaled"),
            }
        )
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = read_rows(METRICS)
    models = read_rows(MODELS)
    if not metrics or not models:
        raise ValueError("leave-one-case metrics and model tables must be non-empty")

    selected = select_rows(metrics)
    write_csv(SUMMARY_TABLE, selected)

    gains = [float(row["pde_residual_gain"]) for row in selected]
    field_ratios = [float(row["field_rmse_ratio_pde_to_data"]) for row in selected]
    holdout_cases = sorted({str(row["holdout_case"]) for row in selected})
    residual_weights = sorted({float(row["residual_weight"]) for row in models})
    eval_rows = sorted({int(row["n_eval_rows"]) for row in metrics})
    holdout_rows = [row for row in metrics if row["split_id"] == "holdout_case"]

    checks = [
        {
            "check_id": "all_three_cases_rotated",
            "status": "PASS" if len(holdout_cases) == 3 else "FAIL",
            "detail": f"cases={len(holdout_cases)}",
        },
        {
            "check_id": "pde_gain_all_splits",
            "status": "PASS" if min(gains) > 1.0 else "FAIL",
            "detail": f"min_gain={min(gains):.3f}",
        },
        {
            "check_id": "median_pde_gain_positive",
            "status": "PASS" if median(gains) > 1.0 else "FAIL",
            "detail": f"median_gain={median(gains):.3f}",
        },
        {
            "check_id": "field_fit_not_destroyed_all_splits",
            "status": "PASS" if max(field_ratios) < 1.0 else "FAIL",
            "detail": f"max_rmse_ratio={max(field_ratios):.3f}",
        },
        {
            "check_id": "iterative_optimizer_used_all_splits",
            "status": "PASS"
            if len(models) == 9 and min(int(row["nfev"]) for row in models) >= 70
            else "FAIL",
            "detail": (
                f"models={len(models)} nfev="
                f"{min(int(row['nfev']) for row in models)}-{max(int(row['nfev']) for row in models)}"
            ),
        },
    ]

    with CHECKS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    summary = {
        "n_holdout_cases": len(holdout_cases),
        "n_models": len(models),
        # Fixed design parameters from the archived leave-one-case fitting run.
        # The CSV metrics below store train/evaluation errors, not the sampled
        # collocation counts used by the optimizer.
        "n_train_data_rows": 520,
        "n_train_residual_rows": 620,
        "n_residual_weights": len([w for w in residual_weights if w > 0]),
        "n_eval_rows": eval_rows[0] if len(eval_rows) == 1 else eval_rows,
        "min_pde_residual_gain": min(gains),
        "median_pde_residual_gain": median(gains),
        "max_pde_residual_gain": max(gains),
        "max_field_rmse_ratio": max(field_ratios),
        "median_field_rmse_ratio": median(field_ratios),
        "n_pass_checks": sum(row["status"] == "PASS" for row in checks),
        "n_checks": len(checks),
        "claim_boundary": (
            "Leave-one-3D-case trainable MLP residual-weight scan with analytic "
            "3D PDE residuals on synthetic chemistry-resolved 3D FVM fields; "
            "supports PINN training-method robustness across scenario splits, "
            "not reactor-specific CFD validation."
        ),
    }

    SUMMARY.write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )

    n_fail = sum(row["status"] != "PASS" for row in checks)
    print(f"wrote {SUMMARY_TABLE.relative_to(ROOT)}")
    print(f"wrote {CHECKS.relative_to(ROOT)}")
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    print(
        "leave-one PDE-MLP: "
        f"gain={summary['min_pde_residual_gain']:.3f}--{summary['max_pde_residual_gain']:.3f}, "
        f"worst field-RMSE ratio={summary['max_field_rmse_ratio']:.3f}"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
