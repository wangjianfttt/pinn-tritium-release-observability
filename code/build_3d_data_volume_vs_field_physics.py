#!/usr/bin/env python3
"""Rebuild the 3D training-data volume versus target-field summary.

The manuscript uses this analysis to separate two effects that are easy to
confuse in synthetic PINN studies:

1. More computed histories from non-target 3D fields.
2. Histories generated under the calibrated target transport/thermal/chemistry
   operator.

This script intentionally starts from the split-level detail table rather than
from the manuscript summary numbers. It recomputes the protocol-level maxima,
pass checks and JSON/CSV summaries used by the paper.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "fvm3d_data_volume_vs_field_physics"
DETAIL = RESULT_DIR / "fvm3d_data_volume_vs_field_physics_detail.csv"
SUMMARY_CSV = RESULT_DIR / "fvm3d_data_volume_vs_field_physics_summary.csv"
CHECKS_CSV = RESULT_DIR / "fvm3d_data_volume_vs_field_physics_checks.csv"
SUMMARY_JSON = RESULT_DIR / "fvm3d_data_volume_vs_field_physics_summary.json"


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not DETAIL.exists():
        raise FileNotFoundError(f"missing detail table: {DETAIL}")

    rows: list[dict[str, str]] = []
    with DETAIL.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty detail table: {DETAIL}")

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    fields: set[str] = set()
    sources: set[str] = set()
    noise_values: set[float] = set()
    for row in rows:
        grouped[(row["protocol_id"], int(row["n_train"]))].append(row)
        fields.add(row["holdout_field"])
        sources.add(row["holdout_source_family"])
        noise_values.add(f(row, "noise_sigma_fraction"))

    summary_rows: list[dict[str, object]] = []
    for protocol_id, n_train in sorted(grouped, key=lambda item: (item[0], item[1])):
        chunk = grouped[(protocol_id, n_train)]
        summary_rows.append(
            {
                "protocol_id": protocol_id,
                "n_train": n_train,
                "worst_closed_p95_error_pct": max(f(r, "closed_p95_error_pct") for r in chunk),
                "worst_zone_l1_p95_error_pct": max(f(r, "zone_l1_p95_error_pct") for r in chunk),
                "min_closed_pass_rate_5pct": min(f(r, "closed_pass_rate_5pct") for r in chunk),
                "max_operator_mismatch_rmse": max(f(r, "operator_mismatch_rmse") for r in chunk),
                "n_splits": len(chunk),
                "passes_joint_gate": (
                    max(f(r, "closed_p95_error_pct") for r in chunk) < 5.0
                    and max(f(r, "zone_l1_p95_error_pct") for r in chunk) < 5.0
                    and min(f(r, "closed_pass_rate_5pct") for r in chunk) >= 0.95
                ),
            }
        )

    pooled = [r for r in summary_rows if r["protocol_id"] == "leave_field_out_pooled_library"]
    target = [r for r in summary_rows if r["protocol_id"] == "target_field_calibrated"]
    if not pooled or not target:
        raise ValueError("detail table must contain pooled and target-field protocols")

    pooled_max_train = max(int(r["n_train"]) for r in pooled)
    pooled_best = min(pooled, key=lambda r: float(r["worst_closed_p95_error_pct"]))
    target_passing = [r for r in target if bool(r["passes_joint_gate"])]
    if not target_passing:
        raise ValueError("target-field protocol never satisfies the joint source criterion")
    target_first = min(target_passing, key=lambda r: int(r["n_train"]))
    target_520 = next((r for r in target if int(r["n_train"]) == 520), None)
    if target_520 is None:
        raise ValueError("target-field n_train=520 row is required by the manuscript")

    summary = {
        "n_field_families": len(fields),
        "n_source_families": len(sources),
        "n_splits_per_row": max(int(r["n_splits"]) for r in summary_rows),
        "noise_sigma_fraction": noise_values.pop() if len(noise_values) == 1 else sorted(noise_values),
        "pooled_max_train_size": pooled_max_train,
        "pooled_best_worst_closed_p95_pct": float(pooled_best["worst_closed_p95_error_pct"]),
        "pooled_best_worst_zone_l1_p95_pct": float(pooled_best["worst_zone_l1_p95_error_pct"]),
        "pooled_any_joint_pass": any(bool(r["passes_joint_gate"]) for r in pooled),
        "target_first_joint_pass_n_train": int(target_first["n_train"]),
        "target_first_joint_pass_closed_p95_pct": float(target_first["worst_closed_p95_error_pct"]),
        "target_first_joint_pass_zone_l1_p95_pct": float(target_first["worst_zone_l1_p95_error_pct"]),
        "target_520_worst_closed_p95_pct": float(target_520["worst_closed_p95_error_pct"]),
        "target_520_worst_zone_l1_p95_pct": float(target_520["worst_zone_l1_p95_error_pct"]),
        "pooled_to_target_520_closed_ratio": float(pooled_best["worst_closed_p95_error_pct"])
        / float(target_520["worst_closed_p95_error_pct"]),
        "claim": (
            "Data volume cannot replace a matched 3D thermal/chemistry field. "
            "Synthetic augmentation is used only after the target-field "
            "operator is calibrated or selected by metadata and the spatial "
            "HT/HTO observation family is retained."
        ),
    }

    checks = [
        {
            "check_id": "pooled_large_library_does_not_pass",
            "status": "PASS" if not summary["pooled_any_joint_pass"] else "FAIL",
            "detail": (
                f"best pooled closed={summary['pooled_best_worst_closed_p95_pct']:.2f}%; "
                f"zone={summary['pooled_best_worst_zone_l1_p95_pct']:.2f}%"
            ),
        },
        {
            "check_id": "target_field_has_finite_passing_size",
            "status": "PASS" if summary["target_first_joint_pass_n_train"] == 80 else "FAIL",
            "detail": f"first pass n={summary['target_first_joint_pass_n_train']}",
        },
        {
            "check_id": "matched_field_beats_pooled_by_factor",
            "status": "PASS" if summary["pooled_to_target_520_closed_ratio"] > 5.0 else "FAIL",
            "detail": f"closed ratio={summary['pooled_to_target_520_closed_ratio']:.2f}",
        },
        {
            "check_id": "all_field_source_splits_used",
            "status": "PASS" if summary["n_splits_per_row"] == len(fields) * len(sources) else "FAIL",
            "detail": f"splits={summary['n_splits_per_row']}",
        },
    ]

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "protocol_id",
            "n_train",
            "worst_closed_p95_error_pct",
            "worst_zone_l1_p95_error_pct",
            "min_closed_pass_rate_5pct",
            "max_operator_mismatch_rmse",
            "n_splits",
            "passes_joint_gate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with CHECKS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    SUMMARY_JSON.write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )

    n_fail = sum(check["status"] != "PASS" for check in checks)
    print(f"wrote {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"wrote {CHECKS_CSV.relative_to(ROOT)}")
    print(f"wrote {SUMMARY_JSON.relative_to(ROOT)}")
    print(
        "target-field training: "
        f"pooled best closed={summary['pooled_best_worst_closed_p95_pct']:.2f}%, "
        f"target first pass n={summary['target_first_joint_pass_n_train']}, "
        f"target 520 closed={summary['target_520_worst_closed_p95_pct']:.2f}%"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
