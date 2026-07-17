#!/usr/bin/env python3
"""Rebuild the 3D training-operator selection summary.

The manuscript uses this analysis to decide when computed 3D histories are
acceptable PINN training data for a target pebble-bed measurement. The script
starts from the split-level table and recomputes the worst-case source errors
for four choices:

* selected matched 3D operator with spatial/species observations,
* residual-only mixed-outlet selector,
* larger pooled non-target library,
* nearest wrong single-field library.

No training labels are created here; the script only aggregates the completed
field/source withheld tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "fvm3d_training_data_selection_gate"
DETAIL = RESULT_DIR / "fvm3d_training_data_selection_gate.csv"
CHECKS_CSV = RESULT_DIR / "fvm3d_training_data_selection_gate_checks.csv"
SUMMARY_JSON = RESULT_DIR / "fvm3d_training_data_selection_gate_summary.json"


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def b(row: dict[str, str], key: str) -> bool:
    return row[key].strip().lower() == "true"


def subset(rows: list[dict[str, str]], selector_id: str) -> list[dict[str, str]]:
    chunk = [row for row in rows if row["selector_id"] == selector_id]
    if not chunk:
        raise ValueError(f"missing selector_id={selector_id!r}")
    return chunk


def maxf(rows: list[dict[str, str]], key: str) -> float:
    return max(f(row, key) for row in rows)


def minf(rows: list[dict[str, str]], key: str) -> float:
    return min(f(row, key) for row in rows)


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not DETAIL.exists():
        raise FileNotFoundError(f"missing detail table: {DETAIL}")

    with DETAIL.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty detail table: {DETAIL}")

    fields = sorted({row["holdout_field"] for row in rows})
    sources = sorted({row["holdout_source_family"] for row in rows})
    expected_splits = len(fields) * len(sources)

    selected = subset(rows, "operator_residual_plus_observability")
    residual_only = subset(rows, "operator_residual_only")
    pooled = subset(rows, "larger_pooled_library_no_gate")
    wrong = subset(rows, "nearest_wrong_single_field_no_gate")
    oracle = subset(rows, "matched_field_oracle_upper_bound")

    selected_allowed = [row for row in selected if b(row, "allowed_for_training_claim")]
    if len(selected_allowed) != len(selected):
        raise ValueError("selected matched-operator rows must all be allowed")

    summary = {
        "n_holdout_field_source_splits": expected_splits,
        "residual_gate": 0.001,
        "minimum_observation_rank": min(int(row["n_projection_features"]) for row in selected),
        "admissible_selector_worst_closed_p95_pct": maxf(selected, "closed_p95_error_pct"),
        "admissible_selector_worst_zone_p95_pct": maxf(selected, "zone_l1_p95_error_pct"),
        "admissible_selector_min_closed_pass_rate": minf(selected, "closed_pass_rate_5pct"),
        "admissible_selector_allowed_splits": len(selected_allowed),
        "residual_only_worst_closed_p95_pct": maxf(residual_only, "closed_p95_error_pct"),
        "residual_only_worst_zone_p95_pct": maxf(residual_only, "zone_l1_p95_error_pct"),
        "residual_only_observability_fail_splits": sum(
            not b(row, "observability_pass") for row in residual_only
        ),
        "pooled_no_gate_worst_closed_p95_pct": maxf(pooled, "closed_p95_error_pct"),
        "pooled_no_gate_worst_zone_p95_pct": maxf(pooled, "zone_l1_p95_error_pct"),
        "wrong_no_gate_worst_closed_p95_pct": maxf(wrong, "closed_p95_error_pct"),
        "wrong_no_gate_worst_zone_p95_pct": maxf(wrong, "zone_l1_p95_error_pct"),
        "oracle_worst_closed_p95_pct": maxf(oracle, "closed_p95_error_pct"),
        "oracle_worst_zone_p95_pct": maxf(oracle, "zone_l1_p95_error_pct"),
    }
    summary["pooled_to_admissible_closed_ratio"] = (
        summary["pooled_no_gate_worst_closed_p95_pct"]
        / summary["admissible_selector_worst_closed_p95_pct"]
    )
    summary["residual_only_to_admissible_zone_ratio"] = (
        summary["residual_only_worst_zone_p95_pct"]
        / summary["admissible_selector_worst_zone_p95_pct"]
    )
    summary["claim_boundary"] = (
        "Synthetic 3D FVM neural-inverse training-data selection test. It "
        "supports a physics-plus-observability selection rule for trainable 3D "
        "operators; it is not an experimental or reactor-specific validation result."
    )

    checks = [
        {
            "check_id": "all_field_source_splits_present",
            "status": "PASS" if len(selected) == expected_splits else "FAIL",
            "detail": f"splits={len(selected)}",
        },
        {
            "check_id": "admissible_selector_passes_joint_source_criterion",
            "status": "PASS"
            if (
                summary["admissible_selector_worst_closed_p95_pct"] < 5.0
                and summary["admissible_selector_worst_zone_p95_pct"] < 5.0
                and summary["admissible_selector_min_closed_pass_rate"] >= 0.95
            )
            else "FAIL",
            "detail": (
                f"closed={summary['admissible_selector_worst_closed_p95_pct']:.3f}%; "
                f"zone={summary['admissible_selector_worst_zone_p95_pct']:.3f}%"
            ),
        },
        {
            "check_id": "larger_pooled_library_is_rejected",
            "status": "PASS"
            if summary["pooled_no_gate_worst_closed_p95_pct"] > 5.0
            else "FAIL",
            "detail": (
                f"closed={summary['pooled_no_gate_worst_closed_p95_pct']:.2f}%; "
                f"zone={summary['pooled_no_gate_worst_zone_p95_pct']:.2f}%"
            ),
        },
        {
            "check_id": "nearest_wrong_operator_is_rejected",
            "status": "PASS"
            if summary["wrong_no_gate_worst_closed_p95_pct"] > 5.0
            else "FAIL",
            "detail": (
                f"closed={summary['wrong_no_gate_worst_closed_p95_pct']:.2f}%; "
                f"zone={summary['wrong_no_gate_worst_zone_p95_pct']:.2f}%"
            ),
        },
        {
            "check_id": "operator_residual_alone_is_insufficient",
            "status": "PASS"
            if summary["residual_only_observability_fail_splits"] == expected_splits
            else "FAIL",
            "detail": (
                "observability_fail_splits="
                f"{summary['residual_only_observability_fail_splits']}"
            ),
        },
    ]

    with CHECKS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    SUMMARY_JSON.write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )

    n_fail = sum(check["status"] != "PASS" for check in checks)
    print(f"wrote {CHECKS_CSV.relative_to(ROOT)}")
    print(f"wrote {SUMMARY_JSON.relative_to(ROOT)}")
    print(
        "training-operator selection: "
        f"selected closed={summary['admissible_selector_worst_closed_p95_pct']:.2f}%, "
        f"selected zone={summary['admissible_selector_worst_zone_p95_pct']:.2f}%, "
        f"residual-only zone={summary['residual_only_worst_zone_p95_pct']:.2f}%, "
        f"pooled closed={summary['pooled_no_gate_worst_closed_p95_pct']:.2f}%"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
