#!/usr/bin/env python3
"""Rebuild the external time-scale to system-accountancy summary.

The input table contains the commissioning-budget sweep used to test whether
the sparse Park/FEC HT/HTO time-scale information changes target-field
reference selection at module/system scale. This script recomputes the compact
JSON summary and checks used by the manuscript.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "external_timescale_to_system_accountancy"
TABLE = OUT / "external_timescale_to_system_accountancy.csv"
FRONTIER = OUT / "external_timescale_to_system_accountancy_frontier.csv"
CHECKS = OUT / "external_timescale_to_system_accountancy_checks.csv"
SUMMARY = OUT / "external_timescale_to_system_accountancy_summary.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def b(row: dict[str, str], key: str) -> bool:
    return row[key].strip().lower() == "true"


def row_for(rows: list[dict[str, str]], strategy: str, k: int) -> dict[str, str]:
    for row in rows:
        if row["strategy"] == strategy and int(row["k"]) == k:
            return row
    raise KeyError(f"missing row {strategy} K={k}")


def first_pass(rows: list[dict[str, str]], strategy: str) -> int:
    candidates = [
        int(row["k"])
        for row in rows
        if row["strategy"] == strategy and b(row, "system_budget_pass")
    ]
    return min(candidates) if candidates else -1


def main() -> int:
    rows = read_rows(TABLE)
    frontier = read_rows(FRONTIER)

    k0 = row_for(rows, "time_scale_guided", 0)
    guided16 = row_for(rows, "time_scale_guided", 16)
    random16 = row_for(rows, "random", 16)
    random64 = row_for(rows, "random", 64)
    prefix16 = row_for(rows, "prefix", 16)
    space16 = row_for(rows, "space_filling", 16)

    frontier_by_strategy = {row["strategy"]: row for row in frontier}
    summary = {
        "n_rows": len(rows),
        "source_reference_threshold_pct": f(guided16, "system_source_threshold_pct"),
        "accountancy_radius": f(guided16, "accountancy_radius"),
        "net_recovery_half_width": f(guided16, "net_recovery_gate"),
        "k0_radius": f(k0, "combined_net_recovery_radius"),
        "time_scale_guided_first_system_k": first_pass(rows, "time_scale_guided"),
        "prefix_first_system_k": first_pass(rows, "prefix"),
        "space_filling_first_system_k": first_pass(rows, "space_filling"),
        "random_first_system_k": first_pass(rows, "random"),
        "time_scale_guided_k16_joint_p95_pct": f(guided16, "joint_p95_worst_pct"),
        "time_scale_guided_k16_radius": f(guided16, "combined_net_recovery_radius"),
        "random_k16_joint_p95_worst_pct": f(random16, "joint_p95_worst_pct"),
        "random_k16_trial_pass_rate": f(random16, "pass_rate_trials"),
        "random_k16_radius": f(random16, "combined_net_recovery_radius"),
        "random_k64_joint_p95_worst_pct": f(random64, "joint_p95_worst_pct"),
        "prefix_k16_joint_p95_pct": f(prefix16, "joint_p95_worst_pct"),
        "space_filling_k16_joint_p95_pct": f(space16, "joint_p95_worst_pct"),
        "frontier": frontier,
        "claim": (
            "Sparse HT/HTO time-scale information changes the target-field "
            "reference selection used by the source inverse before system "
            "tritium-accountancy propagation."
        ),
        "claim_boundary": (
            "The calculation propagates digitized Park/FEC time-scale information "
            "through the existing 3D curriculum and 288-component accountancy "
            "surrogate; it is not direct reactor-specific validation."
        ),
    }

    checks = [
        {
            "check_id": "external_timescale_alone_not_enough",
            "status": "PASS" if summary["k0_radius"] > 0.1 else "FAIL",
            "detail": f"K=0 radius={summary['k0_radius']:.5f}",
        },
        {
            "check_id": "time_scale_guided_k16_passes_system_budget",
            "status": "PASS"
            if summary["time_scale_guided_first_system_k"] == 16
            and summary["time_scale_guided_k16_joint_p95_pct"]
            < summary["source_reference_threshold_pct"]
            else "FAIL",
            "detail": (
                f"K={summary['time_scale_guided_first_system_k']}; "
                f"joint={summary['time_scale_guided_k16_joint_p95_pct']:.2f}%"
            ),
        },
        {
            "check_id": "random_k16_not_robust",
            "status": "PASS"
            if summary["random_k16_joint_p95_worst_pct"]
            > summary["source_reference_threshold_pct"]
            and summary["random_k16_trial_pass_rate"] < 1.0
            else "FAIL",
            "detail": (
                f"random K=16 worst={summary['random_k16_joint_p95_worst_pct']:.2f}%; "
                f"pass rate={summary['random_k16_trial_pass_rate']:.2f}"
            ),
        },
        {
            "check_id": "random_requires_larger_budget",
            "status": "PASS" if summary["random_first_system_k"] == 64 else "FAIL",
            "detail": f"random first system K={summary['random_first_system_k']}",
        },
        {
            "check_id": "frontier_matches_source_table",
            "status": "PASS"
            if int(frontier_by_strategy["time_scale_guided"]["first_system_k"]) == 16
            and int(frontier_by_strategy["random"]["first_system_k"]) == 64
            else "FAIL",
            "detail": (
                f"frontier time-scale={frontier_by_strategy['time_scale_guided']['first_system_k']}; "
                f"random={frontier_by_strategy['random']['first_system_k']}"
            ),
        },
    ]
    summary["n_checks"] = len(checks)
    summary["n_pass_checks"] = sum(item["status"] == "PASS" for item in checks)
    summary["all_checks_pass"] = summary["n_checks"] == summary["n_pass_checks"]

    SUMMARY.write_text(json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    with CHECKS.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    print(
        "external time-scale system pass: "
        f"K16={summary['time_scale_guided_k16_joint_p95_pct']:.2f}%, "
        f"random K16={summary['random_k16_joint_p95_worst_pct']:.2f}%"
    )
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
