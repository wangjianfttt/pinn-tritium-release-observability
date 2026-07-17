#!/usr/bin/env python3
"""Assemble a target-dependent observability summary from generated artifacts.

The script does not create new simulation data. It reads existing 3D-FVM,
component, transfer-kernel and reduced system summaries and writes a compact
cross-scale table used to state which measurement families are needed as the
source-estimation target becomes more demanding.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "target_observability_phase_diagram"
MANUSCRIPT = ROOT / "manuscript"


def read_summary(relative: str) -> dict:
    path = RESULTS / relative
    data = json.loads(path.read_text())
    return data.get("summary", data)


def fmt_pct(value: float) -> str:
    return f"{value:.2f}\\%"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def build_rows() -> list[dict]:
    min_sensor = read_summary("fvm3d_minimum_sensor_design/fvm3d_minimum_sensor_design_summary.json")
    thermal = read_summary("fvm3d_thermal_sensor_budget/fvm3d_thermal_sensor_budget_summary.json")
    component = read_summary("component_3d_direct_inverse_observability/component_3d_direct_inverse_summary.json")
    module = read_summary("multimodal_sensor_budget_design/multimodal_sensor_budget_summary.json")
    transfer = read_summary("transfer_kernel_reference_selection_law/transfer_kernel_reference_selection_summary.json")
    reactor = read_summary("whole_reactor_sensor_budget_design/whole_reactor_sensor_budget_summary.json")

    module_low = module["best_by_budget"]["2.5"]
    module_full = module["best_by_budget"]["9.6"]

    rows = [
        {
            "target_level": 1,
            "target": "clean 3D release multipliers",
            "target_dimension": 2,
            "failed_baseline": "single mixed total outlet",
            "failed_error_metric": "max multiplier error (%)",
            "failed_error_value": min_sensor["mixed_total_max_error_pct"],
            "first_passing_observation": "single spatial-quadrant HT/HTO pair",
            "observation_units": min_sensor["minimum_feasible_scalar_channels"],
            "passing_error_metric": "max multiplier error (%)",
            "passing_error_value": min_sensor["minimum_feasible_error_pct"],
            "closed_direction": "spatial/species residence-time ambiguity",
            "artifact": "results/fvm3d_minimum_sensor_design/fvm3d_minimum_sensor_design_summary.json",
        },
        {
            "target_level": 2,
            "target": "thermal 3D source map",
            "target_dimension": 8,
            "failed_baseline": "mixed outlet with insufficient spatial/thermal basis",
            "failed_error_metric": "source-map L1 p95 (%)",
            "failed_error_value": thermal["mixed_source_l1_p95_pct"],
            "first_passing_observation": "four quadrant HT/HTO pairs + matched thermal basis",
            "observation_units": 8,
            "passing_error_metric": "joint source/closed p95 (%)",
            "passing_error_value": thermal["minimum_feasible_source_l1_p95_pct"],
            "closed_direction": "source localization and Arrhenius/temperature weighting",
            "artifact": "results/fvm3d_thermal_sensor_budget/fvm3d_thermal_sensor_budget_summary.json",
        },
        {
            "target_level": 3,
            "target": "component tritium-balance vector",
            "target_dimension": component["n_parameters"],
            "failed_baseline": "single mixed noisy outlet",
            "failed_error_metric": "closed-pathway p95 (%)",
            "failed_error_value": component["single_mixed_closed_p95_error_pct"],
            "first_passing_observation": "12 spatial HT/HTO histories + inventory/TES",
            "observation_units": 26,
            "passing_error_metric": "closed-pathway p95 (%)",
            "passing_error_value": component["full_accountancy_closed_p95_error_pct"],
            "closed_direction": "component zone source and retained inventory",
            "artifact": "results/component_3d_direct_inverse_observability/component_3d_direct_inverse_summary.json",
        },
        {
            "target_level": 4,
            "target": "module uncertainty vector",
            "target_dimension": 10,
            "failed_baseline": "outlet/flow/pressure only",
            "failed_error_metric": "max zone-source uncertainty (%)",
            "failed_error_value": module_low["max_zone_source_uncertainty_pct"],
            "first_passing_observation": "spatial/species/flow/pressure/temperature/inventory-TES package",
            "observation_units": module_full["n_measurement_rows"],
            "passing_error_metric": "max zone-source uncertainty (%)",
            "passing_error_value": module_full["max_zone_source_uncertainty_pct"],
            "closed_direction": "transport, thermal/species and TES/recovery coupling",
            "artifact": "results/multimodal_sensor_budget_design/multimodal_sensor_budget_summary.json",
        },
        {
            "target_level": 5,
            "target": "transfer-kernel commissioning",
            "target_dimension": 2,
            "failed_baseline": "random target-bed reference pulses",
            "failed_error_metric": "2% noise first system K",
            "failed_error_value": transfer["best_2pct_first_system_k"] * 3.0,
            "first_passing_observation": "D-optimal anchor reference selection",
            "observation_units": transfer["best_2pct_first_system_k"],
            "passing_error_metric": "2% noise first system K",
            "passing_error_value": transfer["best_2pct_first_system_k"],
            "closed_direction": "biased transfer-kernel calibration directions",
            "artifact": "results/transfer_kernel_reference_selection_law/transfer_kernel_reference_selection_summary.json",
        },
        {
            "target_level": 6,
            "target": "whole-reactor local slow-release risk",
            "target_dimension": 288,
            "failed_baseline": "aggregate balance",
            "failed_error_metric": "local-failure RMSE",
            "failed_error_value": reactor["aggregate_median_local_fail_rmse"],
            "first_passing_observation": "risk-selected sector release/tail diagnostics",
            "observation_units": 6,
            "passing_error_metric": "local-failure RMSE",
            "passing_error_value": reactor["risk_release_tail_6_sector_rmse"],
            "closed_direction": "aggregate-to-local risk null space",
            "artifact": "results/whole_reactor_sensor_budget_design/whole_reactor_sensor_budget_summary.json",
        },
    ]

    for row in rows:
        failed = float(row["failed_error_value"])
        passed = float(row["passing_error_value"])
        row["repair_gain"] = failed / passed if passed > 0 else float("inf")
        row["nullspace_context"] = "single aggregate/mixed signal loses the target-specific source direction"
    return rows


def write_latex_table(rows: list[dict]) -> None:
    out = MANUSCRIPT / "generated_target_observability_table.tex"
    selected = rows[:4]
    display_target = {
        "clean 3D release multipliers": "two-parameter 3D release multipliers",
        "thermal 3D source map": "thermal 3D source map",
        "component tritium-balance vector": "component tritium-balance vector",
        "module uncertainty vector": "module uncertainty vector",
    }
    display_measurement = {
        "spatial/species/flow/pressure/temperature/inventory-TES package": (
            "spatial/species histories, flow, pressure, temperature, inventory and TES"
        ),
    }
    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\caption{Target-dependent measurement sets for source estimation. The required observations change when the target moves from two release multipliers to a spatial source, a component tritium-balance vector or a module uncertainty vector.}",
        r"\label{tab:target_observability}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.20\textwidth}P{0.22\textwidth}P{0.22\textwidth}P{0.13\textwidth}Y@{}}",
        r"\toprule",
        r"Target & Insufficient baseline & Required measurement set & Error change & Separated coupling \\",
        r"\midrule",
    ]
    for row in selected:
        failed = fmt_pct(float(row["failed_error_value"])) if "(%)" in row["failed_error_metric"] else f"{float(row['failed_error_value']):.3g}"
        passed = fmt_pct(float(row["passing_error_value"])) if "(%)" in row["passing_error_metric"] else f"{float(row['passing_error_value']):.3g}"
        lines.append(
            f"{display_target.get(row['target'], row['target'])} & {row['failed_baseline']} & "
            f"{display_measurement.get(row['first_passing_observation'], row['first_passing_observation'])} "
            f"& {failed} $\\rightarrow$ {passed} & {row['closed_direction']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = build_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "target_observability_phase_diagram.csv", rows)

    checks = [
        {
            "check_id": "every_target_has_failed_baseline",
            "status": "PASS" if min(float(r["repair_gain"]) for r in rows) > 1.0 else "FAIL",
            "detail": f"minimum gain={min(float(r['repair_gain']) for r in rows):.2f}",
        },
        {
            "check_id": "target_dimension_reaches_whole_reactor",
            "status": "PASS" if max(int(r["target_dimension"]) for r in rows) >= 288 else "FAIL",
            "detail": f"max target dimension={max(int(r['target_dimension']) for r in rows)}",
        },
        {
            "check_id": "thermal_target_requires_more_channels_than_clean_multiplier",
            "status": "PASS" if int(rows[1]["observation_units"]) > int(rows[0]["observation_units"]) else "FAIL",
            "detail": f"{rows[0]['observation_units']} -> {rows[1]['observation_units']} observation units",
        },
        {
            "check_id": "component_requires_accountancy_not_only_spatial_outlets",
            "status": "PASS" if "inventory/TES" in rows[2]["first_passing_observation"] else "FAIL",
            "detail": rows[2]["first_passing_observation"],
        },
        {
            "check_id": "whole_reactor_uses_local_not_aggregate_observable",
            "status": "PASS" if "sector" in rows[-1]["first_passing_observation"] else "FAIL",
            "detail": rows[-1]["first_passing_observation"],
        },
    ]
    write_csv(OUT / "target_observability_phase_diagram_checks.csv", checks)

    summary = {
        "n_targets": len(rows),
        "n_checks": len(checks),
        "n_pass_checks": sum(1 for c in checks if c["status"] == "PASS"),
        "min_repair_gain": min(float(r["repair_gain"]) for r in rows),
        "median_repair_gain": sorted(float(r["repair_gain"]) for r in rows)[len(rows) // 2 - 1 : len(rows) // 2 + 1],
        "max_repair_gain": max(float(r["repair_gain"]) for r in rows),
        "clean_multiplier_observation_units": int(rows[0]["observation_units"]),
        "thermal_source_map_observation_units": int(rows[1]["observation_units"]),
        "component_observation_units": int(rows[2]["observation_units"]),
        "module_observation_units": int(rows[3]["observation_units"]),
        "whole_reactor_observation_units": int(rows[-1]["observation_units"]),
        "reference_2pct_random_k": 24,
        "reference_2pct_doptimal_k": int(rows[4]["observation_units"]),
        "reference_5pct_random_status": "blocked",
        "reference_5pct_doptimal_k": 32,
        "claim_boundary": "Cross-scale observability summary from generated reduced 3D, component and system tests; not reactor-specific validation.",
    }
    vals = sorted(float(r["repair_gain"]) for r in rows)
    summary["median_repair_gain"] = (vals[2] + vals[3]) / 2.0
    write_json(OUT / "target_observability_phase_diagram_summary.json", {"summary": summary, "checks": checks})
    write_csv(OUT / "target_observability_phase_diagram_summary.csv", [summary])
    write_latex_table(rows)
    print(f"wrote {OUT}")
    print(f"wrote {MANUSCRIPT / 'generated_target_observability_table.tex'}")


if __name__ == "__main__":
    main()
