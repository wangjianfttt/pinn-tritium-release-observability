#!/usr/bin/env python3
"""Build the main-text uncertainty and operating-envelope table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}\\%"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def main() -> None:
    li2_noise = read_json(RESULTS / "li2_detector_uncertainty_robustness.json")["summary"]
    cfdem_pressure = read_json(
        RESULTS
        / "cfdem_outlet_pressure_profile_likelihood"
        / "cfdem_outlet_pressure_profile_likelihood_summary.json"
    )
    detector_cal = read_json(
        RESULTS
        / "component_3d_neural_calibration_uncertainty_training"
        / "component_3d_neural_calibration_uncertainty_training_summary.json"
    )["summary"]
    detector_memory = read_json(
        RESULTS
        / "dynamic_detector_memory_sensitivity"
        / "dynamic_detector_memory_summary.json"
    )
    thermal = read_json(
        RESULTS / "fvm3d_thermal_sensor_budget" / "fvm3d_thermal_sensor_budget_summary.json"
    )["summary"]
    external = read_json(
        RESULTS
        / "external_curve_uncertainty_propagation"
        / "external_curve_uncertainty_propagation_summary.json"
    )["summary"]
    source_profile = read_json(
        RESULTS
        / "fvm3d_source_uncertainty_profile_gate"
        / "fvm3d_source_uncertainty_profile_gate_summary.json"
    )
    module_rows = read_csv(RESULTS / "module_uncertainty_propagation" / "module_uncertainty_summary.csv")
    module = {row["method"]: row for row in module_rows}
    whole = read_json(
        RESULTS
        / "whole_reactor_pinn_uncertainty_calibration"
        / "whole_reactor_pinn_uncertainty_calibration_summary.json"
    )["summary"]

    rows = [
        {
            "source": "Outlet noise and fine-tuning",
            "case": "Li$_2$TiO$_3$ HT/HTO outlet, 1\\% noise seeds",
            "effect": (
                f"staged max {pct(li2_noise['recommended_closed_error_max_percent'])}; "
                f"full fine-tune max {pct(li2_noise['full_finetune_closed_error_max_percent'])}"
            ),
            "use": "staged inverse is retained; unconstrained fine-tuning is the control calculation",
        },
        {
            "source": "Velocity and pressure/RTD",
            "case": "static-bed CFD-DEM sparse outlets plus 2\\% pressure noise",
            "effect": (
                f"2\\% outlet-only source p95 {pct(cfdem_pressure['two_pct_outlet_only_p95_source_error_pct'])}; "
                f"pressure/RTD source p95 {pct(cfdem_pressure['two_pct_pressure_p95_source_error_pct'])}"
            ),
            "use": "pressure/RTD anchors residence time within the covered static-bed data set",
        },
        {
            "source": "Detector calibration",
            "case": "3D component calibration draws plus dynamic HTO sampling-line memory",
            "effect": (
                f"finite calibration p95 {pct(detector_cal['calibration_augmented_twenty_mixed_2pct_worst_closed_p95_pct'])}; "
                f"metadata-calibrated p95 {pct(detector_cal['calibration_metadata_twenty_mixed_2pct_worst_closed_p95_pct'])}; "
                f"static memory bias {pct(detector_memory['static_matrix_closed_error_pct'])} $\\rightarrow$ "
                f"pulse-calibrated p95 {pct(detector_memory['pulse_calibrated_closed_error_p95_pct'])}"
            ),
            "use": "detector matrix and calibration metadata are fitted separately from release multipliers",
        },
        {
            "source": "Thermal-field identity",
            "case": f"3D thermal sensor set, span {thermal['temperature_span_k']:.1f} K",
            "effect": (
                f"mixed source p95 {pct(thermal['mixed_source_l1_p95_pct'])}; "
                f"four-quadrant source/closed {pct(thermal['four_quadrant_source_l1_p95_pct'])}/"
                f"{pct(thermal['four_quadrant_closed_p95_pct'])}"
            ),
            "use": "temperature measurement or reconstruction is included; average-temperature fits remain high",
        },
        {
            "source": "External-curve digitization",
            "case": "Park/FEC 8 points and literature TDS/image bounds",
            "effect": (
                f"{int(external['park_points'])} points; "
                f"time-scale inventory reduction {external['real_prior_inventory_reduction_pp']:.2f} percentage points; "
                f"trend pass {int(external['digitization_trend_pass'])}/{int(external['digitization_trend_total'])}"
            ),
            "use": "sparse curves provide weak time-scale and mechanism constraints only",
        },
        {
            "source": "Source-structure uncertainty",
            "case": f"{int(source_profile['n_bootstrap'])} bootstrap/profile source-map fits",
            "effect": (
                f"full source $L_1$ p95 {pct(source_profile['full_source_map_l1_p95_pct'])}; "
                f"wrong-family p95 {pct(source_profile['best_wrong_source_map_l1_p95_pct'])}; "
                f"incompatible families passing 0"
            ),
            "use": "source point error is reported with interval width and field-family separation",
        },
        {
            "source": "Module propagation",
            "case": "2000 module uncertainty samples",
            "effect": (
                f"within-limit fraction {float(module['uncorrected_1d']['pass_rate']) * 100:.1f}\\% "
                f"$\\rightarrow$ {float(module['full_multimodal']['pass_rate']) * 100:.1f}\\%; "
                f"$t_{{90}}$ half-width {float(module['uncorrected_1d']['t90_h_half_width_p05_p95']):.2f} h "
                f"$\\rightarrow$ {float(module['full_multimodal']['t90_h_half_width_p05_p95']):.2f} h"
            ),
            "use": "local source uncertainty is propagated with retained inventory and extraction balance",
        },
        {
            "source": "System-scale coverage",
            "case": "288-component computed tritium-balance uncertainty calibration",
            "effect": (
                f"covered transfer paths {int(whole['n_bounded_whole_reactor_ood_paths'])}; "
                f"outside-range paths {int(whole['n_blocked_whole_reactor_ood_paths'])}"
            ),
            "use": "system result is a reduced source-estimation calculation; blanket uncertainty requires reactor-specific fields",
        },
    ]

    out_csv = RESULTS / "uncertainty_operating_envelope_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Uncertainty sources and operating envelope used by the source inverse. Each row states which uncertainty is varied, what it does to the source estimate and how the result is used.}",
        r"\label{tab:uncertainty_operating_envelope}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.18\textwidth}P{0.27\textwidth}P{0.25\textwidth}Y@{}}",
        r"\toprule",
        r"Uncertainty source & Case or envelope & Effect on source estimate & Use in the paper \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["source"]),
                    row["case"],
                    row["effect"],
                    tex_escape(row["use"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_uncertainty_operating_envelope_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
