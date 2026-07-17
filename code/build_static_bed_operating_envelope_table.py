#!/usr/bin/env python3
"""Build the static-bed/component operating-envelope table for the SI."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def main() -> None:
    cfdem = read_json(RESULTS / "cfdem_ergun_envelope" / "cfdem_ergun_envelope_summary.json")["summary"]
    wall = read_json(RESULTS / "wall_bypass_radial_rtd_gate" / "wall_bypass_radial_rtd_summary.json")["summary"]
    mesh = read_json(RESULTS / "fvm3d_mesh_observation_sensitivity" / "fvm3d_mesh_observation_summary.json")
    chem = read_json(RESULTS / "fvm3d_chemistry_transport_pressure" / "fvm3d_chemistry_transport_summary.json")
    module = read_json(RESULTS / "module_operating_envelope" / "module_operating_envelope_summary.json")
    component = read_json(RESULTS / "component_3d_module_network" / "component_3d_summary.json")
    diagnostic = read_json(RESULTS / "diagnostic_operating_envelope" / "diagnostic_operating_envelope_summary.json")

    nominal = module["nominal"]
    worst = module["worst_tail"]
    best = module["best_release"]
    grid_cells = [g["n_cells"] for g in mesh["grids"]]

    rows = [
        {
            "quantity": "static-bed velocity support",
            "covered_range": "normalized inlet velocity 0.3, 1.0 and 2.0",
            "evidence": f"pressure shape fit uses {int(cfdem['n_velocity_levels'])} velocity levels",
            "boundary": "dimensional Reynolds number requires target gas properties and bed dimensions",
        },
        {
            "quantity": "pressure-drop envelope",
            "covered_range": f"inertial fraction {cfdem['inertial_fraction_min']:.3f}--{cfdem['inertial_fraction_max']:.3f}",
            "evidence": "Ergun-shape max relative error below 0.1\\%",
            "boundary": "shape comparison; dimensional Ergun comparison uses target bed units",
        },
        {
            "quantity": "radial porosity/RTD heterogeneity",
            "covered_range": f"wall fraction median {wall['wall_fraction_median']:.2f}; bypass ratio median {wall['bypass_ratio_median']:.2f}",
            "evidence": "RTD-constrained inverse keeps closed p95 near the calibrated-RTD result",
            "boundary": "computed annular wall-bypass case",
        },
        {
            "quantity": "3D-FVM grid family",
            "covered_range": f"{min(grid_cells)}--{max(grid_cells)} cells",
            "evidence": "matched 3D closed error stays in the matched-bed error range",
            "boundary": "source-estimation mesh sensitivity for coarse porous/effective fields",
        },
        {
            "quantity": "HT/HTO chemistry field",
            "covered_range": f"{int(chem['n_cases'])} dry/wet-bypass transport-chemistry cases",
            "evidence": f"wrong chemistry {chem['heterogeneous_wet_bypass_matched_transport_dry_chemistry_closed_error_percent']:.2f}\\%; matched about 0.3\\%",
            "boundary": "computed chemistry-resolved 3D-FVM pressure calculation",
        },
        {
            "quantity": "module velocity/temperature/TES envelope",
            "covered_range": "velocity scale 0.5--2.0; temperature offset -80--80 degC; TES 0.92--0.99",
            "evidence": f"{int(module['n_pass'])}/{int(module['n_scenarios'])} reduced scenarios stay within source limits",
            "boundary": "reduced module operating envelope",
        },
        {
            "quantity": "release-time envelope",
            "covered_range": f"t90 {best['t90_h']:.2f}--{worst['t90_h']:.2f} h across module scenarios",
            "evidence": f"nominal 48 h release {100.0 * nominal['released_fraction_48h']:.2f}\\%",
            "boundary": "computed component transfer-function response",
        },
        {
            "quantity": "component discretization",
            "covered_range": f"{component['grid']['n_cells']} reduced zones and {component['grid']['n_outlet_groups']} outlet groups",
            "evidence": f"source-weighted zone t90 {component['component']['source_weighted_cell_t90_h']:.2f} h",
            "boundary": "reduced component network; CAD/neutronics fields replace it for a named design",
        },
        {
            "quantity": "measurement-noise envelope",
            "covered_range": f"{int(diagnostic['fisher_noise_levels'])} noise levels and {int(diagnostic['fisher_observable_sets'])} measurement sets",
            "evidence": f"complete measurement set remains within source limits to noise scale {diagnostic['full_multimodal_max_noise_scale_passing_all_gates']:.2f}",
            "boundary": "measurement design envelope only",
        },
        {
            "quantity": "Re/Pe/Damkohler reporting",
            "covered_range": "reported for each target gas, packing, humidity and chemistry package",
            "evidence": "computed when dimensional target fields are supplied",
            "boundary": "used to define the dimensional extrapolation range",
        },
    ]

    out_csv = RESULTS / "static_bed_operating_envelope_table.csv"
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
        r"\caption{Operating envelope of the computed static-bed and component calculations. The table states the covered range and what remains a reactor-specific input requirement.}",
        r"\label{tab:static_bed_operating_envelope}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.20\textwidth}P{0.30\textwidth}P{0.25\textwidth}Y@{}}",
        r"\toprule",
        r"Quantity & Covered range & Evidence used & Use range \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["quantity"]),
                    row["covered_range"],
                    row["evidence"],
                    tex_escape(row["boundary"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_static_bed_operating_envelope_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
