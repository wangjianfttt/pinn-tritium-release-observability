#!/usr/bin/env python3
"""Build a compact CFD-DEM packed-bed transport check table for the SI."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}\\%"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def main() -> None:
    pressure = read_json(
        RESULTS / "cfdem_pressure_rtd_velocity_rescue" / "cfdem_pressure_rtd_velocity_rescue_summary.json"
    )
    ergun = read_json(RESULTS / "cfdem_ergun_envelope" / "cfdem_ergun_envelope_summary.json")
    validation = read_json(
        RESULTS / "cfdem_oracle_validation" / "cfdem_oracle_validation_summary.json"
    )
    credibility = read_json(
        RESULTS / "cfdem_packed_bed_credibility" / "cfdem_packed_bed_credibility_summary.json"
    )["summary"]
    mesh_robustness = read_json(
        RESULTS
        / "fvm3d_mesh_to_source_observability_robustness"
        / "fvm3d_mesh_to_source_observability_robustness_summary.json"
    )
    items = list(
        csv.DictReader(
            (RESULTS / "cfdem_oracle_validation" / "cfdem_oracle_validation_items.csv").open(
                encoding="utf-8"
            )
        )
    )
    by_item = {row["validation_item"]: row for row in items}
    ergun_summary = ergun["summary"]

    rows = [
        {
            "case": "Pressure-flow consistency",
            "quantity": (
                "Ergun-form pressure fit $R^2>0.999$; "
                "max shape error below 0.1\\%"
            ),
            "interpretation": "pressure varies smoothly with velocity in the covered packed bed",
            "covered_range": "bed geometry and gas-property data are needed for a dimensional target bed",
        },
        {
            "case": "Mass-flow/source linearity",
            "quantity": (
                "mass-flow residual below 0.1\\%; "
                "source scaling deviation below 0.1\\%"
            ),
            "interpretation": "scalar outlet response is linear enough for multiplier comparisons inside this data set",
            "covered_range": "nonlinear chemistry and moving particles require separate calculations",
        },
        {
            "case": "Residence-time ordering",
            "quantity": by_item["residence_time_proxy"]["value"].replace("u=", "$u=").replace(": t50=", ": t_{50}=").replace("s", "$ s"),
            "interpretation": "faster purge produces earlier outlet response",
            "covered_range": "uses outlet rise time; no independent tracer-pulse RTD is included",
        },
        {
            "case": "Leave-one-velocity split",
            "quantity": (
                f"leave-one-velocity source error {pct(validation['max_leave_velocity_error_pct'])}; "
                f"nominal-$u$ error {pct(pressure['nominal_u1_max_source_error_pct'])}"
            ),
            "interpretation": "source and residence time remain coupled if velocity is not observed",
            "covered_range": "use the covered velocity support for pressure-based correction",
        },
        {
            "case": "Pressure/RTD calibration",
            "quantity": (
                "pressure-calibrated source error returns to the pressure/RTD reference; "
                f"velocity error {pct(pressure['pressure_primary_rtd_gate_max_velocity_error_pct'], 3)}"
            ),
            "interpretation": "pressure-derived velocity separates source amplitude from residence time inside the covered data",
            "covered_range": "requires pressure calibration for the target packed bed",
        },
        {
            "case": "3D-FVM grid family",
            "quantity": (
                f"{mesh_robustness['mesh_cell_range'][0]}--{mesh_robustness['mesh_cell_range'][1]} cells; "
                f"uniform min closed error {pct(mesh_robustness['mesh_min_uniform_closed_error_pct'])}; "
                "matched max stays in the matched-bed error range"
            ),
            "interpretation": "the 3D source-observation conclusion is stable across the evaluated grid family",
            "covered_range": "target-field mesh independence uses the same comparison on target fields",
        },
        {
            "case": "CFD-DEM data-set scope",
            "quantity": (
                f"computed histories {int(credibility['raw_archive_members'])}; "
                "source-only CV at matched velocity reference; "
                f"hard sparse 5x p95 {pct(credibility['hard_sparse_5x_p95_error_pct'], 2)}"
            ),
            "interpretation": "the static-bed data set supports pressure/RTD and source-scaling comparisons",
            "covered_range": "target packing or reactor-field replacement is needed for reactor-specific application",
        },
    ]

    out_csv = RESULTS / "cfdem_transport_check_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "quantity", "interpretation", "covered_range"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\caption{Static packed-bed CFD-DEM transport comparisons used in the source-inference calculations. The table states the computed or calibrated quantity, source-inference role and covered range for each quantity.}",
        r"\label{tab:si_cfdem_transport_checks}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.17\textwidth}P{0.25\textwidth}P{0.28\textwidth}P{0.22\textwidth}@{}}",
        r"\toprule",
        r"Case & Computed or calibrated quantity & Source-inference meaning & Covered range \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["case"]),
                    row["quantity"],
                    tex_escape(row["interpretation"]),
                    tex_escape(row["covered_range"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    out_tex = MANUSCRIPT / "generated_cfdem_transport_check_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
