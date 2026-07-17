#!/usr/bin/env python3
"""Build a table for 3D residence-time/kernel uncertainty sensitivity."""

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
    return f"{float(value):.{digits}f}\\%"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def main() -> None:
    kernel = read_json(
        RESULTS
        / "component_3d_neural_kernel_mismatch"
        / "component_3d_neural_kernel_mismatch_summary.json"
    )["summary"]
    wall = read_json(
        RESULTS / "wall_bypass_radial_rtd_gate" / "wall_bypass_radial_rtd_summary.json"
    )["summary"]
    cfdem = read_json(RESULTS / "cfdem_oracle_validation" / "cfdem_oracle_validation_summary.json")

    rows = [
        {
            "perturbation": "nominal 3D response",
            "case": "same computed kernel family used for training and evaluation",
            "effect": f"worst closed p95 {pct(kernel['nominal_worst_closed_p95_pct'])}",
            "decision": "matched-bed consistency reference",
        },
        {
            "perturbation": "unmodelled slow kernel",
            "case": "nominal training, 25\\% slower residence-time kernel at evaluation",
            "effect": f"worst closed p95 {pct(kernel['slow_mismatch_worst_closed_p95_pct'])}",
            "decision": "large residence-time mismatch case",
        },
        {
            "perturbation": "unmodelled fast kernel",
            "case": "nominal training, 20\\% faster residence-time kernel at evaluation",
            "effect": f"worst closed p95 {pct(kernel['fast_mismatch_worst_closed_p95_pct'])}",
            "decision": "large residence-time mismatch case",
        },
        {
            "perturbation": "field-specific slow/fast kernels",
            "case": "same slow/fast kernel used for field-matched training and evaluation",
            "effect": (
                f"slow/fast worst closed p95 {pct(kernel['slow_matched_worst_closed_p95_pct'])}/"
                f"{pct(kernel['fast_matched_worst_closed_p95_pct'])}"
            ),
            "decision": "reduced by field-specific kernel calibration",
        },
        {
            "perturbation": "wall-bypass RTD shape",
            "case": "annular wall-bypass computed RTD versus average transport",
            "effect": (
                f"uniform closed p95 {pct(wall['uniform_closed_p95_pct'])}; "
                "RTD closed p95 near the calibrated-RTD result"
            ),
            "decision": "requires RTD or equivalent radial transport measurement",
        },
        {
            "perturbation": "velocity-support extrapolation",
            "case": "static-bed CFD-DEM leave-one-velocity cross-validation",
            "effect": f"leave-velocity source error {pct(cfdem['max_leave_velocity_error_pct'])}",
            "decision": "velocity support is checked for the target bed before source estimation is used",
        },
    ]

    out_csv = RESULTS / "kernel_uncertainty_sensitivity_table.csv"
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
        r"\caption{Residence-time response transfer tests defining the field-calibrated source-recovery scale.}",
        r"\label{tab:kernel_uncertainty}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.20\textwidth}P{0.33\textwidth}P{0.24\textwidth}Y@{}}",
        r"\toprule",
        r"Residence-time response & Perturbation case & Source effect & Role in source estimation \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["perturbation"]),
                    row["case"],
                    row["effect"],
                    tex_escape(row["decision"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_kernel_uncertainty_sensitivity_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
