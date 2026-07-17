#!/usr/bin/env python3
"""Build the main-text data-source/use table."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def main() -> None:
    rows = [
        {
            "data_class": "1D FVM release histories",
            "truth_status": "computed reduced-equation histories",
            "fit_role": "condition design and same-equation inverse reference",
            "selection_role": "outlet-only SVD/profile analysis",
            "heldout_role": "withheld reduced-condition cases",
            "limit": "reduced-model source identification; packed-bed transport is evaluated separately",
        },
        {
            "data_class": "3D-FVM packed-bed histories",
            "truth_status": "computed 3D packed-bed response family",
            "fit_role": "3D residence-time kernels and field-matched PINN fitting",
            "selection_role": "transport-response family selection",
            "heldout_role": "withheld 3D fields and release histories",
            "limit": "computed packed-bed transport; named designs require target fields",
        },
        {
            "data_class": "Static CFD-DEM histories",
            "truth_status": "computed fixed-bed transport response",
            "fit_role": "pressure/RTD sensitivity and transport-response calculations",
            "selection_role": "velocity scale, source scaling and residence-time comparison",
            "heldout_role": "velocity, source-location and sparse-outlet comparisons inside the data set",
            "limit": "static bed only; no moving pebbles, thermal expansion or reactor-specific manifold",
        },
        {
            "data_class": "Park/FEC Li2TiO3 HT/HTO points",
            "truth_status": "digitized external experimental curve",
            "fit_role": "low-weight time-scale and species-fraction constraint",
            "selection_role": "field-matched reference-history selection",
            "heldout_role": "leave-one-point predictive check",
            "limit": "8 points constrain effective time scale; pathway separation uses designed histories",
        },
        {
            "data_class": "TDS curves",
            "truth_status": "digitized literature spectra",
            "fit_role": "mechanism support without dense PINN labels",
            "selection_role": "delayed high-temperature pathway prior",
            "heldout_role": "mechanism consistency check",
            "limit": "mechanism support only; no isothermal closed-pathway calibration",
        },
        {
            "data_class": "Component and generic 288-component reduced balance histories",
            "truth_status": "computed generic 288-component reduced tritium-balance histories",
            "fit_role": "module inventory and extraction-balance propagation",
            "selection_role": "minimum measurement set and sector-resolution calculations",
            "heldout_role": "withheld dynamic balance cases",
            "limit": "generic reduced source-estimation calculation; reactor-specific fields are required for blanket performance estimates",
        },
    ]

    out_csv = RESULTS / "data_source_use_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\caption{Data provenance and validation hierarchy for the source inverse. The table separates author-generated numerical truth, digitized external measurements, fitting roles, model-selection roles and held-out checks.}",
        r"\label{tab:data_source_use}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.20\textwidth}P{0.32\textwidth}P{0.22\textwidth}Y@{}}",
        r"\toprule",
        r"Data class & Inverse use & Prediction check & Use range \\",
        r"\midrule",
    ]
    for row in rows:
        inverse_use = (
            f"{row['truth_status']}; {row['fit_role']}; "
            f"{row['selection_role']}"
        )
        lines.append(
            " & ".join(
                [
                    tex_escape(row["data_class"]),
                    tex_escape(inverse_use),
                    tex_escape(row["heldout_role"]),
                    tex_escape(row["limit"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_data_source_use_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
