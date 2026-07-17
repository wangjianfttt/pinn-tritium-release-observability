#!/usr/bin/env python3
"""Build a compact definition table for manuscript error metrics."""

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
            "metric": "closed-pathway multiplier error",
            "definition": "relative error of the inferred delayed/closed release multiplier",
            "use": "Li$_2$TiO$_3$ condition design, nuisance profiles and 3D transport comparisons",
        },
        {
            "metric": "source-zone p95",
            "definition": "95th percentile error of local source-zone amplitudes over independent cases",
            "use": "3D source-family, transport-field and measurement-family comparisons",
        },
        {
            "metric": "full-source $L_1$ p95",
            "definition": "95th percentile normalized integral error of the full source map",
            "use": "source-structure and wrong-family rejection comparisons",
        },
        {
            "metric": "tail-fraction p95",
            "definition": "95th percentile error in the retained or delayed release-tail fraction",
            "use": "module and generic 288-component reduced balance calculation",
        },
        {
            "metric": "module-inventory half-width",
            "definition": "half-width of the propagated retained-inventory uncertainty interval",
            "use": "module-scale propagation and retained-inventory measurement value",
        },
        {
            "metric": "system p95",
            "definition": "95th percentile system-level balance or source metric over withheld dynamic cases",
            "use": "generic 288-component reduced tritium-balance calculation",
        },
        {
            "metric": "slow-tail sector accuracy",
            "definition": "classification accuracy for the sector with the largest delayed-tail contribution",
            "use": "sector versus system-total source comparison",
        },
        {
            "metric": "matched 3D-response result",
            "definition": "source error for a specified computed transport response",
            "use": "closed computed comparisons with the declared transport response",
        },
    ]

    RESULTS.mkdir(exist_ok=True)
    out_csv = RESULTS / "error_metric_definition_table.csv"
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
        r"\caption{Error metrics used in the source-estimation and source-comparison calculations.}",
        r"\label{tab:error_metrics}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.23\textwidth}P{0.39\textwidth}Y@{}}",
        r"\toprule",
        r"Metric & Definition & Where it is used \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['metric']} & {row['definition']} & {row['use']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_error_metric_definition_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
