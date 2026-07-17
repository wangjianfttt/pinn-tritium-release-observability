#!/usr/bin/env python3
"""Build sensitivity table for the 5\% source-error reference.

The table uses the already generated minimum-measurement-set results.  It does
not introduce new simulations; it compares the paper's measurement ordering
depends on choosing 5% as the comparison scale.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"
INPUT = RESULTS / "minimum_measurement_set_table.csv"
OUT_CSV = RESULTS / "source_error_threshold_sensitivity_table.csv"
OUT_JSON = RESULTS / "source_error_threshold_sensitivity_summary.json"
OUT_TEX = MANUSCRIPT / "generated_source_error_threshold_sensitivity_table.tex"

METRICS = [
    "local_source_p95_pct",
    "module_inventory_half_width_pct",
    "system_2pct_p95_pct",
]

DISPLAY = {
    "single mixed total": "single mixed total",
    "field-calibrated spatial HT/HTO": "spatial HT/HTO",
    "field-calibrated spatial HT/HTO + inventory": "spatial HT/HTO + inventory",
    "field-calibrated spatial HT/HTO + inventory + TES": "spatial HT/HTO + inventory + TES",
    "pooled field + inventory + TES": "pooled field + inventory + TES",
    "nearest mismatched field + inventory + TES": "nearest mismatched field + inventory + TES",
}


def fmt_pct(value: float) -> str:
    return f"{value:.2f}\\%"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def metric_triplet(row: pd.Series) -> str:
    return "/".join(fmt_pct(float(row[col])) for col in METRICS)


def main() -> None:
    df = pd.read_csv(INPUT)
    thresholds = [2.0, 5.0, 10.0]
    rows: list[dict[str, str | int | float]] = []

    for threshold in thresholds:
        complete = df[(df[METRICS] <= threshold).all(axis=1)].copy()
        still_incomplete = df[(df[METRICS] > threshold).any(axis=1)].copy()
        if complete.empty:
            best_idx = df[METRICS].max(axis=1).idxmin()
            best = df.loc[best_idx]
            complete_sets = "none"
            reference_values = metric_triplet(best)
            interpretation = (
                f"best row remains above this scale: {DISPLAY[best['measurement_set']]}"
            )
        else:
            complete["max_metric"] = complete[METRICS].max(axis=1)
            best = complete.sort_values("max_metric").iloc[0]
            complete_sets = "; ".join(
                DISPLAY[str(name)] for name in complete["measurement_set"].tolist()
            )
            reference_values = metric_triplet(best)
            interpretation = (
                "first complete row uses spatial/species histories plus balance information"
                if threshold <= 5.0
                else "spatial/species histories need retained inventory for this looser scale"
            )

        rows.append(
            {
                "comparison_scale_pct": threshold,
                "complete_rows": int(len(complete)),
                "complete_measurement_sets": complete_sets,
                "best_local_module_system_pct": reference_values,
                "incomplete_rows": int(len(still_incomplete)),
                "interpretation": interpretation,
            }
        )

    RESULTS.mkdir(exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input": str(INPUT.relative_to(ROOT)),
        "metrics": METRICS,
        "thresholds_pct": thresholds,
        "rows": rows,
        "conclusion": (
            "The single mixed outlet and mismatched-field rows remain incomplete "
            "for 2%, 5% and 10%.  At 5%, the first complete row is spatial "
            "HT/HTO + inventory + TES."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Sensitivity of the 5\% source-error reference. Complete rows satisfy local source p95, module-inventory half-width and system p95 at 2\% balance noise.}",
        r"\label{tab:source_error_threshold_sensitivity}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.11\textwidth}P{0.11\textwidth}P{0.35\textwidth}P{0.18\textwidth}Y@{}}",
        r"\toprule",
        r"Scale & Complete rows & Complete measurement sets & Best source/module/system & Interpretation \\",
        r"\midrule",
    ]
    for row in rows:
        scale = fmt_pct(float(row["comparison_scale_pct"]))
        complete_rows = str(row["complete_rows"])
        complete_sets = tex_escape(str(row["complete_measurement_sets"]))
        values = str(row["best_local_module_system_pct"])
        interp = tex_escape(str(row["interpretation"]))
        lines.append(f"{scale} & {complete_rows} & {complete_sets} & {values} & {interp} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
