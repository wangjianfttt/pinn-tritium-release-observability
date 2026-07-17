#!/usr/bin/env python3
"""Build a manuscript table for generated-tritium source-map uncertainty."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"
SOURCE_DIR = RESULTS / "neutronics_source_shape_accountancy"


def pct(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}\\%"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def read_summary_rows() -> list[dict[str, str]]:
    with (SOURCE_DIR / "neutronics_source_shape_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def protocol(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    for row in rows:
        if row["protocol"] == name:
            return row
    raise KeyError(name)


def main() -> None:
    source_summary = json.loads(
        (SOURCE_DIR / "neutronics_source_shape_summary.json").read_text(encoding="utf-8")
    )["summary"]
    rows = read_summary_rows()

    table_rows = [
        {
            "case": "mixed total outlet",
            "source_constraint": "source free in the mixed release fit",
            "source_l1": float(protocol(rows, "mixed_total_free_source_release")["source_l1_p95_pct"]),
            "closed": float(protocol(rows, "mixed_total_free_source_release")["closed_p95_pct"]),
            "tail": float(protocol(rows, "mixed_total_free_source_release")["tail_p95_pct"]),
            "interpretation": "generation map and delayed release are not separated",
        },
        {
            "case": "spatial HT/HTO, wrong source map",
            "source_constraint": "fixed nominal source distribution",
            "source_l1": float(
                protocol(rows, "spatial_hthto_wrong_nominal_neutronics")["source_l1_p95_pct"]
            ),
            "closed": float(
                protocol(rows, "spatial_hthto_wrong_nominal_neutronics")["closed_p95_pct"]
            ),
            "tail": float(
                protocol(rows, "spatial_hthto_wrong_nominal_neutronics")["tail_p95_pct"]
            ),
            "interpretation": "wrong generated-source shape is absorbed by release multipliers",
        },
        {
            "case": "spatial HT/HTO, heat/activation proxy",
            "source_constraint": "source-map proxy only",
            "source_l1": float(
                protocol(rows, "spatial_hthto_heat_activation_proxy")["source_l1_p95_pct"]
            ),
            "closed": float(
                protocol(rows, "spatial_hthto_heat_activation_proxy")["closed_p95_pct"]
            ),
            "tail": float(protocol(rows, "spatial_hthto_heat_activation_proxy")["tail_p95_pct"]),
            "interpretation": "source shape improves; release remains biased",
        },
        {
            "case": "source proxy plus release/balance",
            "source_constraint": "proxy, release reference, inventory and TES",
            "source_l1": float(
                protocol(rows, "source_proxy_release_reference_inventory_tes")[
                    "source_l1_p95_pct"
                ]
            ),
            "closed": float(
                protocol(rows, "source_proxy_release_reference_inventory_tes")[
                    "closed_p95_pct"
                ]
            ),
            "tail": float(
                protocol(rows, "source_proxy_release_reference_inventory_tes")["tail_p95_pct"]
            ),
            "interpretation": "source map, release multiplier and retained tail are separated",
        },
    ]

    out_csv = RESULTS / "neutronics_source_shape_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Generated-tritium source-map uncertainty in the reduced component calculation. The rows perturb the spatial source distribution that would normally come from neutronics or an experimentally constrained heat/activation proxy. The reported values are p95 errors over "
        + f"{int(source_summary['n_trials'])}"
        + r" computed trials.}",
        r"\label{tab:neutronics_source_shape}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.20\textwidth}P{0.24\textwidth}P{0.12\textwidth}P{0.12\textwidth}P{0.10\textwidth}Y@{}}",
        r"\toprule",
        r"Observation case & Source information & Source-map $L_1$ p95 & Closed p95 & Tail p95 & Meaning \\",
        r"\midrule",
    ]
    for row in table_rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["case"]),
                    tex_escape(row["source_constraint"]),
                    pct(row["source_l1"]),
                    pct(row["closed"]) if row["closed"] < 999.0 else r"$>10^3$\%",
                    pct(row["tail"]),
                    tex_escape(row["interpretation"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_neutronics_source_shape_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
