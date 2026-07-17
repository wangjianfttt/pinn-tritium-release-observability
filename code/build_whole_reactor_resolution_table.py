#!/usr/bin/env python3
"""Build the whole-reactor observation-resolution table for the SI."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"

SRC = (
    RESULTS
    / "whole_reactor_diagnostic_resolution_convergence"
    / "whole_reactor_diagnostic_resolution_convergence.csv"
)
SUMMARY = (
    RESULTS
    / "whole_reactor_diagnostic_resolution_convergence"
    / "whole_reactor_diagnostic_resolution_convergence_summary.json"
)
OUT_CSV = RESULTS / "whole_reactor_resolution_table.csv"
OUT_TEX = MANUSCRIPT / "generated_whole_reactor_resolution_table.tex"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}\\%"


def fmt_signal(value: float) -> str:
    value = float(value)
    if abs(value) < 1e-4:
        return r"\(<10^{-4}\)"
    return f"{value:.3f}"


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if not SUMMARY.exists():
        raise FileNotFoundError(SUMMARY)

    rows = read_rows(SRC)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))["summary"]
    selected_counts = {1, 3, 12, 18, 72, 288}
    selected = [row for row in rows if int(row["group_count"]) in selected_counts]
    if len(selected) != len(selected_counts):
        found = {int(row["group_count"]) for row in selected}
        raise ValueError(f"missing group counts: {sorted(selected_counts - found)}")

    out_rows: list[dict[str, str]] = []
    for row in selected:
        group_count = int(row["group_count"])
        out_rows.append(
            {
                "group_count": str(group_count),
                "components_per_group_mean": f"{float(row['components_per_group_mean']):.1f}",
                "relative_detection_signal": fmt_signal(row["relative_detection_signal"]),
                "cancellation_fraction_pct": pct(100.0 * float(row["cancellation_fraction"]), 1),
                "nonzero_group_count": row["nonzero_group_count"],
                "fixed_budget_coverage_pct": pct(row["new_failure_coverage_pct_top_budget"], 1),
            }
        )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\caption{Observation-resolution convergence in the generic 288-component reduced tritium-balance calculation. The table groups the same component-level slow-tail perturbation into progressively finer outlet signals.}",
        r"\label{tab:si_whole_reactor_resolution}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.13\textwidth}P{0.17\textwidth}P{0.19\textwidth}P{0.18\textwidth}P{0.16\textwidth}Y@{}}",
        r"\toprule",
        r"Groups & Components per group & Relative signal & Cancellation & Nonzero groups & 12-outlet coverage \\",
        r"\midrule",
    ]
    for row in out_rows:
        lines.append(
            " & ".join(
                [
                    row["group_count"],
                    row["components_per_group_mean"],
                    row["relative_detection_signal"],
                    row["cancellation_fraction_pct"],
                    row["nonzero_group_count"],
                    row["fixed_budget_coverage_pct"],
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            (
                r"\vspace{2pt}\footnotesize Aggregate release is below the displayed signal scale "
                + r"because positive and negative local tail shifts cancel. "
                + r"Twelve grouped outlet signals cover the selected local slow-tail regions in this case; "
                + r"eighteen grouped outlet signals give nonzero slow-tail response in all active groups. "
                + r"The calculation is a reduced observation-resolution study on the current 288-component field."
            ),
            r"\end{table}",
            "",
        ]
    )
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
