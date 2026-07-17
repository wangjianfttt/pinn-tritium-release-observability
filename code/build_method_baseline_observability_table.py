#!/usr/bin/env python3
"""Build the main-text method baseline and source-estimation table."""

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


def find_row(rows: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in conditions.items()):
            return row
    raise KeyError(conditions)


def main() -> None:
    inverse_rows = read_csv(RESULTS / "inverse_baseline_comparison.csv")
    exact_fvm = find_row(inverse_rows, method="classical FVM LS: exact outlet")
    noisy_fvm = find_row(inverse_rows, method="classical FVM LS: 1% noisy outlet")
    staged_pinn = find_row(inverse_rows, method="PINN: extended_tail frozen")
    zero_physics = find_row(inverse_rows, method="PINN: physics weight = 0.0")

    profile = read_json(
        RESULTS / "li2_outlet_profile_likelihood" / "li2_outlet_profile_likelihood_summary.json"
    )["summary"]
    classical = read_json(
        RESULTS
        / "component_3d_neural_classical_baseline"
        / "component_3d_neural_classical_baseline_summary.json"
    )["summary"]
    selection = read_json(
        RESULTS
        / "fvm3d_training_data_selection_gate"
        / "fvm3d_training_data_selection_gate_summary.json"
    )["summary"]
    source_profile = read_json(
        RESULTS
        / "fvm3d_source_uncertainty_profile_gate"
        / "fvm3d_source_uncertainty_profile_gate_summary.json"
    )
    threshold = read_json(
        RESULTS / "observability_threshold_analysis" / "observability_threshold_summary.json"
    )

    rows = [
        {
            "calculation": "Same-equation classical inverse",
            "data": "reduced Li$_2$TiO$_3$ extended-tail outlet",
            "result": (
                "exact FVM at matched reduced-equation reference; "
                f"1\\% noisy FVM {pct(float(noisy_fvm['k_closed_error_percent']), 3)}"
            ),
            "meaning": "sets the optimizer reference under the governing equations",
        },
        {
            "calculation": "PINN same-equation comparison",
            "data": "same reduced histories with physics residual",
            "result": (
                f"staged PINN closed error {pct(float(staged_pinn['k_closed_error_percent']))}; "
                f"zero-physics control {pct(float(zero_physics['k_closed_error_percent']))}"
            ),
            "meaning": "the neural inverse uses the same measurement design as the numerical reference",
        },
        {
            "calculation": "Outlet-only profile likelihood",
            "data": "outlet histories with velocity, porosity and detector nuisance terms",
            "result": (
                f"withheld-condition open/closed {pct(profile['max_holdout_open_error_percent'])}/"
                f"{pct(profile['max_holdout_closed_error_percent'])}; broad nuisance surfaces"
            ),
            "meaning": "the outlet fit exchanges release with flow, porosity and detector response",
        },
        {
            "calculation": "Classical and neural 3D baselines",
            "data": "same component features and family splits",
            "result": (
                f"full vector linear/neural {pct(classical['linear_full_worst_closed_p95_pct'])}/"
                f"{pct(classical['neural_full_worst_closed_p95_pct'])}; "
                f"total-only {pct(classical['linear_total_accountancy_worst_closed_p95_pct'])}/"
                f"{pct(classical['neural_total_accountancy_worst_closed_p95_pct'])}"
            ),
            "meaning": "measurement completeness controls both classical and neural inverses",
        },
        {
            "calculation": "3D source-estimation withheld split",
            "data": f"{int(selection['n_holdout_field_source_splits'])} field/source withheld splits",
            "result": (
                f"selected closed/zone {pct(selection['admissible_selector_worst_closed_p95_pct'])}/"
                f"{pct(selection['admissible_selector_worst_zone_p95_pct'])}; "
                f"residual-only zone {pct(selection['residual_only_worst_zone_p95_pct'])}"
            ),
            "meaning": "source estimation uses both a field residual and withheld source prediction",
        },
        {
            "calculation": "Bootstrap/profile source interval",
            "data": f"{int(source_profile['n_bootstrap'])} bootstrap source-map fits",
            "result": (
                f"full source $L_1$ p95 {pct(source_profile['full_source_map_l1_p95_pct'])}; "
                f"interval width {pct(source_profile['full_source_interval_l1_width_p95_pct'])}; "
                f"incompatible families passing 0"
            ),
            "meaning": "source error is reported with interval width and wrong-family comparisons",
        },
        {
            "calculation": "Measurement-set source estimation",
            "data": "single outlet, 12 outlets and complete spatial/species/balance set",
            "result": (
                f"zone uncertainty {pct(threshold['single_mixed_zone_uncertainty_pct'])} "
                f"$\\rightarrow$ {pct(threshold['full12_total_zone_uncertainty_pct'])} "
                f"$\\rightarrow$ {pct(threshold['full_multimodal_zone_uncertainty_pct'])}"
            ),
            "meaning": "spatial, species and balance measurements reduce the source uncertainty",
        },
        {
            "calculation": "System source estimation",
            "data": "module and generic 288-component reduced balance histories",
            "result": (
                f"six-measurement set resolves source only; full measurement set resolves all listed quantities; "
                f"TES uncertainty {pct(threshold['budget96_tes_uncertainty_pct'])}"
            ),
            "meaning": "local source estimates are paired with inventory and TES information",
        },
    ]

    out_csv = RESULTS / "method_baseline_observability_table.csv"
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
        r"\caption{Classical baselines, PINN controls and source-estimation comparisons. The table separates same-equation optimizer accuracy from source estimation under nuisance parameters, 3D transport and system-balance measurements.}",
        r"\label{tab:method_baseline_observability}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.20\textwidth}P{0.25\textwidth}P{0.27\textwidth}Y@{}}",
        r"\toprule",
        r"Calculation & Data or split & Result & Interpretation \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["calculation"]),
                    row["data"],
                    row["result"],
                    tex_escape(row["meaning"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_method_baseline_observability_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
