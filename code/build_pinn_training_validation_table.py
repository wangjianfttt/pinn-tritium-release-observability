#!/usr/bin/env python3
"""Generate a compact PINN fitting and withheld-case table for the SI."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}\\%"


def esc(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def hidden_leakage_from_release_summary(release: dict) -> dict[str, float]:
    hidden = release.get("hidden_leakage")
    if hidden is not None:
        return hidden
    closure = read_json(
        RESULTS
        / "pinn_training_data_science_closure"
        / "pinn_training_data_science_closure_summary.json"
    )
    return {
        "hidden_oracle_available_worst_closed_p95_pct": closure[
            "hidden_oracle_worst_closed_p95_pct"
        ],
        "hidden_oracle_missing_worst_closed_p95_pct": closure[
            "hidden_missing_worst_closed_p95_pct"
        ],
    }


def main() -> None:
    science = read_json(
        RESULTS / "pinn_training_scientific_map" / "pinn_training_scientific_map_summary.json"
    )
    informed = read_json(
        RESULTS / "fvm3d_informed_pinn_inverse" / "fvm3d_informed_pinn_inverse_summary.json"
    )["summary"]
    pde_leave_one = read_json(
        RESULTS
        / "fvm3d_trainable_pde_mlp_leave_one_case"
        / "fvm3d_trainable_pde_mlp_leave_one_case_summary.json"
    )["summary"]
    pde_source = read_json(
        RESULTS
        / "fvm3d_pde_mlp_source_operator_recovery"
        / "fvm3d_pde_mlp_source_operator_recovery_summary.json"
    )["summary"]
    source_head = read_json(
        RESULTS
        / "fvm3d_source_operator_head_recovery"
        / "fvm3d_source_operator_head_recovery_summary.json"
    )["summary"]
    release = read_json(
        RESULTS / "pinn_training_release_evidence" / "pinn_training_release_evidence_summary.json"
    )
    hidden_leakage = hidden_leakage_from_release_summary(release)
    classical = read_json(
        RESULTS
        / "component_3d_neural_classical_baseline"
        / "component_3d_neural_classical_baseline_summary.json"
    )["summary"]

    rows = [
        {
            "role": "Reduced-model representation",
            "fit_information": f"{science['n_training_rows']} reduced-model rows; solid-state histories only for representation comparison",
            "withheld_or_stress_test": "separate from the sparse external-curve fit",
            "result": f"reduced-model FVM noisy closed error {pct(science['same_model_fvm_noisy_closed_error_pct'], 3)}; staged PINN {pct(science['staged_pinn_noisy_closed_error_max_pct'])}",
        },
        {
            "role": "Sparse external curves",
            "fit_information": f"{science['n_external_primary_training_rows']} primary PINN training rows from external curves",
            "withheld_or_stress_test": f"{science['n_external_weak_prior_rows']} time-scale constraint; no interpolated dense label",
            "result": "time-scale/mechanism support only",
        },
        {
            "role": "3D-FVM-informed inverse",
            "fit_information": f"{informed['n_train_cases']} 3D field families used for fitting",
            "withheld_or_stress_test": f"{informed['n_holdout_cases']} withheld field family",
            "result": f"matched 3D closed error returns to the matched-bed error range; mismatched chemistry {pct(informed['wrong_chemistry_closed_error_pct'])}",
        },
        {
            "role": "3D PDE-residual fit",
            "fit_information": f"{pde_leave_one['n_train_data_rows']} data rows plus {pde_leave_one['n_train_residual_rows']} residual rows",
            "withheld_or_stress_test": f"{pde_leave_one['n_holdout_cases']} leave-one-3D-case splits; {pde_leave_one['n_eval_rows']} evaluation rows",
            "result": f"PDE residual gain {pde_leave_one['min_pde_residual_gain']:.3f}--{pde_leave_one['max_pde_residual_gain']:.3f}; worst field-RMSE ratio {pde_leave_one['max_field_rmse_ratio']:.3f}",
        },
        {
            "role": "Source-term recovery",
            "fit_information": "PDE-derived source from concentration fields is the comparison path",
            "withheld_or_stress_test": f"{source_head['n_holdout_cases']} withheld 3D source-term splits",
            "result": f"PDE-derived source multiplier error {pct(pde_source['median_pde_total_multiplier_error_pct'])}; explicit source RMS ratio {source_head['median_head_source_rms_ratio']:.3f}",
        },
        {
            "role": "Solid-inventory sensitivity",
            "fit_information": "experiment-visible measurements only for the source inverse",
            "withheld_or_stress_test": "solid inventory unavailable to outlet prediction",
            "result": f"solid-inventory representation closes the representation error; prediction without that input {pct(hidden_leakage['hidden_oracle_missing_worst_closed_p95_pct'])}",
        },
        {
            "role": "Classical-vs-neural comparison",
            "fit_information": "same 3D component features and family splits",
            "withheld_or_stress_test": f"{classical['n_families']} leave-family evaluations",
            "result": f"full vector: linear {pct(classical['linear_full_worst_closed_p95_pct'])}, neural {pct(classical['neural_full_worst_closed_p95_pct'])}; total only: linear {pct(classical['linear_total_accountancy_worst_closed_p95_pct'])}, neural {pct(classical['neural_total_accountancy_worst_closed_p95_pct'])}",
        },
    ]

    out_csv = RESULTS / "pinn_training_validation_separation.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\caption{PINN fitting inputs and withheld physical calculations. The table separates reduced-model representation comparison, sparse external curves, 3D transport comparisons, source-term estimation and solid-inventory sensitivity.}",
        r"\label{tab:si_pinn_training_validation}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.17\textwidth}P{0.28\textwidth}P{0.23\textwidth}X@{}}",
        r"\toprule",
        r"Calculation & Fitting information & Withheld physical case & Key result \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    esc(row["role"]),
                    esc(row["fit_information"]),
                    esc(row["withheld_or_stress_test"]),
                    row["result"],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    (MANUSCRIPT / "generated_pinn_training_validation_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"wrote {out_csv}")
    print(f"wrote {MANUSCRIPT / 'generated_pinn_training_validation_table.tex'}")


if __name__ == "__main__":
    main()
