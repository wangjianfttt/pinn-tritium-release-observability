#!/usr/bin/env python3
"""Build the main-text PINN training and validation separation table."""

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


def esc(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def main() -> None:
    science = read_json(
        RESULTS / "pinn_training_scientific_map" / "pinn_training_scientific_map_summary.json"
    )
    informed = read_json(
        RESULTS / "fvm3d_informed_pinn_inverse" / "fvm3d_informed_pinn_inverse_summary.json"
    )["summary"]
    volume = read_json(
        RESULTS / "fvm3d_data_volume_vs_field_physics" / "fvm3d_data_volume_vs_field_physics_summary.json"
    )["summary"]
    selection = read_json(
        RESULTS
        / "fvm3d_training_data_selection_gate"
        / "fvm3d_training_data_selection_gate_summary.json"
    )["summary"]
    commissioning = read_json(
        RESULTS
        / "fvm3d_source_family_commissioning_rescue"
        / "fvm3d_source_family_commissioning_rescue_summary.json"
    )["summary"]
    domain = read_json(
        RESULTS
        / "fvm3d_training_domain_applicability_gate"
        / "fvm3d_training_domain_applicability_gate_summary.json"
    )["summary"]
    external = read_json(
        RESULTS / "external_curve_to_source_claim_law" / "external_curve_to_source_claim_law_summary.json"
    )
    leave_one = read_json(
        RESULTS
        / "real_curve_leave_one_point_predictive"
        / "real_curve_leave_one_point_summary.json"
    )["summary"]
    real_curve_rows = read_csv(
        RESULTS
        / "real_curve_regularized_3d_inverse"
        / "real_curve_regularized_3d_inverse_summary.csv"
    )
    real_curve = {
        (row["prior_mode"], row["protocol_id"]): row for row in real_curve_rows
    }
    weak_matched = real_curve[("weak_real_prior", "matched_3d_ht_hto_spatial")]
    weak_total = real_curve[("weak_real_prior", "negative_total_uniform_basis")]
    weak_wrong = real_curve[("weak_real_prior", "negative_matched_transport_wrong_chemistry")]
    pde_source = read_json(
        RESULTS
        / "fvm3d_pde_mlp_source_operator_recovery"
        / "fvm3d_pde_mlp_source_operator_recovery_summary.json"
    )["summary"]
    source_head = read_json(
        RESULTS / "fvm3d_source_operator_head_recovery" / "fvm3d_source_operator_head_recovery_summary.json"
    )["summary"]
    training_closure = read_json(
        RESULTS
        / "pinn_training_data_science_closure"
        / "pinn_training_data_science_closure_summary.json"
    )

    rows = [
        {
            "question": "1D FVM optimizer reference",
            "fit_data": "reduced Li$_2$TiO$_3$ FVM outlet histories",
            "independent_data": "1\\% noisy outlet histories",
            "main_result": (
                f"FVM closed error {pct(science['same_model_fvm_noisy_closed_error_pct'], 3)}; "
                f"staged PINN max {pct(science['staged_pinn_noisy_closed_error_max_pct'])}; "
                f"full fine-tune max {pct(science['full_finetune_closed_error_max_pct'])}"
            ),
        },
        {
            "question": "Sparse external release curves",
            "fit_data": "Park/FEC HT/HTO points used as a broad delayed-time constraint",
            "independent_data": "leave-one-point prediction and matched/mismatched 3D histories",
            "main_result": (
                f"leave-one median/max {float(leave_one['median_point_abs_error_norm']):.3f}/"
                f"{float(leave_one['max_point_abs_error_norm']):.3f}; "
                f"matched 3D closed error {pct(weak_matched['closed_error_percent'], 3)}; "
                f"total-only RMSE {float(weak_total['holdout_rmse_normalized']):.4f}; "
                f"mismatched chemistry {pct(weak_wrong['closed_error_percent'])}; "
                f"interpolated dense labels {pct(external['pseudo_dense_matched_closed_error_pct'])}"
            ),
        },
        {
            "question": "Withheld 3D transport field",
            "fit_data": f"{informed['n_train_cases']} fixed chemistry-resolved 3D transport bases",
            "independent_data": f"{informed['n_holdout_cases']} withheld 3D wet-bypass field",
            "main_result": (
                "matched 3D closed error returns to the matched-bed error range; "
                f"mismatched chemistry {pct(informed['wrong_chemistry_closed_error_pct'])}"
            ),
        },
        {
            "question": "Field-matched training",
            "fit_data": f"pooled non-matching 3D training set up to {int(volume['pooled_max_train_size'])} histories",
            "independent_data": "inferred transport/chemistry field excluded from pooled training",
            "main_result": (
                f"pooled worst p95 {pct(volume['pooled_best_worst_closed_p95_pct'])}; "
                f"field-matched computed 520 histories {pct(volume['target_520_worst_closed_p95_pct'])}"
            ),
        },
        {
            "question": "3D transport model selection",
            "fit_data": f"{selection['n_holdout_field_source_splits']} field/source withheld splits",
            "independent_data": "3D residual, measured-outlet sensitivity and source estimation evaluated on the same splits",
            "main_result": (
                f"selected closed/zone p95 {pct(selection['admissible_selector_worst_closed_p95_pct'])}/"
                f"{pct(selection['admissible_selector_worst_zone_p95_pct'])}; "
                f"residual-only zone {pct(selection['residual_only_worst_zone_p95_pct'])}; "
                f"pooled closed {pct(selection['pooled_no_gate_worst_closed_p95_pct'])}"
            ),
        },
        {
            "question": "New source-family calibration",
            "fit_data": f"{commissioning['n_base_train']} computed base 3D histories plus new-family pilot histories",
            "independent_data": (
                f"{commissioning['n_family_severity_cells']} severe source-family cells; "
                f"pilot counts {', '.join(str(k) for k in commissioning['pilot_counts'])}"
            ),
            "main_result": (
                f"uncalibrated zone/closed p95 {pct(commissioning['uncalibrated_worst_zone_l1_p95_pct'])}/"
                f"{pct(commissioning['uncalibrated_worst_closed_p95_pct'])}; "
                f"K=16 zone/closed p95 {pct(commissioning['pilot16_worst_zone_l1_p95_pct'])}/"
                f"{pct(commissioning['pilot16_worst_closed_p95_pct'])}; "
                f"first source-zone below 5\\% at K={int(commissioning['first_passing_pilot_count'])}"
            ),
        },
        {
            "question": "Training-field coverage",
            "fit_data": f"{domain['n_train']} 3D histories with {domain['k_neighbours']}-neighbour source/field distance",
            "independent_data": (
                f"{domain['n_validation']} evaluation histories and "
                f"{domain['n_ood_cells']} new-family source/field cells"
            ),
            "main_result": (
                f"evaluation inside domain {100.0 * domain['validation_accept_rate']:.2f}\\%; "
                f"severe high-error capture {100.0 * domain['pathological_failed_sample_capture_rate']:.2f}\\%; "
                f"in-domain severe closed/zone p95 {pct(domain['pathological_worst_accepted_closed_p95_pct'])}/"
                f"{pct(domain['pathological_worst_accepted_zone_l1_p95_pct'])}"
            ),
        },
        {
            "question": "Source recovery from the PDE field",
            "fit_data": "PDE-constrained concentration field and explicit source-term layer",
            "independent_data": f"{source_head['n_holdout_cases']} leave-one-3D-case source evaluations",
            "main_result": (
                f"field-only PDE differentiation error {pct(pde_source['median_pde_total_multiplier_error_pct'])}; "
                f"explicit source-term RMS ratio {source_head['median_head_source_rms_ratio']:.3f}"
            ),
        },
        {
            "question": "Solid-inventory sensitivity",
            "fit_data": "measured outlets, kernels and calibration variables",
            "independent_data": "solid inventory unavailable to outlet prediction",
            "main_result": (
                f"full experiment-visible set {pct(training_closure['legitimate_full_worst_closed_p95_pct'])}; "
                "solid-inventory representation closes the representation error; "
                f"prediction without that input {pct(training_closure['hidden_missing_worst_closed_p95_pct'])}"
            ),
        },
    ]

    out_csv = RESULTS / "main_pinn_validation_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{PINN training, differentiable inverse and source-estimation calculations. Each row separates the data used to fit or constrain the inverse from the histories kept for prediction or solid-inventory sensitivity. The fixed-basis 3D release-layer row evaluates withheld curve prediction; the trainable neural rows evaluate PDE-field fitting, source-term estimation and solid-inventory representation sensitivity.}",
        r"\label{tab:pinn_validation_separation}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.18\textwidth}P{0.24\textwidth}P{0.25\textwidth}Y@{}}",
        r"\toprule",
        r"Calculation & Fitting or constraint data & Prediction data & Source-recovery result \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            " & ".join(
                [
                    row["question"],
                    row["fit_data"],
                    row["independent_data"],
                    row["main_result"],
                ]
            )
            + r" \\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_main_pinn_validation_table.tex"
    out_tex.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
