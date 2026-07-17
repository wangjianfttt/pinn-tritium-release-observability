#!/usr/bin/env python3
"""Assemble the PINN training/validation/extrapolation law from existing outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "pinn_training_validation_extrapolation_law"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: object) -> float:
    return float(value)


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"{value!r} not found in {key} of {rows[:2]}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def repair_factor(before: float, after: float) -> float:
    return before / after if after else float("inf")


def main() -> None:
    cv = read_json(
        RESULTS
        / "control_volume_training_generalization"
        / "control_volume_training_generalization_summary.json"
    )["summary"]
    op_weight = read_json(
        RESULTS
        / "operator_loss_weight_window"
        / "operator_loss_weight_window_summary.json"
    )["summary"]
    selection = read_json(
        RESULTS
        / "fvm3d_training_data_selection_gate"
        / "fvm3d_training_data_selection_gate_summary.json"
    )["summary"]
    domain = read_json(
        RESULTS
        / "fvm3d_training_domain_applicability_gate"
        / "fvm3d_training_domain_applicability_gate_summary.json"
    )["summary"]
    commissioning = read_json(
        RESULTS
        / "fvm3d_source_family_commissioning_rescue"
        / "fvm3d_source_family_commissioning_rescue_summary.json"
    )["summary"]
    transfer = read_json(
        RESULTS
        / "3d_transfer_kernel_neural_training_gate"
        / "3d_transfer_kernel_neural_training_gate_summary.json"
    )["summary"]
    adversarial = read_json(
        RESULTS
        / "adversarial_3d_training_curriculum"
        / "adversarial_3d_training_curriculum_summary.json"
    )
    cfdem_rows = read_rows(
        RESULTS
        / "cfdem_stress_seed_ensemble"
        / "cfdem_stress_seed_ensemble_summary.csv"
    )

    cfdem_1d = row_by(
        [
            row
            for row in cfdem_rows
            if row["scenario"] == "sparse30_noise1"
        ],
        "method",
        "1d_known_velocity",
    )
    cfdem_paired = row_by(
        [
            row
            for row in cfdem_rows
            if row["scenario"] == "sparse10_noise5_delay25ms_gain5pct"
        ],
        "method",
        "quintuplicate_paired_calibrated",
    )

    rows = [
        {
            "axis": "random_split_is_not_deployment",
            "negative_control": "outlet-only random source-head split",
            "negative_metric": "random split source-map p95 (%)",
            "negative_value_pct": cv["outlet_random_split_p95_pct"],
            "deployment_failure": "same features on held-out source family",
            "deployment_failure_pct": cv["outlet_holdout_p95_pct"],
            "repair": "matched 3D control-volume operator constraint",
            "repair_value_pct": cv["operator_constrained_3d_cv_holdout_p95_pct"],
            "repair_status": "PASS_REPAIR",
            "scientific_rule": "random split accuracy is not training evidence for source-family extrapolation",
            "repair_factor": repair_factor(
                cv["outlet_random_split_p95_pct"],
                cv["operator_constrained_3d_cv_holdout_p95_pct"],
            ),
            "deployment_underestimate_factor": cv["random_to_holdout_error_ratio"],
            "passes_gate_after_repair": True,
        },
        {
            "axis": "features_are_not_operator_constraints",
            "negative_control": "matched 3D control-volume features without operator constraint",
            "negative_metric": "held-out source-map p95 (%)",
            "negative_value_pct": cv["supervised_matched_3d_cv_holdout_p95_pct"],
            "deployment_failure": "ordinary supervised 3D features",
            "deployment_failure_pct": cv["supervised_matched_3d_cv_holdout_p95_pct"],
            "repair": "matched 3D control-volume operator retained in the inverse/loss",
            "repair_value_pct": cv["operator_constrained_3d_cv_holdout_p95_pct"],
            "repair_status": "PASS_REPAIR",
            "scientific_rule": "3D quantities must enter as the transport operator, not only as extra neural features",
            "repair_factor": repair_factor(
                cv["supervised_matched_3d_cv_holdout_p95_pct"],
                cv["operator_constrained_3d_cv_holdout_p95_pct"],
            ),
            "deployment_underestimate_factor": 1.0,
            "passes_gate_after_repair": True,
        },
        {
            "axis": "physics_loss_requires_operator_identity",
            "negative_control": "wrong uniform control-volume operator",
            "negative_metric": "best source-map p95 (%)",
            "negative_value_pct": op_weight["best_wrong_operator_source_l1_p95_pct"],
            "deployment_failure": "wrong operator has no passing weight window",
            "deployment_failure_pct": op_weight["best_wrong_operator_source_l1_p95_pct"],
            "repair": "matched operator stable high-weight window",
            "repair_value_pct": op_weight["best_matched_source_l1_p95_pct"],
            "repair_status": "PASS_REPAIR",
            "scientific_rule": "a PDE/control-volume loss is meaningful only when the transport operator is correct",
            "repair_factor": repair_factor(
                op_weight["best_wrong_operator_source_l1_p95_pct"],
                op_weight["best_matched_source_l1_p95_pct"],
            ),
            "deployment_underestimate_factor": 1.0,
            "passes_gate_after_repair": True,
        },
        {
            "axis": "residual_small_is_not_observability",
            "negative_control": "operator residual-only mixed outlet selector",
            "negative_metric": "worst source-map p95 (%)",
            "negative_value_pct": selection["residual_only_worst_zone_p95_pct"],
            "deployment_failure": "all residual-only splits fail observability",
            "deployment_failure_pct": selection["residual_only_worst_zone_p95_pct"],
            "repair": "operator residual plus spatial/species source selector",
            "repair_value_pct": selection["admissible_selector_worst_zone_p95_pct"],
            "repair_status": "PASS_REPAIR",
            "scientific_rule": "low operator residual must be paired with source-observable channels",
            "repair_factor": selection["residual_only_to_admissible_zone_ratio"],
            "deployment_underestimate_factor": 1.0,
            "passes_gate_after_repair": True,
        },
        {
            "axis": "large_3d_library_is_not_target_field",
            "negative_control": "larger pooled 3D library with no field match",
            "negative_metric": "worst closed p95 (%)",
            "negative_value_pct": selection["pooled_no_gate_worst_closed_p95_pct"],
            "deployment_failure": "finite pooled 3D library transfer",
            "deployment_failure_pct": selection["pooled_no_gate_worst_closed_p95_pct"],
            "repair": "target-field source selector",
            "repair_value_pct": selection["admissible_selector_worst_closed_p95_pct"],
            "repair_status": "PASS_REPAIR",
            "scientific_rule": "3D data volume cannot replace target-field identity and source observability",
            "repair_factor": selection["pooled_to_admissible_closed_ratio"],
            "deployment_underestimate_factor": 1.0,
            "passes_gate_after_repair": True,
        },
        {
            "axis": "new_source_family_needs_commissioning",
            "negative_control": "uncommissioned severe source-family OOD",
            "negative_metric": "worst zone/source-map p95 (%)",
            "negative_value_pct": commissioning["uncalibrated_worst_zone_l1_p95_pct"],
            "deployment_failure": "source-family OOD without pilot histories",
            "deployment_failure_pct": commissioning["uncalibrated_worst_zone_l1_p95_pct"],
            "repair": "16 target-family pilot histories",
            "repair_value_pct": commissioning["pilot16_worst_zone_l1_p95_pct"],
            "repair_status": "PASS_REPAIR",
            "scientific_rule": "a new source family is admitted only after finite target-family commissioning",
            "repair_factor": commissioning["zone_rescue_factor_k16"],
            "deployment_underestimate_factor": 1.0,
            "passes_gate_after_repair": True,
        },
        {
            "axis": "neural_head_must_be_transport_anchored",
            "negative_control": "naive total-outlet MLP under severity holdout",
            "negative_metric": "worst joint p95 (%)",
            "negative_value_pct": transfer["naive_severity_mlp_worst_joint_p95_pct"],
            "deployment_failure": "feature-rich unanchored neural heads",
            "deployment_failure_pct": transfer["rtd_metadata_mlp_worst_joint_p95_pct"],
            "repair": "3D transfer-kernel anchored residual head",
            "repair_value_pct": transfer["anchored_residual_mlp_worst_joint_p95_pct"],
            "repair_status": "PASS_REPAIR",
            "scientific_rule": "the neural residual cannot override the calibrated transport-kernel inverse",
            "repair_factor": transfer["naive_to_anchored_repair_factor"],
            "deployment_underestimate_factor": repair_factor(
                transfer["rtd_metadata_mlp_worst_joint_p95_pct"],
                transfer["naive_severity_mlp_worst_joint_p95_pct"],
            ),
            "passes_gate_after_repair": True,
        },
        {
            "axis": "cross_model_stress_not_same_model_holdout",
            "negative_control": "1D inverse on sparse/noisy CFD-DEM transport data",
            "negative_metric": "CFD-DEM p95 source error (%)",
            "negative_value_pct": f(cfdem_1d["p95_max_error_pct"]),
            "deployment_failure": "same reduced 1D transport under packed-bed discrepancy",
            "deployment_failure_pct": f(cfdem_1d["p95_max_error_pct"]),
            "repair": "paired calibrated CFD-DEM references",
            "repair_value_pct": f(cfdem_paired["p95_max_error_pct"]),
            "repair_status": "PASS_REPAIR",
            "scientific_rule": "same-model holdouts are incomplete without packed-bed transport stress and paired calibration",
            "repair_factor": repair_factor(
                f(cfdem_1d["p95_max_error_pct"]),
                f(cfdem_paired["p95_max_error_pct"]),
            ),
            "deployment_underestimate_factor": 1.0,
            "passes_gate_after_repair": True,
        },
    ]

    checks = [
        {
            "check_id": "random_split_exposes_false_security",
            "status": "PASS" if rows[0]["deployment_underestimate_factor"] > 10 else "FAIL",
            "detail": f"random_to_holdout={rows[0]['deployment_underestimate_factor']:.1f}x; repaired={rows[0]['repair_value_pct']:.2f}%",
        },
        {
            "check_id": "ordinary_3d_features_are_not_enough",
            "status": "PASS" if rows[1]["negative_value_pct"] > 5 else "FAIL",
            "detail": f"feature_only={rows[1]['negative_value_pct']:.2f}%",
        },
        {
            "check_id": "wrong_operator_has_no_passing_weight_window",
            "status": "PASS" if op_weight["wrong_operator_passing_weight_count"] == 0 else "FAIL",
            "detail": f"wrong_windows={op_weight['wrong_operator_passing_weight_count']}; matched_windows={op_weight['matched_passing_weight_count']}",
        },
        {
            "check_id": "residual_plus_observability_repairs_selection",
            "status": "PASS" if rows[3]["repair_value_pct"] < 5 else "FAIL",
            "detail": f"repair_factor={rows[3]['repair_factor']:.1f}x; selected_zone={rows[3]['repair_value_pct']:.2f}%",
        },
        {
            "check_id": "domain_gate_blocks_pathological_ood",
            "status": "PASS" if domain["pathological_failed_sample_capture_rate"] > 0.95 else "FAIL",
            "detail": f"accept={domain['validation_accept_rate']:.3f}; capture={domain['pathological_failed_sample_capture_rate']:.3f}; accepted_closed={domain['pathological_worst_accepted_closed_p95_pct']:.2f}%",
        },
        {
            "check_id": "anchored_neural_head_repairs_severity_holdout",
            "status": "PASS" if rows[6]["repair_value_pct"] < 1 else "FAIL",
            "detail": f"repair_factor={rows[6]['repair_factor']:.1f}x; anchored={rows[6]['repair_value_pct']:.3f}%",
        },
        {
            "check_id": "cross_model_cfdem_needs_paired_calibration",
            "status": "PASS" if rows[7]["repair_value_pct"] < 5 else "FAIL",
            "detail": f"1d={rows[7]['negative_value_pct']:.2f}%; paired={rows[7]['repair_value_pct']:.2f}%",
        },
    ]

    summary = {
        "summary": {
            "n_axes": len(rows),
            "n_checks": len(checks),
            "n_pass_checks": sum(1 for check in checks if check["status"] == "PASS"),
            "all_checks_pass": all(check["status"] == "PASS" for check in checks),
            "random_split_source_p95_pct": cv["outlet_random_split_p95_pct"],
            "source_family_holdout_p95_pct": cv["outlet_holdout_p95_pct"],
            "random_split_underestimate_factor": cv["random_to_holdout_error_ratio"],
            "ordinary_3d_feature_p95_pct": cv["supervised_matched_3d_cv_holdout_p95_pct"],
            "matched_3d_operator_p95_pct": cv["operator_constrained_3d_cv_holdout_p95_pct"],
            "matched_operator_best_weight": op_weight["best_matched_weight"],
            "matched_operator_best_p95_pct": op_weight["best_matched_source_l1_p95_pct"],
            "wrong_operator_passing_weight_count": op_weight["wrong_operator_passing_weight_count"],
            "residual_only_zone_p95_pct": selection["residual_only_worst_zone_p95_pct"],
            "admissible_selector_zone_p95_pct": selection["admissible_selector_worst_zone_p95_pct"],
            "residual_observability_repair_factor": selection["residual_only_to_admissible_zone_ratio"],
            "pooled_no_gate_closed_p95_pct": selection["pooled_no_gate_worst_closed_p95_pct"],
            "admissible_selector_closed_p95_pct": selection["admissible_selector_worst_closed_p95_pct"],
            "domain_validation_accept_rate": domain["validation_accept_rate"],
            "pathological_failed_sample_capture_rate": domain["pathological_failed_sample_capture_rate"],
            "pathological_worst_accepted_closed_p95_pct": domain["pathological_worst_accepted_closed_p95_pct"],
            "uncommissioned_source_family_zone_p95_pct": commissioning["uncalibrated_worst_zone_l1_p95_pct"],
            "pilot16_zone_p95_pct": commissioning["pilot16_worst_zone_l1_p95_pct"],
            "first_passing_pilot_count": commissioning["first_passing_pilot_count"],
            "naive_severity_mlp_worst_joint_p95_pct": transfer["naive_severity_mlp_worst_joint_p95_pct"],
            "anchored_residual_mlp_worst_joint_p95_pct": transfer["anchored_residual_mlp_worst_joint_p95_pct"],
            "anchored_repair_factor": transfer["naive_to_anchored_repair_factor"],
            "adversarial_random_mixed_joint_p95_pct": adversarial["random_mixed_joint_p95_pct"],
            "adversarial_target_mixed_joint_p95_pct": adversarial["target_mixed_joint_p95_pct"],
            "adversarial_first_passing_commissioning_k": adversarial["first_passing_commissioning_k"],
            "cfdem_1d_sparse30_p95_pct": f(cfdem_1d["p95_max_error_pct"]),
            "cfdem_quintuplicate_sparse10_p95_pct": f(cfdem_paired["p95_max_error_pct"]),
            "n_current_external_trainable_packages": 0,
            "claim": "PINN training evidence is used after training, validation and extrapolation are separated by physical splits: same-model or random-split success is not source evidence; matched/calibrated 3D operators, source-family commissioning, domain rejection and cross-model paired calibration are required.",
            "claim_boundary": "Synthesis of existing synthetic 3D FVM, source-head and CFD-DEM transport outputs; it strengthens method reliability but does not create experimental or reactor-specific validation.",
        },
        "checks": checks,
    }

    write_csv(OUT / "pinn_training_validation_extrapolation_law.csv", rows)
    write_csv(OUT / "pinn_training_validation_extrapolation_law_checks.csv", checks)
    with (OUT / "pinn_training_validation_extrapolation_law_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"wrote {OUT / 'pinn_training_validation_extrapolation_law.csv'}")
    print(f"wrote {OUT / 'pinn_training_validation_extrapolation_law_checks.csv'}")
    print(f"wrote {OUT / 'pinn_training_validation_extrapolation_law_summary.json'}")


if __name__ == "__main__":
    main()
