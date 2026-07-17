#!/usr/bin/env python3
"""Trace headline manuscript numbers to current generated artifacts.

This compact check covers the numbers that appear in the abstract, conclusions
and main figure captions. It complements the broader consistency checker without
reintroducing the older project-wide tracking workflow.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "headline_claim_traceability.csv"
SUMMARY = RESULTS / "headline_claim_traceability.json"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def close(value: float, target: float, tol: float = 0.01) -> bool:
    return abs(float(value) - target) <= tol


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct_from_text(text: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\\?%", text)
    if not match:
        raise ValueError(f"no percentage found in {text!r}")
    return float(match.group(1))


def add(
    rows: list[dict[str, str]],
    claim_id: str,
    marker: str,
    source_artifact: str,
    source_field: str,
    source_value: str,
    ok: bool,
    note: str,
) -> None:
    manuscript = read(MANUSCRIPT / "manuscript_iop_submission_closure_en.tex")
    marker_present = marker in manuscript
    source_paths = [item.strip() for item in source_artifact.split(";")]
    sources_present = all((ROOT / item).exists() for item in source_paths)
    rows.append(
        {
            "claim_id": claim_id,
            "status": "PASS" if ok and marker_present and sources_present else "FAIL",
            "manuscript_marker": marker,
            "marker_present": str(marker_present),
            "source_artifact": source_artifact,
            "source_field": source_field,
            "source_value": source_value,
            "note": note,
        }
    )


def main() -> int:
    rows: list[dict[str, str]] = []

    obs = load_csv(RESULTS / "observation_ambiguity_table.csv")
    release_row = next(row for row in obs if row["ambiguity"] == "Open and delayed release")
    one_d_row = next(row for row in obs if row["ambiguity"] == "Packed-bed residence time")
    li2_start = pct_from_text(release_row["under_observed_result"])
    li2_end = pct_from_text(release_row["separated_result"])
    one_d_error = pct_from_text(one_d_row["under_observed_result"])

    add(
        rows,
        "abstract_li2_condition_design",
        "7.29\\% to 0.57\\%",
        "results/observation_ambiguity_table.csv",
        "Open and delayed release: under_observed_result -> separated_result",
        f"{li2_start:.2f}% -> {li2_end:.2f}%",
        close(li2_start, 7.29) and close(li2_end, 0.57),
        "Li2TiO3 late-tail/slow-fast design headline.",
    )
    add(
        rows,
        "abstract_1d_on_3d_transport_bias",
        "32.1\\% closed-pathway error",
        "results/same_operator_cross_operator_failure_experiment/same_operator_cross_operator_failure_summary.json",
        "one_d_on_3d_closed_p95_pct",
        f"{one_d_error:.2f}%",
        close(one_d_error, 32.07),
        "1D averaged inverse applied to heterogeneous 3D histories.",
    )

    kernel_rows = load_csv(RESULTS / "kernel_uncertainty_sensitivity_table.csv")
    field_row = next(row for row in kernel_rows if row["perturbation"] == "field-specific slow/fast kernels")
    slow_row = next(row for row in kernel_rows if row["perturbation"] == "unmodelled slow kernel")
    fast_row = next(row for row in kernel_rows if row["perturbation"] == "unmodelled fast kernel")
    field_values = [float(v) for v in re.findall(r"([0-9]+(?:\.[0-9]+)?)\\?%", field_row["effect"])]
    slow_error = pct_from_text(slow_row["effect"])
    fast_error = pct_from_text(fast_row["effect"])
    add(
        rows,
        "abstract_kernel_calibrated_scale",
        "field-calibrated recovery scale of about 3--4\\%",
        "results/kernel_uncertainty_sensitivity_table.csv",
        "field-calibrated slow/fast responses: effect",
        "/".join(f"{v:.2f}%" for v in field_values),
        len(field_values) == 2 and close(field_values[0], 3.21) and close(field_values[1], 3.93),
        "Field-calibrated residence-time-response error range.",
    )
    add(
        rows,
        "abstract_severe_kernel_mismatch",
        "Severe unmodelled slow/fast residence-time mismatch drives errors above 60\\%",
        "results/kernel_uncertainty_sensitivity_table.csv",
        "unmodelled slow/fast residence-time response: effect",
        f"{slow_error:.2f}%/{fast_error:.2f}%",
        slow_error > 60.0 and fast_error > 60.0,
        "Severe residence-time mismatch bound.",
    )

    pinn = json.loads(read(RESULTS / "pinn_training_validation_extrapolation_law" / "pinn_training_validation_extrapolation_law_summary.json"))["summary"]
    add(
        rows,
        "figure_physical_split_underestimate",
        "Random partitions within one release mechanism give 3.2\\% p95 error",
        "results/pinn_training_validation_extrapolation_law/pinn_training_validation_extrapolation_law_summary.json",
        "random_split_source_p95_pct/source_family_holdout_p95_pct/matched_3d_operator_p95_pct",
        f"{pinn['random_split_source_p95_pct']:.2f}%/{pinn['source_family_holdout_p95_pct']:.2f}%/{pinn['matched_3d_operator_p95_pct']:.2f}%",
        close(pinn["random_split_source_p95_pct"], 3.20)
        and close(pinn["source_family_holdout_p95_pct"], 280.98)
        and close(pinn["matched_3d_operator_p95_pct"], 2.47),
        "Physical split result retained in the Results figure caption.",
    )

    pde_source = json.loads(
        read(RESULTS / "fvm3d_pde_mlp_source_operator_recovery" / "fvm3d_pde_mlp_source_operator_recovery_summary.json")
    )["summary"]
    source_head = json.loads(
        read(RESULTS / "fvm3d_source_operator_head_recovery" / "fvm3d_source_operator_head_recovery_summary.json")
    )["summary"]
    pde_error = float(pde_source["median_pde_total_multiplier_error_pct"])
    head_gain = float(source_head["min_head_vs_pde_source_rms_reduction"])
    add(
        rows,
        "main_pinn_source_output_negative_control",
        "Direct differentiation of a trained concentration field gives a large median source-multiplier error",
        "results/fvm3d_pde_mlp_source_operator_recovery/fvm3d_pde_mlp_source_operator_recovery_summary.json; results/fvm3d_source_operator_head_recovery/fvm3d_source_operator_head_recovery_summary.json",
        "median_pde_total_multiplier_error_pct/min_head_vs_pde_source_rms_reduction",
        f"{pde_error:.2f}%/{head_gain:.2f}x",
        close(pde_error, 93.84) and close(head_gain, 2.57, 0.01),
        "PDE-field differentiation negative control and explicit source-term output layer result in the Results text.",
    )

    wr = json.loads(read(RESULTS / "whole_reactor_dynamic_tritium_observability" / "whole_reactor_dynamic_tritium_observability_summary.json"))["summary"]
    aggregate_acc = 100.0 * float(wr["aggregate_total_top_tail_accuracy"])
    sector_acc = float(wr["sector_release_top_tail_accuracy"]) * 100.0
    add(
        rows,
        "abstract_sector_localization",
        "aggregate release histories identify the slow-tail sector with 16.7\\% accuracy, whereas sector release-tail histories reach 96.0\\%",
        "results/whole_reactor_dynamic_tritium_observability/whole_reactor_dynamic_tritium_observability_summary.json",
        "aggregate_total_top_tail_accuracy/sector_release_top_tail_accuracy",
        f"{aggregate_acc:.1f}%/{sector_acc:.1f}%",
        close(aggregate_acc, 16.7, 0.05) and close(sector_acc, 96.0, 0.05),
        "Generic 288-component sector-localization headline.",
    )

    minimum_rows = load_csv(RESULTS / "minimum_measurement_set_table.csv")
    usable = next(row for row in minimum_rows if row["main_decision"] == "usable")
    half_width = float(usable["module_inventory_half_width_pct"])
    add(
        rows,
        "abstract_module_inventory_half_width",
        "module-inventory half-width to about 2.6\\%",
        "results/minimum_measurement_set_table.csv",
        "usable row: module_inventory_half_width_pct",
        f"{half_width:.2f}%",
        close(half_width, 2.56),
        "Generic system measurement-set propagation headline.",
    )

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_pass = sum(1 for row in rows if row["status"] == "PASS")
    SUMMARY.write_text(
        json.dumps({"n_claims": len(rows), "n_pass": n_pass, "all_pass": n_pass == len(rows)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"HEADLINE_CLAIM_TRACEABILITY=PASS={n_pass}/{len(rows)} output={OUTPUT.relative_to(ROOT)}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
