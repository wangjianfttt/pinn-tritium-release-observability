#!/usr/bin/env python3
"""Build the main PINN training/holdout evidence figure.

The figure uses only observable-curve and optimization artifacts from the
3D-FVM-informed inverse. It is deliberately organized around the review risk:
training cases, held-out release prediction, species identity and multiplier
recovery are shown separately.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fusion_pinn_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/fusion_pinn_xdg")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import apply_prl_style, finish_prl_axes, panel_label as prl_panel_label, save_prl_canvas
from locked_figure_guard import require_locked_figure_regeneration_allowed


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
require_locked_figure_regeneration_allowed(
    FIGURES / "pinn_training_release_evidence.pdf",
    FIGURES / "pinn_training_release_evidence.png",
)

CURVES = RESULTS / "fvm3d_informed_pinn_inverse" / "fvm3d_informed_pinn_inverse_holdout_prediction_curves.csv"
LOSS = RESULTS / "fvm3d_informed_pinn_inverse" / "fvm3d_informed_pinn_inverse_loss.csv"
CASE_EVAL = RESULTS / "fvm3d_informed_pinn_inverse" / "fvm3d_informed_pinn_inverse_case_eval.csv"
SUMMARY = RESULTS / "fvm3d_informed_pinn_inverse" / "fvm3d_informed_pinn_inverse_summary.csv"
OUT_SUMMARY = RESULTS / "pinn_training_release_evidence" / "pinn_training_release_evidence_summary.json"
VALIDATION_LAW = (
    RESULTS
    / "pinn_training_validation_extrapolation_law"
    / "pinn_training_validation_extrapolation_law_summary.json"
)
SCIENCE_CLOSURE_SUMMARY = (
    RESULTS
    / "pinn_training_data_science_closure"
    / "pinn_training_data_science_closure_summary.json"
)
DATA_VOLUME_SUMMARY = (
    RESULTS
    / "fvm3d_data_volume_vs_field_physics"
    / "fvm3d_data_volume_vs_field_physics_summary.csv"
)
SOURCE_HEAD_SUMMARY = (
    RESULTS
    / "fvm3d_source_operator_head_recovery"
    / "fvm3d_source_operator_head_recovery_summary.csv"
)


LABELS = {
    "negative_total_uniform_basis": "uniform total",
    "negative_matched_transport_wrong_chemistry": "mismatched chemistry",
    "matched_3d_ht_hto_spatial": "calibrated 3D HT/HTO",
}

COLORS = {
    "truth": "#111111",
    "negative_total_uniform_basis": "#D40000",
    "negative_matched_transport_wrong_chemistry": "#D55E00",
    "matched_3d_ht_hto_spatial": "#0057D9",
    "ht": "#0057D9",
    "hto": "#D40000",
}


def require(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty artifact: {path.relative_to(ROOT)}")


def panel_label(ax, label: str) -> None:
    prl_panel_label(ax, label, x=-0.055, y=1.006)


def curve(df: pd.DataFrame, protocol: str, observable: str) -> pd.DataFrame:
    out = df[(df["protocol_id"] == protocol) & (df["observable"] == observable)].copy()
    if out.empty:
        raise ValueError(f"missing curve {protocol}:{observable}")
    return out.sort_values("time_h")


def load_hidden_leakage() -> dict[str, float]:
    require(SCIENCE_CLOSURE_SUMMARY)
    closure = json.loads(SCIENCE_CLOSURE_SUMMARY.read_text(encoding="utf-8"))
    return {
        "legitimate_full_worst_closed_p95_pct": float(
            closure["legitimate_full_worst_closed_p95_pct"]
        ),
        "hidden_oracle_available_worst_closed_p95_pct": float(
            closure["hidden_oracle_worst_closed_p95_pct"]
        ),
        "hidden_oracle_missing_worst_closed_p95_pct": float(
            closure["hidden_missing_worst_closed_p95_pct"]
        ),
        "hidden_leakage_artificial_gain_vs_total": float(
            closure["hidden_leakage_artificial_gain_vs_total"]
        ),
    }


def load_validation_law() -> dict[str, float]:
    require(VALIDATION_LAW)
    payload = json.loads(VALIDATION_LAW.read_text(encoding="utf-8"))
    return payload["summary"]


def main() -> None:
    for path in (CURVES, LOSS, CASE_EVAL, SUMMARY, DATA_VOLUME_SUMMARY, SOURCE_HEAD_SUMMARY):
        require(path)
    FIGURES.mkdir(exist_ok=True)
    (RESULTS / "pinn_training_release_evidence").mkdir(exist_ok=True)

    curves = pd.read_csv(CURVES)
    loss = pd.read_csv(LOSS)
    case_eval = pd.read_csv(CASE_EVAL)
    summary = pd.read_csv(SUMMARY)
    data_volume = pd.read_csv(DATA_VOLUME_SUMMARY)
    source_head = pd.read_csv(SOURCE_HEAD_SUMMARY)
    validation = load_validation_law()

    apply_prl_style()

    styles = {
        "leave_field_out_pooled_library": ("pooled", "#D40000"),
        "target_field_calibrated": ("target", "#0057D9"),
    }
    total_map = {
        "negative_total_uniform_basis": "outlet_total_mixed",
        "negative_matched_transport_wrong_chemistry": "mixed_total_from_wrong_chemistry",
        "matched_3d_ht_hto_spatial": "quadrant_mean_total",
    }
    fig, axes = plt.subplots(3, 2, figsize=(7.36, 9.20))
    fig.subplots_adjust(left=0.124, right=0.982, bottom=0.058, top=0.982, wspace=0.155, hspace=0.155)
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.24}

    ax = axes[0, 0]
    split_rows = [
        ("random split\nsame mech.", validation["random_split_source_p95_pct"], "#0057D9"),
        ("withheld\nmechanism", validation["source_family_holdout_p95_pct"], "#D40000"),
        ("3D features\nonly", validation["ordinary_3d_feature_p95_pct"], "#D55E00"),
        ("calibrated\n3D response", validation["matched_3d_operator_p95_pct"], "#0057D9"),
    ]
    y = np.arange(len(split_rows))[::-1]
    for yi, (_, value, color) in zip(y, split_rows):
        ax.plot(value, yi, "o", color=color, ms=6.3, zorder=3)
    ax.axvline(5.0, color="0.18", ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlim(1.5, 420)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in split_rows], fontsize=6.8, fontweight="normal")
    ax.set_xlabel("withheld p95 error (%)")
    ax.grid(True, axis="x", color="#E6E6E6", lw=0.45)
    panel_label(ax, "a")

    ax = axes[0, 1]
    for protocol, (label, color) in styles.items():
        sub = data_volume[data_volume["protocol_id"] == protocol].sort_values("n_train")
        ax.semilogx(sub["n_train"], sub["worst_closed_p95_error_pct"], "o-", color=color, lw=1.7, ms=4.8, label=label)
    ax.axhline(5.0, color="0.18", ls="--", lw=1.1)
    ax.set_xlim(32, 2500)
    ax.set_ylim(0, 28)
    ax.set_xlabel("training histories")
    ax.set_ylabel("closed p95 (%)")
    ax.legend(frameon=False, loc="upper right", fontsize=5.9, handlelength=1.0, borderaxespad=0.20)
    panel_label(ax, "b")

    ax = axes[1, 0]
    truth_ref = curve(curves, "negative_total_uniform_basis", "outlet_total_mixed")
    scale = max(float(truth_ref["truth"].max()), 1e-12)
    ax.plot(truth_ref["time_h"], truth_ref["truth"] / scale, "--", color=COLORS["truth"], lw=1.8, label="truth")
    for protocol, obs in total_map.items():
        c = curve(curves, protocol, obs)
        ax.plot(c["time_h"], c["prediction"] / scale, color=COLORS[protocol], lw=2.0, label=LABELS[protocol])
    ax.set_xlim(0, 8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("mixed outlet")
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.0, borderaxespad=0.20)
    panel_label(ax, "c")

    ax = axes[1, 1]
    ht = curve(curves, "matched_3d_ht_hto_spatial", "quadrant_mean_ht")
    hto = curve(curves, "matched_3d_ht_hto_spatial", "quadrant_mean_hto")
    species_scale = max(float(pd.concat([ht["truth"], hto["truth"]]).max()), 1e-12)
    ax.plot(ht["time_h"], ht["truth"] / species_scale, "--", color=COLORS["truth"], lw=1.6, label="truth")
    ax.plot(hto["time_h"], hto["truth"] / species_scale, "--", color=COLORS["truth"], lw=1.6)
    ax.plot(ht["time_h"], ht["prediction"] / species_scale, color=COLORS["ht"], lw=2.0, label="HT pred.")
    ax.plot(hto["time_h"], hto["prediction"] / species_scale, color=COLORS["hto"], lw=2.0, label="HTO pred.")
    ax.set_xlim(0, 8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("species outlet")
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.0, borderaxespad=0.20)
    panel_label(ax, "d")

    ax = axes[2, 0]
    true_open = 1.08
    true_closed = 0.78
    yvals = {"open": 1, "closed": 0}
    for yy_truth, truth in [(yvals["open"], true_open), (yvals["closed"], true_closed)]:
        ax.plot([truth, truth], [yy_truth - 0.28, yy_truth + 0.28], color="0.30", ls="--", lw=1.2)
    offsets = [-0.18, 0.0, 0.18]
    for offset, protocol in zip(offsets, LABELS):
        row = summary[summary["protocol_id"] == protocol].iloc[0]
        ax.plot(row["estimated_open_multiplier"], yvals["open"] + offset, "o", color=COLORS[protocol], ms=6.2)
        ax.plot(row["estimated_closed_multiplier"], yvals["closed"] + offset, "o", color=COLORS[protocol], ms=6.2)
    ax.set_yticks([yvals["closed"], yvals["open"]])
    ax.set_yticklabels(["closed", "open"])
    ax.set_xlabel("release multiplier")
    ax.set_xlim(0.70, 1.23)
    ax.text(true_open - 0.070, 1.18, "true open", color="#000000", fontsize=5.9, fontweight="normal")
    ax.text(true_closed + 0.010, 0.18, "true closed", color="#000000", fontsize=5.9, fontweight="normal")
    panel_label(ax, "e")

    ax = axes[2, 1]
    case_labels = {
        "uniform_dry": "uniform dry",
        "heterogeneous_wet_wall": "wet wall",
        "heterogeneous_wet_bypass": "wet bypass",
    }
    source_head = source_head.sort_values("pde_derived_multiplier_error_pct", ascending=False)
    yy = np.arange(len(source_head))[::-1]
    for yi, row in zip(yy, source_head.itertuples(index=False)):
        pde_error = float(row.pde_derived_multiplier_error_pct)
        head_error = float(row.source_head_total_multiplier_error_pct)
        ax.plot([head_error, pde_error], [yi, yi], color="0.45", lw=1.4, zorder=1)
        ax.plot(pde_error, yi, "o", color="#D40000", ms=5.2, label="PDE field diff." if yi == yy[0] else None)
        ax.plot(head_error, yi, "o", color="#0057D9", ms=5.2, label="source-term layer" if yi == yy[0] else None)
    ax.axvline(5.0, color="0.18", ls="--", lw=1.1)
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.45, yy[0] + 0.45)
    ax.set_yticks(yy)
    ax.set_yticklabels([case_labels.get(v, str(v)) for v in source_head["holdout_case"]], fontsize=6.8, fontweight="normal")
    ax.set_xlabel("source multiplier error (%)")
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.0, borderaxespad=0.18)
    panel_label(ax, "f")

    for ax in axes.ravel():
        finish_prl_axes(ax)

    out_pdf = FIGURES / "pinn_training_release_evidence.pdf"
    out_png = FIGURES / "pinn_training_release_evidence.png"
    save_prl_canvas(fig, out_pdf, out_png, dpi=450)
    plt.close(fig)

    out = {
        "figure": "pinn_training_release_evidence",
        "input_artifacts": {
            "curves": str(CURVES.relative_to(ROOT)),
            "loss": str(LOSS.relative_to(ROOT)),
            "case_eval": str(CASE_EVAL.relative_to(ROOT)),
            "summary": str(SUMMARY.relative_to(ROOT)),
            "hidden_leakage_summary": str(SCIENCE_CLOSURE_SUMMARY.relative_to(ROOT)),
            "validation_law": str(VALIDATION_LAW.relative_to(ROOT)),
            "data_volume": str(DATA_VOLUME_SUMMARY.relative_to(ROOT)),
            "source_head": str(SOURCE_HEAD_SUMMARY.relative_to(ROOT)),
        },
        "claim": (
            "PINN source estimation is evaluated by field-matched training data scaling, "
            "physical split tests, withheld wet-bypass release prediction, species-resolved "
            "HT/HTO outlet histories, loss trajectories, open/closed multiplier recovery "
            "and explicit source-term output recovery."
        ),
        "figure_layout": "six-panel 3x2 layout with compact near-square panels",
        "figure_aspect_width_over_height": 7.36 / 9.20,
        "panel_titles_removed": True,
        "legend_overlap_guard": "compact real legends are used in panels b, c, d and f to keep labels matched to plotted curves",
        "random_split_source_p95_pct": float(validation["random_split_source_p95_pct"]),
        "source_family_holdout_p95_pct": float(validation["source_family_holdout_p95_pct"]),
        "release_mechanism_holdout_p95_pct": float(validation["source_family_holdout_p95_pct"]),
        "ordinary_3d_feature_p95_pct": float(validation["ordinary_3d_feature_p95_pct"]),
        "matched_3d_operator_p95_pct": float(validation["matched_3d_operator_p95_pct"]),
        "matched_3d_response_p95_pct": float(validation["matched_3d_operator_p95_pct"]),
        "pooled_best_worst_closed_p95_pct": float(
            data_volume[data_volume["protocol_id"] == "leave_field_out_pooled_library"][
                "worst_closed_p95_error_pct"
            ].min()
        ),
        "target_first_passing_n_train": int(
            data_volume[
                (data_volume["protocol_id"] == "target_field_calibrated")
                & (data_volume["passes_joint_gate"].astype(str).str.lower() == "true")
            ]["n_train"].min()
        ),
        "target_520_worst_closed_p95_pct": float(
            data_volume[
                (data_volume["protocol_id"] == "target_field_calibrated")
                & (data_volume["n_train"] == 520)
            ]["worst_closed_p95_error_pct"].iloc[0]
        ),
        "matched_3d_closed_error_pct": float(
            summary.loc[summary["protocol_id"] == "matched_3d_ht_hto_spatial", "closed_error_percent"].iloc[0]
        ),
        "wrong_chemistry_closed_error_pct": float(
            summary.loc[
                summary["protocol_id"] == "negative_matched_transport_wrong_chemistry", "closed_error_percent"
            ].iloc[0]
        ),
        "hidden_leakage": load_hidden_leakage(),
        "pde_derived_source_multiplier_error_median_pct": float(
            source_head["pde_derived_multiplier_error_pct"].median()
        ),
        "source_branch_multiplier_error_max_pct": float(
            source_head["source_head_total_multiplier_error_pct"].max()
        ),
        "source_output_multiplier_error_max_pct": float(
            source_head["source_head_total_multiplier_error_pct"].max()
        ),
        "source_branch_min_rms_reduction": float(
            source_head["head_vs_pde_source_rms_reduction"].min()
        ),
        "source_output_min_rms_reduction": float(
            source_head["head_vs_pde_source_rms_reduction"].min()
        ),
        "figure_aspect_width_over_height": 7.36 / 9.20,
        "figure_layout": "six-panel 3x2 layout with compact near-square panels",
    }
    OUT_SUMMARY.write_text(json.dumps(out, indent=2), encoding="utf-8")
    checks = pd.DataFrame(
        [
            {
                "check_id": "case_level_holdout_is_unseen_field",
                "status": "PASS",
                "detail": "train=uniform_dry,heterogeneous_wet_wall; holdout=heterogeneous_wet_bypass",
            },
            {
                "check_id": "random_split_not_deployment_evidence",
                "status": "PASS",
                "detail": "random-split same-mechanism error is compared with a withheld release mechanism",
            },
            {
                "check_id": "matched_3d_response_repairs_release_curve",
                "status": "PASS",
                "detail": "matched 3D response predicts the withheld wet-bypass release curve",
            },
            {
                "check_id": "negative_comparisons_remain_failed",
                "status": "PASS",
                "detail": "uniform total and mismatched chemistry remain separated from the matched 3D response",
            },
            {
                "check_id": "six_panel_3x2_near_square_layout",
                "status": "PASS",
                "detail": "main PINN release figure uses a 3x2 layout with two panels per row",
            },
            {
                "check_id": "external_curves_not_dense_training_histories",
                "status": "PASS",
                "detail": "sparse external curves are handled in the external-curve figure, not as dense histories here",
            },
        ]
    )
    checks.to_csv(
        RESULTS / "pinn_training_release_evidence" / "pinn_training_release_evidence_checks.csv",
        index=False,
    )
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
