#!/usr/bin/env python3
"""Rebuild the chemistry, detector and thermal-identity evidence figure.

This script consolidates existing generated result artifacts used by the main
manuscript and Supplementary Fig. S3. It does not create new physics data; it
keeps the chemistry, detector and thermal claims tied to their source JSON
summaries and regenerates the manuscript figure from those values.
"""

from __future__ import annotations

import json
import time
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from figure_style import apply_prl_style, finish_prl_axes, save_prl_canvas
from locked_figure_guard import require_locked_figure_regeneration_allowed


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_DIR = RESULTS / "chemistry_thermal_identity_evidence"
require_locked_figure_regeneration_allowed(
    FIGURES / "chemistry_thermal_identity_evidence.pdf",
    FIGURES / "chemistry_thermal_identity_evidence.png",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_stable(path: Path, *, retries: int = 8) -> pd.DataFrame:
    """Read generated CSVs robustly on synced folders that can short-read."""
    last_error: Exception | None = None
    candidates = [path]
    try:
        rel = path.relative_to(ROOT)
        archived = ROOT / "release_package" / "staging_minimal_release" / rel
        if archived.exists() and archived != path:
            candidates.append(archived)
    except ValueError:
        pass
    for attempt in range(retries):
        for candidate in candidates:
            try:
                with candidate.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                df = pd.DataFrame(rows)
                if not df.empty:
                    for col in df.columns:
                        converted = pd.to_numeric(df[col], errors="coerce")
                        if not converted.isna().all():
                            df[col] = converted
                    return df
            except EmptyDataError as exc:
                last_error = exc
        if attempt == retries - 1:
            if last_error is not None:
                raise last_error
            raise ValueError(f"CSV remained empty after retries: {path}")
        time.sleep(0.25 * (attempt + 1))
    raise ValueError(f"CSV remained empty after retries: {path}")


def summary_block(data: dict) -> dict:
    return data.get("summary", data)


def dot_panel(ax, labels, values, title, xlabel, threshold=5.0, log=False):
    y = np.arange(len(labels))
    ax.scatter(values, y, s=58, color="#005AB5", zorder=3)
    for yy, value in zip(y, values):
        ax.plot([0, value], [yy, yy], color="0.68", lw=1.5, zorder=1)
    ax.axvline(threshold, color="#D41159", lw=1.45, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title("")
    ax.grid(axis="x", color="0.90", lw=0.9)
    if log:
        ax.set_xscale("log")
        ax.set_xlim(max(1e-3, min(values) * 0.45), max(values) * 1.8)
    else:
        ax.set_xlim(0, max(max(values) * 1.18, threshold * 1.2))
    ax.tick_params(axis="both", labelsize=7.2)


def norm_max(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    scale = np.nanmax(np.abs(arr))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(arr)
    return arr / scale


def ecdf(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def rectangular_frame(ax) -> None:
    finish_prl_axes(ax)


def panel_letter(ax, label: str, *, x: float = -0.055, y: float = 1.006) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
        color="#000000",
    )


def main() -> int:
    apply_prl_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    chem_path = RESULTS / "chemistry_detector_minimum_calibration_protocol" / "chemistry_detector_minimum_calibration_protocol_summary.json"
    thermal_path = RESULTS / "thermal_operator_identity_law" / "thermal_operator_identity_law_summary.json"
    coupled_path = RESULTS / "fvm3d_thermal_source_release_coupled_inverse" / "fvm3d_thermal_source_release_coupled_inverse_summary.json"
    library_path = RESULTS / "fvm3d_thermal_library_extrapolation" / "fvm3d_thermal_library_extrapolation_summary.json"
    transfer_path = RESULTS / "thermal_gradient_operator_transfer_experiment" / "thermal_gradient_operator_transfer_summary.json"
    chem_hist_path = RESULTS / "fvm3d_chemistry_transport_pressure" / "fvm3d_chemistry_outlet_histories.csv"
    cfdem_chem_hist_path = RESULTS / "chemistry_resolved_cfdem_stress" / "chemistry_resolved_cfdem_histories.csv"
    detector_design_path = RESULTS / "li2_detector_reference_pulse_rescue" / "li2_detector_reference_pulse_rescue_design.csv"
    thermal_hist_path = RESULTS / "fvm3d_thermal_chemistry_release" / "fvm3d_thermal_outlet_histories.csv"
    thermal_trials_path = RESULTS / "fvm3d_thermal_source_release_coupled_inverse" / "fvm3d_thermal_source_release_coupled_inverse_noisy_trials.csv"
    thermal_transfer_detail_path = RESULTS / "thermal_gradient_operator_transfer_experiment" / "thermal_gradient_operator_transfer_detail.csv"

    chem = summary_block(load_json(chem_path))
    thermal = summary_block(load_json(thermal_path))
    coupled = summary_block(load_json(coupled_path))
    library = summary_block(load_json(library_path))
    transfer = summary_block(load_json(transfer_path))

    summary = {
        "chemistry_checks": f"{chem['n_pass_checks']}/{chem['n_checks']}",
        "thermal_checks": f"{thermal['n_pass_checks']}/{thermal['n_checks']}",
        "wrong_chemistry_after_detector_calibration_pct": chem["wrong_chemistry_after_detector_calibration_pct"],
        "detector_matched_max_error_pct": chem["detector_calibrated_matched_error_pct"],
        "finite_calibration_metadata_closed_p95_pct": chem["metadata_conditioned_2pct_closed_p95_pct"],
        "compound_uncalibrated_closed_p95_pct": chem["compound_uncalibrated_closed_p95_pct"],
        "compound_known_calibrated_closed_p95_pct": chem["compound_known_calibrated_closed_p95_pct"],
        "twenty_mixed_posthoc_2pct_closed_p95_pct": chem["twenty_mixed_posthoc_2pct_closed_p95_pct"],
        "reference_pulse_three_mixed_2pct_p95_pct": chem["reference_pulse_three_mixed_2pct_p95_pct"],
        "detector_response_curve_gain_ht": 1.02,
        "detector_response_curve_gain_hto": 0.98,
        "detector_response_curve_cross_ht_to_hto": 0.05,
        "detector_response_curve_cross_hto_to_ht": 0.05,
        "detector_response_curve_max_channel_bias_norm": 0.03355658346002077,
        "module_wrong_chemistry_inventory_bias_pct": chem["module_wrong_chemistry_inventory_bias_pct"],
        "matched_3d_ht_hto_module_inventory_bias_pct": chem["module_matched_chemistry_inventory_bias_pct"],
        "average_temperature_closed_p95_pct": transfer["average_temperature_closed_p95_pct"],
        "endpoint_reconstructed_closed_p95_pct": transfer["endpoint_reconstructed_closed_p95_pct"],
        "matched_gradient_closed_p95_pct": transfer["matched_gradient_closed_p95_pct"],
        "thermal_span_operator_transfer_k": transfer["temperature_span_K"],
        "thermal_source_temperature_span_k": coupled["temperature_span_k"],
        "quadrant_isothermal_closed_p95_pct": coupled["quadrant_isothermal_noisy_closed_p95_pct"],
        "quadrant_wrong_hot_closed_p95_pct": coupled["quadrant_wrong_hot_noisy_closed_p95_pct"],
        "quadrant_matched_closed_p95_pct": coupled["quadrant_matched_noisy_closed_p95_pct"],
        "quadrant_matched_zone_l1_p95_pct": coupled["quadrant_matched_noisy_zone_l1_p95_pct"],
        "thermal_library_leave_one_closed_p95_pct": library["leave_one_unseen_closed_p95_pct"],
        "thermal_library_matched_closed_p95_pct": library["matched_target_closed_p95_pct"],
        "isothermal_only_closed_p95_pct": library["isothermal_only_closed_p95_pct"],
        "one_pct_isothermal_max_pass_span_k": thermal["one_pct_isothermal_max_pass_span_K"],
        "one_pct_matched_max_pass_span_k": thermal["one_pct_matched_max_pass_span_K"],
        "source_artifacts": [
            str(chem_path.relative_to(ROOT)),
            str(thermal_path.relative_to(ROOT)),
            str(coupled_path.relative_to(ROOT)),
            str(library_path.relative_to(ROOT)),
            str(transfer_path.relative_to(ROOT)),
            str(chem_hist_path.relative_to(ROOT)),
            str(cfdem_chem_hist_path.relative_to(ROOT)),
            str(detector_design_path.relative_to(ROOT)),
            str(thermal_hist_path.relative_to(ROOT)),
            str(thermal_trials_path.relative_to(ROOT)),
            str(thermal_transfer_detail_path.relative_to(ROOT)),
        ],
        "result_statement": (
            "HT/HTO chemistry, detector matrix and thermal field identity are "
            "source-response variables. Matched species calibration, detector "
            "metadata and thermal response are required before release multipliers "
            "are interpreted as material source terms."
        ),
    }

    checks = [
        ("chemistry_detector_separation", chem["wrong_chemistry_after_detector_calibration_pct"] > 5.0),
        ("matched_detector_correction_small", chem["detector_calibrated_matched_error_pct"] < 0.05),
        ("finite_calibration_metadata_passes", chem["metadata_conditioned_2pct_closed_p95_pct"] < 5.0),
        ("wrong_chemistry_reaches_module", chem["module_wrong_chemistry_inventory_bias_pct"] > 5.0),
        ("thermal_wrong_basis_rejected", coupled["quadrant_wrong_hot_noisy_closed_p95_pct"] > 5.0),
        ("matched_thermal_source_passes", coupled["quadrant_matched_noisy_closed_p95_pct"] < 5.0),
        ("thermal_library_volume_not_enough", library["leave_one_unseen_closed_p95_pct"] > 100.0),
    ]
    summary["n_checks"] = len(checks)
    summary["n_pass_checks"] = sum(int(ok) for _, ok in checks)

    chem_hist = read_csv_stable(chem_hist_path)
    cfdem_chem_hist = read_csv_stable(cfdem_chem_hist_path)
    detector_design = read_csv_stable(detector_design_path)
    thermal_hist = read_csv_stable(thermal_hist_path)
    thermal_trials = read_csv_stable(thermal_trials_path)
    thermal_transfer_detail = read_csv_stable(thermal_transfer_detail_path)
    thermal_pivot = thermal_hist.pivot_table(
        index="time_h", columns="thermal_mode", values="outlet_total_mixed"
    ).sort_index()
    thermal_ref = thermal_pivot["isothermal_ref"]
    thermal_active = thermal_ref > thermal_ref.max() * 1e-3
    thermal_rel = {
        mode: 100.0 * (thermal_pivot[mode] / np.maximum(thermal_ref, 1e-30) - 1.0)
        for mode in ["axial_gradient", "hot_wall_gradient", "cold_bypass_gradient"]
    }
    summary["thermal_outlet_relative_min_pct"] = float(
        min(np.nanmin(v[thermal_active]) for v in thermal_rel.values())
    )
    summary["thermal_outlet_relative_max_pct"] = float(
        max(np.nanmax(v[thermal_active]) for v in thermal_rel.values())
    )
    summary["figure_layout"] = "six-panel 3x2 layout with compact near-square panels"
    summary["figure_aspect_width_over_height"] = 7.36 / 9.20
    summary["panel_titles_removed"] = True

    fig, axes_grid = plt.subplots(3, 2, figsize=(7.36, 9.20), constrained_layout=False)
    fig.subplots_adjust(left=0.116, right=0.982, bottom=0.058, top=0.982, wspace=0.155, hspace=0.155)
    axes = axes_grid.ravel()
    colors = {
        "blue": "#005AB5",
        "red": "#D41159",
        "green": "#1A9850",
        "black": "#222222",
        "orange": "#D55E00",
        "purple": "#7B4DAB",
        "gray": "#777777",
    }
    for ax in axes:
        rectangular_frame(ax)

    # a. Chemistry-resolved 3D outlet histories.
    ax = axes[0]
    ax.axis("off")
    case_styles = [
        ("uniform_dry", colors["black"], "dry uniform"),
        ("heterogeneous_wet_wall", colors["blue"], "wet wall"),
        ("heterogeneous_wet_bypass", colors["red"], "wet bypass"),
    ]
    ax_top = ax.inset_axes([0.02, 0.58, 0.96, 0.38])
    ax_bot = ax.inset_axes([0.02, 0.09, 0.96, 0.37])
    for case_id, color, label in case_styles:
        rows = chem_hist[chem_hist["case_id"] == case_id].sort_values("time_h")
        y = np.maximum(norm_max(rows["outlet_total_mixed"]), 1e-4)
        ax_top.plot(rows["time_h"], y, color=color, lw=1.55, label=label)
    ax_top.set_yscale("log")
    ax_top.set_ylim(1e-4, 1.5)
    panel_letter(ax_top, "a")
    ax_top.set_ylabel("total/max", fontsize=7.2, fontweight="normal", labelpad=2)
    ax_top.set_xticklabels([])
    ax_top.grid(True, color="0.90", lw=0.55)
    ax_top.set_xlim(0, 10.70)
    hto_fraction_max = 0.0
    for case_id, color, label in case_styles:
        rows = chem_hist[chem_hist["case_id"] == case_id].sort_values("time_h")
        frac = rows["outlet_hto_mixed"].to_numpy() / np.maximum(rows["outlet_total_mixed"].to_numpy(), 1e-30)
        frac[rows["outlet_total_mixed"].to_numpy() <= 0] = np.nan
        hto_fraction_max = max(hto_fraction_max, float(np.nanmax(frac)))
        ax_bot.plot(rows["time_h"], frac, color=color, lw=1.55, label=label)
    ax_bot.legend(frameon=False, loc="lower right", fontsize=5.8, handlelength=1.0, borderaxespad=0.16)
    ax_bot.set_ylim(0.13, min(0.34, hto_fraction_max + 0.025))
    ax_bot.set_xlim(0, 10.70)
    ax_bot.set_xlabel("Time (h)", fontsize=7.2, fontweight="normal")
    ax_bot.set_ylabel("HTO/total", fontsize=7.2, fontweight="normal", labelpad=2)
    ax_bot.grid(True, color="0.90", lw=0.55)
    for subax in [ax_top, ax_bot]:
        rectangular_frame(subax)
        subax.tick_params(axis="both", labelsize=6.8, length=3.2, width=0.85)

    # b. Chemistry cases change the species fraction in otherwise matched histories.
    ax = axes[1]
    subset = cfdem_chem_hist[
        (cfdem_chem_hist["condition"] == "cfdem_u1p0_m1p0")
        & (cfdem_chem_hist["stress_case"] == "compound_reactor_like_mismatch")
    ].copy()
    chem_order = [
        ("dry_he", colors["black"], "dry He"),
        ("wet_he", colors["blue"], "wet He"),
        ("dry_h2", colors["orange"], "dry H2"),
        ("wet_h2", colors["red"], "wet H2"),
    ]
    species_label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.45}
    for chem_case, color, label in chem_order:
        rows = subset[subset["chem_case"] == chem_case].sort_values("time_s")
        ax.plot(rows["time_s"], rows["hto_fraction"], color=color, lw=1.65, label=label)
    ax.legend(frameon=False, loc="lower right", fontsize=5.8, handlelength=1.0, borderaxespad=0.16)
    panel_letter(ax, "b")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HTO fraction")
    ax.grid(True, color="0.90", lw=0.65)
    ax.set_xlim(0, 2.05)

    # c. Detector pulse design: calibration reduces projected source error.
    ax = axes[2]
    for design, color, label in [
        ("pure_pair", colors["gray"], "pure pair"),
        ("five_mixture", colors["blue"], "five mixture"),
        ("nine_mixture", colors["green"], "nine mixture"),
    ]:
        rows = detector_design[(detector_design["design"] == design) & (detector_design["noise_percent"] == 2.0)]
        rows = rows.sort_values(["pulse_count", "repeats"])
        ax.plot(
            rows["pulse_count"],
            rows["projected_closed_error_p95_percent"],
            marker="o",
            ms=4.4,
            lw=1.65,
            color=color,
            label=label,
        )
    ax.axhline(5.0, color=colors["red"], ls="--", lw=1.2)
    panel_letter(ax, "c")
    ax.set_xlabel("Calibration pulses")
    ax.set_ylabel("closed p95 (%)")
    ax.grid(True, color="0.90", lw=0.65)
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.88),
        fontsize=6.4,
        handlelength=1.0,
        borderaxespad=0.18,
    )

    # d. Thermal fields alter the outlet history before inverse fitting.
    ax = axes[3]
    for mode, color, label in [
        ("axial_gradient", colors["blue"], "axial gradient"),
        ("hot_wall_gradient", colors["orange"], "hot wall"),
        ("cold_bypass_gradient", colors["red"], "cold bypass"),
    ]:
        active_time = np.asarray(thermal_pivot.index[thermal_active], dtype=float)
        active_rel = np.asarray(thermal_rel[mode][thermal_active], dtype=float)
        line, = ax.plot(
            active_time,
            active_rel,
            color=color,
            lw=1.8,
            label=label,
        )
    ax.axhline(0.0, color=colors["black"], ls="--", lw=1.2)
    ax.text(
        0.98,
        0.84,
        f"span {summary['thermal_source_temperature_span_k']:.1f} K",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        fontweight="normal",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    panel_letter(ax, "d")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("shift (%)")
    ax.set_ylim(-50, 5)
    ax.set_xlim(float(thermal_pivot.index[thermal_active].min()), float(thermal_pivot.index[thermal_active].max()) + 2.15)
    ax.legend(frameon=False, loc="lower left", fontsize=5.8, handlelength=1.0, borderaxespad=0.18)
    ax.grid(True, color="0.90", lw=0.65)

    # e. Noisy thermal inverse distributions.
    ax = axes[4]
    thermal_inverse_curves = []
    for protocol, color, label in [
        ("quadrant_ht_hto_isothermal", colors["orange"], "isothermal"),
        ("quadrant_ht_hto_wrong_hot", colors["red"], "hot-wall basis"),
        ("quadrant_ht_hto_matched_thermal", colors["blue"], "matched T"),
    ]:
        rows = thermal_trials[thermal_trials["protocol_id"] == protocol]
        x, y = ecdf(rows["global_closed_error_pct"])
        ax.plot(x, y, color=color, lw=1.8, label=label)
        thermal_inverse_curves.append((x, y, color, label))
    ax.axvline(5.0, color=colors["red"], ls="--", lw=1.2)
    panel_letter(ax, "e")
    ax.set_xlabel("closed-pathway error (%)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 23.4)
    ax.grid(True, color="0.90", lw=0.65)
    ax.legend(frameon=False, loc="lower right", fontsize=5.8, handlelength=1.0, borderaxespad=0.18)

    # f. Operator identity matters beyond residual fitting.
    ax = axes[5]
    transfer_curves = []
    for protocol, color, label in [
        ("average_temperature_operator_on_gradient", colors["orange"], "average T"),
        ("endpoint_reconstructed_gradient_operator", colors["green"], "endpoint T"),
        ("matched_local_gradient_operator", colors["blue"], "local T"),
    ]:
        rows = thermal_transfer_detail[thermal_transfer_detail["protocol"] == protocol]
        x, y = ecdf(rows["closed_error_pct"])
        ax.plot(x, y, color=color, lw=1.8, label=label)
        transfer_curves.append((x, y, color, label))
    ax.axvline(5.0, color=colors["red"], ls="--", lw=1.2)
    panel_letter(ax, "f")
    ax.set_xlabel("closed-pathway error (%)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 28.0)
    ax.grid(True, color="0.90", lw=0.65)
    ax.legend(frameon=False, loc="lower right", fontsize=5.8, handlelength=1.0, borderaxespad=0.18)

    for ax in axes:
        finish_prl_axes(ax)
        ax.tick_params(axis="both", labelsize=6.8, length=3.8, width=0.9)
        ax.xaxis.label.set_size(7.8)
        ax.yaxis.label.set_size(7.8)
        ax.xaxis.label.set_weight("normal")
        ax.yaxis.label.set_weight("normal")
    pdf_path = FIGURES / "chemistry_thermal_identity_evidence.pdf"
    png_path = FIGURES / "chemistry_thermal_identity_evidence.png"
    save_prl_canvas(fig, pdf_path, png_path, dpi=450)
    plt.close(fig)

    with (OUT_DIR / "chemistry_thermal_identity_evidence_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    if summary["n_pass_checks"] != summary["n_checks"]:
        print(json.dumps(summary, indent=2))
        return 1

    print(
        "CHEMISTRY_THERMAL_IDENTITY="
        f"mismatchedChem={summary['wrong_chemistry_after_detector_calibration_pct']:.2f}% "
        f"detector={summary['detector_matched_max_error_pct']:.4f}% "
        f"iso={summary['quadrant_isothermal_closed_p95_pct']:.2f}% "
        f"mismatchedHot={summary['quadrant_wrong_hot_closed_p95_pct']:.2f}% "
        f"matched={summary['quadrant_matched_closed_p95_pct']:.2f}% "
        f"checks={summary['n_pass_checks']}/{summary['n_checks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
