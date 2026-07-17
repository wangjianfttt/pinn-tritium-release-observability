#!/usr/bin/env python3
"""Build the compact packed-bed transport figure from result artifacts.

This figure is used to support the paper's central method point: a release
multiplier is only meaningful after the packed-bed residence-time response,
velocity scale and HT/HTO chemistry have been constrained.  The panels are
therefore rebuilt from CSV/JSON outputs rather than from hand-entered numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from figure_style import apply_prl_style, finish_prl_axes, panel_label as prl_panel_label, save_prl_canvas
from locked_figure_guard import require_locked_figure_regeneration_allowed


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
require_locked_figure_regeneration_allowed(
    FIGURES / "packed_bed_transport_tests_current.pdf",
    FIGURES / "packed_bed_transport_tests_current.png",
)


def read_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty JSON artifact: {path}")
    return json.loads(text)


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty artifact: {path}")
    return path


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    inverse = pd.read_csv(
        require(RESULTS / "fvm3d_transport_pressure_test" / "fvm3d_inverse_comparison.csv")
    )
    histories = pd.read_csv(
        require(RESULTS / "fvm3d_transport_pressure_test" / "fvm3d_outlet_histories.csv")
    )
    mesh_inverse = pd.read_csv(
        require(RESULTS / "fvm3d_mesh_observation_sensitivity" / "fvm3d_mesh_inverse_comparison.csv")
    )
    rtd = pd.read_csv(require(RESULTS / "cfdem_oracle_validation" / "cfdem_oracle_rtd_proxy.csv"))
    pressure_rows = pd.read_csv(
        require(
            RESULTS
            / "cfdem_pressure_rtd_velocity_rescue"
            / "cfdem_pressure_rtd_velocity_rescue_rows.csv"
        )
    )
    chem = pd.read_csv(
        require(
            RESULTS
            / "chemistry_resolved_cfdem_stress"
            / "chemistry_resolved_cfdem_stress_summary.csv"
        )
    )
    ergun = pd.read_csv(
        require(RESULTS / "cfdem_ergun_envelope" / "cfdem_ergun_pressure_fit.csv")
    )
    ergun_summary = read_json(
        require(RESULTS / "cfdem_ergun_envelope" / "cfdem_ergun_envelope_summary.json")
    )["summary"]
    validation = read_json(
        require(RESULTS / "cfdem_oracle_validation" / "cfdem_oracle_validation_summary.json")
    )
    pressure_summary = read_json(
        require(
            RESULTS
            / "cfdem_pressure_rtd_velocity_rescue"
            / "cfdem_pressure_rtd_velocity_rescue_summary.json"
        )
    )
    pressure_error_curves = pd.read_csv(
        require(
            RESULTS
            / "packed_bed_transport_admissibility_evidence"
            / "pressure_rtd_velocity_source_error_curves.csv"
        )
    )

    apply_prl_style()
    blue = "#0057D9"
    red = "#D40000"
    green = "#008A00"
    black = "#111111"
    orange = "#D55E00"
    purple = "#6A3D9A"

    display_axes = plt.subplots(3, 2, figsize=(7.36, 9.20))[1]
    fig = display_axes.ravel()[0].figure
    axes = display_axes
    fig.subplots_adjust(left=0.116, right=0.978, bottom=0.058, top=0.982, wspace=0.155, hspace=0.155)

    def panel_label(ax, label: str) -> None:
        prl_panel_label(ax, label, x=-0.055, y=1.006)

    # a. Transport mismatch and inferred source bias.
    ax = axes[0, 0]
    mixed_uniform = inverse[
        (inverse["basis_type"] == "uniform_1d_basis") & (inverse["observables"] == "mixed")
    ].copy()
    matched = inverse[
        (inverse["basis_type"] == "matched_3d_basis") & (inverse["observables"] == "q1+q2+q3+q4")
    ].copy()
    ax.semilogy(
        mixed_uniform["rmse_normalized"],
        mixed_uniform["closed_error_percent"],
        "o",
        color=red,
        ms=6.2,
        label="mixed outlet, 1D basis",
    )
    ax.semilogy(
        matched["rmse_normalized"],
        matched["closed_error_percent"],
        "s",
        color=blue,
        ms=5.8,
        label="4 outlets, matched 3D",
    )
    ax.axhline(5, color="0.20", ls="--", lw=1.1)
    ax.set_xlabel("outlet mismatch RMSE")
    ax.set_ylabel("closed error (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=5.9, handlelength=1.0, borderaxespad=0.20)
    panel_label(ax, "a")

    # b. Heterogeneous-bypass inverse protocols as an observation/model matrix.
    ax = axes[0, 1]
    bypass = inverse[inverse["case_id"] == "heterogeneous_bypass"].copy()
    row_defs = [("uniform_1d_basis", "1D basis"), ("matched_3d_basis", "3D basis")]
    col_defs = [("mixed", "mixed"), ("q1+q2+q3+q4", "4 outlets")]
    matrix = np.zeros((len(row_defs), len(col_defs)))
    vals = []
    for i, (basis, _) in enumerate(row_defs):
        for j, (obs, _) in enumerate(col_defs):
            value = float(
                bypass[(bypass["basis_type"] == basis) & (bypass["observables"] == obs)][
                    "closed_error_percent"
                ].iloc[0]
            )
            matrix[i, j] = max(value, 0.02)
            vals.append(value)
    im = ax.imshow(matrix, cmap="magma_r", norm=LogNorm(vmin=0.02, vmax=30), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = vals[i * matrix.shape[1] + j]
            color = "white" if value > 5 else "black"
            ax.text(j, i, f"{value:.2g}%", ha="center", va="center", fontsize=7.0, color=color, fontweight="normal")
    ax.set_xticks(range(len(col_defs)))
    ax.set_xticklabels([x[1] for x in col_defs])
    ax.set_yticks(range(len(row_defs)))
    ax.set_yticklabels([x[1] for x in row_defs], fontsize=6.4, fontweight="normal")
    ax.tick_params(axis="y", pad=1.2)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("error (%)", fontsize=6.8, fontweight="normal")
    cbar.ax.tick_params(labelsize=6.8, length=3.0, width=0.9)
    panel_label(ax, "b")

    # c. Mesh-size observation sensitivity from the actual mesh table.
    ax = axes[1, 0]
    for protocol, color, label in [
        ("mixed_uniform_basis", red, "mixed outlet, 1D"),
        ("four_outlet_matched_3d_basis", blue, "4 outlets, 3D"),
    ]:
        sub = mesh_inverse[mesh_inverse["protocol"] == protocol].sort_values("n_cells")
        ax.semilogy(sub["n_cells"], sub["closed_error_percent"], "o-", color=color, lw=1.9, ms=5.2, label=label)
    ax.axhline(5, color="0.20", ls="--", lw=1.1)
    ax.set_xlabel("3D FVM cells")
    ax.set_ylabel("closed error (%)")
    ax.set_xlim(mesh_inverse["n_cells"].min() * 0.84, mesh_inverse["n_cells"].max() * 1.38)
    ax.legend(frameon=False, loc="lower right", fontsize=5.9, handlelength=1.0, borderaxespad=0.20)
    panel_label(ax, "c")

    # d. Actual 3D quadrant deviation hidden by the mixed outlet.
    ax = axes[1, 1]
    hist = histories[histories["case_id"] == "heterogeneous_bypass"].copy()
    q_cols = ["outlet_q1", "outlet_q2", "outlet_q3", "outlet_q4"]
    active = hist["outlet_mixed"] > hist["outlet_mixed"].max() * 1e-3
    h_active = hist.loc[active].copy()
    ratios = h_active[q_cols].div(h_active["outlet_mixed"], axis=0)
    q_min = ratios.min(axis=1)
    q_max = ratios.max(axis=1)
    spread_pct = 100.0 * (q_max - q_min)
    ax.fill_between(
        h_active["time_h"],
        q_min,
        q_max,
        color="#BDBDBD",
        alpha=0.65,
        lw=0,
        label="quadrant envelope",
    )
    ax.plot(h_active["time_h"], ratios["outlet_q1"], color=blue, lw=1.65, label="q1/mixed")
    ax.plot(h_active["time_h"], ratios["outlet_q4"], color=orange, lw=1.65, label="q4/mixed")
    ax.axhline(1.0, color=black, lw=1.2, ls="--")
    ax.text(
        0.98,
        0.08,
        f"max spread {spread_pct.max():.1f}%",
        transform=ax.transAxes,
        fontsize=6.9,
        fontweight="normal",
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    ax.set_ylim(0.90, 1.10)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("quadrant / mixed outlet")
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.0, borderaxespad=0.15)
    panel_label(ax, "d")

    # e. CFD-DEM pressure and RTD scale checks.
    ax = axes[2, 0]
    ergun_sorted = ergun.sort_values("inlet_Uz")
    pressure_norm = ergun_sorted["pressure_drop_mean_Pa"] / ergun_sorted["pressure_drop_mean_Pa"].max()
    ax.plot(
        ergun_sorted["inlet_Uz"],
        pressure_norm,
        "o-",
        color=black,
        lw=1.9,
        ms=5.2,
    )
    rtd_med = rtd.groupby("inlet_Uz", as_index=False)["t50_s"].mean().sort_values("inlet_Uz")
    t50_norm = rtd_med["t50_s"] / rtd_med["t50_s"].max()
    ax.plot(rtd_med["inlet_Uz"], t50_norm, "s-", color=blue, lw=1.9, ms=5.2)
    ax.set_xlabel("inlet velocity (relative)")
    ax.set_ylabel("normalized scale")
    ax.set_ylim(0.05, 1.08)
    ax.set_xlim(0.22, ergun_sorted["inlet_Uz"].max() * 1.12)
    ax.legend(["pressure drop", r"$t_{50}$"], frameon=False, loc="center right", fontsize=5.9, handlelength=1.1, borderaxespad=0.20)
    ax.text(
        0.04,
        0.92,
        f"Ergun shape error {ergun_summary['max_abs_relative_error_pct']:.3g}%",
        transform=ax.transAxes,
        fontsize=6.8,
        fontweight="normal",
        color="#222222",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
    )
    panel_label(ax, "e")

    # f. Chemistry-resolved source-estimation matrix from CFD-DEM chemistry tables.
    ax = axes[2, 1]
    chem_methods = [
        ("dual_wrong_dry_he_clean_reference", "wrong"),
        ("dual_clean_reference", "dual"),
        ("dual_paired_same_physics", "paired"),
        ("dual_fivefold_paired_same_physics", "5x"),
    ]
    stress_order = [
        ("baseline_partition_only", "baseline"),
        ("mild_adsorption_exchange", "mild\nads."),
        ("strong_hto_retention", "HTO\nret."),
        ("wall_catalytic_wet_shift", "wet\nwall"),
        ("compound_reactor_like_mismatch", "compound"),
    ]
    chem_matrix = np.zeros((len(stress_order), len(chem_methods)))
    for i, (stress, _) in enumerate(stress_order):
        for j, (method, _) in enumerate(chem_methods):
            chem_matrix[i, j] = float(
                chem[(chem["stress_case"] == stress) & (chem["method"] == method)]["p95_abs_error_pct"].iloc[0]
            )
    im2 = ax.imshow(chem_matrix, cmap="magma_r", norm=LogNorm(vmin=1.0, vmax=30.0), aspect="auto")
    for i in range(chem_matrix.shape[0]):
        for j in range(chem_matrix.shape[1]):
            value = chem_matrix[i, j]
            color = "white" if value > 5 else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7.0, color=color, fontweight="normal")
    ax.set_xticks(range(len(chem_methods)))
    ax.set_xticklabels([x[1] for x in chem_methods])
    ax.set_yticks(range(len(stress_order)))
    ax.set_yticklabels([x[1] for x in stress_order], fontsize=6.9, fontweight="normal")
    ax.tick_params(length=0)
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.045, pad=0.02)
    cbar2.set_label("p95 error (%)", fontsize=6.8, fontweight="normal")
    cbar2.ax.tick_params(labelsize=6.8, length=3.0, width=0.9)
    panel_label(ax, "f")

    for ax in axes.ravel():
        finish_prl_axes(ax)

    summary = {
        "figure_pdf": str(FIGURES / "packed_bed_transport_tests_current.pdf"),
        "heterogeneous_bypass_uniform_mixed_closed_error_pct": vals[0],
        "heterogeneous_bypass_matched_3d_four_outlet_closed_error_pct": vals[-1],
        "mesh_uniform_mixed_max_closed_error_pct": float(
            mesh_inverse[mesh_inverse["protocol"] == "mixed_uniform_basis"]["closed_error_percent"].max()
        ),
        "mesh_matched_3d_four_outlet_max_closed_error_pct": float(
            mesh_inverse[mesh_inverse["protocol"] == "four_outlet_matched_3d_basis"][
                "closed_error_percent"
            ].max()
        ),
        "pressure_drop_normalized_at_max_velocity": float(pressure_norm.iloc[-1]),
        "t50_normalized_at_max_velocity": float(t50_norm.iloc[-1]),
        "spatial_quadrant_max_spread_pct": float(spread_pct.max()),
        "ergun_shape_max_relative_error_pct": float(ergun_summary["max_abs_relative_error_pct"]),
        "leave_velocity_max_error_pct": float(validation["max_leave_velocity_error_pct"]),
        "pressure_rtd_gate_max_source_error_pct": float(
            pressure_summary["pressure_primary_rtd_gate_max_source_error_pct"]
        ),
        "nominal_velocity_max_source_error_pct": float(
            pressure_error_curves[pressure_error_curves["method"] == "nominal_u1"][
                "max_abs_source_error_pct"
            ].max()
        ),
        "wrong_chemistry_max_p95_error_pct": float(chem_matrix[:, 0].max()),
        "paired_hthto_max_p95_error_pct": float(chem_matrix[:, -1].max()),
        "figure_layout": "six-panel 3x2 layout with near-square panels",
        "figure_aspect_width_over_height": 7.36 / 9.20,
        "panel_titles_removed": True,
        "legend_overlap_guard": "crowded panels use compact real legends or small labels in open regions to avoid mismatched or overlapping legend boxes",
    }

    out_pdf = FIGURES / "packed_bed_transport_tests_current.pdf"
    out_png = FIGURES / "packed_bed_transport_tests_current.png"
    out_json = RESULTS / "packed_bed_transport_tests_current_summary.json"
    save_prl_canvas(fig, out_pdf, out_png, dpi=450)
    plt.close(fig)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
