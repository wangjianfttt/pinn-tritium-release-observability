#!/usr/bin/env python3
"""Build the SI packed-bed transport supporting-test figure."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def pct_from_tex(text: str) -> float:
    return float(text.replace("\\%", "").split()[-1])


def source_scaling_pct(text: str) -> float:
    fragment = text.split("source scaling deviation ")[1].replace("\\%", "")
    fragment = fragment.replace("below ", "")
    return float(fragment)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    ergun = pd.read_csv(require(RESULTS / "cfdem_ergun_envelope" / "cfdem_ergun_pressure_fit.csv"))
    rtd = pd.read_csv(require(RESULTS / "cfdem_oracle_validation" / "cfdem_oracle_rtd_proxy.csv"))
    velocity = pd.read_csv(
        require(RESULTS / "cfdem_pressure_rtd_velocity_rescue" / "cfdem_pressure_rtd_velocity_rescue_rows.csv")
    )
    chem = pd.read_csv(require(RESULTS / "chemistry_resolved_cfdem_stress" / "chemistry_resolved_cfdem_stress_summary.csv"))
    mesh = pd.read_csv(require(RESULTS / "fvm3d_mesh_observation_sensitivity" / "fvm3d_mesh_inverse_comparison.csv"))
    checks = pd.read_csv(require(RESULTS / "cfdem_transport_check_table.csv"))
    ergun_summary = json.loads(
        require(RESULTS / "cfdem_ergun_envelope" / "cfdem_ergun_envelope_summary.json").read_text(
            encoding="utf-8"
        )
    )["summary"]
    pressure_summary = json.loads(
        require(
            RESULTS
            / "cfdem_pressure_rtd_velocity_rescue"
            / "cfdem_pressure_rtd_velocity_rescue_summary.json"
        ).read_text(encoding="utf-8")
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.linewidth": 0.85,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
        }
    )
    blue = "#0057D9"
    red = "#D40000"
    green = "#008A00"
    orange = "#D55E00"
    black = "#111111"

    fig, axes = plt.subplots(2, 3, figsize=(7.75, 4.65))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.91, wspace=0.58, hspace=0.70)

    def label(ax, letter: str, title: str) -> None:
        ax.text(-0.18, 1.10, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
        ax.set_title(title, loc="left", fontsize=8, pad=5)

    ax = axes[0, 0]
    ergun = ergun.sort_values("inlet_Uz")
    ax.plot(ergun["inlet_Uz"], ergun["pressure_drop_mean_Pa"], "o", color=black, ms=4.5, label="CFD-DEM")
    ax.plot(ergun["inlet_Uz"], ergun["ergun_shape_predicted_Pa"], "-", color=blue, lw=1.5, label="Ergun-form fit")
    ax.set_xlabel("inlet velocity")
    ax.set_ylabel("pressure drop (Pa)")
    ax.legend(fontsize=7, loc="upper left")
    label(ax, "a", "Pressure-flow relation")

    ax = axes[0, 1]
    rtd_med = rtd.groupby("inlet_Uz", as_index=False)[["t10_s", "t50_s", "t90_s"]].mean().sort_values("inlet_Uz")
    ax.plot(rtd_med["inlet_Uz"], rtd_med["t10_s"], "o-", color=green, lw=1.3, ms=4, label="$t_{10}$")
    ax.plot(rtd_med["inlet_Uz"], rtd_med["t50_s"], "s-", color=blue, lw=1.3, ms=4, label="$t_{50}$")
    ax.plot(rtd_med["inlet_Uz"], rtd_med["t90_s"], "^-", color=red, lw=1.3, ms=4, label="$t_{90}$")
    ax.set_xlabel("inlet velocity")
    ax.set_ylabel("outlet rise time (s)")
    ax.legend(fontsize=7, loc="upper right")
    label(ax, "b", "RTD proxy ordering")

    ax = axes[0, 2]
    methods = ["nominal_u1", "rtd_only", "pressure_primary_rtd_gate", "true_velocity_oracle"]
    labels = ["nominal\nvelocity", "RTD\nonly", "pressure\n+ RTD", "true\nvelocity"]
    vals = [velocity[velocity["method"] == m]["source_error_pct"].abs().max() for m in methods]
    x = np.arange(len(vals))
    ax.semilogy(x, vals, "o", color=black, ms=4)
    for i, (v, c) in enumerate(zip(vals, [red, orange, blue, green])):
        ax.vlines(i, 0.005, v, color=c, lw=1.8)
        ax.scatter(i, v, color=c, s=24, zorder=3)
    ax.axhline(5, color="0.35", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("max source error (%)")
    label(ax, "c", "Velocity calibration")

    ax = axes[1, 0]
    chem_methods = [
        ("total_clean_reference", "total\nclean", red),
        ("dual_clean_reference", "HT/HTO\nclean", orange),
        ("dual_wrong_dry_he_clean_reference", "mismatch\nchem.", red),
        ("dual_fivefold_paired_same_physics", "paired\nHT/HTO", blue),
    ]
    vals = []
    labels = []
    colors = []
    for method, lab, color in chem_methods:
        vals.append(float(chem[chem["method"] == method]["p95_abs_error_pct"].max()))
        labels.append(lab)
        colors.append(color)
    x = np.arange(len(vals))
    ax.semilogy(x, vals, "o", color=black, ms=4)
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.vlines(i, 0.5, v, color=c, lw=1.8)
        ax.scatter(i, v, color=c, s=24, zorder=3)
    ax.axhline(5, color="0.35", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.set_ylabel("p95 source error (%)")
    label(ax, "d", "Chemistry/species calibration")

    ax = axes[1, 1]
    for protocol, color, lab in [
        ("mixed_uniform_basis", red, "mixed outlet, 1D"),
        ("four_outlet_matched_3d_basis", blue, "4 outlets, 3D"),
    ]:
        sub = mesh[mesh["protocol"] == protocol].sort_values("n_cells")
        ax.semilogy(sub["n_cells"], sub["closed_error_percent"], "o-", color=color, lw=1.4, ms=4, label=lab)
    ax.axhline(5, color="0.35", ls="--", lw=0.9)
    ax.set_xlabel("3D FVM cells")
    ax.set_ylabel("closed-pathway error (%)")
    ax.legend(fontsize=7, loc="lower left")
    label(ax, "e", "3D grid family")

    ax = axes[1, 2]
    row_names = ["Pressure-flow consistency", "Mass-flow/source linearity", "Leave-one-velocity test", "Pressure/RTD calibration"]
    short = ["pressure", "flow", "velocity\ntest", "pressure\nRTD"]
    values = []
    for name in row_names:
        text = checks.loc[checks["check"] == name, "quantity"].iloc[0]
        if name == "Pressure-flow consistency":
            values.append(float(ergun_summary["max_abs_relative_error_pct"]))
        elif name == "Mass-flow/source linearity":
            values.append(source_scaling_pct(text))
        elif name == "Leave-one-velocity test":
            values.append(float(text.split("leave-one-velocity source error ")[1].split("\\%")[0]))
        else:
            values.append(float(pressure_summary["pressure_primary_rtd_gate_max_source_error_pct"]))
    x = np.arange(len(values))
    ax.semilogy(x, values, "o", color=black, ms=4)
    for i, (v, c) in enumerate(zip(values, [green, green, red, blue])):
        ax.vlines(i, 1e-4, v, color=c, lw=1.8)
        ax.scatter(i, v, color=c, s=24, zorder=3)
    ax.axhline(5, color="0.35", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("relative error (%)")
    label(ax, "f", "Static-bed support checks")

    for ax in axes.ravel():
        ax.grid(True, color="#E6E6E6", lw=0.45)
        ax.tick_params(length=3, width=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out_pdf = FIGURES / "packed_bed_transport_supporting_tests.pdf"
    out_png = FIGURES / "packed_bed_transport_supporting_tests.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
