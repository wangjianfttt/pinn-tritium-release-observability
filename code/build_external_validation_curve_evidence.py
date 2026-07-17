#!/usr/bin/env python3
"""Build the external-curve evidence figure from archived CSV/JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import BLACK, apply_prl_style, finish_prl_axes, panel_label as prl_panel_label


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CURVES = ROOT / "literature_data" / "curves"


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty artifact: {path}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(require(path).read_text(encoding="utf-8"))


def panel_label(ax, label: str) -> None:
    prl_panel_label(ax, label, x=-0.065, y=1.025)


def style_axes(ax) -> None:
    finish_prl_axes(ax)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    out_dir = RESULTS / "external_validation_curve_evidence"
    out_dir.mkdir(exist_ok=True)

    park = pd.read_csv(require(CURVES / "park_2025_fig5_li2tio3_800c_release_digitized_v1.csv"))
    pred = pd.read_csv(require(out_dir / "park_fec_posterior_predictive_curves.csv"))
    boot = pd.read_csv(require(RESULTS / "real_curve_bootstrap_inverse" / "real_curve_bootstrap_inverse_summary.csv"))
    prior = pd.read_csv(require(RESULTS / "real_curve_prior_weight_sweep" / "real_curve_prior_weight_sweep_rows.csv"))
    leave = pd.read_csv(require(RESULTS / "real_curve_leave_one_point_predictive" / "real_curve_leave_one_point_detail.csv"))
    timescale = pd.read_csv(require(RESULTS / "external_timescale_to_system_accountancy" / "external_timescale_to_system_accountancy.csv"))
    tds = pd.read_csv(require(CURVES / "kobayashi_2018_fig4_li2tio3_tds_digitized_v1.csv"))
    noise = pd.read_csv(
        require(
            RESULTS
            / "real_curve_prior_decisive_validation"
            / "real_curve_prior_decisive_validation_protocol_summary.csv"
        )
    )
    summary = read_json(require(out_dir / "external_validation_curve_evidence_summary.json"))

    apply_prl_style()

    blue = "#0057D9"
    red = "#D40000"
    green = "#008A00"
    black = "#111111"

    fig, axes_grid = plt.subplots(3, 2, figsize=(7.70, 8.10))
    axes = np.array(
        [
            [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 0]],
            [axes_grid[1, 1], axes_grid[2, 0], axes_grid[2, 1]],
        ]
    )
    fig.subplots_adjust(left=0.095, right=0.996, bottom=0.052, top=0.982, wspace=0.20, hspace=0.20)

    # a. Sparse Park/FEC points and bootstrap effective-time-scale fit.
    ax = axes[0, 0]
    curve_end_labels = []
    for series, color, marker in [("HT", blue, "o"), ("HTO", red, "s")]:
        raw = park[park["series"] == series].sort_values("time_min").copy()
        scale = float(raw["tritium_concentration_bq_l"].iloc[0])
        y = raw["tritium_concentration_bq_l"] / scale
        yerr = np.vstack([raw["error_minus_bq_l"] / scale, raw["error_plus_bq_l"] / scale])
        pp = pred[pred["series"] == series]
        ax.fill_between(
            pp["time_min"],
            pp["posterior_p05_norm"],
            pp["posterior_p95_norm"],
            color=color,
            alpha=0.12,
            lw=0,
        )
        ax.plot(pp["time_min"], pp["posterior_p50_norm"], color=color, lw=1.5, label=f"{series} fit")
        curve_end_labels.append((float(pp["time_min"].iloc[-1]), float(pp["posterior_p50_norm"].iloc[-1]), color, series))
        ax.errorbar(
            raw["time_min"],
            y,
            yerr=yerr,
            fmt=marker,
            mfc="white",
            mec=color,
            mew=1.0,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            ms=4,
            label=f"{series} points [18]",
        )
    ht = boot[boot["series"] == "HT"].iloc[0]
    hto = boot[boot["series"] == "HTO"].iloc[0]
    for x_end, y_end, color, series in curve_end_labels:
        ax.text(x_end + 3.0, y_end, f"{series} fit", color=color, fontsize=6.8, ha="left", va="center")
    ax.text(
        0.96,
        0.88,
        f"8 digitized points\n$\\tau_{{HT}}$ {ht.tau_eff_median_min:.0f} min\n"
        f"$\\tau_{{HTO}}$ {hto.tau_eff_median_min:.0f} min",
        transform=ax.transAxes,
        fontsize=6.8,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    ax.set_xlabel("hold time at 800 $^\\circ$C (min)")
    ax.set_ylabel("normalized outlet")
    ax.set_xlim(park["time_min"].min() - 8, max(pred["time_min"].max(), park["time_min"].max()) + 28)
    ax.set_ylim(-0.05, 1.17)
    panel_label(ax, "a")

    # b. Prior-weight sweep.
    ax = axes[0, 1]
    sub = prior[prior["family_id"] == "cold_rear_compound"].copy()
    label_map = {
        "full_accountancy": ("full spatial/species/accountancy", blue, "o"),
        "twelve_total_accountancy": ("total + accountancy", red, "s"),
    }
    for protocol, (label, color, marker) in label_map.items():
        rows = sub[sub["protocol_id"] == protocol].sort_values("prior_weight_fraction")
        x = rows["prior_weight_fraction"].replace(0, 1e-5)
        ax.semilogx(x, rows["closed_error_p95_pct"], marker + "-", color=color, lw=1.5, ms=4)
        if protocol == "full_accountancy":
            ax.text(
                1.3e-5,
                rows["closed_error_p95_pct"].iloc[0] - 0.75,
                "spatial/species",
                color=color,
                fontsize=6.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            )
        else:
            ax.text(
                1.3e-5,
                rows["closed_error_p95_pct"].iloc[0] + 1.25,
                "total + account.",
                color=color,
                fontsize=6.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            )
    ax.axhline(5, color="0.25", lw=0.9, ls="--")
    ax.axvspan(1e-5, 0.08, color="#DCEBFF", alpha=0.45, lw=0)
    ax.text(0.0018, 14.4, "weak-prior\nrange", color=blue, fontsize=6.8)
    ax.text(0.22, 15.4, "over-weighted", color=red, fontsize=6.8, ha="left")
    ax.set_xlabel("real-curve prior weight fraction")
    ax.set_ylabel("closed p95 error (%)")
    ax.set_ylim(0, 17.0)
    panel_label(ax, "b")

    # c. Leave-one-point posterior predictive check.
    ax = axes[0, 2]
    for series, color, marker, offset in [("HT", blue, "o", -1.5), ("HTO", red, "s", 1.5)]:
        rows = leave[leave["series"] == series].sort_values("holdout_time_min")
        x = rows["holdout_time_min"] + offset
        y = rows["bootstrap_pred_median_norm"]
        yerr = np.vstack([y - rows["expanded_predictive_p05_norm"], rows["expanded_predictive_p95_norm"] - y])
        ax.errorbar(x, y, yerr=yerr, fmt=marker + "-", color=color, lw=1.2, ms=3.6, capsize=2, label=f"{series} pred.")
        ax.plot(x, rows["holdout_norm"], marker, mfc="white", mec=color, mew=1.0, ms=4, ls="none", label=f"{series} held-out")
    ax.text(
        0.04,
        0.07,
        f"coverage {summary['leave_one_point_predictive_coverage']:.3f}\n"
        f"max error {summary['leave_one_point_max_abs_error_norm']:.3f}",
        transform=ax.transAxes,
        fontsize=6.8,
    )
    ax.set_xlabel("leave-one holdout time (min)")
    ax.set_ylabel("normalized held-out")
    ax.set_ylim(0.0, 0.88)
    ax.legend(fontsize=6.2, loc="upper right", ncol=2, handlelength=0.9, columnspacing=0.45)
    panel_label(ax, "c")

    # d. Time-scale-guided commissioning references.
    ax = axes[1, 0]
    style_map = {
        "prefix": (black, "o", "prefix"),
        "random": (red, "s", "random references"),
        "space_filling": (blue, "^", "space-filling"),
        "time_scale_guided": (green, "o", "time-scale guided"),
    }
    for strategy, (color, marker, label) in style_map.items():
        rows = timescale[timescale["strategy"] == strategy].sort_values("k")
        ax.plot(rows["k"], rows["combined_net_recovery_radius"], marker + "-", color=color, lw=1.35, ms=3.4, label=label)
    ax.axhline(0.02, color="0.25", ls="--", lw=0.9)
    ax.set_xlabel("target-bed reference histories $K$")
    ax.set_ylabel("net-recovery half-width")
    ax.set_ylim(0, 0.21)
    ax.legend(fontsize=6.3, loc="upper right", handlelength=0.9)
    panel_label(ax, "d")

    # e. Density-dependent TDS spectra without inset occlusion.
    ax = axes[1, 1]
    cmap = plt.get_cmap("viridis")
    densities = sorted(tds["relative_density"].dropna().unique(), reverse=True)
    for idx, rho in enumerate(densities):
        rows = tds[tds["relative_density"] == rho].sort_values("x_value")
        y = rows["y_value"] / max(rows["y_value"].max(), 1e-12)
        color = cmap(idx / max(len(densities) - 1, 1))
        ax.plot(rows["x_value"], y, color=color, lw=1.15)
        ax.text(792, 0.12 + 0.055 * idx, f"{rho:.1f}", color=color, fontsize=6.8, ha="right", va="center")
    ax.axvspan(500, 640, color="#DCEBFF", alpha=0.55, lw=0)
    ax.axvspan(700, 800, color="#FFE1E1", alpha=0.55, lw=0)
    ax.text(548, 1.04, "low-T window", color=blue, fontsize=6.8, ha="center", va="top")
    ax.text(748, 1.04, "delayed high-T", color=red, fontsize=6.8, ha="center", va="top")
    ax.set_xlabel("TDS temperature (K)")
    ax.set_ylabel("normalized TDS rate")
    ax.set_xlim(430, 805)
    ax.set_ylim(0, 1.12)
    ax.text(792, 0.40, "Kobayashi and Oya [11]\n$\\rho/\\rho_{th}$", ha="right", va="bottom", fontsize=6.8, color="0.05")
    panel_label(ax, "e")

    # f. Weak-prior noise response.
    ax = axes[1, 2]
    ns = noise[(noise["prior_mode"] == "park_fec_weak_prior") & (noise["n_repeats"] == 1)].copy()
    noise_map = {
        "single_mixed_total": ("single mixed", red, "o"),
        "twelve_ht_hto": ("12 HT/HTO", blue, "s"),
        "full_accountancy": ("full accountancy", green, "^"),
    }
    for pid, (label, color, marker) in noise_map.items():
        rows = ns[ns["protocol_id"] == pid].sort_values("outlet_noise_pct")
        ax.plot(rows["outlet_noise_pct"], rows["joint_error_p95_pct"], marker + "-", color=color, lw=1.5, ms=4, label=label)
    ax.axhline(5, color="0.25", ls="--", lw=0.9)
    ax.set_xlabel("outlet noise (%)")
    ax.set_ylabel("joint p95 error (%)")
    ax.set_xlim(1.35, 3.20)
    ax.set_ylim(0, 22.5)
    ax.legend(fontsize=6.3, loc="upper left", handlelength=1.1, borderaxespad=0.25)
    panel_label(ax, "f")

    for ax in axes.ravel():
        style_axes(ax)

    out_pdf = FIGURES / "external_validation_curve_evidence.pdf"
    out_png = FIGURES / "external_validation_curve_evidence.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(out_png, dpi=450, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    summary["figure_pdf"] = str(out_pdf)
    summary["figure_png"] = str(out_png)
    summary["builder_script"] = str(ROOT / "code" / "build_external_validation_curve_evidence.py")
    summary["figure_layout"] = "six-panel 3x2 layout with compact near-square panels"
    summary["figure_aspect_width_over_height"] = 7.70 / 8.10
    summary["panel_titles_removed"] = True
    (out_dir / "external_validation_curve_evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
