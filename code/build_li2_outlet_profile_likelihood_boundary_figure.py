#!/usr/bin/env python3
"""Build the SI Li2TiO3 outlet-only profile-likelihood figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "li2_outlet_profile_likelihood"
FIGURES = ROOT / "figures"
import sys

sys.path.insert(0, str(ROOT / "code"))
from figure_style import apply_prl_style, finish_prl_axes, panel_label, save_prl_canvas


SURFACE_TITLES = {
    "open_vs_closed": "open--closed",
    "closed_vs_porosity": "closed--porosity",
    "closed_vs_velocity": "closed--velocity",
    "closed_vs_detector_gain": "closed--detector gain",
    "closed_vs_detector_crosstalk": "closed--cross-talk",
}

SURFACE_LABELS = {
    "open_vs_closed": ("open multiplier", "closed multiplier"),
    "closed_vs_porosity": ("porosity", "closed multiplier"),
    "closed_vs_velocity": ("velocity", "closed multiplier"),
    "closed_vs_detector_gain": ("detector gain", "closed multiplier"),
    "closed_vs_detector_crosstalk": ("detector cross-talk", "closed multiplier"),
}


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def draw_representative_surface(ax, frame: pd.DataFrame) -> None:
    table = frame.pivot(index="y_value", columns="x_value", values="delta_chi2")
    x = table.columns.to_numpy(dtype=float)
    y = table.index.to_numpy(dtype=float)
    z = table.to_numpy(dtype=float)
    accepted = (z <= 3.84).astype(float)
    mesh = ax.pcolormesh(
        x,
        y,
        accepted,
        shading="nearest",
        cmap=ListedColormap(["#D95F02", "#F4F4F4"]),
        vmin=0,
        vmax=1,
        zorder=1,
    )
    ax.contour(x, y, z, levels=[3.84], colors="#D95F02", linewidths=1.05, zorder=3)
    ax.plot(0, 0, marker="*", ms=6.5, color="#0057D9", mec="black", mew=0.45, zorder=4)
    ax.text(
        0.04,
        0.92,
        r"$\Delta\chi^2 \leq 3.84$ region",
        transform=ax.transAxes,
        fontsize=6.7,
        va="top",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "0.75", "linewidth": 0.35},
    )
    ax.text(0.05, 0.08, "best fit", transform=ax.transAxes, color="#0057D9", fontsize=6.5)
    ax.set_xlabel("velocity perturbation (norm.)")
    ax.set_ylabel("closed multiplier perturbation (norm.)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    return mesh


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    surfaces = pd.read_csv(require(RESULTS / "li2_outlet_profile_likelihood_surfaces.csv"))
    holdout = pd.read_csv(require(RESULTS / "li2_outlet_leave_one_condition_out.csv"))

    apply_prl_style()

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05))
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.105, top=0.965, wspace=0.34, hspace=0.38)

    # a: compact summary of the broad profiled admissible domain.
    ax = axes[0, 0]
    order = list(SURFACE_TITLES)
    labels = [SURFACE_TITLES[k] for k in order]
    inside = [
        100.0 * float(surfaces.loc[surfaces["surface_id"] == k, "inside_delta_3p84"].mean())
        for k in order
    ]
    y = np.arange(len(order))
    ax.hlines(y, 98.7, inside, color="0.68", lw=1.2, zorder=1)
    ax.plot(inside, y, "o", color="#0057D9", ms=4.6, zorder=2)
    for yi, val in zip(y, inside):
        ax.text(min(val + 0.03, 100.16), yi, f"{val:.1f}%", va="center", fontsize=6.5)
    ax.axvline(100, color="0.25", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(98.7, 100.25)
    ax.set_xlabel(r"profile domain inside $\Delta\chi^2\leq3.84$ (%)")
    ax.invert_yaxis()
    panel_label(ax, "a")
    ax.text(
        0.02,
        0.06,
        "near-complete admissible domain",
        transform=ax.transAxes,
        fontsize=6.6,
        color="0.25",
    )

    # b: one representative two-parameter profile map; the others are summarized in a.
    ax = axes[0, 1]
    representative = surfaces[surfaces["surface_id"] == "closed_vs_velocity"]
    draw_representative_surface(ax, representative)
    panel_label(ax, "b")

    # c: holdout prediction errors.  Conditions are categorical, so use points instead of joined lines.
    ax = axes[1, 0]
    cond = [v.replace("_", "\n") for v in holdout["heldout_condition"]]
    y = np.arange(len(holdout))
    ax.hlines(y + 0.12, 0, holdout["open_error_percent"], color="#0057D9", lw=1.1)
    ax.hlines(y - 0.12, 0, holdout["closed_error_percent"], color="#D40000", lw=1.1)
    ax.plot(holdout["open_error_percent"], y + 0.12, "o", color="#0057D9", label="open")
    ax.plot(holdout["closed_error_percent"], y - 0.12, "s", color="#D40000", label="closed")
    ax.axvline(5, color="0.30", ls="--", lw=0.85)
    ax.text(5.05, -0.42, "5%", fontsize=6.5, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(cond)
    ax.set_xlim(0, 5.45)
    ax.set_xlabel("held-out multiplier error (%)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.1)
    panel_label(ax, "c")

    # d: recovered multipliers against the true unit multiplier.
    ax = axes[1, 1]
    x = np.arange(len(holdout))
    ax.axhline(1.0, color="0.25", lw=0.85, ls="--")
    ax.plot(x - 0.05, holdout["open_multiplier_estimate"], "o", color="#0057D9", label="open")
    ax.plot(x + 0.05, holdout["closed_multiplier_estimate"], "s", color="#D40000", label="closed")
    for xi, row in holdout.iterrows():
        ax.vlines(xi - 0.05, 1.0, row["open_multiplier_estimate"], color="#0057D9", lw=0.85)
        ax.vlines(xi + 0.05, 1.0, row["closed_multiplier_estimate"], color="#D40000", lw=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cond)
    ax.set_ylim(0.996, 1.022)
    ax.set_ylabel("estimated multiplier")
    ax.text(0.02, 0.91, "true = 1", transform=ax.transAxes, fontsize=6.6, color="0.25")
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.1)
    panel_label(ax, "d")

    for ax in axes.ravel():
        finish_prl_axes(ax, grid=True)

    out_pdf = FIGURES / "li2_outlet_profile_likelihood_boundary.pdf"
    out_png = FIGURES / "li2_outlet_profile_likelihood_boundary.png"
    save_prl_canvas(fig, out_pdf, out_png)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
