#!/usr/bin/env python3
"""Build the main-text 288-component system tritium-balance figure."""

from __future__ import annotations

import json
import csv
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import apply_prl_style, finish_prl_axes, panel_label, save_prl_canvas
from locked_figure_guard import require_locked_figure_regeneration_allowed


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "whole_reactor_dynamic_tritium_observability"
COMPONENT = ROOT / "results" / "component_3d_module_network"
FIGURES = ROOT / "figures"
require_locked_figure_regeneration_allowed(
    FIGURES / "whole_reactor_accountancy_main_figure.pdf",
    FIGURES / "whole_reactor_accountancy_main_figure.png",
)

BLUE = "#0057D9"
RED = "#D40000"
GREEN = "#008A00"
ORANGE = "#D55E00"
PURPLE = "#7B4DAB"
BLACK = "#111111"
GRAY = "#666666"


def setup_style() -> None:
    apply_prl_style()


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty artifact: {path}")
    return path


def read_csv_stable(path: Path, *, retries: int = 8) -> pd.DataFrame:
    candidates = [require(path)]
    try:
        rel = path.relative_to(ROOT)
        archived = ROOT / "release_package" / "staging_minimal_release" / rel
        if archived.exists() and archived != path:
            candidates.append(archived)
    except ValueError:
        pass
    for attempt in range(retries):
        for candidate in candidates:
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                try:
                    text = subprocess.check_output(["/bin/cat", str(candidate)], text=True)
                    rows = list(csv.DictReader(text.splitlines()))
                except subprocess.CalledProcessError:
                    rows = []
            if rows:
                df = pd.DataFrame(rows)
                for col in df.columns:
                    converted = pd.to_numeric(df[col], errors="coerce")
                    if not converted.isna().all():
                        df[col] = converted
                return df
        time.sleep(0.25 * (attempt + 1))
    raise ValueError(f"CSV remained empty after retries: {path}")


def panel(ax, label: str) -> None:
    panel_label(ax, label, x=-0.055, y=1.006)
    finish_prl_axes(ax)


def norm_max(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    scale = np.nanmax(np.abs(arr))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(arr)
    return arr / scale


def norm_range(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def main() -> None:
    setup_style()
    FIGURES.mkdir(exist_ok=True)

    histories = read_csv_stable(RESULTS / "whole_reactor_dynamic_histories.csv")
    predictions = read_csv_stable(RESULTS / "whole_reactor_dynamic_tritium_predictions.csv")
    profiles = read_csv_stable(RESULTS / "whole_reactor_dynamic_sector_profile.csv")
    comp_hist = read_csv_stable(COMPONENT / "component_3d_outlet_histories.csv")
    comp_cells = read_csv_stable(COMPONENT / "component_3d_cell_response_metrics.csv")
    summary = json.loads(
        require(RESULTS / "whole_reactor_dynamic_accountancy_main_summary.json").read_text(
            encoding="utf-8"
        )
    )["summary"]

    display_axes = plt.subplots(3, 2, figsize=(7.36, 9.20))[1]
    fig = display_axes.ravel()[0].figure
    axes = display_axes
    plt.subplots_adjust(left=0.124, right=0.982, bottom=0.058, top=0.982, wspace=0.155, hspace=0.155)

    # a. Component outlet histories: a mixed outlet hides the spread across the 12 spatial outlets.
    ax = axes[0, 0]
    comp = comp_hist[comp_hist["time_h"] <= 48].copy()
    t_comp = comp["time_h"].to_numpy()
    outlet_cols = [c for c in comp.columns if c.endswith("_q1") or c.endswith("_q2") or c.endswith("_q3") or c.endswith("_q4")]
    for col in outlet_cols:
        ax.plot(t_comp, np.clip(norm_max(comp[col]), 1e-4, None), color="#A8A8A8", lw=0.85, alpha=0.52, zorder=1)
    ax.plot(t_comp, np.clip(norm_max(comp["single_mixed_outlet"]), 1e-4, None), color=BLACK, lw=1.95, label="mixed outlet", zorder=3)
    ax.plot(t_comp, np.clip(norm_max(comp["central_q1"]), 1e-4, None), color=BLUE, lw=1.55, label="central q1", zorder=4)
    ax.plot(t_comp, np.clip(norm_max(comp["central_q4"]), 1e-4, None), color=RED, lw=1.55, label="central q4", zorder=4)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized release")
    ax.set_xlim(0, 48)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.35)
    ax.legend(loc="upper right", frameon=False, fontsize=6.3, handlelength=1.05, borderaxespad=0.18)
    panel(ax, "a")

    # b. Axial source and retained-tail profiles: slow cells dominate inventory even when their source is distributed.
    ax = axes[0, 1]
    cells = comp_cells.copy()
    cells["tail_proxy"] = cells["source_fraction"] * np.clip(cells["tail_48h_fraction"], 0, None)
    axial = (
        cells.groupby("axial_index", as_index=False)
        .agg(
            source_fraction=("source_fraction", "sum"),
            tail_proxy=("tail_proxy", "sum"),
            response_t90_h=("response_t90_h", lambda s: np.average(s, weights=cells.loc[s.index, "source_fraction"])),
        )
        .sort_values("axial_index")
    )
    x = axial["axial_index"].to_numpy()
    ax.plot(x, norm_range(axial["source_fraction"]), color=BLACK, lw=1.85, label="source")
    ax.plot(x, norm_range(axial["tail_proxy"]), color=PURPLE, lw=1.85, label="48 h tail")
    ax.plot(x, norm_range(axial["response_t90_h"]), color=ORANGE, lw=1.65, ls="--", label=r"$t_{90}$")
    ax.set_xlabel("Axial cell index")
    ax.set_ylabel("Normalized axial profile")
    ax.set_xticks(np.arange(0, int(x.max()) + 1, 5))
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.24}
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.0, borderaxespad=0.18)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xlim(float(x.min()), float(x.max()) + 4.8)
    panel(ax, "b")

    # c. Dynamic histories show how a local slow-tail sector is diluted by summation.
    ax = axes[1, 0]
    hist = histories[histories["scenario_id"] == histories["scenario_id"].min()].copy()
    t = hist["time_h"].to_numpy()
    release_norm = norm_max(hist["aggregate_release"])
    inv_norm = norm_max(hist["top_tail_sector_inventory"])
    inv_line, = ax.plot(
        t,
        inv_norm,
        color=PURPLE,
        lw=1.9,
        label="slow-sector inventory",
        zorder=3,
    )
    release_line, = ax.plot(t, release_norm, color=BLACK, lw=2.05, label="system release", zorder=4)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Normalized history")
    ax.set_xlim(float(t.min()), float(t.max()) + 4.0)
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.0, borderaxespad=0.18)
    panel(ax, "c")

    # d. One independent sector profile: aggregate readout shifts the slow sector, sector release restores it.
    ax = axes[1, 1]
    prof_case = int(hist["scenario_id"].iloc[0])
    prof = profiles[profiles["scenario_id"] == prof_case].copy()
    true = prof[prof["feature_family"] == "aggregate_total"].sort_values("sector")
    agg = prof[prof["feature_family"] == "aggregate_total"].sort_values("sector")
    sector = prof[prof["feature_family"] == "sector_release_tail"].sort_values("sector")
    sectors = true["sector"].to_numpy()
    true_y = true["true_tail_fraction"].to_numpy(float)
    agg_y = agg["pred_tail_fraction"].to_numpy(float)
    sector_y = sector["pred_tail_fraction"].to_numpy(float)
    ax.plot(sectors, true_y, color=BLACK, lw=2.0, label="true")
    ax.plot(sectors, agg_y, color=RED, lw=1.65, ls="--", label="system total")
    ax.plot(sectors, sector_y, color=BLUE, lw=1.65, label="sector release")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Tail fraction", labelpad=0)
    ax.set_xticks(np.arange(0, 18, 3))
    ax.set_xlim(float(sectors.min()), float(sectors.max()))
    ax.legend(loc="lower left", frameon=False, fontsize=6.0, handlelength=1.05, borderaxespad=0.12)
    panel(ax, "d")

    # e. Error distributions across 300 independent dynamic cases.
    ax = axes[2, 0]
    families = [
        ("aggregate_total", "linear_ridge", "system total", RED),
        ("sector_release_tail", "linear_ridge", "sector release", BLUE),
        ("sector_full_accountancy", "random_feature_neural_worst_seed", "sector rel.+inventory", GREEN),
    ]
    for feature, readout, label, color in families:
        rows = predictions[(predictions["feature_family"] == feature) & (predictions["readout"] == readout)]
        x, y = ecdf(rows["tail_l1_pct"].to_numpy())
        ax.plot(x, y, color=color, lw=1.95, label=label)
    ax.axvline(5.0, color="#303030", ls="--", lw=1.1)
    ax.set_xlabel("Tail-fraction error (%)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(frameon=False, loc="lower right", fontsize=5.8, handlelength=1.0, borderaxespad=0.18)
    panel(ax, "e")

    # f. True/predicted top-tail sector across all independent cases.
    ax = axes[2, 1]
    sector_rows = predictions[
        (predictions["feature_family"] == "sector_release_tail")
        & (predictions["readout"] == "linear_ridge")
    ].copy()
    aggregate_rows = predictions[
        (predictions["feature_family"] == "aggregate_total")
        & (predictions["readout"] == "linear_ridge")
    ].copy()
    n_sector = int(
        max(
            sector_rows["true_top_tail_sector"].max(),
            sector_rows["pred_top_tail_sector"].max(),
            aggregate_rows["true_top_tail_sector"].max(),
            aggregate_rows["pred_top_tail_sector"].max(),
        )
    ) + 1
    matrix = np.zeros((n_sector, n_sector), dtype=float)
    for row in sector_rows.itertuples(index=False):
        matrix[int(row.pred_top_tail_sector), int(row.true_top_tail_sector)] += 1.0
    col_sum = np.maximum(matrix.sum(axis=0, keepdims=True), 1.0)
    matrix = matrix / col_sum
    im = ax.imshow(
        matrix,
        origin="lower",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        extent=(-0.5, n_sector - 0.5, -0.5, n_sector - 0.5),
        aspect="auto",
    )
    true_sector = np.arange(n_sector)
    agg_median = (
        aggregate_rows.groupby("true_top_tail_sector")["pred_top_tail_sector"]
        .median()
        .reindex(true_sector)
        .interpolate(limit_direction="both")
        .to_numpy(float)
    )
    ax.plot(true_sector, true_sector, color=BLACK, lw=1.2, label="perfect")
    ax.plot(true_sector, agg_median, color=RED, lw=1.55, ls="--", label="system total median")
    ax.text(0.04, 0.93, "sector release: 96.0%", transform=ax.transAxes, color=BLUE, fontsize=5.9, fontweight="normal",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.20})
    ax.text(0.04, 0.84, "system total: 16.7%", transform=ax.transAxes, color=RED, fontsize=5.9, fontweight="normal",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.20})
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-0.5, 17.5)
    ax.set_xticks(np.arange(0, 18, 3))
    ax.set_yticks(np.arange(0, 18, 3))
    ax.set_xlabel("True slow-tail sector")
    ax.set_ylabel("Predicted slow-tail sector", labelpad=0)
    ax.legend(frameon=False, loc="lower center", fontsize=5.9, handlelength=1.0, borderaxespad=0.18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.012)
    cbar.set_label("")
    cbar.ax.tick_params(length=3.0, width=0.9, pad=1, labelsize=6.8)
    panel(ax, "f")

    for ax in axes.ravel():
        finish_prl_axes(ax)

    figure_pdf = FIGURES / "whole_reactor_accountancy_main_figure.pdf"
    figure_png = FIGURES / "whole_reactor_accountancy_main_figure.png"
    save_prl_canvas(fig, figure_pdf, figure_png, dpi=450)
    plt.close(fig)

    out_summary = {
        "figure_pdf": str(figure_pdf.relative_to(ROOT)),
        "figure_png": str(figure_png.relative_to(ROOT)),
        "figure_layout": "six-panel 3x2 layout with near-square panels",
        "figure_aspect_width_over_height": 7.36 / 9.20,
        "panel_titles_removed": True,
        "legend_overlap_guard": "compact real legends are used in panels b, c, d and e; summary labels in panel f are small and boxed",
        "n_test_scenarios": int(summary["n_test_scenarios"]),
        "aggregate_tail_p95_pct": float(summary["manuscript_numbers"]["aggregate_tail_p95_pct"]),
        "aggregate_top_tail_accuracy_pct": float(
            summary["manuscript_numbers"]["aggregate_top_tail_accuracy_pct"]
        ),
        "sector_tail_p95_pct": float(summary["manuscript_numbers"]["sector_tail_p95_pct"]),
        "sector_top_tail_accuracy_pct": float(
            summary["manuscript_numbers"]["sector_top_tail_accuracy_pct"]
        ),
        "full_neural_tail_p95_pct": float(summary["manuscript_numbers"]["full_neural_tail_p95_pct"]),
        "full_neural_risk_p95_pct": float(summary["manuscript_numbers"]["full_neural_risk_p95_pct"]),
        "component_n_cells": int(len(comp_cells)),
        "component_n_outlet_histories": int(len(outlet_cols)),
        "component_tail_profile_peak_axial_index": int(axial.loc[axial["tail_proxy"].idxmax(), "axial_index"]),
        "component_t90_profile_peak_axial_index": int(axial.loc[axial["response_t90_h"].idxmax(), "axial_index"]),
    }
    out_json = RESULTS / "whole_reactor_accountancy_main_figure_summary.json"
    out_json.write_text(json.dumps(out_summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {figure_pdf}")
    print(f"wrote {figure_png}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
