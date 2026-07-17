#!/usr/bin/env python3
"""Draw the manuscript PINN architecture figure.

The schematic is source-grounded. It shows the source inverse as a shared
Li2TiO3 PINN trunk with gas, inventory and source-term outputs, then separates
inference flow, training loss terms and solid-inventory availability checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from locked_figure_guard import require_locked_figure_regeneration_allowed


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results" / "pinn_source_head_architecture"
require_locked_figure_regeneration_allowed(
    FIG / "pinn_source_head_architecture.pdf",
    FIG / "pinn_source_head_architecture.png",
    FIG / "pinn_source_head_architecture.svg",
)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.0,
    }
)

COL = {
    "ink": "#111111",
    "muted": "#555555",
    "line": "#1C1C1C",
    "blue": "#003CFF",
    "red": "#C00000",
    "green": "#08751A",
    "input": "#E9EFF8",
    "operator": "#F3F3F3",
    "head": "#EFE8F7",
    "source": "#F7D36A",
    "loss": "#FFE6E6",
    "ok": "#E8F4EA",
    "withheld": "#FCECEC",
}


def add_rect(ax, x, y, w, h, title, subtitle="", fc="#FFFFFF", ec="#111111", lw=1.45, fs=9.7):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=fs, weight="bold")
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=fs - 1.4,
            color=COL["muted"],
        )
    return rect


def arrow(ax, p0, p1, lw=2.0, color=None, dashed=False, mutation_scale=13):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color or COL["line"],
        linestyle=(0, (5, 3)) if dashed else "solid",
        connectionstyle="arc3,rad=0",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    return arr


def line(ax, p0, p1, lw=2.0, color=None, dashed=False):
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color or COL["line"],
        lw=lw,
        linestyle=(0, (5, 3)) if dashed else "solid",
        solid_capstyle="butt",
    )


def ortho_arrow(ax, points, lw=2.0, color=None, dashed=False, mutation_scale=13):
    for p0, p1 in zip(points[:-2], points[1:-1]):
        line(ax, p0, p1, lw=lw, color=color, dashed=dashed)
    arrow(ax, points[-2], points[-1], lw=lw, color=color, dashed=dashed, mutation_scale=mutation_scale)


def panel_title(ax, letter, title, x, y):
    ax.text(x, y, letter, ha="left", va="top", fontsize=18, weight="bold")
    ax.text(x + 0.035, y - 0.003, title, ha="left", va="top", fontsize=12, weight="bold")


def axes_circle_dims(ax, r):
    fig = ax.figure
    pos = ax.get_position()
    axes_w = fig.get_size_inches()[0] * pos.width
    axes_h = fig.get_size_inches()[1] * pos.height
    return 2 * r, 2 * r * axes_w / axes_h


def draw_node(ax, x, y, r=0.0118, lw=1.25):
    w, h = axes_circle_dims(ax, r)
    node = mpl.patches.Ellipse(
        (x, y),
        width=w,
        height=h,
        transform=ax.transAxes,
        facecolor="#FFFFFF",
        edgecolor=COL["line"],
        linewidth=lw,
        zorder=4,
    )
    ax.add_patch(node)


def draw_mlp(ax, x0, y0, w, h):
    add_rect(ax, x0, y0, w, h, "", fc="#FFFFFF", ec=COL["line"], lw=1.5)
    ax.text(
        x0 + w / 2,
        y0 + h + 0.035,
        r"shared PINN trunk $\theta_{\psi}(z,t,\xi)$",
        ha="center",
        va="bottom",
        fontsize=11.3,
        weight="bold",
        color=COL["green"],
    )
    ax.text(
        x0 + w / 2,
        y0 + h + 0.012,
        "4 hidden layers | 96 neurons/layer | tanh",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color=COL["muted"],
        weight="bold",
    )

    layer_counts = [5, 5, 5, 5, 5, 4]
    layer_names = ["input", "H1", "H2", "H3", "H4", "features"]
    xs = [x0 + w * f for f in [0.08, 0.255, 0.42, 0.585, 0.75, 0.925]]
    node_cols = []
    for n, xx in zip(layer_counts, xs):
        ys = [y0 + h * (0.18 + 0.64 * i / (n - 1)) for i in range(n)]
        node_cols.append([(xx, yy) for yy in ys])

    for left, right in zip(node_cols[:-1], node_cols[1:]):
        for p in left:
            for q in right:
                line(ax, p, q, lw=0.62, color="#595959")

    for col in node_cols:
        for xx, yy in col:
            draw_node(ax, xx, yy, r=0.0102, lw=1.25)

    for xx, name in zip(xs, layer_names):
        ax.text(
            xx,
            y0 - 0.030,
            name,
            ha="center",
            va="top",
            fontsize=8.5,
            weight="bold" if name.startswith("H") else "normal",
        )

    return xs[0], xs[-1], y0 + h / 2


def draw_panel_a(ax):
    panel_title(ax, "a.", "PINN source-inverse architecture", 0.030, 0.930)

    inputs = [
        (0.055, 0.775, "coordinates", r"$z,\ t$"),
        (0.055, 0.675, "operating variables", r"$u,\ T,\ \phi,\ \xi$"),
        (0.055, 0.575, "3D kernels", r"$h_z^r(t)$, RTD"),
        (0.055, 0.475, "chemistry", "HT/HTO"),
        (0.055, 0.375, "calibration", "detector pulses"),
    ]
    for x, y, title, sub in inputs:
        add_rect(ax, x, y, 0.135, 0.064, title, sub, fc=COL["input"], fs=9.5)

    bus_x = 0.225
    mlp_x, mlp_y, mlp_w, mlp_h = 0.285, 0.385, 0.290, 0.375
    for _, y, _, _ in inputs:
        ymid = y + 0.032
        line(ax, (0.190, ymid), (bus_x, ymid), lw=2.0)
    line(ax, (bus_x, inputs[-1][1] + 0.032), (bus_x, inputs[0][1] + 0.032), lw=2.0)
    arrow(ax, (bus_x, mlp_y + mlp_h / 2), (mlp_x, mlp_y + mlp_h / 2), lw=2.2)

    _, _, mlp_mid_y = draw_mlp(ax, mlp_x, mlp_y, mlp_w, mlp_h)

    heads = [
        (0.635, 0.700, "gas-field branch", r"$C_{\rm HT}, C_{\rm HTO}$", COL["head"]),
        (0.635, 0.590, "inventory branch", r"$I_o,\ I_c$", COL["head"]),
        (0.635, 0.480, "source-term layer", r"$m_o,\ m_c$; 8 basis terms", COL["source"]),
    ]
    split_x = 0.605
    line(ax, (mlp_x + mlp_w, mlp_mid_y), (split_x, mlp_mid_y), lw=2.2)
    line(ax, (split_x, 0.512), (split_x, 0.732), lw=2.2)
    for x, y, title, sub, fc in heads:
        ymid = y + 0.032
        arrow(ax, (split_x, ymid), (x, ymid), lw=2.2)
        add_rect(ax, x, y, 0.130, 0.064, title, sub, fc=fc, fs=9.3)

    ops = [
        (0.825, 0.740, "transport residual", "3D residence time"),
        (0.825, 0.650, "detector response", r"$G,\eta$, cross-talk"),
        (0.825, 0.515, "tritium balance", "inventory/source"),
    ]
    for x2, y2, title2, sub2 in ops:
        add_rect(ax, x2, y2, 0.130, 0.064, title2, sub2, fc=COL["operator"], fs=9.0)

    gas_y = heads[0][1] + 0.032
    inv_y = heads[1][1] + 0.032
    src_y = heads[2][1] + 0.032
    trans_y = ops[0][1] + 0.032
    det_y = ops[1][1] + 0.032
    bal_y = ops[2][1] + 0.032
    bus2_x = 0.795

    line(ax, (0.765, gas_y), (bus2_x, gas_y), lw=2.1)
    line(ax, (bus2_x, det_y), (bus2_x, trans_y), lw=2.1)
    arrow(ax, (bus2_x, trans_y), (0.825, trans_y), lw=2.1)
    arrow(ax, (bus2_x, det_y), (0.825, det_y), lw=2.1)

    line(ax, (0.765, inv_y), (bus2_x, inv_y), lw=2.1)
    line(ax, (0.765, src_y), (bus2_x, src_y), lw=2.1)
    line(ax, (bus2_x, src_y), (bus2_x, inv_y), lw=2.1)
    arrow(ax, (bus2_x, bal_y), (0.825, bal_y), lw=2.1)

    ax.text(
        0.888,
        0.470,
        "predicted release histories\nand source multipliers",
        ha="center",
        va="top",
        fontsize=9.0,
        weight="bold",
    )


def draw_panel_b(ax):
    panel_title(ax, "b.", "Training losses and observable inputs", 0.030, 0.335)

    loss_y = 0.165
    losses = [
        (0.070, "outlet", "HT/HTO histories"),
        (0.245, "PDE", "gas + solid residuals"),
        (0.420, "IC/BC", "inlet, outlet, initial"),
        (0.595, "calibration", "detector + RTD"),
        (0.770, "balance", "tritium balance"),
    ]
    for x, title, sub in losses:
        add_rect(ax, x, loss_y, 0.135, 0.074, title, sub, fc=COL["loss"], ec=COL["red"], fs=9.2)

    bus_y = 0.272
    line(ax, (0.138, bus_y), (0.838, bus_y), lw=2.0, color=COL["red"], dashed=True)
    for x, _, _ in losses:
        xc = x + 0.0675
        arrow(ax, (xc, bus_y), (xc, loss_y + 0.074), lw=1.9, color=COL["red"], dashed=True, mutation_scale=11)

    ax.text(
        0.500,
        0.120,
        r"optimize $\theta$:  $w_y\mathcal{L}_{y}+w_r\mathcal{L}_{r}+w_b\mathcal{L}_{\rm IC/BC}+w_c\mathcal{L}_{\rm cal}+w_m\mathcal{L}_{\rm bal}$",
        ha="center",
        va="center",
        fontsize=11.4,
        color=COL["red"],
        weight="bold",
    )

    add_rect(
        ax,
        0.070,
        0.035,
        0.255,
        0.060,
        "observable inputs",
        "outlets, RTD/pressure, T, detector calibration",
        fc=COL["ok"],
        ec=COL["green"],
        fs=9.0,
    )
    add_rect(
        ax,
        0.375,
        0.035,
        0.230,
        0.060,
        "field-fit checks",
        "same-model fields and residual rows",
        fc="#F6F6F6",
        ec="#777777",
        fs=9.0,
    )
    add_rect(
        ax,
        0.655,
        0.035,
        0.275,
        0.060,
        "solid-state checks",
        "inventories/source labels only for checks",
        fc=COL["withheld"],
        ec=COL["red"],
        fs=9.0,
    )
    ax.text(
        0.792,
        0.020,
        "excluded from outlet-inverse inputs",
        ha="center",
        va="top",
        color=COL["red"],
        fontsize=8.8,
        weight="bold",
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    spec = {
        "figure_goal": "Explain how the PINN source inverse uses observable histories, 3D transport response, chemistry and calibration data, while keeping solid-state checks excluded from outlet-inverse inputs.",
        "model_name": "Li2TiO3 3D packed-bed constrained PINN inverse with explicit source-term output",
        "visual_grammar": "pipeline + module inset",
        "inputs": ["z,t", "u,T,phi,xi", "3D RTD kernel h_z^r(t)", "HT/HTO chemistry", "detector calibration"],
        "trunk_architecture": "four hidden layers, width 96, tanh activation; representative nodes drawn for readability",
        "branches": ["gas-field branch", "inventory branch", "source-term layer"],
        "source_branch": "eight physics-basis terms in the 3D source-recovery test",
        "loss_terms": ["outlet HT/HTO", "gas/material PDE residual", "IC/BC", "detector/RTD calibration", "tritium balance"],
        "solid_inventory_checks": ["solid inventories", "target source labels"],
        "assumptions": ["The schematic draws representative neurons while the implemented hidden-layer width is 96."],
    }
    (RES / "pinn_source_head_architecture_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11.2, 8.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.985,
        "Physics-informed neural-network inverse for delayed tritium-release sources",
        ha="center",
        va="top",
        fontsize=14.2,
        weight="bold",
    )
    ax.text(
        0.5,
        0.955,
        "solid arrows: inference flow    dashed red: training losses    solid inventories: checks excluded from outlet-inverse inputs",
        ha="center",
        va="top",
        fontsize=9.2,
        color=COL["muted"],
        weight="bold",
    )

    draw_panel_a(ax)
    draw_panel_b(ax)

    fig.subplots_adjust(left=0.015, right=0.995, top=0.995, bottom=0.015)
    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(FIG / f"pinn_source_head_architecture.{suffix}", dpi=420)
    plt.close(fig)

    summary = {
        "figure": "pinn_source_head_architecture",
        "generated_files": [
            "figures/pinn_source_head_architecture.pdf",
            "figures/pinn_source_head_architecture.png",
            "figures/pinn_source_head_architecture.svg",
            "results/pinn_source_head_architecture/pinn_source_head_architecture_spec.json",
        ],
        "network_inputs": spec["inputs"],
        "trunk_architecture": spec["trunk_architecture"],
        "network_branches": spec["branches"],
        "source_branch": spec["source_branch"],
        "loss_terms_drawn": spec["loss_terms"],
        "solid_inventory_checks": spec["solid_inventory_checks"],
        "qa_notes": [
            "main data-flow arrows are horizontal or vertical",
            "network layers are fully connected in the representative-node display",
            "loss signals are separated from inference flow",
            "fixed canvas save prevents later auto-crop and aspect-ratio regressions",
        ],
    }
    (RES / "pinn_source_head_architecture_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIG / 'pinn_source_head_architecture.pdf'}")
    print(f"wrote {FIG / 'pinn_source_head_architecture.svg'}")
    print(f"wrote {RES / 'pinn_source_head_architecture_spec.json'}")


if __name__ == "__main__":
    main()
