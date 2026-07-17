#!/usr/bin/env python3
"""Draw the manuscript-level PINN observability workflow figure.

The figure is intentionally schematic. It shows the logical interfaces that
make the inverse problem defensible: material release evidence, designed
observations and 3D packed-bed fields jointly constrain a PINN source inverse;
independent physical tests then feed the blanket tritium-balance calculation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from locked_figure_guard import require_locked_figure_regeneration_allowed


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results" / "methodology_overview_figure"
require_locked_figure_regeneration_allowed(
    FIG / "methodology_overview_figure.pdf",
    FIG / "methodology_overview_figure.png",
    FIG / "methodology_overview_figure.svg",
)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

COL = {
    "ink": "#101010",
    "muted": "#5B5B5B",
    "grid": "#D8D8D8",
    "blue": "#0057D9",
    "red": "#D40000",
    "green": "#008A2E",
    "amber": "#D89000",
    "purple": "#6B4AB5",
    "source": "#F9F1DA",
    "measure": "#F8F8F8",
    "field": "#EAF2FC",
    "inverse": "#F0EAF8",
    "test": "#FCECEC",
    "scale": "#E8F4EC",
}


def rect(ax, x, y, w, h, fc="white", ec=None, lw=1.0, z=1):
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec or COL["ink"], linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def text(ax, x, y, s, fs=7.2, color=None, weight=None, ha="center", va="center", **kwargs):
    ax.text(
        x,
        y,
        s,
        fontsize=fs,
        color=color or COL["ink"],
        weight=weight,
        ha=ha,
        va=va,
        linespacing=kwargs.pop("linespacing", 1.05),
        **kwargs,
    )


def line(ax, p0, p1, color=None, lw=1.2, dashed=False, z=2):
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color or COL["ink"],
        lw=lw,
        linestyle=(0, (5, 3)) if dashed else "solid",
        solid_capstyle="butt",
        zorder=z,
    )


def arrow(ax, p0, p1, color=None, lw=1.8, dashed=False, scale=12, z=4):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color or COL["ink"],
            linestyle=(0, (5, 3)) if dashed else "solid",
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0",
            zorder=z,
        )
    )


def ortho_arrow(ax, pts, color=None, lw=1.45, dashed=False, scale=11):
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        line(ax, p0, p1, color=color, lw=lw, dashed=dashed, z=3)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, dashed=dashed, scale=scale, z=4)


def dot(ax, x, y, size=65, fc="white", ec=None, lw=0.8, z=5):
    ax.scatter([x], [y], s=size, marker="o", facecolors=fc, edgecolors=ec or COL["ink"], linewidths=lw, zorder=z)


def card(ax, x, y, w, h, title, subtitle, fc, accent):
    rect(ax, x, y, w, h, fc=fc, lw=1.35)
    rect(ax, x, y + h - 0.026, w, 0.026, fc=accent, ec=accent, lw=0, z=2)
    text(ax, x + 0.014, y + h - 0.055, title, fs=8.2, weight="bold", ha="left")
    if subtitle:
        text(ax, x + 0.014, y + h - 0.087, subtitle, fs=5.8, color=COL["muted"], ha="left")


def label_box(ax, x, y, w, h, label, color, fs=5.8):
    rect(ax, x, y, w, h, fc="white", ec=color, lw=0.95, z=5)
    text(ax, x + w / 2, y + h / 2, label, fs=fs, color=COL["ink"], zorder=6)


def connector_label(ax, x, y, label, color, w=0.118):
    rect(ax, x - w / 2, y - 0.020, w, 0.040, fc="white", ec=color, lw=0.80, z=7)
    text(ax, x, y, label, fs=5.35, color=color, weight="bold", zorder=8)


def draw_pebble_source(ax, x, y, w, h):
    centers = [
        (0.17, 0.30, 0.030),
        (0.29, 0.33, 0.037),
        (0.42, 0.30, 0.029),
        (0.23, 0.46, 0.026),
        (0.37, 0.48, 0.030),
        (0.49, 0.43, 0.024),
    ]
    for cx, cy, r in centers:
        dot(ax, x + w * cx, y + h * cy, size=760 * r, fc="#FFF8D7", lw=0.78, z=3)
    # Two delayed reservoirs are shown as material states, not as fitted boxes.
    label_box(ax, x + w * 0.58, y + h * 0.36, w * 0.32, h * 0.24, "open\npath", COL["red"])
    label_box(ax, x + w * 0.58, y + h * 0.13, w * 0.32, h * 0.19, "delayed\npath", COL["red"], fs=5.35)
    ortho_arrow(
        ax,
        [(x + w * 0.47, y + h * 0.39), (x + w * 0.52, y + h * 0.39), (x + w * 0.52, y + h * 0.48), (x + w * 0.58, y + h * 0.48)],
        color=COL["red"],
        lw=1.05,
        scale=8,
    )
    ortho_arrow(
        ax,
        [(x + w * 0.47, y + h * 0.31), (x + w * 0.52, y + h * 0.31), (x + w * 0.52, y + h * 0.22), (x + w * 0.58, y + h * 0.22)],
        color=COL["red"],
        lw=1.05,
        scale=8,
    )
    text(ax, x + w * 0.18, y + h * 0.13, "TDS peaks\nexternal curve", fs=5.6, color=COL["muted"], ha="left")


def draw_observations(ax, x, y, w, h):
    # Four designed histories are encoded as parallel measurement tracks rather
    # than repeated curve sketches.
    tracks = [("24 h", COL["ink"]), ("48 h", COL["ink"]), ("slow", COL["amber"]), ("fast", COL["amber"])]
    for i, (lab, col) in enumerate(tracks):
        yy = y + h * (0.19 + 0.105 * i)
        line(ax, (x + w * 0.13, yy), (x + w * 0.49, yy), color=col, lw=2.15)
        text(ax, x + w * 0.54, yy, lab, fs=5.8, ha="left", color=col)
    label_box(ax, x + w * 0.68, y + h * 0.31, w * 0.22, h * 0.23, "HT\nHTO", COL["blue"])
    label_box(ax, x + w * 0.68, y + h * 0.09, w * 0.22, h * 0.16, "p, T", COL["amber"])


def draw_3d_response(ax, x, y, w, h):
    # A simple voxel slab with flow and heat vectors.
    base_y = y - h * 0.015
    front = np.array(
        [
            [x + w * 0.16, base_y + h * 0.13],
            [x + w * 0.58, base_y + h * 0.13],
            [x + w * 0.58, base_y + h * 0.38],
            [x + w * 0.16, base_y + h * 0.38],
        ]
    )
    offset = np.array([w * 0.10, h * 0.073])
    back = front + offset
    ax.add_patch(Polygon(back, closed=True, facecolor="#F6FBFF", edgecolor=COL["blue"], lw=0.9, zorder=2))
    ax.add_patch(Polygon(front, closed=True, facecolor="#FFFFFF", edgecolor=COL["blue"], lw=1.05, zorder=3))
    for p, q in zip(front, back):
        line(ax, p, q, color=COL["blue"], lw=0.75, z=2)
    for i in range(4):
        for j in range(3):
            xx = x + w * (0.245 + 0.082 * i)
            yy = base_y + h * (0.230 + 0.064 * j)
            rect(ax, xx, yy, w * 0.060, h * 0.045, fc="#DDEBFF", ec=COL["blue"], lw=0.45, z=4)
    for yy in [0.20, 0.30, 0.38]:
        arrow(ax, (x + w * 0.08, base_y + h * yy), (x + w * 0.73, base_y + h * yy), color=COL["blue"], lw=0.9, scale=7)
    text(ax, x + w * 0.80, base_y + h * 0.36, "RTD\nheat\nchem.", fs=5.35, color=COL["blue"], ha="center")


def draw_network(ax, x, y, w, h):
    # Four visible layers match the manuscript-level MLP representation while
    # keeping the architecture schematic readable at journal scale.
    layers = [3, 5, 5, 3]
    xs = np.linspace(x + w * 0.14, x + w * 0.80, len(layers))
    r = min(w, h) * 0.022
    layer_nodes = []
    for li, n in enumerate(layers):
        ys = np.linspace(y + h * 0.28, y + h * 0.66, n)
        nodes = []
        for yy in ys:
            dot(ax, xs[li], yy, size=64, fc="white", lw=0.75, z=5)
            nodes.append((xs[li], yy))
        layer_nodes.append(nodes)
    for left, right in zip(layer_nodes[:-1], layer_nodes[1:]):
        for p0 in left:
            for p1 in right:
                line(ax, (p0[0] + r, p0[1]), (p1[0] - r, p1[1]), color="#A9A9A9", lw=0.35, z=4)
    text(ax, x + w * 0.14, y + h * 0.21, "$z,t,\\xi$", fs=6.1, color=COL["muted"])
    text(ax, x + w * 0.80, y + h * 0.21, "$C_{HT}, C_{HTO}$\n$S_o,S_c$", fs=5.7, color=COL["muted"])
    label_box(ax, x + w * 0.17, y + h * 0.69, w * 0.62, h * 0.105, "PDE residual + outlet loss", COL["red"], fs=6.1)
    label_box(ax, x + w * 0.18, y + h * 0.07, w * 0.60, h * 0.11, "source terms: open / delayed", COL["purple"], fs=6.0)


def draw_validation(ax, x, y, w, h):
    rows = [("source family", COL["red"]), ("3D field", COL["blue"]), ("chemistry", COL["amber"])]
    for i, (lab, col) in enumerate(rows):
        yy = y + h * (0.475 - 0.175 * i)
        label_box(ax, x + w * 0.085, yy - h * 0.045, w * 0.43, h * 0.090, lab, col, fs=5.45)
        label_box(ax, x + w * 0.655, yy - h * 0.045, w * 0.235, h * 0.090, "test", col, fs=5.35)
        arrow(ax, (x + w * 0.515, yy), (x + w * 0.655, yy), color=col, lw=0.95, scale=7)


def draw_balance(ax, x, y, w, h):
    gx = x + w * 0.12
    gy = y + h * 0.39
    for i in range(5):
        for j in range(3):
            fc = "#DDF1E2" if (i + j) % 2 else "#FFFFFF"
            rect(ax, gx + i * w * 0.070, gy + j * h * 0.075, w * 0.054, h * 0.052, fc=fc, ec=COL["green"], lw=0.48, z=3)
    label_box(ax, x + w * 0.60, y + h * 0.31, w * 0.31, h * 0.155, "$G=R+I+L$", COL["green"], fs=6.15)
    text(ax, x + w * 0.50, y + h * 0.14, "sector outlets\ninventory and TES", fs=5.55, color=COL["muted"])


def draw_workflow(ax) -> None:
    # Left evidence column.
    lx, lwid, ch = 0.030, 0.285, 0.215
    ys = [0.655, 0.390, 0.125]
    cards = [
        ("Material release basis", "Li$_2$TiO$_3$ pathway prior", COL["source"], COL["red"], draw_pebble_source),
        ("Designed observations", "late time and residence time", COL["measure"], COL["amber"], draw_observations),
        ("3D packed-bed field", "velocity, heat and chemistry", COL["field"], COL["blue"], draw_3d_response),
    ]
    for y, (title, subtitle, fc, accent, drawer) in zip(ys, cards):
        card(ax, lx, y, lwid, ch, title, subtitle, fc, accent)
        drawer(ax, lx, y, lwid, ch)

    # Central PINN inverse.
    cx, cy, cw, cH = 0.425, 0.235, 0.255, 0.560
    card(ax, cx, cy, cw, cH, "PINN source inverse", "physics-informed multi-condition model", COL["inverse"], COL["red"])
    draw_network(ax, cx, cy, cw, cH)

    # Right validation and system propagation.
    rx, rwid = 0.755, 0.220
    card(ax, rx, 0.530, rwid, 0.305, "Independent histories", "source, 3D field and chemistry", COL["test"], COL["red"])
    draw_validation(ax, rx, 0.530, rwid, 0.305)
    card(ax, rx, 0.140, rwid, 0.285, "Blanket balance", "module and component propagation", COL["scale"], COL["green"])
    draw_balance(ax, rx, 0.140, rwid, 0.285)

    # Interfaces into the inverse model.
    source_mid = (lx + lwid, ys[0] + ch * 0.50)
    obs_mid = (lx + lwid, ys[1] + ch * 0.50)
    field_mid = (lx + lwid, ys[2] + ch * 0.50)
    targets = [
        (cx, source_mid[1], "release\nbasis", COL["red"]),
        (cx, obs_mid[1], "observed\nhistories", COL["amber"]),
        (cx, field_mid[1], "3D\nresponse", COL["blue"]),
    ]
    label_x = 0.362
    label_offsets = [-0.046, 0.046, 0.046]
    for p0, (tx, ty, lab, col), dy in zip([source_mid, obs_mid, field_mid], targets, label_offsets):
        arrow(ax, (p0[0] + 0.014, ty), (tx - 0.020, ty), color=col, lw=1.85, scale=10)
        connector_label(ax, label_x, ty + dy, lab, col, w=0.078)

    # Main prediction and source-to-balance flow.
    pred_y = 0.690
    arrow(ax, (cx + cw, pred_y), (rx - 0.010, pred_y), color=COL["red"], lw=2.05, scale=12)
    connector_label(ax, 0.720, pred_y - 0.058, "independent\nprediction", COL["red"], w=0.088)
    arrow(ax, (rx + rwid * 0.50, 0.530), (rx + rwid * 0.50, 0.425), color=COL["green"], lw=1.65, scale=10)
    connector_label(ax, rx + rwid * 0.50, 0.475, "source estimate\nfor balance", COL["green"], w=0.100)

    # Feedback loop from physical tests to the observation and field-design cards.
    # It is routed through the blank gutter so no line passes through a card.
    top = 0.922
    fb_x = 0.015
    fb_right_x = 0.990
    line(ax, (rx + rwid, 0.760), (fb_right_x, 0.760), color=COL["red"], lw=1.35, dashed=True)
    line(ax, (fb_right_x, 0.760), (fb_right_x, top), color=COL["red"], lw=1.35, dashed=True)
    line(ax, (fb_right_x, top), (fb_x, top), color=COL["red"], lw=1.35, dashed=True)
    line(ax, (fb_x, top), (fb_x, ys[2] + ch + 0.020), color=COL["red"], lw=1.35, dashed=True)
    for y_card_top in [ys[1] + ch, ys[2] + ch]:
        stub_x = lx + 0.032
        line(ax, (fb_x, y_card_top + 0.020), (stub_x, y_card_top + 0.020), color=COL["red"], lw=1.35, dashed=True)
        arrow(ax, (stub_x, y_card_top + 0.020), (stub_x, y_card_top + 0.002), color=COL["red"], lw=1.35, dashed=True, scale=9)
    text(ax, 0.520, top + 0.030, "prediction feedback to purge histories and bed-field coverage", fs=6.7, color=COL["red"], weight="bold")

    # Reported outputs: visually tied to inverse and balance, not a decorative footer.
    ox, oy, ow, oh = 0.365, 0.030, 0.485, 0.075
    rect(ax, ox, oy, ow, oh, fc="#FFFFFF", ec=COL["green"], lw=1.35)
    text(ax, ox + ow / 2, oy + oh * 0.66, "tritium source estimate", fs=8.0, color=COL["green"], weight="bold")
    text(ax, ox + ow / 2, oy + oh * 0.30, "release multipliers, outlet histories, retained inventory and extraction balance", fs=5.9, color=COL["muted"])
    arrow(ax, (cx + cw * 0.50, cy), (cx + cw * 0.50, oy + oh), color=COL["green"], lw=1.25, scale=9)
    ortho_arrow(ax, [(rx + rwid * 0.50, 0.140), (rx + rwid * 0.50, oy + oh + 0.020), (ox + ow * 0.82, oy + oh + 0.020), (ox + ow * 0.82, oy + oh)], color=COL["green"], lw=1.25, scale=9)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    spec = {
        "figure_goal": "Manuscript-level PINN delayed-source observability workflow with explicit measurement interfaces.",
        "workflow": [
            "material release basis",
            "designed outlet and auxiliary observations",
            "3D packed-bed transport, chemistry and thermal fields",
            "PINN source inverse with PDE residual and outlet loss",
            "withheld release histories",
            "module and component tritium balance propagation",
        ],
        "interfaces": [
            "release basis",
            "observed histories",
            "3D response",
            "withheld prediction",
            "source estimate for balance",
            "tritium source estimate",
        ],
        "style": "single workflow, no panel letters, no decorative repeated curves, explicit connector labels, orthogonal main arrows",
    }
    (RES / "methodology_overview_figure_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.85, 3.70))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_workflow(ax)

    fig.subplots_adjust(left=0.010, right=0.995, top=0.995, bottom=0.012)
    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(FIG / f"methodology_overview_figure.{suffix}", dpi=430)
    plt.close(fig)

    summary = {
        "figure": "methodology_overview_figure",
        "generated_files": [
            "figures/methodology_overview_figure.pdf",
            "figures/methodology_overview_figure.png",
            "figures/methodology_overview_figure.svg",
        ],
        "workflow": spec["workflow"],
        "interfaces": spec["interfaces"],
        "qa_notes": [
            "replaced repetitive internal release curves with evidence-specific icons",
            "main workflow arrows are horizontal or vertical orthogonal connectors",
            "screen-space circular markers avoid compressed nodes",
            "central PINN block receives three separate physical inputs",
            "each connector names the transferred quantity",
            "feedback loop links withheld-prediction error to observation and bed-field coverage",
            "bottom output is connected to inverse estimates and blanket balance propagation",
            "fixed canvas save prevents later auto-crop and aspect-ratio regressions",
        ],
    }
    (RES / "methodology_overview_figure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIG / 'methodology_overview_figure.pdf'}")
    print(f"wrote {FIG / 'methodology_overview_figure.svg'}")
    print(f"wrote {RES / 'methodology_overview_figure_summary.json'}")


if __name__ == "__main__":
    main()
