"""Small plotting helpers for Nuclear Fusion style PINN figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


BLACK = "#111111"
BLUE = "#0057D9"
RED = "#D40000"
GREEN = "#008A00"
GRAY = "#666666"


def setup_nf_pinn_style(font_size: float = 8.2) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.linewidth": 0.85,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax) -> None:
    ax.grid(True, color="#E6E6E6", lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=12, fontweight="bold")


def bounded_legend(ax, **kwargs) -> None:
    legend = ax.legend(**kwargs)
    if legend is not None:
        legend.get_frame().set_linewidth(0.0)
