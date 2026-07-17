"""Shared publication figure style for the tritium PINN manuscript."""

from __future__ import annotations

import matplotlib.pyplot as plt


BLACK = "#000000"
GRID = "#EFEFEF"


def apply_prl_style() -> None:
    """Use a compact PRL-like style for manuscript figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.4,
            "mathtext.fontset": "stix",
            "font.weight": "normal",
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "axes.labelsize": 7.8,
            "axes.titlesize": 7.8,
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.linewidth": 0.95,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.72,
            "ytick.minor.width": 0.72,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
            "lines.linewidth": 1.22,
            "lines.markersize": 3.8,
            "legend.fontsize": 5.9,
            "legend.labelcolor": BLACK,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_prl_axes(ax, *, grid: bool = True) -> None:
    """Apply consistent black rectangular frames and inward ticks."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.95)
        spine.set_color(BLACK)
    ax.tick_params(axis="both", which="major", length=4.0, width=0.9, pad=2.2, colors=BLACK)
    ax.tick_params(axis="both", which="minor", length=2.4, width=0.72, colors=BLACK)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_color(BLACK)
        tick_label.set_fontweight("normal")
    ax.xaxis.label.set_color(BLACK)
    ax.yaxis.label.set_color(BLACK)
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")
    ax.title.set_color(BLACK)
    ax.title.set_fontweight("normal")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(BLACK)
            text.set_fontweight("normal")
        legend.get_frame().set_edgecolor(BLACK)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.42, zorder=0)


def panel_label(ax, label: str, *, x: float = -0.065, y: float = 1.02) -> None:
    """Place a bold outside panel label; captions carry panel descriptions."""
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
        color=BLACK,
    )
    ax.set_title("")


def save_prl_canvas(fig, pdf_path, png_path, *, dpi: int = 450) -> None:
    """Save with the declared figure canvas instead of auto-cropping.

    The manuscript scales figures by their PDF bounding box.  Using
    ``bbox_inches='tight'`` makes that box drift when labels or legends move,
    which previously reintroduced stretched multi-panel figures after a
    regeneration.  Keep the canvas fixed and control whitespace through
    ``subplots_adjust`` in each figure script.
    """
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
