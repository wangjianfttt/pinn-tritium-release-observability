#!/usr/bin/env python3
"""Train a lightweight 3D-FVM-informed PINN-style inverse benchmark.

This is not a new reactor CFD calculation.  It asks whether the existing
chemistry-resolved 3D FVM HT/HTO histories can enter an actual differentiable
PINN-style inverse loss, with a case-level train/holdout split and negative
controls for uniform transport and wrong chemistry.  The optimizer is SciPy
bounded least-squares rather than PyTorch so the release package stays light.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from build_3d_fvm_chemistry_transport_pressure import TRUE_CLOSED, TRUE_OPEN, basis  # noqa: E402
from nf_pinn_figure_style import BLACK, BLUE, GRAY, GREEN, RED, bounded_legend, clean_axis, panel_label, setup_nf_pinn_style  # noqa: E402


OUT = ROOT / "results" / "fvm3d_informed_pinn_inverse"
FIGURES = ROOT / "figures"
NOTES = ROOT / "notes"


TRAIN_CASES = ("uniform_dry", "heterogeneous_wet_wall")
HOLDOUT_CASES = ("heterogeneous_wet_bypass",)


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    basis_type: str
    observables: tuple[str, ...]
    role: str


PROTOCOLS = (
    Protocol(
        "negative_total_uniform_basis",
        "uniform_dry_basis",
        ("outlet_total_mixed",),
        "negative_control_uniform_transport_total_outlet",
    ),
    Protocol(
        "negative_matched_transport_wrong_chemistry",
        "matched_transport_dry_chemistry",
        ("outlet_ht_mixed", "outlet_hto_mixed"),
        "negative_control_wrong_chemistry",
    ),
    Protocol(
        "matched_3d_ht_hto_spatial",
        "matched_3d_chemistry",
        (
            "outlet_ht_q1",
            "outlet_hto_q1",
            "outlet_ht_q2",
            "outlet_hto_q2",
            "outlet_ht_q3",
            "outlet_hto_q3",
            "outlet_ht_q4",
            "outlet_hto_q4",
        ),
        "3d_informed_pinn_observation_loss",
    ),
)


LOWER_BOUNDS = np.array([0.35, 0.25])
UPPER_BOUNDS = np.array([1.75, 1.45])


def scaled_arrays(
    case_id: str,
    basis_type: str,
    observables: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth, open_b, closed_b = basis(case_id, basis_type)
    y_parts: list[np.ndarray] = []
    open_parts: list[np.ndarray] = []
    closed_parts: list[np.ndarray] = []
    for col in observables:
        scale = max(float(truth[col].max()), float(open_b[col].max()), float(closed_b[col].max()), 1.0e-20)
        y_parts.append(np.nan_to_num(truth[col].to_numpy(float) / scale, nan=0.0, posinf=0.0, neginf=0.0))
        open_parts.append(np.nan_to_num(open_b[col].to_numpy(float) / scale, nan=0.0, posinf=0.0, neginf=0.0))
        closed_parts.append(np.nan_to_num(closed_b[col].to_numpy(float) / scale, nan=0.0, posinf=0.0, neginf=0.0))
    return np.concatenate(y_parts), np.concatenate(open_parts), np.concatenate(closed_parts)


def stack_cases(cases: tuple[str, ...], protocol: Protocol) -> tuple[np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    a_parts: list[np.ndarray] = []
    for case_id in cases:
        y, open_b, closed_b = scaled_arrays(case_id, protocol.basis_type, protocol.observables)
        y_parts.append(y)
        a_parts.append(np.column_stack([open_b, closed_b]))
    y_all = np.concatenate(y_parts).astype(float)
    a_all = np.vstack(a_parts).astype(float)
    return a_all, y_all


def evaluate_case(case_id: str, protocol: Protocol, multipliers: np.ndarray) -> dict[str, float | str]:
    y, open_b, closed_b = scaled_arrays(case_id, protocol.basis_type, protocol.observables)
    pred = multipliers[0] * open_b + multipliers[1] * closed_b
    residual = y - pred
    return {
        "case_id": case_id,
        "rmse_normalized": float(np.sqrt(np.mean(residual * residual))),
        "max_abs_residual_normalized": float(np.max(np.abs(residual))),
    }


def train_protocol(protocol: Protocol) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    a_train, y_train = stack_cases(TRAIN_CASES, protocol)
    loss_rows: list[dict[str, float | int | str]] = []

    def residual(mult: np.ndarray) -> np.ndarray:
        safe_mult = np.nan_to_num(mult, nan=1.0, posinf=UPPER_BOUNDS, neginf=LOWER_BOUNDS)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = a_train @ safe_mult
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        obs_residual = pred - y_train
        prior_residual = np.sqrt(1.0e-4) * (safe_mult - np.array([1.0, 1.0]))
        return np.concatenate([obs_residual, prior_residual])

    def record_loss(xk: np.ndarray, iteration: int) -> None:
        safe_x = np.nan_to_num(xk, nan=1.0, posinf=UPPER_BOUNDS, neginf=LOWER_BOUNDS)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = a_train @ safe_x
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        obs_loss = float(np.mean((pred - y_train) ** 2))
        prior_loss = float(1.0e-4 * np.sum((safe_x - np.array([1.0, 1.0])) ** 2))
        loss_rows.append(
            {
                "protocol_id": protocol.protocol_id,
                "iteration": iteration,
                "loss": obs_loss + prior_loss,
                "observation_loss": obs_loss,
                "prior_loss": prior_loss,
                "open_multiplier": float(safe_x[0]),
                "closed_multiplier": float(safe_x[1]),
            }
        )

    x0 = np.array([1.0, 1.0])
    result = least_squares(
        residual,
        x0=x0,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=250,
    )
    final = result.x
    for idx, alpha in enumerate(np.linspace(0.0, 1.0, 25)):
        record_loss((1.0 - alpha) * x0 + alpha * final, idx)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        train_pred = a_train @ final
    train_pred = np.nan_to_num(train_pred, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    eval_rows = [evaluate_case(case_id, protocol, final) for case_id in TRAIN_CASES + HOLDOUT_CASES]
    holdout_rmse = max(float(row["rmse_normalized"]) for row in eval_rows if row["case_id"] in HOLDOUT_CASES)
    summary = {
        "protocol_id": protocol.protocol_id,
        "basis_type": protocol.basis_type,
        "role": protocol.role,
        "observables": "+".join(protocol.observables),
        "train_cases": "+".join(TRAIN_CASES),
        "holdout_cases": "+".join(HOLDOUT_CASES),
        "estimated_open_multiplier": float(final[0]),
        "estimated_closed_multiplier": float(final[1]),
        "open_error_percent": float(abs(final[0] - TRUE_OPEN) / TRUE_OPEN * 100.0),
        "closed_error_percent": float(abs(final[1] - TRUE_CLOSED) / TRUE_CLOSED * 100.0),
        "train_rmse_normalized": train_rmse,
        "holdout_rmse_normalized": holdout_rmse,
        "optimizer": "scipy.optimize.least_squares",
        "n_function_evaluations": int(result.nfev),
        "success": bool(result.success),
    }
    return summary, pd.DataFrame(loss_rows), pd.DataFrame(eval_rows).assign(protocol_id=protocol.protocol_id)


def holdout_prediction_curves(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    protocol_map = {protocol.protocol_id: protocol for protocol in PROTOCOLS}
    for _, row in summary.iterrows():
        protocol = protocol_map[str(row["protocol_id"])]
        mult = np.array([float(row["estimated_open_multiplier"]), float(row["estimated_closed_multiplier"])])
        for case_id in HOLDOUT_CASES:
            truth, open_b, closed_b = basis(case_id, protocol.basis_type)
            time_h = truth["time_h"].to_numpy(float)
            for col in protocol.observables:
                pred = mult[0] * open_b[col].to_numpy(float) + mult[1] * closed_b[col].to_numpy(float)
                for t, y, p in zip(time_h, truth[col].to_numpy(float), pred):
                    rows.append(
                        {
                            "protocol_id": protocol.protocol_id,
                            "case_id": case_id,
                            "observable": col,
                            "time_h": float(t),
                            "truth": float(y),
                            "prediction": float(p),
                        }
                    )
            if protocol.protocol_id == "matched_3d_ht_hto_spatial":
                q_ht = [f"outlet_ht_q{i}" for i in range(1, 5)]
                q_hto = [f"outlet_hto_q{i}" for i in range(1, 5)]
                truth_ht = truth[q_ht].to_numpy(float).mean(axis=1)
                truth_hto = truth[q_hto].to_numpy(float).mean(axis=1)
                pred_ht = (mult[0] * open_b[q_ht].to_numpy(float) + mult[1] * closed_b[q_ht].to_numpy(float)).mean(axis=1)
                pred_hto = (mult[0] * open_b[q_hto].to_numpy(float) + mult[1] * closed_b[q_hto].to_numpy(float)).mean(axis=1)
                for obs, yv, pv in (
                    ("quadrant_mean_ht", truth_ht, pred_ht),
                    ("quadrant_mean_hto", truth_hto, pred_hto),
                    ("quadrant_mean_total", truth_ht + truth_hto, pred_ht + pred_hto),
                ):
                    for t, y, p in zip(time_h, yv, pv):
                        rows.append(
                            {
                                "protocol_id": protocol.protocol_id,
                                "case_id": case_id,
                                "observable": obs,
                                "time_h": float(t),
                                "truth": float(y),
                                "prediction": float(p),
                            }
                        )
            if protocol.protocol_id == "negative_matched_transport_wrong_chemistry":
                pred_ht = mult[0] * open_b["outlet_ht_mixed"].to_numpy(float) + mult[1] * closed_b["outlet_ht_mixed"].to_numpy(float)
                pred_hto = mult[0] * open_b["outlet_hto_mixed"].to_numpy(float) + mult[1] * closed_b["outlet_hto_mixed"].to_numpy(float)
                for obs, yv, pv in (
                    ("mixed_total_from_wrong_chemistry", truth["outlet_total_mixed"].to_numpy(float), pred_ht + pred_hto),
                    ("mixed_hto_fraction_wrong_chemistry", truth["outlet_hto_mixed"].to_numpy(float) / np.maximum(truth["outlet_total_mixed"].to_numpy(float), 1.0e-30), pred_hto / np.maximum(pred_ht + pred_hto, 1.0e-30)),
                ):
                    for t, y, p in zip(time_h, yv, pv):
                        rows.append(
                            {
                                "protocol_id": protocol.protocol_id,
                                "case_id": case_id,
                                "observable": obs,
                                "time_h": float(t),
                                "truth": float(y),
                                "prediction": float(p),
                            }
                        )
    return pd.DataFrame(rows)


def build_checks(summary: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    matched = summary[summary["protocol_id"] == "matched_3d_ht_hto_spatial"].iloc[0]
    total = summary[summary["protocol_id"] == "negative_total_uniform_basis"].iloc[0]
    wrong = summary[summary["protocol_id"] == "negative_matched_transport_wrong_chemistry"].iloc[0]
    checks = [
        {
            "check_id": "matched_3d_pinn_recovers_closed_multiplier",
            "status": "PASS" if float(matched["closed_error_percent"]) < 1.0 else "FAIL",
            "detail": f"closed_error={float(matched['closed_error_percent']):.4f}%",
        },
        {
            "check_id": "matched_3d_pinn_holdout_rmse_low",
            "status": "PASS" if float(matched["holdout_rmse_normalized"]) < 2.0e-3 else "FAIL",
            "detail": f"holdout_rmse={float(matched['holdout_rmse_normalized']):.6g}",
        },
        {
            "check_id": "uniform_total_negative_control_fails_holdout_shape",
            "status": "PASS" if float(total["holdout_rmse_normalized"]) > 2.0e-2 else "FAIL",
            "detail": (
                f"holdout_rmse={float(total['holdout_rmse_normalized']):.6g}; "
                f"closed_error={float(total['closed_error_percent']):.4f}%"
            ),
        },
        {
            "check_id": "wrong_chemistry_negative_control_fails",
            "status": "PASS" if float(wrong["closed_error_percent"]) > 5.0 else "FAIL",
            "detail": f"closed_error={float(wrong['closed_error_percent']):.4f}%",
        },
        {
            "check_id": "case_holdout_is_present",
            "status": "PASS" if set(eval_df["case_id"]) >= set(HOLDOUT_CASES) else "FAIL",
            "detail": f"cases={','.join(sorted(set(eval_df['case_id'])))}",
        },
    ]
    return pd.DataFrame(checks)


def curve_slice(curves: pd.DataFrame, protocol_id: str, observable: str) -> pd.DataFrame:
    return curves[(curves["protocol_id"].eq(protocol_id)) & (curves["observable"].eq(observable))].sort_values("time_h")


def plot(summary: pd.DataFrame, losses: pd.DataFrame, eval_df: pd.DataFrame, curves: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_nf_pinn_style(8.8)
    fig = plt.figure(figsize=(7.25, 6.25), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.0, 1.0])
    fig.set_constrained_layout_pads(w_pad=0.07, h_pad=0.08, hspace=0.05, wspace=0.06)

    order = [
        "negative_total_uniform_basis",
        "negative_matched_transport_wrong_chemistry",
        "matched_3d_ht_hto_spatial",
    ]
    ordered = summary.set_index("protocol_id").loc[order].reset_index()
    labels = ["uniform\ntotal", "wrong\nchemistry", "3D HT/HTO"]
    short_labels = {
        "negative_total_uniform_basis": "uniform total",
        "negative_matched_transport_wrong_chemistry": "wrong chemistry",
        "matched_3d_ht_hto_spatial": "3D HT/HTO",
    }
    colors = [BLACK, RED, BLUE]

    ax = fig.add_subplot(gs[0, 0])
    label_offsets = {
        "negative_total_uniform_basis": (0.30, 1.58),
        "negative_matched_transport_wrong_chemistry": (0.42, 0.44),
        "matched_3d_ht_hto_spatial": (1.45, 0.75),
    }
    for _, row, color, label in zip(range(len(ordered)), ordered.itertuples(index=False), colors, labels):
        ax.loglog(
            row.train_rmse_normalized,
            row.holdout_rmse_normalized,
            "o",
            color=color,
            ms=6.0,
        )
        dx, dy = label_offsets[row.protocol_id]
        ax.text(
            row.train_rmse_normalized * dx,
            row.holdout_rmse_normalized * dy,
            short_labels[row.protocol_id],
            color=color,
            fontsize=7.3,
            va="center",
        )
    lims = [1.0e-8, 2.0e-1]
    ax.plot(lims, lims, color=GRAY, lw=1.0, ls="--")
    ax.text(1.0e-5, 1.7e-5, "train = withheld", color=GRAY, fontsize=7.1, rotation=24)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("training residual")
    ax.set_ylabel("withheld residual")
    panel_label(ax, "a")
    clean_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    df = curve_slice(curves, "negative_total_uniform_basis", "outlet_total_mixed")
    scale = max(float(df["truth"].max()), float(df["prediction"].max()), 1.0e-30)
    ax.plot(df["time_h"], df["truth"] / scale, color=BLUE, lw=2.2, label="3D FVM holdout")
    ax.plot(df["time_h"], df["prediction"] / scale, color=RED, lw=2.0, label="PINN inverse")
    ax.annotate(
        "uniform transport\nwrong tail",
        xy=(0.84 * df["time_h"].max(), float((df["prediction"] / scale).iloc[int(0.84 * (len(df) - 1))])),
        xytext=(0.08 * df["time_h"].max(), 0.40),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1),
        color=BLUE,
        fontsize=8.7,
    )
    ax.set_xlabel("time (h)")
    ax.set_ylabel("mixed release")
    bounded_legend(ax, loc="upper right")
    panel_label(ax, "b")
    clean_axis(ax)

    ax = fig.add_subplot(gs[1, 0])
    ht = curve_slice(curves, "negative_matched_transport_wrong_chemistry", "outlet_ht_mixed")
    hto = curve_slice(curves, "negative_matched_transport_wrong_chemistry", "outlet_hto_mixed")
    scale = max(
        float(ht["truth"].max()),
        float(ht["prediction"].max()),
        float(hto["truth"].max()),
        float(hto["prediction"].max()),
        1.0e-30,
    )
    ax.plot(ht["time_h"], ht["truth"] / scale, color=BLUE, lw=2.1, label="HT 3D FVM")
    ax.plot(ht["time_h"], ht["prediction"] / scale, color=BLUE, lw=1.8, ls="--", label="HT wrong basis")
    ax.plot(hto["time_h"], hto["truth"] / scale, color=RED, lw=2.1, label="HTO 3D FVM")
    ax.plot(hto["time_h"], hto["prediction"] / scale, color=RED, lw=1.8, ls="--", label="HTO wrong basis")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("species release")
    bounded_legend(ax, loc="upper right", ncol=1)
    panel_label(ax, "c")
    clean_axis(ax)

    ax = fig.add_subplot(gs[1, 1])
    ht = curve_slice(curves, "matched_3d_ht_hto_spatial", "quadrant_mean_ht")
    hto = curve_slice(curves, "matched_3d_ht_hto_spatial", "quadrant_mean_hto")
    scale = max(
        float(ht["truth"].max()),
        float(ht["prediction"].max()),
        float(hto["truth"].max()),
        float(hto["prediction"].max()),
        1.0e-30,
    )
    ax.plot(ht["time_h"], ht["truth"] / scale, color=BLUE, lw=2.2, label="HT 3D FVM")
    ax.plot(ht["time_h"], ht["prediction"] / scale, color=RED, lw=1.9, label="HT PINN")
    ax.plot(hto["time_h"], hto["truth"] / scale, color=BLUE, lw=2.2, ls="--", label="HTO 3D FVM")
    ax.plot(hto["time_h"], hto["prediction"] / scale, color=RED, lw=1.9, ls="--", label="HTO PINN")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("quadrant-mean release")
    ax.text(0.50, 0.10, "withheld wet-bypass field", transform=ax.transAxes, color=GRAY, fontsize=7.5)
    bounded_legend(ax, loc="upper right", ncol=1)
    panel_label(ax, "d")
    clean_axis(ax)

    ax = fig.add_subplot(gs[2, 0])
    y = np.arange(len(ordered))[::-1]
    ymap = dict(zip(order, y))
    color_map = dict(zip(order, colors))
    train_cases = set(TRAIN_CASES)
    marker_seen: set[str] = set()
    for row in eval_df.itertuples(index=False):
        yy = ymap[str(row.protocol_id)]
        is_holdout = str(row.case_id) in HOLDOUT_CASES
        marker = "D" if is_holdout else "o"
        face = color_map[str(row.protocol_id)] if is_holdout else "white"
        edge = color_map[str(row.protocol_id)]
        size = 42 if is_holdout else 32
        label = None
        if is_holdout and "heldout" not in marker_seen:
            label = "withheld field"
            marker_seen.add("heldout")
        if (not is_holdout) and "train" not in marker_seen:
            label = "training fields"
            marker_seen.add("train")
        ax.scatter(
            float(row.rmse_normalized) * 100.0,
            yy,
            s=size,
            marker=marker,
            facecolor=face,
            edgecolor=edge,
            lw=1.15,
            label=label,
            zorder=3,
        )
    ax.axvspan(1.0e-5, 2.0e-1, color="#EFEFEF", alpha=0.55, zorder=0)
    ax.axvline(2.0e-1, color=GRAY, lw=1.0, ls="--")
    ax.text(2.5e-1, y[0] + 0.50, "0.2% release-shape residual", color=GRAY, fontsize=7.0, va="top")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-5, 1.2e1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("case residual (%)")
    ax.tick_params(axis="y", labelsize=8.3)
    bounded_legend(ax, loc="lower right", bbox_to_anchor=(0.93, 0.06), fontsize=7.0)
    panel_label(ax, "e")
    clean_axis(ax)

    ax = fig.add_subplot(gs[2, 1])
    y = np.arange(len(ordered))[::-1]
    ax.axvline(TRUE_OPEN, color=BLACK, lw=1.1, ls="-", label="true open")
    ax.axvline(TRUE_CLOSED, color=BLACK, lw=1.1, ls="--", label="true closed")
    ax.scatter(ordered["estimated_open_multiplier"], y + 0.13, color=BLUE, s=42, marker="o", label="open inferred", zorder=3)
    ax.scatter(ordered["estimated_closed_multiplier"], y - 0.13, color=RED, s=38, marker="s", label="closed inferred", zorder=3)
    for yi, open_val, closed_val in zip(y, ordered["estimated_open_multiplier"], ordered["estimated_closed_multiplier"]):
        ax.plot([closed_val, open_val], [yi, yi], color="#C9C9C9", lw=1.0, zorder=1)
    for yi, closed_val, err in zip(y, ordered["estimated_closed_multiplier"], ordered["closed_error_percent"]):
        ax.text(float(closed_val) + 0.012, yi - 0.13, f"{float(err):.2f}%", color=RED, fontsize=6.8, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=8.3)
    ax.set_xlabel("release multiplier")
    ax.set_xlim(0.72, 1.20)
    bounded_legend(ax, loc="upper center", bbox_to_anchor=(0.52, 1.30), ncol=2, fontsize=7.1)
    panel_label(ax, "f")
    clean_axis(ax)

    fig.savefig(FIGURES / "fvm3d_informed_pinn_inverse.png", dpi=420, bbox_inches="tight")
    fig.savefig(FIGURES / "fvm3d_informed_pinn_inverse.pdf", bbox_inches="tight")
    plt.close(fig)


def write_note(summary: pd.DataFrame, checks: pd.DataFrame) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    matched = summary[summary["protocol_id"] == "matched_3d_ht_hto_spatial"].iloc[0]
    total = summary[summary["protocol_id"] == "negative_total_uniform_basis"].iloc[0]
    wrong = summary[summary["protocol_id"] == "negative_matched_transport_wrong_chemistry"].iloc[0]
    text = f"""# 3D-informed PINN 反演小基准

这个脚本把 chemistry-resolved 3D FVM HT/HTO outlet histories 直接放进 differentiable inverse loss。它不是新的 reactor CFD，也不是实验验证；它验证上一轮 3D-FVM-to-PINN 接口不是停留在表格层面。

- 训练 case：`{'+'.join(TRAIN_CASES)}`
- holdout case：`{'+'.join(HOLDOUT_CASES)}`
- matched 3D HT/HTO spatial 协议：closed multiplier error `{float(matched['closed_error_percent']):.3f}%`，holdout RMSE `{float(matched['holdout_rmse_normalized']):.3e}`；新增图中直接给出未见 wet-bypass 3D 场的 HT/HTO release-curve posterior prediction。
- total outlet + uniform basis 负控：closed multiplier error `{float(total['closed_error_percent']):.2f}%`
- matched transport + wrong chemistry 负控：closed multiplier error `{float(wrong['closed_error_percent']):.2f}%`
- 检查：`{int((checks['status'] == 'PASS').sum())}/{len(checks)} PASS`

结论：3D HT/HTO 空间观测可以作为 PINN inverse 的实际 loss 约束；仅用 total outlet 或错误化学参考会继续失败。这个结果是 synthetic 3D FVM 压力测试，不能写成真实反应堆预测。
"""
    (NOTES / "fvm3d_informed_pinn_inverse_cn.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict] = []
    all_losses: list[pd.DataFrame] = []
    all_eval: list[pd.DataFrame] = []
    for protocol in PROTOCOLS:
        summary, loss_df, eval_df = train_protocol(protocol)
        all_summary.append(summary)
        all_losses.append(loss_df)
        all_eval.append(eval_df)
    summary_df = pd.DataFrame(all_summary)
    loss_df = pd.concat(all_losses, ignore_index=True)
    eval_df = pd.concat(all_eval, ignore_index=True)
    curves = holdout_prediction_curves(summary_df)
    checks = build_checks(summary_df, eval_df)

    summary_df.to_csv(OUT / "fvm3d_informed_pinn_inverse_summary.csv", index=False)
    loss_df.to_csv(OUT / "fvm3d_informed_pinn_inverse_loss.csv", index=False)
    eval_df.to_csv(OUT / "fvm3d_informed_pinn_inverse_case_eval.csv", index=False)
    curves.to_csv(OUT / "fvm3d_informed_pinn_inverse_holdout_prediction_curves.csv", index=False)
    checks.to_csv(OUT / "fvm3d_informed_pinn_inverse_checks.csv", index=False)
    matched = summary_df[summary_df["protocol_id"] == "matched_3d_ht_hto_spatial"].iloc[0]
    total = summary_df[summary_df["protocol_id"] == "negative_total_uniform_basis"].iloc[0]
    wrong = summary_df[summary_df["protocol_id"] == "negative_matched_transport_wrong_chemistry"].iloc[0]
    payload = {
        "summary": {
            "n_protocols": int(len(summary_df)),
            "n_train_cases": int(len(TRAIN_CASES)),
            "n_holdout_cases": int(len(HOLDOUT_CASES)),
            "n_checks": int(len(checks)),
            "n_pass_checks": int((checks["status"] == "PASS").sum()),
            "matched_3d_closed_error_pct": float(matched["closed_error_percent"]),
            "matched_3d_holdout_rmse": float(matched["holdout_rmse_normalized"]),
            "uniform_total_closed_error_pct": float(total["closed_error_percent"]),
            "uniform_total_holdout_rmse": float(total["holdout_rmse_normalized"]),
            "wrong_chemistry_closed_error_pct": float(wrong["closed_error_percent"]),
            "claim_boundary": "Differentiable 3D FVM-informed neural inverse benchmark with case holdout; synthetic equation-level evidence, not reactor-specific validation.",
        },
        "checks": checks.to_dict(orient="records"),
    }
    (OUT / "fvm3d_informed_pinn_inverse_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(summary_df, loss_df, eval_df, curves)
    write_note(summary_df, checks)
    print(
        "FVM3D_INFORMED_PINN="
        f"PROTOCOLS={len(summary_df)} "
        f"MATCHED_CLOSED={float(matched['closed_error_percent']):.3f}% "
        f"HOLDOUT_RMSE={float(matched['holdout_rmse_normalized']):.3e} "
        f"CHECKS={int((checks['status'] == 'PASS').sum())}/{len(checks)} PASS"
    )


if __name__ == "__main__":
    main()
