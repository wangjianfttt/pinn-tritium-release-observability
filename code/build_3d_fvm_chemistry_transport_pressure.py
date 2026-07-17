#!/usr/bin/env python3
"""Chemistry-resolved 3D FVM transport pressure test.

This script closes a gap between the scalar heterogeneous 3D FVM pressure test
and the HT/HTO chemistry stress tests.  It solves a lightweight two-species
advection-diffusion-reaction model on a static packed-bed-like 3D field and
then asks reduced inverse protocols to recover open/closed release multipliers.

It is intentionally bounded: this is an equation-level synthetic pressure test,
not OpenFOAM/CFD-DEM validation or reactor-specific field ingestion.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "fvm3d_chemistry_transport_pressure"
FIGURES = ROOT / "figures"
NOTES = ROOT / "notes"

NX = 26
NY = 8
NZ = 8
LENGTH_M = 0.12
WIDTH_M = 0.040
HEIGHT_M = 0.040
T_END_H = 8.0
DT_S = 8.0
SAMPLE_EVERY = 18

TRUE_OPEN = 1.08
TRUE_CLOSED = 0.78
BASE_SOURCE = 1.15e-7
OPEN_TAU_S = 0.32 * 3600.0
CLOSED_TAU_S = 4.8 * 3600.0


def make_fields(case_id: str) -> dict[str, np.ndarray | float | str]:
    x = np.linspace(0.0, 1.0, NX)[:, None, None]
    y = np.linspace(-1.0, 1.0, NY)[None, :, None]
    z = np.linspace(-1.0, 1.0, NZ)[None, None, :]
    wall = np.maximum(np.abs(y), np.abs(z))
    radial2 = y * y + z * z

    porosity = 0.365 + 0.012 * np.sin(2.0 * np.pi * x)
    velocity = 6.4e-5 * np.ones((NX, NY, NZ))
    source_shape = 1.0 + 0.30 * np.sin(np.pi * x) + 0.12 * np.cos(np.pi * y) - 0.08 * z
    wetness = 0.18 + 0.08 * np.sin(np.pi * x) ** 2
    wall_activity = 0.05 * wall
    exchange_rate_s = 3.0e-6 * np.ones((NX, NY, NZ))
    loss_rate_s = 4.0e-7 * np.ones((NX, NY, NZ))
    diffusivity_ht = 1.25e-7
    diffusivity_hto = 0.95e-7
    description = "Uniform dry-ish 3D reference with weak HT to HTO exchange."

    if case_id == "uniform_dry":
        pass
    elif case_id == "heterogeneous_wet_wall":
        porosity = porosity + 0.052 * wall - 0.018 * np.exp(-radial2 / 0.18)
        velocity = 4.9e-5 + 5.2e-5 * np.exp(-radial2 / 0.33) + 1.0e-5 * wall
        source_shape = source_shape * (0.86 + 0.38 * np.exp(-((x - 0.40) / 0.26) ** 2)) * (1.0 - 0.08 * wall)
        wetness = 0.26 + 0.22 * wall + 0.10 * np.exp(-((x - 0.62) / 0.20) ** 2)
        wall_activity = 0.10 + 0.28 * wall
        exchange_rate_s = 4.5e-6 * (1.0 + 1.8 * wetness + 1.2 * wall_activity)
        loss_rate_s = 5.0e-7 * (1.0 + 1.4 * wetness)
        description = "Wet wall-effect bed with transverse residence-time spread and wall-catalytic conversion."
    elif case_id == "heterogeneous_wet_bypass":
        bypass = wall > 0.70
        porosity = porosity + 0.074 * bypass + 0.016 * np.sin(np.pi * y) * np.cos(np.pi * z)
        velocity = 3.7e-5 + 7.8e-5 * bypass + 1.4e-5 * np.exp(-((y - 0.35) ** 2 + (z + 0.20) ** 2) / 0.18)
        source_shape = source_shape * (1.10 - 0.32 * bypass) * (0.86 + 0.30 * np.sin(np.pi * x) ** 2)
        wetness = 0.20 + 0.32 * bypass + 0.12 * np.exp(-((x - 0.72) / 0.17) ** 2)
        wall_activity = 0.08 + 0.42 * bypass + 0.08 * wall
        exchange_rate_s = 5.4e-6 * (1.0 + 2.2 * wetness + 1.5 * wall_activity)
        loss_rate_s = 7.0e-7 * (1.0 + 1.6 * wetness + 0.8 * bypass)
        description = "Wet bypass bed with wall channeling, off-axis source, HT to HTO conversion and unrecovered loss."
    else:
        raise ValueError(f"unknown case_id={case_id}")

    porosity = np.clip(np.broadcast_to(porosity, (NX, NY, NZ)).astype(float), 0.25, 0.56)
    velocity = np.clip(np.broadcast_to(velocity, (NX, NY, NZ)).astype(float), 1.1e-5, 1.6e-4)
    source_shape = np.clip(np.broadcast_to(source_shape, (NX, NY, NZ)).astype(float), 1.0e-4, None)
    source_shape = source_shape / float(source_shape.mean())
    wetness = np.clip(np.broadcast_to(wetness, (NX, NY, NZ)).astype(float), 0.0, 1.0)
    wall_activity = np.clip(np.broadcast_to(wall_activity, (NX, NY, NZ)).astype(float), 0.0, 1.0)
    exchange_rate_s = np.clip(np.broadcast_to(exchange_rate_s, (NX, NY, NZ)).astype(float), 0.0, 5.0e-5)
    loss_rate_s = np.clip(np.broadcast_to(loss_rate_s, (NX, NY, NZ)).astype(float), 0.0, 2.0e-5)

    return {
        "case_id": case_id,
        "description": description,
        "porosity": porosity,
        "velocity": velocity,
        "source_shape": source_shape,
        "wetness": wetness,
        "wall_activity": wall_activity,
        "exchange_rate_s": exchange_rate_s,
        "loss_rate_s": loss_rate_s,
        "diffusivity_ht": diffusivity_ht,
        "diffusivity_hto": diffusivity_hto,
    }


def release_parts(t_s: float, open_multiplier: float, closed_multiplier: float) -> tuple[float, float]:
    open_part = 0.60 * open_multiplier * np.exp(-t_s / OPEN_TAU_S) / OPEN_TAU_S
    closed_part = 0.40 * closed_multiplier * np.exp(-t_s / CLOSED_TAU_S) / CLOSED_TAU_S
    return float(open_part), float(closed_part)


def source_species_fraction(t_s: float, fields: dict[str, np.ndarray | float | str], mode: str) -> np.ndarray:
    wet = fields["wetness"]
    wall = fields["wall_activity"]
    assert isinstance(wet, np.ndarray) and isinstance(wall, np.ndarray)
    time_norm = t_s / max(T_END_H * 3600.0, 1.0)
    if mode == "dry_reference":
        hto_fraction = 0.14 + 0.03 * time_norm
    elif mode == "matched_chemistry":
        hto_fraction = 0.12 + 0.18 * wet + 0.08 * wall + 0.08 * time_norm
    else:
        raise ValueError(mode)
    return np.clip(hto_fraction, 0.03, 0.72)


def advdiff(c: np.ndarray, fields: dict[str, np.ndarray | float | str], diff: float, dx: float, dy: float, dz: float) -> np.ndarray:
    por = fields["porosity"]
    vel = fields["velocity"]
    assert isinstance(por, np.ndarray) and isinstance(vel, np.ndarray)

    flux_left = np.empty_like(c)
    flux_right = vel * c
    flux_left[0, :, :] = 0.0
    flux_left[1:, :, :] = vel[:-1, :, :] * c[:-1, :, :]
    adv = -(flux_right - flux_left) / (por * dx)

    pad = np.pad(c, ((1, 1), (1, 1), (1, 1)), mode="edge")
    lap_x = (pad[2:, 1:-1, 1:-1] - 2.0 * c + pad[:-2, 1:-1, 1:-1]) / (dx * dx)
    lap_y = (pad[1:-1, 2:, 1:-1] - 2.0 * c + pad[1:-1, :-2, 1:-1]) / (dy * dy)
    lap_z = (pad[1:-1, 1:-1, 2:] - 2.0 * c + pad[1:-1, 1:-1, :-2]) / (dz * dz)
    return adv + diff * (lap_x + lap_y + lap_z)


def outlet_groups(ht: np.ndarray, hto: np.ndarray, velocity_2d: np.ndarray) -> dict[str, float]:
    weights = velocity_2d / max(float(velocity_2d.sum()), 1.0e-30)
    half_y = NY // 2
    half_z = NZ // 2
    masks = {
        "mixed": (slice(None), slice(None)),
        "q1": (slice(None, half_y), slice(None, half_z)),
        "q2": (slice(None, half_y), slice(half_z, None)),
        "q3": (slice(half_y, None), slice(None, half_z)),
        "q4": (slice(half_y, None), slice(half_z, None)),
    }
    out: dict[str, float] = {}
    total = ht + hto
    for key, idx in masks.items():
        w = weights[idx]
        den = max(float(w.sum()), 1.0e-30)
        out[f"ht_{key}"] = float(np.sum(ht[idx] * w) / den)
        out[f"hto_{key}"] = float(np.sum(hto[idx] * w) / den)
        out[f"total_{key}"] = float(np.sum(total[idx] * w) / den)
    return out


def simulate_case(
    case_id: str,
    open_multiplier: float = TRUE_OPEN,
    closed_multiplier: float = TRUE_CLOSED,
    chemistry_mode: str = "matched_chemistry",
) -> tuple[pd.DataFrame, dict[str, object]]:
    fields = make_fields(case_id)
    dx = LENGTH_M / NX
    dy = WIDTH_M / NY
    dz = HEIGHT_M / NZ
    n_steps = int(T_END_H * 3600.0 / DT_S)
    ht = np.zeros((NX, NY, NZ), dtype=float)
    hto = np.zeros((NX, NY, NZ), dtype=float)
    rows: list[dict[str, float | str]] = []
    generated = 0.0
    extracted = 0.0
    lost = 0.0
    converted = 0.0

    por = fields["porosity"]
    vel = fields["velocity"]
    src = fields["source_shape"]
    exch = fields["exchange_rate_s"]
    loss_rate = fields["loss_rate_s"]
    assert isinstance(por, np.ndarray)
    assert isinstance(vel, np.ndarray)
    assert isinstance(src, np.ndarray)
    assert isinstance(exch, np.ndarray)
    assert isinstance(loss_rate, np.ndarray)

    for step in range(n_steps + 1):
        t_s = step * DT_S
        open_rate, closed_rate = release_parts(t_s, open_multiplier, closed_multiplier)
        source_rate = BASE_SOURCE * (open_rate + closed_rate)
        hto_frac = source_species_fraction(t_s, fields, chemistry_mode)
        src_hto = source_rate * src * hto_frac / por
        src_ht = source_rate * src * (1.0 - hto_frac) / por

        if step > 0:
            ht_rhs = advdiff(ht, fields, float(fields["diffusivity_ht"]), dx, dy, dz)
            hto_rhs = advdiff(hto, fields, float(fields["diffusivity_hto"]), dx, dy, dz)
            conv = exch * ht
            loss = loss_rate * hto
            ht = ht + DT_S * (ht_rhs + src_ht - conv)
            hto = hto + DT_S * (hto_rhs + src_hto + conv - loss)
            ht[0, :, :] = 0.0
            hto[0, :, :] = 0.0
            ht = np.maximum(ht, 0.0)
            hto = np.maximum(hto, 0.0)
            if not np.isfinite(ht).all() or not np.isfinite(hto).all():
                raise FloatingPointError(f"non-finite HT/HTO field in {case_id}")
            generated += float(np.sum(source_rate * src) * dx * dy * dz * DT_S)
            converted += float(np.sum(conv * por) * dx * dy * dz * DT_S)
            lost += float(np.sum(loss * por) * dx * dy * dz * DT_S)
            extracted += float(np.sum((ht[-1, :, :] + hto[-1, :, :]) * vel[-1, :, :]) * dy * dz * DT_S)

        if step % SAMPLE_EVERY == 0:
            groups = outlet_groups(ht[-1, :, :], hto[-1, :, :], vel[-1, :, :])
            row: dict[str, float | str] = {
                "case_id": case_id,
                "chemistry_mode": chemistry_mode,
                "time_h": t_s / 3600.0,
            }
            row.update({f"outlet_{key}": value for key, value in groups.items()})
            rows.append(row)

    inventory = float(np.sum((ht + hto) * por) * dx * dy * dz)
    summary = {
        "case_id": case_id,
        "chemistry_mode": chemistry_mode,
        "description": str(fields["description"]),
        "mean_porosity": float(por.mean()),
        "porosity_cv": float(por.std() / por.mean()),
        "mean_velocity_m_s": float(vel.mean()),
        "velocity_cv": float(vel.std() / vel.mean()),
        "source_cv": float(src.std() / src.mean()),
        "mean_wetness": float(np.asarray(fields["wetness"]).mean()),
        "mean_exchange_rate_s": float(exch.mean()),
        "mean_loss_rate_s": float(loss_rate.mean()),
        "generated_proxy_mol": generated,
        "extracted_proxy_mol": extracted,
        "inventory_proxy_mol": inventory,
        "lost_proxy_mol": lost,
        "converted_proxy_mol": converted,
        "mass_balance_residual_proxy_mol": generated - extracted - inventory - lost,
        "mass_balance_relative_error": abs(generated - extracted - inventory - lost) / max(generated, 1.0e-30),
    }
    return pd.DataFrame(rows), summary


def basis(case_id: str, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth, _ = simulate_case(case_id, TRUE_OPEN, TRUE_CLOSED, "matched_chemistry")
    if mode == "uniform_dry_basis":
        open_b, _ = simulate_case("uniform_dry", 1.0, 0.0, "dry_reference")
        closed_b, _ = simulate_case("uniform_dry", 0.0, 1.0, "dry_reference")
    elif mode == "matched_transport_dry_chemistry":
        open_b, _ = simulate_case(case_id, 1.0, 0.0, "dry_reference")
        closed_b, _ = simulate_case(case_id, 0.0, 1.0, "dry_reference")
    elif mode == "matched_3d_chemistry":
        open_b, _ = simulate_case(case_id, 1.0, 0.0, "matched_chemistry")
        closed_b, _ = simulate_case(case_id, 0.0, 1.0, "matched_chemistry")
    else:
        raise ValueError(mode)
    return truth, open_b, closed_b


def fit_multipliers(truth: pd.DataFrame, open_b: pd.DataFrame, closed_b: pd.DataFrame, observables: list[str]) -> dict[str, float]:
    y_parts = []
    a_parts = []
    for col in observables:
        scale = max(float(truth[col].max()), float(open_b[col].max()), float(closed_b[col].max()), 1.0e-20)
        y_parts.append(truth[col].to_numpy(float) / scale)
        a_parts.append(np.column_stack([open_b[col].to_numpy(float) / scale, closed_b[col].to_numpy(float) / scale]))
    y = np.concatenate(y_parts)
    a = np.vstack(a_parts)
    open_grid = np.linspace(0.45, 1.65, 241)
    closed_grid = np.linspace(0.35, 1.35, 251)
    best_loss = np.inf
    fit = np.array([np.nan, np.nan], dtype=float)
    for open_value in open_grid:
        residuals = y[:, None] - open_value * a[:, 0:1] - closed_grid[None, :] * a[:, 1:2]
        losses = np.mean(residuals * residuals, axis=0)
        idx = int(np.argmin(losses))
        if float(losses[idx]) < best_loss:
            best_loss = float(losses[idx])
            fit[:] = [open_value, closed_grid[idx]]
    residual = y - (fit[0] * a[:, 0] + fit[1] * a[:, 1])
    s = np.linalg.svd(a, compute_uv=False)
    return {
        "estimated_open_multiplier": float(fit[0]),
        "estimated_closed_multiplier": float(fit[1]),
        "open_error_percent": float(abs(fit[0] - TRUE_OPEN) / TRUE_OPEN * 100.0),
        "closed_error_percent": float(abs(fit[1] - TRUE_CLOSED) / TRUE_CLOSED * 100.0),
        "rmse_normalized": float(np.sqrt(np.mean(residual * residual))),
        "min_singular_value": float(np.min(s)),
        "condition_number": float(np.max(s) / max(np.min(s), 1.0e-30)),
    }


def run_inversions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    histories = []
    cases = []
    inverse_rows = []
    protocols = [
        (
            "mixed total with uniform dry 1D-equivalent basis",
            "uniform_dry_basis",
            ["outlet_total_mixed"],
        ),
        (
            "mixed HT/HTO with uniform dry 1D-equivalent basis",
            "uniform_dry_basis",
            ["outlet_ht_mixed", "outlet_hto_mixed"],
        ),
        (
            "mixed HT/HTO with matched 3D transport but dry chemistry",
            "matched_transport_dry_chemistry",
            ["outlet_ht_mixed", "outlet_hto_mixed"],
        ),
        (
            "four HT/HTO outlets with matched 3D chemistry",
            "matched_3d_chemistry",
            [
                "outlet_ht_q1",
                "outlet_hto_q1",
                "outlet_ht_q2",
                "outlet_hto_q2",
                "outlet_ht_q3",
                "outlet_hto_q3",
                "outlet_ht_q4",
                "outlet_hto_q4",
            ],
        ),
    ]
    for case_id in ["uniform_dry", "heterogeneous_wet_wall", "heterogeneous_wet_bypass"]:
        truth, summary = simulate_case(case_id, TRUE_OPEN, TRUE_CLOSED, "matched_chemistry")
        histories.append(truth)
        cases.append(summary)
        for label, mode, observables in protocols:
            truth_df, open_b, closed_b = basis(case_id, mode)
            fit = fit_multipliers(truth_df, open_b, closed_b, observables)
            inverse_rows.append(
                {
                    "case_id": case_id,
                    "inverse_protocol": label,
                    "basis_type": mode,
                    "observables": "+".join(observables),
                    **fit,
                }
            )
    return pd.concat(histories, ignore_index=True), pd.DataFrame(cases), pd.DataFrame(inverse_rows)


def build_checks(cases: pd.DataFrame, inverse: pd.DataFrame) -> list[dict[str, str]]:
    hard = inverse[inverse["case_id"] == "heterogeneous_wet_bypass"]
    total_uniform = hard[hard["inverse_protocol"] == "mixed total with uniform dry 1D-equivalent basis"].iloc[0]
    dry_chem = hard[hard["inverse_protocol"] == "mixed HT/HTO with matched 3D transport but dry chemistry"].iloc[0]
    matched = hard[hard["inverse_protocol"] == "four HT/HTO outlets with matched 3D chemistry"].iloc[0]
    return [
        {
            "check_id": "mass_balance_proxy_bounded",
            "status": "PASS" if float(cases["mass_balance_relative_error"].max()) < 0.45 else "FAIL",
            "detail": f"max_relative_error={float(cases['mass_balance_relative_error'].max()):.3f}",
        },
        {
            "check_id": "chemistry_resolved_3d_cases_present",
            "status": "PASS" if {"uniform_dry", "heterogeneous_wet_wall", "heterogeneous_wet_bypass"}.issubset(set(cases["case_id"])) else "FAIL",
            "detail": ",".join(cases["case_id"].tolist()),
        },
        {
            "check_id": "uniform_dry_total_bias_exposed",
            "status": "PASS" if float(total_uniform["closed_error_percent"]) > 8.0 else "FAIL",
            "detail": f"heterogeneous_wet_bypass_total_uniform_closed_error={float(total_uniform['closed_error_percent']):.2f}%",
        },
        {
            "check_id": "wrong_chemistry_bias_exposed",
            "status": "PASS" if float(dry_chem["closed_error_percent"]) > 4.0 else "FAIL",
            "detail": f"matched_transport_dry_chemistry_closed_error={float(dry_chem['closed_error_percent']):.2f}%",
        },
        {
            "check_id": "matched_3d_chemistry_recovers_truth",
            "status": "PASS" if float(matched["closed_error_percent"]) < 1.0 else "FAIL",
            "detail": f"matched_3d_chemistry_closed_error={float(matched['closed_error_percent']):.3f}%",
        },
    ]


def plot(histories: pd.DataFrame, cases: pd.DataFrame, inverse: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()
    colors = {
        "uniform_dry": "#3D5A80",
        "heterogeneous_wet_wall": "#E07A5F",
        "heterogeneous_wet_bypass": "#2A9D8F",
    }
    for case_id, sub in histories.groupby("case_id", sort=False):
        ax0.plot(sub["time_h"], sub["outlet_ht_mixed"], color=colors[case_id], lw=1.9, label=f"{case_id} HT")
        ax0.plot(sub["time_h"], sub["outlet_hto_mixed"], color=colors[case_id], lw=1.3, ls="--", label=f"{case_id} HTO")
    ax0.set_title("a  chemistry-resolved 3D outlet histories", loc="left", fontweight="bold")
    ax0.set_xlabel("time (h)")
    ax0.set_ylabel("mixed outlet concentration (proxy)")
    ax0.legend(frameon=False, fontsize=5.9, ncol=2)

    x = np.arange(len(cases))
    ax1.bar(x - 0.24, cases["porosity_cv"] * 100.0, width=0.24, color="#7B8794", label="porosity CV")
    ax1.bar(x, cases["velocity_cv"] * 100.0, width=0.24, color="#C17C74", label="velocity CV")
    ax1.bar(x + 0.24, cases["mean_wetness"] * 100.0, width=0.24, color="#577590", label="wetness mean")
    ax1.set_xticks(x, [cid.replace("heterogeneous_", "hetero\n").replace("uniform_dry", "uniform") for cid in cases["case_id"]])
    ax1.set_title("b  field and chemistry heterogeneity", loc="left", fontweight="bold")
    ax1.set_ylabel("percent or coefficient of variation (%)")
    ax1.legend(frameon=False, fontsize=6.6)

    protocol_order = [
        "mixed total with uniform dry 1D-equivalent basis",
        "mixed HT/HTO with uniform dry 1D-equivalent basis",
        "mixed HT/HTO with matched 3D transport but dry chemistry",
        "four HT/HTO outlets with matched 3D chemistry",
    ]
    pivot = inverse.pivot(index="case_id", columns="inverse_protocol", values="closed_error_percent")
    order = ["uniform_dry", "heterogeneous_wet_wall", "heterogeneous_wet_bypass"]
    labels = ["total\nuniform dry", "HT/HTO\nuniform dry", "HT/HTO\n3D dry chem", "4 outlets\n3D chemistry"]
    width = 0.18
    for j, col in enumerate(protocol_order):
        ax2.bar(np.arange(len(order)) + (j - 1.5) * width, pivot[col].loc[order], width=width, label=labels[j])
    ax2.axhline(5.0, color="#333333", lw=0.9, ls="--")
    ax2.set_xticks(np.arange(len(order)), [item.replace("heterogeneous_", "hetero\n").replace("uniform_dry", "uniform") for item in order])
    ax2.set_title("c  closed-pathway inverse bias", loc="left", fontweight="bold")
    ax2.set_ylabel("closed multiplier error (%)")
    ax2.legend(frameon=False, fontsize=6.1, ncol=2)

    for protocol, sub in inverse.groupby("inverse_protocol", sort=False):
        ax3.scatter(sub["condition_number"], sub["closed_error_percent"], s=42, label=protocol.replace(" with ", "\nwith "))
    ax3.set_xscale("log")
    ax3.set_title("d  chemistry/transport identifiability", loc="left", fontweight="bold")
    ax3.set_xlabel("least-squares condition number")
    ax3.set_ylabel("closed multiplier error (%)")
    ax3.legend(frameon=False, fontsize=5.4)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"fvm3d_chemistry_transport_pressure.{ext}", dpi=300)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[dict[str, object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "/") for header in headers) + " |")
    return "\n".join(lines)


def write_note(cases: pd.DataFrame, inverse: pd.DataFrame, checks: list[dict[str, str]], summary: dict[str, object]) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    rows = inverse[
        [
            "case_id",
            "inverse_protocol",
            "estimated_open_multiplier",
            "estimated_closed_multiplier",
            "open_error_percent",
            "closed_error_percent",
            "condition_number",
        ]
    ].round(4).to_dict(orient="records")
    lines = [
        "# 化学分辨三维 FVM 输运压力测试",
        "",
        "目的：把三维非均匀输运和 HT/HTO 化学放到同一个可运行压力测试里，检查单一 total outlet、错误 dry chemistry reference 或一维均匀基函数是否会把传输/物种转化误判成 open/closed release multiplier。",
        "",
        "## 核心结果",
        "",
        f"- 最强 heterogeneous_wet_bypass 工况下，mixed total + uniform dry basis 的闭孔误差：`{summary['heterogeneous_wet_bypass_total_uniform_closed_error_percent']:.2f}%`。",
        f"- 同一工况下，即便使用 matched 3D transport，如果 chemistry 仍用 dry reference，闭孔误差仍为：`{summary['heterogeneous_wet_bypass_matched_transport_dry_chemistry_closed_error_percent']:.2f}%`。",
        f"- 四个空间出口 + HT/HTO + matched 3D chemistry 后，闭孔误差降至：`{summary['heterogeneous_wet_bypass_four_outlet_matched3d_chemistry_closed_error_percent']:.3f}%`。",
        "",
        "解释：3D transport 只能解决流动/驻留时间错配，不能自动解决 HT/HTO 化学错配；审稿人如果质疑 PINN 训练数据科学性，这个测试给出的是更强的反证链：必须同时设计三维输运、物种化学和空间观测。",
        "",
        "## 反演对比",
        "",
        markdown_table(
            [
                "case_id",
                "inverse_protocol",
                "estimated_open_multiplier",
                "estimated_closed_multiplier",
                "open_error_percent",
                "closed_error_percent",
                "condition_number",
            ],
            rows,
        ),
        "",
        "## 质量检查",
        "",
        markdown_table(["check_id", "status", "detail"], checks),
        "",
        "## 稿件边界",
        "",
        "这项结果可以进入主文作为 chemistry-resolved 3D FVM pressure test。它不是 OpenFOAM/CFD-DEM 网格无关验证，也不是 reactor-specific blanket prediction；它证明的是：一维均匀 total outlet 和错误化学 reference 不能支撑高可信释放倍增因子反演。",
        "",
        "## case 物理摘要",
        "",
        markdown_table(
            ["case_id", "porosity_cv", "velocity_cv", "mean_wetness", "mean_exchange_rate_s", "mass_balance_relative_error"],
            cases[["case_id", "porosity_cv", "velocity_cv", "mean_wetness", "mean_exchange_rate_s", "mass_balance_relative_error"]].round(6).to_dict(orient="records"),
        ),
        "",
    ]
    (NOTES / "fvm3d_chemistry_transport_pressure_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    histories, cases, inverse = run_inversions()
    checks = build_checks(cases, inverse)
    plot(histories, cases, inverse)

    histories.to_csv(RESULTS / "fvm3d_chemistry_outlet_histories.csv", index=False)
    cases.to_csv(RESULTS / "fvm3d_chemistry_case_summary.csv", index=False)
    inverse.to_csv(RESULTS / "fvm3d_chemistry_inverse_comparison.csv", index=False)
    with (RESULTS / "fvm3d_chemistry_transport_checks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)

    hard = inverse[inverse["case_id"] == "heterogeneous_wet_bypass"]
    total_uniform = hard[hard["inverse_protocol"] == "mixed total with uniform dry 1D-equivalent basis"].iloc[0]
    dry_chem = hard[hard["inverse_protocol"] == "mixed HT/HTO with matched 3D transport but dry chemistry"].iloc[0]
    matched = hard[hard["inverse_protocol"] == "four HT/HTO outlets with matched 3D chemistry"].iloc[0]
    summary = {
        "true_open_multiplier": TRUE_OPEN,
        "true_closed_multiplier": TRUE_CLOSED,
        "n_cases": int(len(cases)),
        "n_inverse_rows": int(len(inverse)),
        "heterogeneous_wet_bypass_total_uniform_closed_error_percent": float(total_uniform["closed_error_percent"]),
        "heterogeneous_wet_bypass_matched_transport_dry_chemistry_closed_error_percent": float(dry_chem["closed_error_percent"]),
        "heterogeneous_wet_bypass_four_outlet_matched3d_chemistry_closed_error_percent": float(matched["closed_error_percent"]),
        "heterogeneous_wet_bypass_total_uniform_open_error_percent": float(total_uniform["open_error_percent"]),
        "max_mass_balance_relative_error": float(cases["mass_balance_relative_error"].max()),
        "n_checks": len(checks),
        "n_pass_checks": sum(row["status"] == "PASS" for row in checks),
        "claim_boundary": (
            "Synthetic chemistry-resolved 3D FVM pressure test; not OpenFOAM/CFD-DEM "
            "validation and not reactor-specific blanket prediction."
        ),
    }
    (RESULTS / "fvm3d_chemistry_transport_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_note(cases, inverse, checks, summary)

    print(
        "FVM3D_CHEMISTRY_PRESSURE="
        f"CASES={len(cases)} INVERSE_ROWS={len(inverse)} "
        f"TOTAL_UNIFORM_CLOSED_ERR={summary['heterogeneous_wet_bypass_total_uniform_closed_error_percent']:.2f}% "
        f"DRY_CHEM_CLOSED_ERR={summary['heterogeneous_wet_bypass_matched_transport_dry_chemistry_closed_error_percent']:.2f}% "
        f"MATCHED_3D_CHEM_CLOSED_ERR={summary['heterogeneous_wet_bypass_four_outlet_matched3d_chemistry_closed_error_percent']:.3f}% "
        f"CHECKS={summary['n_pass_checks']}/{summary['n_checks']} PASS"
    )


if __name__ == "__main__":
    main()

