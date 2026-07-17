#!/usr/bin/env python3
"""Detector and sampling-line memory sensitivity for Li2TiO3 release inversion.

The main detector model is a 2x2 species gain/cross-response matrix.  This
script evaluates a stronger case in which the HTO channel has a first-order
dynamic memory. It compares fitting such data with a static detector matrix
against standard-gas pulse calibration of the dynamic time constants.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from li_ceramic_bed_1d import Li2TiO3Params, SharedParams, simulate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dynamic_detector_memory_sensitivity"
MANUSCRIPT = ROOT / "manuscript"


def first_order_lag(signal: np.ndarray, time_h: np.ndarray, tau_h: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if tau_h <= 0:
        return signal.copy()
    out = np.zeros_like(signal)
    out[0] = signal[0]
    dt = np.diff(time_h)
    for i in range(1, len(signal)):
        alpha = dt[i - 1] / (tau_h + dt[i - 1])
        out[i] = out[i - 1] + alpha * (signal[i] - out[i - 1])
    return out


def detector_response(
    ht: np.ndarray,
    hto: np.ndarray,
    time_h: np.ndarray,
    tau_ht_h: float,
    tau_hto_h: float,
    gain_ht: float = 1.0,
    gain_hto: float = 1.0,
    hto_to_ht: float = 0.030,
    ht_to_hto: float = 0.020,
) -> tuple[np.ndarray, np.ndarray]:
    ht_lag = first_order_lag(ht, time_h, tau_ht_h)
    hto_lag = first_order_lag(hto, time_h, tau_hto_h)
    y_ht = gain_ht * ht_lag + hto_to_ht * hto_lag
    y_hto = gain_hto * hto_lag + ht_to_hto * ht_lag
    return y_ht, y_hto


def run_li2(k_open_mult: float, k_closed_mult: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = SharedParams()
    base = Li2TiO3Params()
    mat = replace(
        base,
        k_open_release_s_1=base.k_open_release_s_1 * k_open_mult,
        k_closed_release_s_1=base.k_closed_release_s_1 * k_closed_mult,
    )
    _, _, t_s, y = simulate("li2tio3", shared, n=120, t_final=8.0 * 3600.0, n_times=401, mat=mat)
    n = 120
    c_ht = y[:, 0:n]
    c_hto = y[:, n : 2 * n]
    return t_s / 3600.0, c_ht[:, -1], c_hto[:, -1]


def linearized_fit(
    observed: tuple[np.ndarray, np.ndarray],
    basis: dict[str, tuple[np.ndarray, np.ndarray]],
    time_h: np.ndarray,
    assumed_tau_ht_h: float,
    assumed_tau_hto_h: float,
    eps: float,
) -> dict[str, float]:
    base = detector_response(*basis["base"], time_h, assumed_tau_ht_h, assumed_tau_hto_h)
    open_p = detector_response(*basis["open_plus"], time_h, assumed_tau_ht_h, assumed_tau_hto_h)
    closed_p = detector_response(*basis["closed_plus"], time_h, assumed_tau_ht_h, assumed_tau_hto_h)

    d_open = ((open_p[0] - base[0]) / eps, (open_p[1] - base[1]) / eps)
    d_closed = ((closed_p[0] - base[0]) / eps, (closed_p[1] - base[1]) / eps)
    scale_ht = max(float(np.max(np.abs(observed[0]))), 1e-30)
    scale_hto = max(float(np.max(np.abs(observed[1]))), 1e-30)
    a = np.column_stack(
        [
            np.r_[d_open[0] / scale_ht, d_open[1] / scale_hto],
            np.r_[d_closed[0] / scale_ht, d_closed[1] / scale_hto],
        ]
    )
    b = np.r_[(observed[0] - base[0]) / scale_ht, (observed[1] - base[1]) / scale_hto]
    delta, *_ = np.linalg.lstsq(a, b, rcond=None)
    pred = (
        base[0] + delta[0] * d_open[0] + delta[1] * d_closed[0],
        base[1] + delta[0] * d_open[1] + delta[1] * d_closed[1],
    )
    rmse = np.sqrt(
        np.mean(
            np.r_[
                ((pred[0] - observed[0]) / scale_ht) ** 2,
                ((pred[1] - observed[1]) / scale_hto) ** 2,
            ]
        )
    )
    return {
        "k_open_estimate": float(1.0 + delta[0]),
        "k_closed_estimate": float(1.0 + delta[1]),
        "k_open_error_pct": float(abs(delta[0]) * 100.0),
        "k_closed_error_pct": float(abs(delta[1]) * 100.0),
        "normalized_rmse_pct": float(100.0 * rmse),
    }


def estimate_tau_from_pulse(true_tau_h: float, rng: np.random.Generator, noise_frac: float) -> float:
    t = np.linspace(0.0, max(2.5, 8.0 * true_tau_h), 220)
    clean = np.exp(-t / true_tau_h)
    noisy = np.clip(clean + rng.normal(0.0, noise_frac, size=t.shape), 1e-6, None)
    mask = (clean < 0.95) & (clean > 0.06)
    slope, intercept = np.polyfit(t[mask], np.log(noisy[mask]), 1)
    if slope >= -1e-12:
        return true_tau_h
    return float(-1.0 / slope)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eps = 0.05
    time_h, ht_base, hto_base = run_li2(1.0, 1.0)
    _, ht_open, hto_open = run_li2(1.0 + eps, 1.0)
    _, ht_closed, hto_closed = run_li2(1.0, 1.0 + eps)
    basis = {
        "base": (ht_base, hto_base),
        "open_plus": (ht_open, hto_open),
        "closed_plus": (ht_closed, hto_closed),
    }

    true_tau_ht_h = 0.050
    true_tau_hto_h = 0.450
    observed = detector_response(ht_base, hto_base, time_h, true_tau_ht_h, true_tau_hto_h)

    static_fit = linearized_fit(observed, basis, time_h, 0.0, 0.0, eps)
    perfect_dynamic_fit = linearized_fit(
        observed, basis, time_h, true_tau_ht_h, true_tau_hto_h, eps
    )

    rng = np.random.default_rng(260628)
    pulse_rows: list[dict[str, float]] = []
    calibrated_rows: list[dict[str, float]] = []
    for seed in range(400):
        tau_ht = estimate_tau_from_pulse(true_tau_ht_h, rng, noise_frac=0.02)
        tau_hto = estimate_tau_from_pulse(true_tau_hto_h, rng, noise_frac=0.02)
        fit = linearized_fit(observed, basis, time_h, tau_ht, tau_hto, eps)
        pulse_rows.append(
            {
                "seed": seed,
                "tau_ht_est_h": tau_ht,
                "tau_hto_est_h": tau_hto,
                "tau_ht_error_pct": abs(tau_ht / true_tau_ht_h - 1.0) * 100.0,
                "tau_hto_error_pct": abs(tau_hto / true_tau_hto_h - 1.0) * 100.0,
            }
        )
        calibrated_rows.append({"seed": seed, **fit})

    def p95(values: list[float]) -> float:
        return float(np.percentile(values, 95))

    calibrated_summary = {
        "k_open_error_p95_pct": p95([r["k_open_error_pct"] for r in calibrated_rows]),
        "k_closed_error_p95_pct": p95([r["k_closed_error_pct"] for r in calibrated_rows]),
        "normalized_rmse_p95_pct": p95([r["normalized_rmse_pct"] for r in calibrated_rows]),
        "tau_ht_error_p95_pct": p95([r["tau_ht_error_pct"] for r in pulse_rows]),
        "tau_hto_error_p95_pct": p95([r["tau_hto_error_pct"] for r in pulse_rows]),
    }

    rows = [
        {
            "case": "static_matrix_on_dynamic_detector",
            "assumed_response": "2x2 gain/cross matrix, zero memory",
            **static_fit,
        },
        {
            "case": "pulse_calibrated_dynamic_response",
            "assumed_response": "2x2 matrix plus noisy reference-pulse time constants",
            "k_open_estimate": "",
            "k_closed_estimate": "",
            "k_open_error_pct": calibrated_summary["k_open_error_p95_pct"],
            "k_closed_error_pct": calibrated_summary["k_closed_error_p95_pct"],
            "normalized_rmse_pct": calibrated_summary["normalized_rmse_p95_pct"],
        },
        {
            "case": "known_dynamic_response",
            "assumed_response": "2x2 matrix plus known first-order HT/HTO memory",
            **perfect_dynamic_fit,
        },
    ]

    with (OUT / "dynamic_detector_memory_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "dynamic_detector_memory_pulse_trials.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pulse_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pulse_rows)
    with (OUT / "dynamic_detector_memory_calibrated_trials.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(calibrated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calibrated_rows)

    summary = {
        "true_tau_ht_h": true_tau_ht_h,
        "true_tau_hto_h": true_tau_hto_h,
        "n_reference_pulse_trials": len(pulse_rows),
        "pulse_noise_fraction": 0.02,
        "static_matrix_closed_error_pct": static_fit["k_closed_error_pct"],
        "static_matrix_open_error_pct": static_fit["k_open_error_pct"],
        "static_matrix_rmse_pct": static_fit["normalized_rmse_pct"],
        "pulse_calibrated_closed_error_p95_pct": calibrated_summary["k_closed_error_p95_pct"],
        "pulse_calibrated_open_error_p95_pct": calibrated_summary["k_open_error_p95_pct"],
        "pulse_calibrated_rmse_p95_pct": calibrated_summary["normalized_rmse_p95_pct"],
        "pulse_tau_ht_error_p95_pct": calibrated_summary["tau_ht_error_p95_pct"],
        "pulse_tau_hto_error_p95_pct": calibrated_summary["tau_hto_error_p95_pct"],
        "known_dynamic_closed_error_pct": perfect_dynamic_fit["k_closed_error_pct"],
        "known_dynamic_open_error_pct": perfect_dynamic_fit["k_open_error_pct"],
    }
    (OUT / "dynamic_detector_memory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    tex = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\caption{Dynamic detector and sampling-line memory sensitivity. The sensitivity case uses the Li$_2$TiO$_3$ release basis with a first-order HTO memory in the measured channel.}",
        r"\label{tab:dynamic_detector_memory}",
        r"\begin{tabularx}{\columnwidth}{@{}P{0.32\columnwidth}YY@{}}",
        r"\toprule",
        r"Detector response used in inverse & Closed multiplier error & Normalized outlet RMSE \\",
        r"\midrule",
        (
            "Static 2x2 matrix & "
            f"{static_fit['k_closed_error_pct']:.2f}\\% & "
            f"{static_fit['normalized_rmse_pct']:.2f}\\% \\\\"
        ),
        (
            "Reference-pulse dynamic response & "
            f"{calibrated_summary['k_closed_error_p95_pct']:.2f}\\% p95 & "
            f"{calibrated_summary['normalized_rmse_p95_pct']:.2f}\\% p95 \\\\"
        ),
        (
            "Known dynamic response & "
            f"{perfect_dynamic_fit['k_closed_error_pct']:.2f}\\% & "
            f"{perfect_dynamic_fit['normalized_rmse_pct']:.2f}\\% \\\\"
        ),
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    (MANUSCRIPT / "generated_dynamic_detector_memory_table.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )

    checks = [
        {
            "check": "static_memory_bias_visible",
            "status": "PASS" if static_fit["k_closed_error_pct"] > 5.0 else "FAIL",
            "detail": f"closed_error={static_fit['k_closed_error_pct']:.2f}%",
        },
        {
            "check": "pulse_calibration_recovers_source",
            "status": "PASS" if calibrated_summary["k_closed_error_p95_pct"] < 5.0 else "FAIL",
            "detail": f"closed_p95={calibrated_summary['k_closed_error_p95_pct']:.2f}%",
        },
    ]
    with (OUT / "dynamic_detector_memory_checks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
        writer.writeheader()
        writer.writerows(checks)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
