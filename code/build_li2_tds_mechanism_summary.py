#!/usr/bin/env python3
"""Rebuild the Li2TiO3 TDS mechanism-closure summary.

The TDS spectra are used as mechanism evidence for a delayed/high-temperature
release pathway. They are not used to calibrate the isothermal open/closed
release multipliers reported in the inverse benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "li2_tds_closed_pathway_mechanism_closure"


def spearman_like(x: np.ndarray, y: np.ndarray) -> float:
    """Return a simple rank correlation, with constants treated as no trend."""
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])


def norm01(x: np.ndarray) -> np.ndarray:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if np.isclose(xmax, xmin):
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kob = pd.read_csv(RESULTS / "literature_kobayashi_fig4_tds_metrics.csv")
    zhu = pd.read_csv(RESULTS / "literature_zhu2013_fig2_li2_tds_metrics.csv")
    ladder = pd.read_csv(OUT_DIR / "li2_tds_closed_pathway_mechanism_closure_ladder.csv")

    kob = kob.sort_values("relative_density", ascending=False).reset_index(drop=True)
    total_points = int(kob["n_points"].sum())
    high_full = kob["high_to_full_area_fraction"].to_numpy(dtype=float)
    high_low = kob["high_to_low_area_ratio"].to_numpy(dtype=float)
    density_loss = (float(kob["relative_density"].max()) - kob["relative_density"].to_numpy(dtype=float))
    density_loss = density_loss / float(density_loss.max())

    zhu_high_density = zhu[zhu["relative_density_percent"] >= 88.6].copy()
    zhu_high_fraction = zhu_high_density["high_area_fraction"].to_numpy(dtype=float)
    zhu_peak_gap = (
        zhu_high_density["high_650_900_peak_temperature_k"]
        - zhu_high_density["low_peak_temperature_k"]
    ).to_numpy(dtype=float)

    mode_rows: list[dict[str, float | str]] = []
    for mode, mode_df in ladder.groupby("mode", sort=False):
        mode_df = mode_df.sort_values("relative_density", ascending=False).reset_index(drop=True)
        closed = mode_df["closed_solid_fraction_48h"].to_numpy(dtype=float)
        rho_density = spearman_like(mode_df["density_loss"].to_numpy(dtype=float), closed)
        rho_tds = spearman_like(high_full, closed)
        shape_rmse = float(np.sqrt(np.mean((norm01(high_full) - norm01(closed)) ** 2)))
        mode_rows.append(
            {
                "mode": mode,
                "spearman_density_loss_vs_closed_fraction": rho_density,
                "spearman_tds_high_fraction_vs_closed_fraction": rho_tds,
                "normalized_shape_rmse_vs_kobayashi_high_fraction": shape_rmse,
                "closed_solid_fraction_min": float(mode_df["closed_solid_fraction_48h"].min()),
                "closed_solid_fraction_max": float(mode_df["closed_solid_fraction_48h"].max()),
                "tail_outlet_fraction_min": float(mode_df["tail_outlet_area_fraction_after_12h"].min()),
                "tail_outlet_fraction_max": float(mode_df["tail_outlet_area_fraction_after_12h"].max()),
                "late_extracted_fraction_min": float(mode_df["late_extracted_fraction_24_48h"].min()),
                "late_extracted_fraction_max": float(mode_df["late_extracted_fraction_24_48h"].max()),
                "max_mass_balance_error": float(mode_df["relative_mass_balance_error"].max()),
            }
        )

    summary_df = pd.DataFrame(mode_rows)
    closed_row = summary_df[summary_df["mode"] == "closed_response"].iloc[0]
    no_closed_row = summary_df[summary_df["mode"] == "no_closed_response"].iloc[0]

    high_full_gain = float(np.max(high_full) / np.min(high_full))
    high_low_gain = float(np.max(high_low) / np.min(high_low))
    checks = [
        {
            "check_id": "tds_high_tail_increases_with_density_loss",
            "status": "PASS"
            if total_points == 1105 and high_full_gain > 10.0 and high_low_gain > 10.0
            else "FAIL",
            "detail": (
                f"Kobayashi points={total_points}; high/full gain={high_full_gain:.2f}; "
                f"high/low gain={high_low_gain:.2f}"
            ),
        },
        {
            "check_id": "zhu_has_separate_high_temperature_tail",
            "status": "PASS"
            if float(np.min(zhu_high_fraction)) > 0.15
            and float(np.max(zhu_high_fraction)) > 0.30
            and float(np.min(zhu_peak_gap)) > 300.0
            else "FAIL",
            "detail": (
                f"Zhu high fraction={np.min(zhu_high_fraction):.3f}-{np.max(zhu_high_fraction):.3f}; "
                f"high-low peak gap min={np.min(zhu_peak_gap):.1f} K"
            ),
        },
        {
            "check_id": "closed_response_matches_tds_direction",
            "status": "PASS"
            if float(closed_row["spearman_tds_high_fraction_vs_closed_fraction"]) > 0.99
            and float(closed_row["normalized_shape_rmse_vs_kobayashi_high_fraction"]) < 0.25
            else "FAIL",
            "detail": (
                f"rho={closed_row['spearman_tds_high_fraction_vs_closed_fraction']:.3f}; "
                f"shape_rmse={closed_row['normalized_shape_rmse_vs_kobayashi_high_fraction']:.3f}; "
                f"no_closed_rmse={no_closed_row['normalized_shape_rmse_vs_kobayashi_high_fraction']:.3f}"
            ),
        },
        {
            "check_id": "no_closed_negative_control_fails_mechanism_direction",
            "status": "PASS"
            if np.isclose(float(no_closed_row["spearman_tds_high_fraction_vs_closed_fraction"]), 0.0)
            and np.isclose(float(no_closed_row["closed_solid_fraction_max"]), 0.0)
            else "FAIL",
            "detail": (
                f"no_closed rho={no_closed_row['spearman_tds_high_fraction_vs_closed_fraction']:.3f}; "
                f"closed max={no_closed_row['closed_solid_fraction_max']:.3e}"
            ),
        },
        {
            "check_id": "closed_response_has_late_outlet_tail",
            "status": "PASS" if float(closed_row["tail_outlet_fraction_min"]) > 0.8 else "FAIL",
            "detail": (
                f"tail area={closed_row['tail_outlet_fraction_min']:.3f}-"
                f"{closed_row['tail_outlet_fraction_max']:.3f}"
            ),
        },
        {
            "check_id": "mass_balance_remains_controlled",
            "status": "PASS" if float(summary_df["max_mass_balance_error"].max()) < 0.01 else "FAIL",
            "detail": f"max relative mass balance error={summary_df['max_mass_balance_error'].max():.4f}",
        },
    ]

    summary = {
        "n_density_points": int(kob.shape[0]),
        "kobayashi_total_points": total_points,
        "n_modes": int(summary_df.shape[0]),
        "n_checks": len(checks),
        "n_pass_checks": sum(1 for row in checks if row["status"] == "PASS"),
        "kobayashi_high_full_min": float(np.min(high_full)),
        "kobayashi_high_full_max": float(np.max(high_full)),
        "kobayashi_high_full_gain": high_full_gain,
        "kobayashi_high_low_gain": high_low_gain,
        "zhu_high_area_fraction_min": float(np.min(zhu_high_fraction)),
        "zhu_high_area_fraction_max": float(np.max(zhu_high_fraction)),
        "zhu_high_low_peak_gap_min_k": float(np.min(zhu_peak_gap)),
        "closed_response_spearman": float(closed_row["spearman_tds_high_fraction_vs_closed_fraction"]),
        "closed_response_shape_rmse": float(closed_row["normalized_shape_rmse_vs_kobayashi_high_fraction"]),
        "closed_response_closed_solid_min": float(closed_row["closed_solid_fraction_min"]),
        "closed_response_closed_solid_max": float(closed_row["closed_solid_fraction_max"]),
        "closed_response_tail_fraction_min": float(closed_row["tail_outlet_fraction_min"]),
        "closed_response_tail_fraction_max": float(closed_row["tail_outlet_fraction_max"]),
        "no_closed_spearman": float(no_closed_row["spearman_tds_high_fraction_vs_closed_fraction"]),
        "no_closed_closed_solid_max": float(no_closed_row["closed_solid_fraction_max"]),
        "claim_boundary": (
            "TDS curves provide temperature-domain mechanism support for the delayed "
            "high-temperature pathway. They do not calibrate isothermal open/closed "
            "multipliers or replace outlet inverse validation."
        ),
    }

    checks_df = pd.DataFrame(checks)
    summary_df.to_csv(OUT_DIR / "li2_tds_closed_pathway_mechanism_closure_summary.csv", index=False)
    checks_df.to_csv(OUT_DIR / "li2_tds_closed_pathway_mechanism_closure_checks.csv", index=False)
    with (OUT_DIR / "li2_tds_closed_pathway_mechanism_closure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "mode_summary": summary_df.to_dict(orient="records"),
                "checks": checks,
            },
            f,
            indent=2,
        )
        f.write("\n")

    if checks_df["status"].ne("PASS").any():
        print(checks_df.to_string(index=False))
        return 1

    print(
        "LI2_TDS_MECHANISM="
        f"points={total_points} high_full_gain={high_full_gain:.2f} "
        f"high_low_gain={high_low_gain:.2f} "
        f"zhu_high={np.min(zhu_high_fraction):.3f}-{np.max(zhu_high_fraction):.3f} "
        f"closed_rho={summary['closed_response_spearman']:.3f} "
        f"shape_rmse={summary['closed_response_shape_rmse']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
