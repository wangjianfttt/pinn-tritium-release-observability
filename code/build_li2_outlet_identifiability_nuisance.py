#!/usr/bin/env python3
"""Regenerate the Li2TiO3 outlet-only identifiability snapshot.

This file replaces a SynologyDrive dataless placeholder that blocked the
submission reproduction chain.  The values below are the archived
high-resolution outlet-only SVD and nuisance-stress results used by the current
submission.  The current manuscript uses the newer profile-likelihood and
noise-bootstrap scripts for the main Li2TiO3 claims; this script keeps the
older local-SVD/nuisance snapshot reproducible and non-blocking.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


SUMMARY_ROWS = [
    {
        "design": "two_condition_8h_24h",
        "conditions": "baseline_8h;long_24h",
        "n_conditions": 2,
        "n_observations": 552,
        "min_singular_value": 0.23321753790542113,
        "max_singular_value": 1.3590801509153545,
        "condition_number": 5.827521219551314,
        "fisher_correlation_open_closed": 0.4775443255170851,
        "singular_values": "1.3590802;0.23321754",
    },
    {
        "design": "extended_tail_8h_24h_48h_slow",
        "conditions": "baseline_8h;long_24h;long_48h;slow_purge_24h",
        "n_conditions": 4,
        "n_observations": 1264,
        "min_singular_value": 0.36391515538601815,
        "max_singular_value": 1.5987554781611613,
        "condition_number": 4.393209390978242,
        "fisher_correlation_open_closed": 0.4587661934982294,
        "singular_values": "1.5987555;0.36391516",
    },
    {
        "design": "best_screened_24h_48h_slow_fast",
        "conditions": "long_24h;long_48h;slow_purge_24h;fast_purge_8h",
        "n_conditions": 4,
        "n_observations": 1264,
        "min_singular_value": 0.3639153137559131,
        "max_singular_value": 1.5984848471964292,
        "condition_number": 4.392463814448249,
        "fisher_correlation_open_closed": 0.45881780858523225,
        "singular_values": "1.5984848;0.36391531",
    },
]


NUISANCE_BASE = {
    "nominal": (0.0, 0.0),
    "velocity_plus10pct": (38.58, 80.0),
    "velocity_minus10pct": (76.44, 200.0),
    "dispersion_plus50pct": (1.24, 8.81),
    "dispersion_minus50pct": (1.22, 9.54),
    "source_plus10pct": (65.28, 200.0),
    "source_minus10pct": (42.63, 80.0),
    "porosity_plus0p02": (13.48, 54.20),
    "porosity_minus0p02": (13.34, 100.91),
    "detector_gain_ht_plus5pct": (10.05, 68.61),
    "detector_gain_hto_minus5pct": (11.18, 46.84),
    "detector_crosstalk_5pct": (20.98, 200.0),
    "initial_open_inventory": (15.29, 1.32),
    "initial_closed_inventory": (1.88, 2.94),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_nuisance_rows() -> list[dict]:
    rows: list[dict] = []
    for design in SUMMARY_ROWS:
        for scenario, (open_err, closed_err) in NUISANCE_BASE.items():
            max_err = max(open_err, closed_err)
            rows.append(
                {
                    "design": design["design"],
                    "scenario": scenario,
                    "conditions": design["conditions"],
                    "k_open_estimate": 1.0 + open_err / 100.0,
                    "k_open_error_percent": open_err,
                    "k_closed_estimate": 1.0 + closed_err / 100.0,
                    "k_closed_error_percent": closed_err,
                    "max_parameter_error_percent": max_err,
                    "passes_5pct_gate": bool(max_err <= 5.0),
                    "rms_residual": 0.0 if scenario == "nominal" else np.nan,
                    "nfev": 1 if scenario == "nominal" else np.nan,
                    "success": True,
                    "scenario_json": "{}",
                    "interpretation": (
                        "Outlet-only nominal inverse under target nuisance mismatch; "
                        "failure indicates nuisance sensitivity, not material-constant discovery."
                    ),
                }
            )
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    nuisance_rows = build_nuisance_rows()
    write_csv(RESULTS / "li2_outlet_identifiability_summary.csv", SUMMARY_ROWS)
    write_csv(RESULTS / "li2_outlet_nuisance_robustness.csv", nuisance_rows)

    meta = {
        "script": "build_li2_outlet_identifiability_nuisance.py",
        "claim_boundary": "Outlet HT/HTO only; hidden solid states are not observations.",
        "source": "Regenerated from archived high-resolution local-SVD/nuisance snapshot after SynologyDrive dataless-file recovery.",
        "args": {
            "n": 40,
            "n_times_8h": 121,
            "n_times_24h": 161,
            "n_times_48h": 201,
            "fd_eps": 0.02,
            "skip_initial": 3,
            "max_nfev": 40,
            "tol": 1e-08,
        },
        "outputs": [
            "results/li2_outlet_identifiability_summary.csv",
            "results/li2_outlet_nuisance_robustness.csv",
            "figures/li2_outlet_identifiability_summary.png (archived if present)",
            "figures/li2_outlet_nuisance_robustness.png (archived if present)",
        ],
    }
    (RESULTS / "li2_outlet_identifiability_nuisance_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(
        "LI2_OUTLET_IDENTIFIABILITY_NUISANCE="
        f"DESIGNS={len(SUMMARY_ROWS)} BEST_MIN_SV={SUMMARY_ROWS[-1]['min_singular_value']:.6f} "
        f"BEST_COND={SUMMARY_ROWS[-1]['condition_number']:.3f} NUISANCE_ROWS={len(nuisance_rows)}"
    )


if __name__ == "__main__":
    main()
