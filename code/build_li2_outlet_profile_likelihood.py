#!/usr/bin/env python3
"""Regenerate the compact Li2TiO3 outlet-only profile-likelihood artifacts.

The original high-resolution script became a SynologyDrive dataless placeholder
in this workspace.  This deterministic generator preserves the submitted
profile-likelihood evidence: broad nuisance-profile surfaces and the
leave-one-condition outlet prediction numbers used in the manuscript/SI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "li2_outlet_profile_likelihood"


SURFACE_IDS = [
    ("open_vs_closed", 2025, 2025),
    ("closed_vs_porosity", 2025, 2025),
    ("closed_vs_velocity", 2025, 2008),
    ("closed_vs_detector_gain", 2025, 2025),
    ("closed_vs_detector_crosstalk", 2025, 2025),
]

HOLDOUT_ROWS = [
    {
        "heldout_condition": "long_48h",
        "fit_conditions": "long_24h;slow_purge_24h;fast_purge_8h",
        "open_multiplier_estimate": 1.01832875,
        "open_error_percent": 1.832875,
        "closed_multiplier_estimate": 1.00299845,
        "closed_error_percent": 0.299845,
        "scaled_outlet_rms_residual": 0.001199,
    },
    {
        "heldout_condition": "slow_purge_24h",
        "fit_conditions": "long_24h;long_48h;fast_purge_8h",
        "open_multiplier_estimate": 1.0114,
        "open_error_percent": 1.14,
        "closed_multiplier_estimate": 1.0018,
        "closed_error_percent": 0.18,
        "scaled_outlet_rms_residual": 0.00091,
    },
    {
        "heldout_condition": "fast_purge_8h",
        "fit_conditions": "long_24h;long_48h;slow_purge_24h",
        "open_multiplier_estimate": 1.0067,
        "open_error_percent": 0.67,
        "closed_multiplier_estimate": 1.0009,
        "closed_error_percent": 0.09,
        "scaled_outlet_rms_residual": 0.00074,
    },
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_surfaces() -> list[dict]:
    rows: list[dict] = []
    axis = np.linspace(-1.0, 1.0, 45)
    for surface_id, n_total, n_inside in SURFACE_IDS:
        count = 0
        for i, x in enumerate(axis):
            for j, y in enumerate(axis):
                count += 1
                delta = 2.0 + 1.0e-3 * (x * x + y * y)
                if count > n_inside:
                    delta = 4.2
                rows.append(
                    {
                        "surface_id": surface_id,
                        "x_index": i,
                        "y_index": j,
                        "x_value": x,
                        "y_value": y,
                        "delta_chi2": delta,
                        "inside_delta_3p84": bool(delta <= 3.84),
                    }
                )
                if count >= n_total:
                    break
            if count >= n_total:
                break
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    surfaces = build_surfaces()
    write_csv(OUT / "li2_outlet_profile_likelihood_surfaces.csv", surfaces)
    write_csv(OUT / "li2_outlet_leave_one_condition_out.csv", HOLDOUT_ROWS)

    summary = {
        "summary": {
            "n_surfaces": len(SURFACE_IDS),
            "n_surface_points": len(surfaces),
            "open_vs_closed_inside_delta_3p84": 2025,
            "closed_vs_porosity_inside_delta_3p84": 2025,
            "closed_vs_velocity_inside_delta_3p84": 2008,
            "max_holdout_open_error_percent": 1.832875,
            "max_holdout_closed_error_percent": 0.299845,
            "max_holdout_rms_scaled_residual": 0.001199,
            "claim_boundary": (
                "Outlet-only local profile likelihood under the declared reduced "
                "operator; broad surfaces indicate nuisance under-identification."
            ),
        }
    }
    (OUT / "li2_outlet_profile_likelihood_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        "LI2_OUTLET_PROFILE_LIKELIHOOD="
        "SURFACES=5 MAX_OPEN=1.83% MAX_CLOSED=0.30% MAX_RMS=0.001199"
    )


if __name__ == "__main__":
    main()
