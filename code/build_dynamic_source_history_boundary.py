#!/usr/bin/env python3
"""Summarize dynamic source-history limits for module propagation.

The component transfer calculation assumes that local release histories are
propagated through calibrated residence-time kernels. This script aggregates
completed stress tests that decide when this assumption is usable:

* source-amplitude coverage for nonlinear delayed release,
* time-dependent power/source histories,
* gain and time-axis calibration of power telemetry.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "dynamic_source_history_boundary"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value == "" else float(value)


def percentile(values: list[float], q: float) -> float:
    values = sorted(value for value in values if not math.isnan(value))
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[int(pos)]
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def median(values: list[float]) -> float:
    return percentile(values, 50.0)


def protocol(rows_in: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    chunk = [row for row in rows_in if row["protocol"] == name]
    if not chunk:
        raise ValueError(f"missing protocol {name}")
    return chunk


def p95(rows_in: list[dict[str, str]], key: str) -> float:
    return percentile([f(row, key) for row in rows_in], 95.0)


def pass_rate(rows_in: list[dict[str, str]]) -> float:
    vals = [row["pass_gate"].strip().lower() == "true" for row in rows_in]
    return sum(vals) / len(vals)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    amp_rows = rows(
        RESULTS
        / "source_amplitude_nonlinearity_gate"
        / "source_amplitude_nonlinearity_detail.csv"
    )
    power_rows = rows(
        RESULTS
        / "dynamic_power_history_release_gate"
        / "dynamic_power_history_release_gate_detail.csv"
    )
    telemetry_rows = rows(
        RESULTS
        / "dynamic_power_telemetry_calibration_gate"
        / "dynamic_power_telemetry_calibration_detail.csv"
    )

    amp_low = protocol(amp_rows, "low_amplitude_linear_kernel")
    amp_bracket_inv = protocol(amp_rows, "bracketed_references_plus_inventory")
    power_const = protocol(power_rows, "constant_power_mixed_total")
    power_spatial = protocol(power_rows, "known_power_spatial_hthto")
    power_full = protocol(power_rows, "known_power_spatial_hthto_inventory_tes")
    tele_raw = protocol(telemetry_rows, "raw_gain_lag_telemetry")
    tele_gain = protocol(telemetry_rows, "gain_only_calibrated")
    tele_full = protocol(telemetry_rows, "gain_lag_calibrated")
    tele_const = protocol(telemetry_rows, "constant_power_mean")

    summary = {
        "source_amplitude": {
            "n_trials": len({row["trial"] for row in amp_rows}),
            "source_amplitude_median": median(
                [f(row, "test_source_amplitude") for row in amp_rows]
            ),
            "low_kernel_closed_p95_pct": p95(amp_low, "closed_error_pct"),
            "low_kernel_tail_p95_pct": p95(amp_low, "tail_inventory_error_pct"),
            "bracketed_inventory_closed_p95_pct": p95(
                amp_bracket_inv, "closed_error_pct"
            ),
            "bracketed_inventory_tail_p95_pct": p95(
                amp_bracket_inv, "tail_inventory_error_pct"
            ),
            "bracketed_inventory_pass_rate": pass_rate(amp_bracket_inv),
        },
        "dynamic_power_history": {
            "n_trials": len({row["trial"] for row in power_rows}),
            "power_cv_median": median([f(row, "power_cv") for row in power_rows]),
            "constant_mixed_closed_p95_pct": p95(power_const, "closed_error_pct"),
            "constant_mixed_residual_median": median(
                [f(row, "relative_residual") for row in power_const]
            ),
            "known_power_spatial_hthto_closed_p95_pct": p95(
                power_spatial, "closed_error_pct"
            ),
            "known_power_spatial_hthto_source_l1_p95_pct": p95(
                power_spatial, "source_map_l1_pct"
            ),
            "known_power_spatial_hthto_tail_p95_pct": p95(
                power_spatial, "tail_inventory_error_pct"
            ),
            "full_dynamic_tail_p95_pct": p95(power_full, "tail_inventory_error_pct"),
            "full_dynamic_pass_rate": pass_rate(power_full),
        },
        "telemetry_calibration": {
            "n_trials": len({row["trial"] for row in telemetry_rows}),
            "raw_closed_p95_pct": p95(tele_raw, "closed_error_pct"),
            "raw_tail_p95_pct": p95(tele_raw, "tail_inventory_error_pct"),
            "gain_only_closed_p95_pct": p95(tele_gain, "closed_error_pct"),
            "gain_only_tail_p95_pct": p95(tele_gain, "tail_inventory_error_pct"),
            "full_calibrated_closed_p95_pct": p95(tele_full, "closed_error_pct"),
            "full_calibrated_tail_p95_pct": p95(
                tele_full, "tail_inventory_error_pct"
            ),
            "full_calibrated_generated_p95_pct": p95(
                tele_full, "generated_history_error_pct"
            ),
            "full_calibrated_pass_rate": pass_rate(tele_full),
            "constant_power_closed_p95_pct": p95(tele_const, "closed_error_pct"),
            "median_lag_estimation_error_h": median(
                [f(row, "telemetry_lag_abs_error_h") for row in tele_full]
            ),
        },
    }
    summary["claim"] = (
        "Local source estimates can be propagated through component kernels only "
        "after source-amplitude coverage, dynamic power history and telemetry "
        "gain/time-axis calibration have been supplied."
    )

    checks = [
        {
            "check_id": "low_amplitude_kernel_rejected_at_high_source",
            "status": "PASS"
            if summary["source_amplitude"]["low_kernel_closed_p95_pct"] > 5.0
            else "FAIL",
            "detail": (
                "closed="
                f"{summary['source_amplitude']['low_kernel_closed_p95_pct']:.2f}%; "
                "tail="
                f"{summary['source_amplitude']['low_kernel_tail_p95_pct']:.2f}%"
            ),
        },
        {
            "check_id": "bracketed_source_references_pass",
            "status": "PASS"
            if (
                summary["source_amplitude"]["bracketed_inventory_closed_p95_pct"]
                < 5.0
                and summary["source_amplitude"]["bracketed_inventory_tail_p95_pct"]
                < 1.0
            )
            else "FAIL",
            "detail": (
                "closed="
                f"{summary['source_amplitude']['bracketed_inventory_closed_p95_pct']:.2f}%; "
                "tail="
                f"{summary['source_amplitude']['bracketed_inventory_tail_p95_pct']:.4f}%"
            ),
        },
        {
            "check_id": "constant_power_rejected_under_dynamic_history",
            "status": "PASS"
            if summary["dynamic_power_history"]["constant_mixed_closed_p95_pct"] > 50.0
            else "FAIL",
            "detail": (
                "closed="
                f"{summary['dynamic_power_history']['constant_mixed_closed_p95_pct']:.2f}%"
            ),
        },
        {
            "check_id": "known_power_spatial_species_pass",
            "status": "PASS"
            if (
                summary["dynamic_power_history"][
                    "known_power_spatial_hthto_closed_p95_pct"
                ]
                < 1.0
                and summary["dynamic_power_history"][
                    "known_power_spatial_hthto_source_l1_p95_pct"
                ]
                < 1.0
            )
            else "FAIL",
            "detail": (
                "closed="
                f"{summary['dynamic_power_history']['known_power_spatial_hthto_closed_p95_pct']:.2f}%; "
                "source="
                f"{summary['dynamic_power_history']['known_power_spatial_hthto_source_l1_p95_pct']:.3f}%"
            ),
        },
        {
            "check_id": "telemetry_gain_and_lag_calibration_pass",
            "status": "PASS"
            if (
                summary["telemetry_calibration"]["full_calibrated_closed_p95_pct"]
                < 5.0
                and summary["telemetry_calibration"]["raw_closed_p95_pct"] > 20.0
            )
            else "FAIL",
            "detail": (
                "raw="
                f"{summary['telemetry_calibration']['raw_closed_p95_pct']:.2f}%; "
                "calibrated="
                f"{summary['telemetry_calibration']['full_calibrated_closed_p95_pct']:.2f}%"
            ),
        },
    ]

    (OUT / "dynamic_source_history_boundary_summary.json").write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUT / "dynamic_source_history_boundary_checks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    n_fail = sum(check["status"] != "PASS" for check in checks)
    print(f"wrote {OUT / 'dynamic_source_history_boundary_summary.json'}")
    print(f"wrote {OUT / 'dynamic_source_history_boundary_checks.csv'}")
    print(
        "dynamic source history: "
        f"low-kernel closed={summary['source_amplitude']['low_kernel_closed_p95_pct']:.2f}%, "
        f"constant-power closed={summary['dynamic_power_history']['constant_mixed_closed_p95_pct']:.2f}%, "
        f"calibrated telemetry closed={summary['telemetry_calibration']['full_calibrated_closed_p95_pct']:.2f}%"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
