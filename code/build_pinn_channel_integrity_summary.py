#!/usr/bin/env python3
"""Generate the PINN channel-integrity summary used in the manuscript."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(value: float, target: float, tol: float = 1e-10) -> bool:
    return abs(float(value) - target) <= tol


def main() -> None:
    scramble_path = (
        RESULTS
        / "component_3d_neural_information_scrambling"
        / "component_3d_neural_information_scrambling_summary.json"
    )
    sensor_path = (
        RESULTS
        / "component_3d_neural_sensor_degradation"
        / "component_3d_neural_sensor_degradation_summary.json"
    )
    for path in [scramble_path, sensor_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    scramble = read_json(scramble_path)["summary"]
    sensor = read_json(sensor_path)["summary"]

    expected = {
        "full_worst_closed_p95_pct": 4.024409294814114,
        "spatial_scrambled_worst_closed_p95_pct": 92.16325235206823,
        "species_swapped_worst_closed_p95_pct": 93.3206722910905,
        "contrast_collapsed_worst_closed_p95_pct": 86.32987807882854,
        "total_only_worst_closed_p95_pct": 14.524764079062367,
        "worst_scrambled_to_full_ratio": 23.1886633428027,
    }
    for key, target in expected.items():
        if not close(float(scramble[key]), target):
            raise ValueError(f"{key}: expected {target}, got {scramble[key]}")

    expected_sensor = {
        "nominal_full_worst_closed_p95_pct": 3.7443861559343987,
        "outlet_degraded_full_worst_closed_p95_pct": 5.928448837045624,
        "compound_degraded_total_accountancy_worst_closed_p95_pct": 16.886046478215107,
        "compound_degraded_single_worst_closed_p95_pct": 42.95589428240332,
    }
    for key, target in expected_sensor.items():
        if not close(float(sensor[key]), target):
            raise ValueError(f"{key}: expected {target}, got {sensor[key]}")

    main_text = (
        "A channel-identity calculation evaluates whether the neural inverse uses the intended "
        "physical observations. With the full spatial HT/HTO and balance vector, the "
        f"worst closed-pathway p95 error is {scramble['full_worst_closed_p95_pct']:.2f}\\%. "
        "When deployment histories keep the same values with wrong spatial registration, "
        "swapped HT/HTO identity or collapsed outlet-to-outlet contrast, "
        f"the corresponding errors rise to {scramble['spatial_scrambled_worst_closed_p95_pct']:.2f}\\%, "
        f"{scramble['species_swapped_worst_closed_p95_pct']:.2f}\\% and "
        f"{scramble['contrast_collapsed_worst_closed_p95_pct']:.2f}\\%. "
        f"A total-release plus balance vector remains at {scramble['total_only_worst_closed_p95_pct']:.2f}\\%. "
        "Correct spatial and species identities are part of the learned input; "
        "total release amplitude alone is insufficient."
    )
    si_text = (
        "The channel-integrity calculation perturbs deployment-time channel identity after "
        "training. The full physical vector gives "
        f"{scramble['full_worst_closed_p95_pct']:.2f}\\% worst closed-pathway p95 error, "
        f"while spatial registration scrambling, HT/HTO swapping and spatial-contrast collapse give "
        f"{scramble['spatial_scrambled_worst_closed_p95_pct']:.2f}\\%, "
        f"{scramble['species_swapped_worst_closed_p95_pct']:.2f}\\% and "
        f"{scramble['contrast_collapsed_worst_closed_p95_pct']:.2f}\\%. "
        "A separate sensor-quality sensitivity calculation gives "
        f"{sensor['nominal_full_worst_closed_p95_pct']:.2f}\\% for the nominal full vector and "
        f"{sensor['outlet_degraded_full_worst_closed_p95_pct']:.2f}\\% when outlet HT/HTO quality is degraded. "
        "The source files are "
        r"\path{results/component_3d_neural_information_scrambling/component_3d_neural_information_scrambling_summary.json} "
        "and "
        r"\path{results/component_3d_neural_sensor_degradation/component_3d_neural_sensor_degradation_summary.json}."
    )

    (MANUSCRIPT / "generated_pinn_channel_integrity_summary.tex").write_text(
        "\\par\n" + main_text + "\n\\par\n", encoding="utf-8"
    )
    (MANUSCRIPT / "generated_pinn_channel_integrity_si.tex").write_text(
        "\\par\n" + si_text + "\n\\par\n", encoding="utf-8"
    )
    print(
        "PINN channel integrity: "
        f"full={scramble['full_worst_closed_p95_pct']:.2f}% "
        f"spatial={scramble['spatial_scrambled_worst_closed_p95_pct']:.2f}% "
        f"species={scramble['species_swapped_worst_closed_p95_pct']:.2f}% "
        f"contrast={scramble['contrast_collapsed_worst_closed_p95_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
