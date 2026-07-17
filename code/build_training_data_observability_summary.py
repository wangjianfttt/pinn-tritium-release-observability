#!/usr/bin/env python3
"""Generate a concise manuscript summary on PINN training-data observability."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
RESULTS = ROOT / "results"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{float(value):.2f}\\%"


def main() -> None:
    aug_path = (
        RESULTS
        / "augmented_data_vs_observability_closure_law"
        / "augmented_data_vs_observability_closure_law_summary.json"
    )
    closure_path = (
        RESULTS
        / "pinn_training_data_science_closure"
        / "pinn_training_data_science_closure_summary.json"
    )
    aug = read_json(aug_path)["summary"]
    closure = read_json(closure_path)

    required = {
        "wrong_observable_1800_closed_p95_pct": 9.989913145563923,
        "full_multimodal_first_pass_closed_p95_pct": 4.527600783809816,
        "mixed_level_curriculum_severe_closed_pct": 22.307692307692314,
        "field_conditioned_3d_closed_pct": 0.2564102564102566,
        "pde_residual_source_l1_p95_pct": 38.338008040420526,
        "full_neural_source_l1_p95_pct": 3.2471175286015286,
    }
    for key, expected in required.items():
        actual = float(aug[key])
        if abs(actual - expected) > 1e-10:
            raise ValueError(f"{key}: expected {expected}, got {actual}")

    hidden_required = {
        "hidden_oracle_worst_closed_p95_pct": 0.006708671373515475,
        "hidden_missing_worst_closed_p95_pct": 73.46794229287366,
        "hidden_leakage_artificial_gain_vs_total": 1384.955712160209,
    }
    for key, expected in hidden_required.items():
        actual = float(closure[key])
        if abs(actual - expected) > 1e-10:
            raise ValueError(f"{key}: expected {expected}, got {actual}")

    main_text = (
        "The training-data separation calculation uses three history groups. The target-bed pilot set "
        "calibrates the 3D response family. The computed extension set varies source amplitude, chemistry "
        "and temperature inside that calibrated response family. These histories extend the calibrated response "
        "and do not replace the pilot measurements that supply the bed-response information. "
        "The physical prediction set changes one mechanism or field outside fitting. "
        "The comparison tests sample count, measurement content, field identity and source-term representation. "
        f"Using 1800 histories with the total-release measurement leaves a {pct(aug['wrong_observable_1800_closed_p95_pct'])} "
        f"closed-pathway p95 error, while the full spatial HT/HTO and balance measurement set reaches "
        f"{pct(aug['full_multimodal_first_pass_closed_p95_pct'])}. "
        f"Combining mild 3D fields for a severe field leaves {pct(aug['mixed_level_curriculum_severe_closed_pct'])} "
        f"closed-pathway error; using the field-conditioned 3D residence-time response gives {pct(aug['field_conditioned_3d_closed_pct'])}. "
        f"The source-output comparison gives the same conclusion with a source-distribution metric: "
        f"source extraction from the residual field leaves {pct(aug['pde_residual_source_l1_p95_pct'])} p95 error, "
        f"while the explicit source-term estimate gives {pct(aug['full_neural_source_l1_p95_pct'])}. "
        "Sample growth becomes useful after the measured variables, calibrated bed response and explicit source-term estimate are present."
    )
    si_text = (
        "The training-data separation summary combines pilot-response calibration, computed-history extension, "
        "measurement-set, field-identity, source-term representation and solid-inventory representation calculations. Total-release sample growth to 1800 histories remains at "
        f"{pct(aug['wrong_observable_1800_closed_p95_pct'])} closed-pathway p95 error, compared with "
        f"{pct(aug['full_multimodal_first_pass_closed_p95_pct'])} for the full spatial HT/HTO and balance set. "
        f"Combined mild 3D-field training gives {pct(aug['mixed_level_curriculum_severe_closed_pct'])} severe-field "
        f"closed-pathway error, while the field-conditioned 3D residence-time response gives {pct(aug['field_conditioned_3d_closed_pct'])}. "
        f"In the source-output comparison, source extraction from the residual field gives "
        f"{pct(aug['pde_residual_source_l1_p95_pct'])} source-distribution p95 error; the explicit source-term estimate gives "
        f"{pct(aug['full_neural_source_l1_p95_pct'])}. "
        f"A solid-inventory label input gives an artificial {pct(closure['hidden_oracle_worst_closed_p95_pct'])} error "
        f"inside the representation check and {pct(closure['hidden_missing_worst_closed_p95_pct'])} after the input is removed, "
        f"corresponding to a {closure['hidden_leakage_artificial_gain_vs_total']:.0f}-fold artificial gain. "
        r"The underlying JSON outputs are stored in the generated results directory and are rebuilt by the training-data separation script."
    )

    (MANUSCRIPT / "generated_training_data_observability_summary.tex").write_text(
        "\\par\n" + main_text + "\n\\par\n", encoding="utf-8"
    )
    (MANUSCRIPT / "generated_training_data_observability_si.tex").write_text(
        "\\par\n" + si_text + "\n\\par\n", encoding="utf-8"
    )
    print(
        "Training-data observability: "
        f"total1800={aug['wrong_observable_1800_closed_p95_pct']:.2f}% "
        f"full={aug['full_multimodal_first_pass_closed_p95_pct']:.2f}% "
        f"pooled3d={aug['mixed_level_curriculum_severe_closed_pct']:.2f}% "
        f"field3d={aug['field_conditioned_3d_closed_pct']:.3f}%"
    )
    print(f"wrote {MANUSCRIPT / 'generated_training_data_observability_summary.tex'}")
    print(f"wrote {MANUSCRIPT / 'generated_training_data_observability_si.tex'}")


if __name__ == "__main__":
    main()
