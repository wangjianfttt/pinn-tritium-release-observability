#!/usr/bin/env python3
"""Generate the main-text source-structure uncertainty paragraph."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "fvm3d_source_uncertainty_profile_gate" / "fvm3d_source_uncertainty_profile_gate_summary.json"
OUT = ROOT / "manuscript" / "generated_source_structure_summary.tex"


def pct(value: float) -> str:
    return f"{float(value):.2f}\\%"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    paragraph = (
        "The source-structure calculation evaluates whether the measurement set distinguishes the "
        "open/delayed release basis from incompatible source shapes before component calculations. "
        f"It uses {int(data['n_bootstrap'])} profile fits, {int(data['n_train'])} training histories "
        f"and {int(data['n_test_severe'])} severe-field histories. "
        "With spatial HT/HTO, retained inventory and TES histories, the compatible source structure gives "
        f"{pct(data['full_source_map_l1_p95_pct'])} full-source $L_1$ p95 error "
        f"and a source-map interval width of {pct(data['full_source_interval_l1_width_p95_pct'])}. "
        "The closest incompatible source-map family remains at "
        f"{pct(data['best_wrong_source_map_l1_p95_pct'])} p95 error, "
        f"{data['source_point_error_repair_best_wrong_over_full']:.1f} times the compatible case. "
        f"All incompatible source models exceed either the {pct(data['source_gate_pct'])} point-error scale "
        f"or the {pct(data['interval_gate_pct'])} interval-width scale. "
        "The component calculation receives the open/delayed source basis together with a "
        "source-map interval, a release-mechanism split result and a withheld outlet fit."
    )
    OUT.write_text("\\par\n" + paragraph + "\n\\par\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
