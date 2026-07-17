#!/usr/bin/env python3
"""Build a compact table for the minimum measurement set.

The table is derived from the existing reduced 3D, module and 288-component system
artifacts. It does not add new simulations; it makes the final experimental
prescription explicit and traceable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: float | str, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}\\%"


def esc(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def status_phrase(row: dict[str, str]) -> str:
    if row["system_ready_for_bounded_propagation"] == "True":
        return "Yes"
    if row["local_joint_gate_pass"] == "False":
        return "No: 3D source unresolved"
    if row["module_gate_pass"] == "False":
        return "No: module inventory broad"
    if row["p5_2pct_accountancy_gate_pass"] == "False":
        return "No: system p95 above 5%"
    return "No: robustness limit"


def main_status_phrase(row: dict[str, str]) -> str:
    if row["system_ready_for_bounded_propagation"] == "True":
        return "usable"
    if row["local_joint_gate_pass"] == "False":
        return "source unresolved"
    if row["module_gate_pass"] == "False":
        return "inventory broad"
    if row["p5_2pct_accountancy_gate_pass"] == "False":
        return "system error high"
    return "repeat needed"


def main() -> None:
    rows = read_csv(
        RESULTS
        / "minimal_multifield_diagnostic_prescription"
        / "minimal_multifield_diagnostic_prescription.csv"
    )
    summary = read_json(
        RESULTS
        / "minimal_multifield_diagnostic_prescription"
        / "minimal_multifield_diagnostic_prescription_summary.json"
    )["summary"]
    row_by_id = {row["diagnostic_id"]: row for row in rows}
    keep_ids = [
        "single_mixed_total",
        "matched_spatial_hthto_only",
        "matched_spatial_hthto_plus_inventory",
        "matched_spatial_hthto_plus_inventory_tes",
        "pooled_field_plus_inventory_tes",
        "wrong_field_plus_inventory_tes",
    ]

    out_rows = []
    for rid in keep_ids:
        row = row_by_id[rid]
        out_rows.append(
            {
                "measurement_set": row["label"]
                .replace("matched", "field-calibrated")
                .replace("inventory+TES", "inventory + TES"),
                "observables": row["observables"],
                "local_source_p95_pct": float(row["local_after_aux_envelope_pct"]),
                "module_inventory_half_width_pct": float(row["module_inventory_half_width_pct"]),
                "system_2pct_p95_pct": float(row["p5_2pct_accountancy_p95_pct"]),
                "cross_shift_or_repeat": (
                    "yes" if row["p5_cross_shift_or_replicate_robust_pass"] == "True" else "no"
                ),
                "decision": status_phrase(row),
                "main_decision": main_status_phrase(row),
            }
	        )

    for row in out_rows:
        row["measurement_set"] = row["measurement_set"].replace(
            "nearest wrong field", "nearest mismatched field"
        )

    out_csv = RESULTS / "minimum_measurement_set_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Minimum measurement set for carrying a local PINN source estimate into module and the generic 288-component reduced tritium-balance model. The values are derived from the reduced 3D field-family, module inventory and generic system calculations.}",
        r"\label{tab:si_minimum_measurement_set}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.25\textwidth}P{0.17\textwidth}P{0.17\textwidth}P{0.17\textwidth}X@{}}",
        r"\toprule",
        r"Measurement set & Local source error & Module inventory half-width & System p95 at 2\% balance noise & Propagation use \\",
        r"\midrule",
    ]
    for row in out_rows:
        lines.append(
            " & ".join(
                [
                    esc(row["measurement_set"]),
                    pct(row["local_source_p95_pct"]),
                    pct(row["module_inventory_half_width_pct"]),
                    pct(row["system_2pct_p95_pct"]),
                    esc(row["decision"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
        ]
    )
    out_tex = MANUSCRIPT / "generated_minimum_measurement_set_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")

    main_lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Measurement set needed to carry a local PINN source estimate into module and the generic 288-component reduced tritium-balance model.}",
        r"\label{tab:minimum_measurement_set}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.30\textwidth}P{0.15\textwidth}P{0.17\textwidth}P{0.16\textwidth}Y@{}}",
        r"\toprule",
        r"Measurements & Local source & Module inventory & System p95 & Propagation \\",
        r"\midrule",
    ]
    for row in out_rows:
        main_lines.append(
            " & ".join(
                [
                    esc(row["measurement_set"]),
                    pct(row["local_source_p95_pct"]),
                    pct(row["module_inventory_half_width_pct"]),
                    pct(row["system_2pct_p95_pct"]),
                    esc(row["main_decision"]),
                ]
            )
            + r" \\"
        )
    main_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    out_main_tex = MANUSCRIPT / "generated_main_minimum_measurement_set_table.tex"
    out_main_tex.write_text("\n".join(main_lines), encoding="utf-8")

    note = (
        "Minimum measurement set: field-calibrated spatial HT/HTO plus retained "
        f"inventory and TES gives module half-width {summary['inventory_tes_module_half_width_pct']:.2f}% "
        f"and system p95 {summary['inventory_tes_p5_2pct_p95_pct']:.2f}% at 2% balance noise; "
        f"cross-shift worst p95 is {summary['inventory_tes_cross_shift_noise2_worst_p95_pct']:.2f}%."
    )
    print(note)
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")
    print(f"wrote {out_main_tex}")


if __name__ == "__main__":
    main()
