#!/usr/bin/env python3
"""Build a compact module-scale source-propagation table for the SI."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}\\%"


def main() -> None:
    rve = read_json(RESULTS / "rve_transfer_function_bridge" / "rve_transfer_function_summary.json")
    module = read_json(RESULTS / "full_module_network_scaleup" / "full_module_network_summary.json")
    chemistry = read_json(
        RESULTS / "fvm3d_chemistry_module_bridge" / "fvm3d_chemistry_module_bridge_summary.json"
    )["summary"]
    thermal = read_json(
        RESULTS / "fvm3d_thermal_module_bridge" / "fvm3d_thermal_module_bridge_summary.json"
    )["summary"]
    accountancy = read_json(
        RESULTS / "fvm3d_pinn_source_to_system_accountancy_gate" / "fvm3d_pinn_source_to_system_accountancy_gate_summary.json"
    )

    rve_metrics = rve["module_metrics"]
    module_metrics = module["module"]
    grid = module["grid"]

    rows = [
        {
            "calculation": "RVE residence-time kernels",
            "quantity": (
                f"module $t_{{50}}={rve_metrics['t50_h']:.2f}$ h; "
                f"$t_{{90}}={rve_metrics['t90_h']:.2f}$ h; "
                f"48 h tail {pct(100.0 * rve_metrics['tail_fraction_after_48h'])}"
            ),
            "role": "maps local source histories to module outlet and retained inventory",
            "condition": "kernel integrals are normalized before module propagation",
        },
        {
            "calculation": "80-cell module balance",
            "quantity": (
                f"{grid['n_cells']} cells; released fraction {pct(100.0 * module_metrics['released_fraction_48h'])}; "
                f"tail fraction {pct(100.0 * module_metrics['tail_fraction_48h'], 3)}"
            ),
            "role": "compares released tritium with retained tail in the reduced module grid",
            "condition": "reduced module grid for the current component study",
        },
        {
            "calculation": "3D chemistry transport",
            "quantity": (
                f"uniform-total module bias {pct(chemistry['uniform_total_module_inventory_bias_pct'])}; "
                "matched HT/HTO bias reaches the matched-chemistry result"
            ),
            "role": "shows how local chemistry/transport errors change the summed inventory",
            "condition": "requires matched 3D transport and paired species observations",
        },
        {
            "calculation": "3D thermal transport",
            "quantity": (
                f"isothermal closed error {pct(thermal['isothermal_local_closed_error_pct'])}; "
                "matched closed error reaches the calibrated-thermal result"
            ),
            "role": "compares average-temperature fits with the closed-pathway source under a matched thermal field",
            "condition": "requires target thermal field or calibrated temperature reconstruction",
        },
        {
            "calculation": "Summed outlet and inventory response",
            "quantity": (
                f"mixed-total radius {accountancy['mixed_total_combined_radius']:.3f}; "
                f"spatial/species radius {accountancy['full_residual_mlp_combined_radius']:.3f}"
            ),
            "role": "tracks local source-family separation after summation",
            "condition": "uses design-specific fields for named reactor performance analysis",
        },
    ]

    out_csv = RESULTS / "module_scale_interface_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["calculation", "quantity", "role", "condition"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\caption{Module-scale propagation of local source multipliers. The table summarizes how a locally identified source is propagated through residence-time kernels and balance equations into component-level tritium balance.}",
        r"\label{tab:si_module_scale_interface}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.19\textwidth}P{0.25\textwidth}P{0.30\textwidth}P{0.18\textwidth}@{}}",
        r"\toprule",
        r"Calculation & Computed quantity & Scientific role & Use condition \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["calculation"]),
                    row["quantity"],
                    tex_escape(row["role"]),
                    tex_escape(row["condition"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    out_tex = MANUSCRIPT / "generated_module_scale_interface_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
