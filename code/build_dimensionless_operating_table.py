#!/usr/bin/env python3
"""Build dimensionless operating groups from archived model parameters."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from build_3d_fvm_chemistry_transport_pressure import LENGTH_M as FVM3D_LENGTH_M
from build_3d_fvm_chemistry_transport_pressure import make_fields
from li_ceramic_bed_1d import Li2TiO3Params, SharedParams


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def fmt(value: float, digits: int = 2) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def main() -> None:
    shared = SharedParams()
    li2 = Li2TiO3Params()

    tau_adv_1d_s = shared.length_m / shared.superficial_velocity_m_s
    pe_1d = shared.superficial_velocity_m_s * shared.length_m / shared.axial_dispersion_m2_s
    tau_open_h = 1.0 / li2.k_open_release_s_1 / 3600.0
    tau_closed_h = 1.0 / li2.k_closed_release_s_1 / 3600.0

    case_rows = []
    pe_ht_values = []
    pe_hto_values = []
    tau_3d_h = []
    da_exchange = []
    da_loss = []
    por_cv = []
    vel_cv = []
    source_cv = []
    for case_id in ["uniform_dry", "heterogeneous_wet_wall", "heterogeneous_wet_bypass"]:
        fields = make_fields(case_id)
        vel = fields["velocity"]
        por = fields["porosity"]
        exch = fields["exchange_rate_s"]
        loss = fields["loss_rate_s"]
        src = fields["source_shape"]
        assert hasattr(vel, "mean")
        tau_h = FVM3D_LENGTH_M / float(vel.mean()) / 3600.0
        pe_ht = float(vel.mean()) * FVM3D_LENGTH_M / float(fields["diffusivity_ht"])
        pe_hto = float(vel.mean()) * FVM3D_LENGTH_M / float(fields["diffusivity_hto"])
        case_rows.append(
            {
                "case_id": case_id,
                "mean_velocity_m_s": float(vel.mean()),
                "mean_porosity": float(por.mean()),
                "gas_residence_time_h": tau_h,
                "pe_ht": pe_ht,
                "pe_hto": pe_hto,
                "exchange_da": float(exch.mean()) * tau_h * 3600.0,
                "loss_da": float(loss.mean()) * tau_h * 3600.0,
                "porosity_cv_pct": 100.0 * float(por.std() / por.mean()),
                "velocity_cv_pct": 100.0 * float(vel.std() / vel.mean()),
                "source_cv_pct": 100.0 * float(src.std() / src.mean()),
            }
        )
        pe_ht_values.append(case_rows[-1]["pe_ht"])
        pe_hto_values.append(case_rows[-1]["pe_hto"])
        tau_3d_h.append(case_rows[-1]["gas_residence_time_h"])
        da_exchange.append(case_rows[-1]["exchange_da"])
        da_loss.append(case_rows[-1]["loss_da"])
        por_cv.append(case_rows[-1]["porosity_cv_pct"])
        vel_cv.append(case_rows[-1]["velocity_cv_pct"])
        source_cv.append(case_rows[-1]["source_cv_pct"])

    module = json.loads((RESULTS / "module_operating_envelope" / "module_operating_envelope_summary.json").read_text())
    component = json.loads((RESULTS / "component_3d_module_network" / "component_3d_summary.json").read_text())
    t90_values = [
        module["best_release"]["t90_h"],
        module["nominal"]["t90_h"],
        module["worst_tail"]["t90_h"],
    ]

    tau_ht_h = 0.050
    tau_hto_h = 0.450
    rows = [
        {
            "group": r"1D gas Peclet number, \(Pe_L=uL/D_z\)",
            "value": fmt(pe_1d),
            "source": "1D reduced parameters",
            "meaning": "Reduced outlet histories are advection dominated at bed length scale.",
        },
        {
            "group": r"1D open/closed release Damkohler numbers, \(kL/u\)",
            "value": f"{fmt(li2.k_open_release_s_1 * tau_adv_1d_s)}/{fmt(li2.k_closed_release_s_1 * tau_adv_1d_s, 4)}",
            "source": "Li2TiO3/SharedParams",
            "meaning": "Gas residence time is shorter than material release, especially for the delayed path.",
        },
        {
            "group": r"open/closed release time-scale ratio, \(\tau_c/\tau_o\)",
            "value": fmt(tau_closed_h / tau_open_h),
            "source": "Li2TiO3Params",
            "meaning": "The delayed pathway is separated by an order-of-magnitude material time scale.",
        },
        {
            "group": r"3D gas residence time, \(L/\bar u\)",
            "value": f"{fmt(min(tau_3d_h))}--{fmt(max(tau_3d_h))} h",
            "source": "3D-FVM field script",
            "meaning": "3D transport delay is comparable to the early release and shorter than the closed tail.",
        },
        {
            "group": r"3D HT/HTO Peclet numbers",
            "value": f"{fmt(min(pe_ht_values))}--{fmt(max(pe_ht_values))}/{fmt(min(pe_hto_values))}--{fmt(max(pe_hto_values))}",
            "source": "3D-FVM fields",
            "meaning": "The static-bed histories probe advection-dominated species transport.",
        },
        {
            "group": r"HT to HTO exchange Damkohler number",
            "value": f"{fmt(min(da_exchange), 3)}--{fmt(max(da_exchange), 3)}",
            "source": "3D-FVM chemistry fields",
            "meaning": "Species conversion is small per gas residence time and changes the late outlet fraction.",
        },
        {
            "group": r"HTO loss Damkohler number",
            "value": f"{fmt(min(da_loss), 4)}--{fmt(max(da_loss), 4)}",
            "source": "3D-FVM loss fields",
            "meaning": "Unrecovered loss is weak per pass and remains a calibrated nuisance term.",
        },
        {
            "group": "3D porosity/velocity/source CV",
            "value": f"{fmt(min(por_cv))}--{fmt(max(por_cv))}%/{fmt(min(vel_cv))}--{fmt(max(vel_cv))}%/{fmt(min(source_cv))}--{fmt(max(source_cv))}%",
            "source": "3D-FVM field summaries",
            "meaning": "The computed static beds include heterogeneity in transport and source shape.",
        },
        {
            "group": r"HTO memory to 3D residence-time ratio, \(\tau_{\rm HTO}/(L/\bar u)\)",
            "value": f"{fmt(tau_hto_h / max(tau_3d_h))}--{fmt(tau_hto_h / min(tau_3d_h))}",
            "source": "detector-memory and 3D-FVM fields",
            "meaning": "Sampling-line memory can overlap the packed-bed residence-time scale.",
        },
        {
            "group": r"module \(t_{90}/48\) h range",
            "value": f"{fmt(min(t90_values) / 48.0)}--{fmt(max(t90_values) / 48.0)}",
            "source": "module envelope summary",
            "meaning": "Module histories span fast and slow release relative to the 48 h tail.",
        },
        {
            "group": r"component max-cell \(t_{90}/48\) h",
            "value": fmt(component["component"]["max_cell_t90_h"] / 48.0),
            "source": "component summary",
            "meaning": "Some component cells retain delayed release over most of the 48 h history.",
        },
    ]

    out_cases = RESULTS / "dimensionless_operating_case_values.csv"
    with out_cases.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    out_csv = RESULTS / "dimensionless_operating_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Dimensionless operating groups and time-scale ratios computed from the archived reduced, 3D-FVM and component fields.}",
        r"\label{tab:dimensionless_operating}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.25\textwidth}P{0.15\textwidth}P{0.24\textwidth}Y@{}}",
        r"\toprule",
        r"Group & Value or range & Source & Use in this study \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["group"],
                    tex_escape(row["value"]),
                    tex_escape(row["source"]),
                    tex_escape(row["meaning"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
    out_tex = MANUSCRIPT / "generated_dimensionless_operating_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_cases}")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
