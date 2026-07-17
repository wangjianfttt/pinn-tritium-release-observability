#!/usr/bin/env python3
"""Generate the SI reduced-model parameter table from the tracked CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"
INPUT = RESULTS / "reduced_model_parameter_table.csv"
OUTPUT = MANUSCRIPT / "generated_reduced_model_parameter_table.tex"
SUMMARY = RESULTS / "reduced_model_parameter_table_si_summary.json"


GROUP_LABELS = {
    "shared_gas_transport": "Transport and operating variables",
    "li2tio3_grain_trap_open_closed": "Li$_2$TiO$_3$ release model",
    "li4sio4_grain_surface_water": "Li$_4$SiO$_4$ protocol-transfer branch",
    "hthto_chemistry_exchange": "HT/HTO chemistry exchange",
    "detector_response": "Detector response",
    "thermal_response": "Thermal response variables",
}


ROLE_REWRITE = {
    "protocol-transfer multiplier": "protocol-transfer multiplier",
    "nuisance/calibration parameter": "nuisance calib.",
}


SYMBOL_LATEX = {
    "epsilon": r"\epsilon",
    "D_ax": r"D_{\rm ax}",
    "x_H2": r"x_{\rm H2}",
    "k_diff": r"k_{\rm diff}",
    "k_trap": r"k_{\rm trap}",
    "k_detrap": r"k_{\rm detrap}",
    "k_closed,in": r"k_{\rm closed,in}",
    "k_closed,rel": r"k_{\rm closed,rel}",
    "k_open,rel": r"k_{\rm open,rel}",
    "f_HT,0": r"f_{\rm HT,0}",
    "k_ex,HT": r"k_{\rm ex,HT}",
    "k_ex,HTO": r"k_{\rm ex,HTO}",
    "k_ads,H2O": r"k_{\rm ads,H2O}",
    "k_des,H2O": r"k_{\rm des,H2O}",
    "k_ex,H2O": r"k_{\rm ex,H2O}",
    "W_s(0)": r"W_s(0)",
    "f_chi": r"f_{\chi}",
    "alpha_ex": r"\alpha_{\rm ex}",
    "alpha_wall": r"\alpha_{\rm wall}",
    "lambda_loss": r"\lambda_{\rm loss}",
    "g_HT": r"g_{\rm HT}",
    "g_HTO": r"g_{\rm HTO}",
    "eta_HT_to_HTO": r"\eta_{\rm HT\to HTO}",
    "eta_HTO_to_HT": r"\eta_{\rm HTO\to HT}",
    "lambda_HT,det": r"\lambda_{\rm HT,det}",
    "lambda_HTO,det": r"\lambda_{\rm HTO,det}",
    "T_0": r"T_0",
    "DeltaT_3D": r"\Delta T_{\rm 3D}",
    "DeltaT_transfer": r"\Delta T_{\rm transfer}",
    "E_z,min": r"E_{z,\min}",
    "E_z,max": r"E_{z,\max}",
}


UNIT_LATEX = {
    "m s^-1": r"m s$^{-1}$",
    "m^2 s^-1": r"m$^2$ s$^{-1}$",
    "mol m^-3 s^-1": r"mol m$^{-3}$ s$^{-1}$",
    "s^-1": r"s$^{-1}$",
    "kJ mol^-1": r"kJ mol$^{-1}$",
}


def esc(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def compact_value(value: str, unit: str) -> str:
    number = float(value)
    if number == 0:
        shown = "0"
    elif abs(number) < 1e-3 or abs(number) >= 1e4:
        shown = f"{number:.2e}"
    else:
        shown = f"{number:.4g}"
    unit_tex = UNIT_LATEX.get(unit, esc(unit))
    return f"{shown} {unit_tex}".strip()


def clean_meaning(text: str) -> str:
    out = esc(text)
    out = out.replace("Li2TiO3", r"Li$_2$TiO$_3$")
    out = out.replace("Li4SiO4", r"Li$_4$SiO$_4$")
    out = out.replace("reduced benchmark", "reduced reference")
    out = out.replace("m\\_closed", r"$m_{\rm closed}$")
    out = out.replace("m\\_open", r"$m_{\rm open}$")
    out = out.replace("thermal-operator transfer", "thermal-response transfer")
    out = out.replace("profiled in outlet-only tests", "profiled in outlet-only profiles")
    out = out.replace("thermal source test", "thermal source case")
    out = out.replace("thermal-response transfer test", "thermal-response transfer calculation")
    out = out.replace("module pressure test", "module pressure calculation")
    out = out.replace("wall-conversion stress", "wall-conversion response")
    out = out.replace("stress test", "sensitivity calculation")
    out = out.replace("stress tests", "sensitivity calculations")
    out = out.replace("stress case", "response case")
    out = out.replace("H2", r"H$_2$")
    out = out.replace("HTO", "HTO")
    return out


def main() -> int:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("parameter table is empty")

    lines: list[str] = [
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{longtable}{@{}P{0.16\textwidth}P{0.15\textwidth}P{0.16\textwidth}P{0.47\textwidth}@{}}",
        r"\caption{Reduced-model parameters used by the source-inverse calculations. Values are default reference values; inverse multipliers scale the listed kinetic rates or source partitions.}",
        r"\label{tab:si_reduced_parameters}\\",
        r"\toprule",
        r"Parameter & Default value & Role & Meaning \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Reduced-model parameters used by the source-inverse calculations (continued).}\\",
        r"\toprule",
        r"Parameter & Default value & Role & Meaning \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    current_group = None
    for row in rows:
        group = row["block"]
        if group != current_group:
            current_group = group
            lines.append(
                rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{GROUP_LABELS.get(group, esc(group))}}}}} \\"
            )
        symbol = SYMBOL_LATEX.get(row["symbol"], esc(row["symbol"]))
        value_unit = compact_value(row["value"], row["unit"])
        role = esc(ROLE_REWRITE.get(row["role"], row["role"]))
        meaning = clean_meaning(row["meaning"])
        lines.append(rf"${symbol}$ & {value_unit} & {role} & {meaning} \\")

    lines.extend(
        [
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    n_inverse = sum("inverse target" in row["role"] for row in rows)
    n_condition = sum(
        row["role"] in {"nuisance/fixed", "condition variable", "fixed/reference"}
        for row in rows
    )
    summary = {
        "input": str(INPUT.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "n_parameters": len(rows),
        "n_inverse_target_parameters": n_inverse,
        "n_transport_or_condition_parameters": n_condition,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "REDUCED_PARAMETER_TABLE="
        f"rows={len(rows)} inverse_targets={n_inverse} output={OUTPUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
