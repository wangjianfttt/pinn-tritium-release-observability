#!/usr/bin/env python3
"""Generate a compact SI table for transport, source and balance variables."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


ROWS = [
    {
        "symbol": r"$c_s({\bf x},t)$",
        "unit": r"mol m$^{-3}$ or normalized activity",
        "definition": "gas concentration or normalized outlet signal for species s=HT,HTO",
        "use": "state variable in 1D and 3D transport equations",
    },
    {
        "symbol": r"$y_s^r(t)$",
        "unit": r"same as $c_s$",
        "definition": "flow-weighted outlet concentration at outlet or sector r",
        "use": "measurement fitted by outlet, spatial-outlet and sector-history inverses",
    },
    {
        "symbol": r"$u$, ${\bf u}$",
        "unit": r"m s$^{-1}$",
        "definition": "superficial purge velocity in the reduced model and local velocity field in 3D",
        "use": "nuisance variable constrained by purge setting, pressure drop and RTD",
    },
    {
        "symbol": r"$u_{\rm int}=u/\phi$",
        "unit": r"m s$^{-1}$",
        "definition": "interstitial velocity corresponding to the superficial velocity and porosity",
        "use": "used when comparing residence time and Peclet scaling across porosity changes",
    },
    {
        "symbol": r"$D_z$, ${\bf D}$",
        "unit": r"m$^2$ s$^{-1}$",
        "definition": "effective axial dispersion or local effective dispersion tensor",
        "use": "transport parameter profiled with velocity and porosity",
    },
    {
        "symbol": r"$\phi({\bf x})$",
        "unit": r"dimensionless",
        "definition": "bed void fraction or local effective porosity",
        "use": "sets gas storage volume and residence-time response",
    },
    {
        "symbol": r"$q_j(t)$",
        "unit": r"mol m$^{-3}_{\rm bed}$ or normalized state",
        "definition": "reduced material inventory in grain, trap, open, closed, surface or water-associated state j",
        "use": "internal state in the release model; excluded from outlet-inverse labels",
    },
    {
        "symbol": r"$S_s(q;\alpha)$",
        "unit": r"mol m$^{-3}_{\rm bed}$ s$^{-1}$ or normalized source rate",
        "definition": "species source released from the material states into the gas phase",
        "use": "quantity controlled by open and delayed release multipliers",
    },
    {
        "symbol": r"$S_s/\phi$",
        "unit": r"mol m$^{-3}_{\rm gas}$ s$^{-1}$ or normalized gas-source rate",
        "definition": "gas-volume-equivalent source when the storage term is divided by porosity",
        "use": "checks consistency between bed-volume source rates and gas concentration histories",
    },
    {
        "symbol": r"$K_s c_s$",
        "unit": r"mol m$^{-3}$ s$^{-1}$ or normalized loss rate",
        "definition": "unrecovered gas loss, wall loss or effective sink",
        "use": "nuisance term separated from release multipliers by calibration histories",
    },
    {
        "symbol": r"$C_s$",
        "unit": r"mol m$^{-3}$ s$^{-1}$ or normalized conversion rate",
        "definition": "HT/HTO chemical conversion before detector mixing",
        "use": "constrained by humidity, H$_2$ and paired HT/HTO histories",
    },
    {
        "symbol": r"$h_z^r(t)$",
        "unit": r"s$^{-1}$ after unit-area normalization",
        "definition": "outlet response at r to a unit local source pulse in zone z",
        "use": "model-informed 3D residence-time kernel constrained by pressure/RTD and spatial-outlet measurements",
    },
    {
        "symbol": r"$g_i(t)$",
        "unit": r"mol s$^{-1}$ or normalized generation-rate history",
        "definition": "generated-tritium rate assigned to component i",
        "use": "source-shape input from neutronics, activation or heat-deposition proxy",
    },
    {
        "symbol": r"$r_i(t)$",
        "unit": r"mol s$^{-1}$ or normalized release-rate history",
        "definition": "released-tritium rate from component i",
        "use": "sector and system outlet signal before detector response",
    },
    {
        "symbol": r"$\ell_i(t)$",
        "unit": r"mol s$^{-1}$ or normalized loss-rate history",
        "definition": "unrecovered-loss or permeation-equivalent rate",
        "use": "balance uncertainty term in component and system propagation",
    },
    {
        "symbol": r"$G_i(t),R_i(t),L_i(t)$",
        "unit": r"mol or normalized cumulative amount",
        "definition": "time integrals of generated, released and unrecovered-loss rates",
        "use": r"cumulative balance in $I_i(t)=I_i(0)+G_i(t)-R_i(t)-L_i(t)$",
    },
    {
        "symbol": r"$I_i(t)$",
        "unit": r"mol or normalized retained inventory",
        "definition": "retained inventory in component i",
        "use": "inventory measurement and system balance variable",
    },
]


def esc(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    out_csv = RESULTS / "transport_variable_units_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "unit", "definition", "use"])
        writer.writeheader()
        writer.writerows(ROWS)

    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Units and normalization of the main transport, source and balance variables. The code stores many histories in normalized form; dimensional units show the physical quantity represented before normalization. The table also states the superficial/interstitial velocity relation and the bed-volume/source normalization used in the transport equations.}",
        r"\label{tab:si_transport_variable_units}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.16\textwidth}P{0.19\textwidth}P{0.33\textwidth}Y@{}}",
        r"\toprule",
        r"Variable & Unit or normalization & Definition & Use in the inverse \\",
        r"\midrule",
    ]
    for row in ROWS:
        lines.append(
            f"{row['symbol']} & {row['unit']} & {esc(row['definition'])} & {esc(row['use'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out_tex = MANUSCRIPT / "generated_transport_variable_units_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
