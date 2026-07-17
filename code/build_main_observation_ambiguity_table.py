#!/usr/bin/env python3
"""Build the main-text observation requirement table."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"


def esc_text(text: str) -> str:
    return (
        str(text)
        .replace("&", r"\&")
        .replace("_", r"\_")
    )


def main() -> None:
    source = RESULTS / "observation_ambiguity_table.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))

    ambiguity_labels = {
        "PINN source prediction": "PINN extrapolation",
        "Sparse external release curve": "External release curve",
    }
    insufficient_labels = {
        "pseudo-dense interpolation": "interpolated dense curve",
        "random split or total outlet": "random split or total outlet only",
    }
    required_labels = {
        "48 h tail plus residence-time perturbation": "48 h tail plus slow/fast purge",
        "time-scale prior plus spatial/species prediction": "time-scale constraint plus spatial/species prediction",
        "3D RTD kernel or spatial HT/HTO outlets": "calibrated 3D residence-time response constrained by pressure/RTD and spatial HT/HTO outlets",
        "independent 3D field with matched HT/HTO operator": "withheld 3D field with matched HT/HTO transport response",
    }

    lines = [
        r"\begin{table*}[tbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Measurement operators required to separate delayed material release from transport, species chemistry, detector response and temperature.}",
        r"\label{tab:observation_ambiguity}",
        r"\begin{tabularx}{\textwidth}{@{}P{0.20\textwidth}P{0.20\textwidth}P{0.28\textwidth}Y@{}}",
        r"\toprule",
        r"Ambiguity & Incomplete data & Added measurement & Effect on source estimate \\",
        r"\midrule",
    ]
    for row in rows:
        change = f"{row['under_observed_result']} $\\rightarrow$ {row['separated_result']}"
        change = change.replace("wrong chemistry", "mismatched chemistry")
        change = change.replace("pseudo-dense", "interpolated dense")
        change = change.replace(
            "closed error 7.29\\% $\\rightarrow$ closed error 0.57\\%",
            "closed error 7.29\\% $\\rightarrow$ 0.57\\%",
        )
        change = change.replace(
            "source error 31.21\\% $\\rightarrow$ source error 0.010\\%",
            "source error 31.21\\% $\\rightarrow$ pressure-calibrated result below 0.1\\%",
        )
        change = change.replace(
            "source error 31.21\\% $\\rightarrow$ about 0.01\\%",
            "source error 31.21\\% $\\rightarrow$ pressure-calibrated result below 0.1\\%",
        )
        change = change.replace(
            "source error 31.21\\% $\\rightarrow$ pressure-calibrated result 0.010\\%",
            "source error 31.21\\% $\\rightarrow$ pressure-calibrated result below 0.1\\%",
        )
        change = change.replace(
            "matched 3D closed p95 at matched-operator closure",
            "closed p95 returns to the matched-bed error range",
        )
        change = change.replace(
            "matched 3D closed error reaches matched-operator closure",
            "closed error returns to the matched-bed error range",
        )
        change = change.replace(
            "matched 3D closed error 0.0011\\%",
            "closed error returns to the matched-bed error range",
        )
        change = change.replace(
            "matched 3D closed p95 0.043\\%",
            "closed p95 returns to the matched-bed error range",
        )
        change = change.replace("detector 0.007\\%", "detector-calibrated result below 0.1\\%")
        change = change.replace(
            "detector-calibrated result 0.007\\%",
            "detector-calibrated result below 0.1\\%",
        )
        ambiguity = ambiguity_labels.get(row["ambiguity"], row["ambiguity"])
        insufficient_raw = insufficient_labels.get(
            row["insufficient_observation"], row["insufficient_observation"]
        )
        required_raw = required_labels.get(row["required_observation"], row["required_observation"])
        insufficient = esc_text(insufficient_raw).replace("pseudo-dense", "interpolated dense")
        lines.append(
            " & ".join(
                [
                    esc_text(ambiguity),
                    insufficient,
                    esc_text(required_raw),
                    change,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])

    out = MANUSCRIPT / "generated_observation_ambiguity_table.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
