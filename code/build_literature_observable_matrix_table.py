#!/usr/bin/env python3
"""Build a compact SI table of experimental observables reported in literature."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "literature_data" / "observable_matrix.csv"
OUT = ROOT / "manuscript" / "generated_literature_observable_matrix_table.tex"
MAIN_OUT = ROOT / "manuscript" / "generated_literature_observable_summary.tex"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def mat_tex(material: str) -> str:
    return (
        material.replace("Li2TiO3", r"Li$_2$TiO$_3$")
        .replace("Li4SiO4", r"Li$_4$SiO$_4$")
    )


def chem_escape(text: str) -> str:
    text = tex_escape(text)
    return (
        text.replace("Li2TiO3", r"Li$_2$TiO$_3$")
        .replace("Li4SiO4", r"Li$_4$SiO$_4$")
        .replace("grain/trap/open/closed", "grain, trap, open- and closed-pathway")
        .replace("H2O", r"H$_2$O")
        .replace("D2O", r"D$_2$O")
        .replace("H2", r"H$_2$")
        .replace("D2", r"D$_2$")
    )


def ref_tex(reference: str) -> str:
    ref = reference.strip()
    mapping = {
        "main ref. 18": "[18]",
        "main ref. 20": "[20]",
        "main ref. 10": "[10]",
        "main ref. 11": "[11]",
        "main ref. 9": "[9]",
        "main ref. 12": "[12]",
        "main ref. 15": "[15]",
        "main refs. 4-5": "[4,5]",
        "main refs. 6-8": "[6--8]",
        "experimental system records": "(system records)",
    }
    if ref in mapping:
        return mapping[ref]
    if ref.startswith("doi:"):
        doi = tex_escape(ref[4:])
        return r"\href{https://doi.org/" + doi + r"}{DOI}"
    return tex_escape(ref)


CATEGORY_TERMS = {
    "release": [
        "outlet",
        "released activity",
        "release behavior",
        "release histories",
        "trapped",
        "recovery histories",
        "tds",
        "isothermal",
    ],
    "HT/HTO or water": [
        "ht/hto",
        "water-bubbler",
        "bubbler",
        "tritiated-water",
        "hdo",
        "d2o",
        "h2o",
        "hto",
        "ht release",
    ],
    "flow/chem.": [
        "purge",
        "sweep gas",
        "gas composition",
        "h2",
        "hydrogen",
        "water vapor",
        "humidity",
    ],
    "T/TDS": ["temperature", "heating", "thermal", "tds", "isothermal"],
    "material": [
        "microstructure",
        "grain",
        "porosity",
        "density",
        "sem",
        "bet",
        "mercury intrusion",
        "defect",
    ],
    "recovery": ["recovery", "extraction", "trapping", "measurement system", "pipeline", "bred-tritium"],
}


def families_for(record: str) -> str:
    low = record.lower()
    families = [
        family for family, terms in CATEGORY_TERMS.items()
        if any(term in low for term in terms)
    ]
    return ", ".join(families) if families else "reported record"


def main() -> None:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    n_rows = len(rows)
    n_li2 = sum(1 for row in rows if "Li2TiO3" in row["material"])
    n_li4 = sum(1 for row in rows if "Li4SiO4" in row["material"])
    record_text = " ".join(row["reported_records"].lower() for row in rows)
    family_counts = {}
    for category, terms in CATEGORY_TERMS.items():
        count = sum(
            any(term in row["reported_records"].lower() for term in terms)
            for row in rows
        )
        family_counts[category] = count
    if sum(count > 0 for count in family_counts.values()) < 6:
        raise ValueError(f"observable matrix lost expected record families: {family_counts}")

    lines = [
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{longtable}{@{}P{0.17\textwidth}P{0.18\textwidth}P{0.33\textwidth}P{0.24\textwidth}@{}}",
        r"\caption{Experimental records from lithium-ceramic breeder-release literature that motivate the measurement families used in this work. The family column gives the reproducible classification used for the measurement-family summary in the Introduction.}\label{tab:si_literature_observables}\\",
        r"\toprule",
        r"Source & Material and measurement family & Reported records & Use in this work \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Source & Material and measurement family & Reported records & Use in this work \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in rows:
        source = tex_escape(row["source_label"]) + " " + ref_tex(row["main_reference"])
        material = mat_tex(row["material"])
        family = tex_escape(families_for(row["reported_records"]))
        records = chem_escape(row["reported_records"])
        role = (
            chem_escape(row["measurement_role_in_this_paper"])
            .replace("stress test", "sensitivity calculation")
            .replace("material-extension tests", "material-extension calculations")
            .replace("coupled observables", "coupled records")
        )
        lines.append(
            rf"{source} & {material}; {family} & {records} & {role} \\"
        )
    lines.extend([r"\end{longtable}", r"}", ""])
    OUT.write_text("\n".join(lines), encoding="utf-8")

    main_text = (
        "Existing lithium-ceramic release studies report several records "
        "around the purge-gas history. Supplementary Table S1 groups "
        f"{n_rows} literature record rows, including {n_li2} Li$_2$TiO$_3$ and "
        f"{n_li4} Li$_4$SiO$_4$ rows. These grouped records include dynamic "
        f"release or trapped-release histories in {family_counts['release']} rows, "
        f"HT/HTO or water-collection records in "
        f"{family_counts['HT/HTO or water']}, purge-flow or gas-chemistry records in "
        f"{family_counts['flow/chem.']}, temperature or TDS programs in "
        f"{family_counts['T/TDS']}, material-state records in "
        f"{family_counts['material']}, and recovery or extraction-system records in "
        f"{family_counts['recovery']}. The release curve gives the purge-gas time "
        "history. The companion records constrain species chemistry, residence time, "
        "thermal state, material state and tritium balance. The inverse calculation "
        "uses these measurement families as physical inputs for source separation. Computed "
        "PINN histories extend source amplitude, chemistry and temperature variation "
        "inside the target-bed response family calibrated by flow, species and "
        "selected release measurements. Withheld cases change the release mechanism, 3D field, "
        "chemistry field, thermal field or balance history."
    )
    MAIN_OUT.write_text(main_text + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"wrote {MAIN_OUT}")


if __name__ == "__main__":
    main()
