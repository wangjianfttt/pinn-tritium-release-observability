#!/usr/bin/env python3
"""Build manuscript text for the measurement-necessity calculation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_OUT = ROOT / "manuscript" / "generated_observable_necessity_summary.tex"
SI_OUT = ROOT / "manuscript" / "generated_observable_necessity_si.tex"
NECESSITY = ROOT / "results" / "observable_necessity_leave_one_out" / "observable_necessity_summary.json"
NULLSPACE = ROOT / "results" / "observable_nullspace_closure_synthesis" / "observable_nullspace_closure_summary.json"
DESIGNS = ROOT / "results" / "observable_necessity_leave_one_out" / "observable_necessity_designs.csv"
EFFECTS = ROOT / "results" / "observable_necessity_leave_one_out" / "observable_necessity_effects.csv"
CHECKS = ROOT / "results" / "observable_necessity_leave_one_out" / "observable_necessity_checks.csv"


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    necessity = json.loads(NECESSITY.read_text(encoding="utf-8"))
    nullspace = json.loads(NULLSPACE.read_text(encoding="utf-8"))

    full_rank = int(necessity["full_rank_without_regularization"])
    full_null = int(necessity["full_null_dimension"])
    outlet_null = int(nullspace["outlet_only_null_dimension"])
    full_min_sv = float(nullspace["full_multimodal_min_positive_singular_value"])
    full_cond = float(nullspace["full_multimodal_condition_number"])
    n_families = int(necessity["n_observable_families"])
    n_reopened = int(necessity["n_gate_reopening_effects"])

    spatial = float(nullspace["spatial_zone_source_increase_factor"])
    hthto = float(nullspace["hthto_thermal_speciation_increase_factor"])
    pressure = float(nullspace["pressure_transport_increase_factor"])
    inventory = float(nullspace["inventory_tes_recovery_increase_factor"])

    main_text = (
        "\\par\n"
        "The measurement-set calculation gives the same message in algebraic form. "
        f"Outlet-only histories leave {outlet_null} unresolved parameter combinations. "
        f"Using all {n_families} measurement types gives rank {full_rank} and no unresolved "
        "linear direction in the reduced calculation. "
        f"The smallest positive singular value is {full_min_sv:.2f}, and the corresponding "
        f"condition number is {full_cond:.1f}. "
        "Removing one measurement type at a time shows which record carries each part of the source information. "
        f"Removing spatially resolved outlets increases spatial source uncertainty by {pct(spatial)}-fold. "
        f"Removing HT/HTO speciation increases thermal/species uncertainty by {pct(hthto)}-fold. "
        f"Removing pressure information increases transport uncertainty by {pressure:.2f}-fold. "
        f"Removing retained-inventory or TES information increases recovery uncertainty by {pct(inventory)}-fold. "
        f"{n_reopened} of the seven removals leave a weak source direction above 5\\% relative source error. "
        "The retained measurement types carry separate spatial, species, transport and tritium-balance information.\n"
        "\\par\n"
    )

    si_text = (
        "\\par\n"
        "The measurement-set calculation uses the same reduced linear system as the component "
        "and system tritium-balance propagation. "
        f"The outlet-only stage has {outlet_null} unresolved linear combinations. "
        f"The full {n_families}-type system has 17 measurement rows, rank {full_rank}, "
        f"no unresolved linear combination, minimum positive singular value {full_min_sv:.2f} and "
        f"positive-direction condition number {full_cond:.1f}. "
        "Removing spatial outlets, HT/HTO speciation, pressure information and "
        "retained-inventory/TES information increases the dominant uncertainty by "
        f"{pct(spatial)}-, {pct(hthto)}-, {pressure:.2f}- and {pct(inventory)}-fold, respectively. "
        f"The source files are \\path{{{DESIGNS.relative_to(ROOT)}}}, "
        f"\\path{{{EFFECTS.relative_to(ROOT)}}}, "
        f"\\path{{{NECESSITY.relative_to(ROOT)}}}, "
        f"\\path{{{NULLSPACE.relative_to(ROOT)}}} and "
        f"\\path{{{CHECKS.relative_to(ROOT)}}}.\n"
        "\\par\n"
    )

    MAIN_OUT.write_text(main_text, encoding="utf-8")
    SI_OUT.write_text(si_text, encoding="utf-8")
    print(f"wrote {MAIN_OUT}")
    print(f"wrote {SI_OUT}")


if __name__ == "__main__":
    main()
