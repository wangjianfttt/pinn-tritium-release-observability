#!/usr/bin/env python3
"""Lightweight check for sparse external-curve claims used in the submission.

The heavy bootstrap/digitization scripts are kept as provenance artifacts, but
the compact submission reproduction should only verify the numbers that appear
in the current manuscript.
"""

from __future__ import annotations

import json
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "external_curve_use_consistency" / "external_curve_use_consistency_summary.json"
FAST_SUMMARY = ROOT / "notes" / "external_curve_use_consistency_fast.json"
BOOT = ROOT / "results" / "real_curve_bootstrap_inverse" / "real_curve_bootstrap_inverse_summary.csv"
IMPACT = ROOT / "results" / "real_curve_constraint_impact" / "real_curve_constraint_impact_summary.csv"
EXT_FIG_SUMMARY = (
    ROOT
    / "results"
    / "external_validation_curve_evidence"
    / "external_validation_curve_evidence_summary.json"
)
REG3D = (
    ROOT
    / "results"
    / "real_curve_regularized_3d_inverse"
    / "real_curve_regularized_3d_inverse_summary.csv"
)
MANUSCRIPT = ROOT / "manuscript" / "manuscript_iop_submission_closure_en.tex"
SI = ROOT / "manuscript" / "supplementary_information_submission_en.tex"


def fail(message: str) -> None:
    print(f"EXTERNAL_CURVE_CHECK=FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        fail(f"missing_or_empty:{path.relative_to(ROOT)}")


def approx(value: float, target: float, tol: float = 5e-3) -> bool:
    return abs(float(value) - target) <= tol


def main() -> None:
    for path in (SUMMARY, BOOT, IMPACT, EXT_FIG_SUMMARY, REG3D, MANUSCRIPT, SI):
        require_file(path)

    summary_source = FAST_SUMMARY if FAST_SUMMARY.exists() and FAST_SUMMARY.stat().st_size > 0 else SUMMARY
    data = json.loads(summary_source.read_text(encoding="utf-8"))["summary"]
    ext_fig = json.loads(EXT_FIG_SUMMARY.read_text(encoding="utf-8"))
    checks = [
        ("ht_tau_median_min", data["ht_tau_median_min"], 158.48147884855138),
        ("hto_tau_median_min", data["hto_tau_median_min"], 123.26126464196768),
        (
            "uncorrected_inventory_half_width_no_prior_pct",
            data["uncorrected_inventory_half_width_no_prior_pct"],
            58.400577696154585,
        ),
        (
            "uncorrected_inventory_half_width_with_prior_pct",
            data["uncorrected_inventory_half_width_with_prior_pct"],
            42.32107440125046,
        ),
        ("double_identifiable_rate_max", data["double_identifiable_rate_max"], 0.0),
        ("time_scale_guided_k16_radius", ext_fig["time_scale_guided_k16_radius"], 0.017604387029857),
        ("random_k16_radius", ext_fig["random_k16_radius"], 0.0326586643137402),
    ]
    for name, value, target in checks:
        if not approx(value, target):
            fail(f"{name}:{value}!={target}")

    if data["n_pass_checks"] != data["n_checks"]:
        fail(f"checks_not_all_pass:{data['n_pass_checks']}/{data['n_checks']}")

    rows = list(csv.DictReader(REG3D.open(encoding="utf-8", newline="")))
    by_case = {(row["prior_mode"], row["protocol_id"]): row for row in rows}
    weak_matched = by_case[("weak_real_prior", "matched_3d_ht_hto_spatial")]
    weak_total = by_case[("weak_real_prior", "negative_total_uniform_basis")]
    weak_wrong = by_case[("weak_real_prior", "negative_matched_transport_wrong_chemistry")]
    reg3d_checks = [
        ("weak_matched_closed_error", weak_matched["closed_error_percent"], 0.5345176909713991),
        ("weak_total_holdout_rmse", weak_total["holdout_rmse_normalized"], 0.06282508141415992),
        ("weak_wrong_chem_closed_error", weak_wrong["closed_error_percent"], 11.570421474500492),
    ]
    for name, value, target in reg3d_checks:
        if not approx(float(value), target):
            fail(f"{name}:{value}!={target}")

    main_text = MANUSCRIPT.read_text(encoding="utf-8")
    si_text = SI.read_text(encoding="utf-8")
    main_required = [
        r"effective release time scales of 158\.5 min for HT and 123\.3 min for HTO",
        r"Bootstrap fits support a single effective time scale",
        r"no bootstrap sample separates two independent time scales for either species",
        r"matched 3D spatial/species closed-pathway error of about 0\.5\\%",
        r"Total-only and mismatched-chemistry inverses remain biased",
        r"Selection by source time scale and 3D field response gives a smaller source/field distance than random selection",
        r"The sparse Park/FEC curve supports a delayed time scale and improves reference-history selection",
        r"Over-weighting the same curve raises the matched 3D closed-pathway error to 18\.57\\%",
    ]
    missing = [pattern for pattern in main_required if not re.search(pattern, main_text)]
    if missing:
        fail("manuscript_missing:" + ";".join(missing))

    si_required = [
        r"158\.5 min for HT and 123\.3 min for HTO",
        r"single-time-scale fits remain within the error limit in 97\.0\\% of HT samples and 100\.0\\% of HTO samples",
        r"two-time-scale identifiable fraction is 0\.0\\% for both species",
        r"contracts a time-scale-dominated source-uncertainty interval by 16\.08 percentage points",
        r"nearest-neighbour source/field distance is 0\.0176 for time-scale-guided selection and 0\.0327 for random selection",
        r"An over-weighted constraint increases the error to 18\.57\\%",
        r"Treating the sparse curve as interpolated dense training information increases the matched 3D closed-pathway error to 28\.02\\%",
    ]
    missing = [pattern for pattern in si_required if not re.search(pattern, si_text)]
    if missing:
        fail("si_missing:" + ";".join(missing))

    print(
        "EXTERNAL_CURVE_CHECK=PASS "
        f"HT_tau={data['ht_tau_median_min']:.1f}min "
        f"HTO_tau={data['hto_tau_median_min']:.1f}min "
        f"weak3D={float(weak_matched['closed_error_percent']):.3f}% "
        f"mismatchedChem={float(weak_wrong['closed_error_percent']):.2f}% "
        f"inventory={data['uncorrected_inventory_half_width_no_prior_pct']:.2f}->"
        f"{data['uncorrected_inventory_half_width_with_prior_pct']:.2f}% "
        f"k16={ext_fig['time_scale_guided_k16_radius']:.4f}/"
        f"{ext_fig['random_k16_radius']:.4f} "
        f"double_id={data['double_identifiable_rate_max']:.1f}"
    )


if __name__ == "__main__":
    main()
