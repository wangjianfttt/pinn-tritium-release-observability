#!/usr/bin/env python3
"""Recompute the whole-reactor dynamic accountancy numbers used in the paper.

The script does not create new reactor-specific fields. It reads the generated
288-component surrogate readout tables and rewrites a compact summary used by
the manuscript.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "whole_reactor_dynamic_tritium_observability"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_metric(rows: list[dict[str, str]], feature_family: str, readout: str) -> dict[str, float]:
    for row in rows:
        if row["feature_family"] == feature_family and row["readout"] == readout:
            metrics: dict[str, float] = {}
            for key, value in row.items():
                if key in {"feature_family", "readout"}:
                    continue
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
            return metrics
    raise KeyError(f"missing {feature_family}/{readout}")


def main() -> int:
    table = load_rows(RESULTS / "whole_reactor_dynamic_tritium_observability.csv")
    predictions = load_rows(RESULTS / "whole_reactor_dynamic_tritium_predictions.csv")
    histories = load_rows(RESULTS / "whole_reactor_dynamic_histories.csv")

    aggregate = find_metric(table, "aggregate_total", "linear_ridge")
    aggregate_dual = find_metric(table, "aggregate_dual_species", "linear_ridge")
    sector = find_metric(table, "sector_release_tail", "linear_ridge")
    full = find_metric(table, "sector_full_accountancy", "random_feature_neural_worst_seed")

    n_test = len({row["scenario_id"] for row in predictions})
    n_history_scenarios = len({row["scenario_id"] for row in histories})

    summary = {
        "source_table": str((RESULTS / "whole_reactor_dynamic_tritium_observability.csv").relative_to(ROOT)),
        "prediction_table": str((RESULTS / "whole_reactor_dynamic_tritium_predictions.csv").relative_to(ROOT)),
        "history_table": str((RESULTS / "whole_reactor_dynamic_histories.csv").relative_to(ROOT)),
        "n_test_scenarios": n_test,
        "n_history_scenarios": n_history_scenarios,
        "aggregate_total_tail_l1_p95_pct": aggregate["tail_l1_p95_pct"],
        "aggregate_total_top_tail_accuracy": aggregate["top_tail_sector_accuracy"],
        "aggregate_dual_species_tail_l1_p95_pct": aggregate_dual["tail_l1_p95_pct"],
        "sector_release_tail_l1_p95_pct": sector["tail_l1_p95_pct"],
        "sector_release_top_tail_accuracy": sector["top_tail_sector_accuracy"],
        "full_accountancy_neural_tail_l1_p95_pct": full["tail_l1_p95_pct"],
        "full_accountancy_neural_risk_l1_p95_pct": full["risk_l1_p95_pct"],
        "manuscript_numbers": {
            "aggregate_tail_p95_pct": round(aggregate["tail_l1_p95_pct"], 2),
            "aggregate_top_tail_accuracy_pct": round(100.0 * aggregate["top_tail_sector_accuracy"], 1),
            "sector_tail_p95_pct": round(sector["tail_l1_p95_pct"], 2),
            "sector_top_tail_accuracy_pct": round(100.0 * sector["top_tail_sector_accuracy"], 1),
            "full_neural_tail_p95_pct": round(full["tail_l1_p95_pct"], 2),
            "full_neural_risk_p95_pct": round(full["risk_l1_p95_pct"], 2),
        },
        "claim": (
            "Aggregate whole-reactor release histories do not localize slow-tail sectors; "
            "sector release-tail histories and accountancy observables recover local delayed-release risk."
        ),
    }

    checks = [
        {
            "check_id": "aggregate_total_fails_localization",
            "status": "PASS" if aggregate["top_tail_sector_accuracy"] < 0.25 else "FAIL",
            "detail": f"top_tail_accuracy={aggregate['top_tail_sector_accuracy']:.3f}",
        },
        {
            "check_id": "sector_release_tail_repairs_localization",
            "status": "PASS" if sector["top_tail_sector_accuracy"] > 0.90 else "FAIL",
            "detail": f"top_tail_accuracy={sector['top_tail_sector_accuracy']:.3f}",
        },
        {
            "check_id": "full_accountancy_neural_tail_risk_below_1pct",
            "status": "PASS"
            if full["tail_l1_p95_pct"] < 1.0 and full["risk_l1_p95_pct"] < 1.0
            else "FAIL",
            "detail": f"tail={full['tail_l1_p95_pct']:.3f} risk={full['risk_l1_p95_pct']:.3f}",
        },
    ]

    out_json = RESULTS / "whole_reactor_dynamic_accountancy_main_summary.json"
    out_csv = RESULTS / "whole_reactor_dynamic_accountancy_main_checks.csv"
    out_json.write_text(json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    n_fail = sum(row["status"] == "FAIL" for row in checks)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    print(
        "WHOLE_REACTOR_DYNAMIC="
        f"aggregate_tail={summary['manuscript_numbers']['aggregate_tail_p95_pct']:.2f}% "
        f"aggregate_acc={summary['manuscript_numbers']['aggregate_top_tail_accuracy_pct']:.1f}% "
        f"sector_tail={summary['manuscript_numbers']['sector_tail_p95_pct']:.2f}% "
        f"sector_acc={summary['manuscript_numbers']['sector_top_tail_accuracy_pct']:.1f}% "
        f"full_tail={summary['manuscript_numbers']['full_neural_tail_p95_pct']:.2f}% "
        f"full_risk={summary['manuscript_numbers']['full_neural_risk_p95_pct']:.2f}%"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
