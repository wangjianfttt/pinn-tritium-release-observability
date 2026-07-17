#!/usr/bin/env python3
"""Rebuild the 288-component dynamic tritium-balance summary.

This reader-facing entry point keeps the manuscript and Supplement terminology
aligned with tritium balance. It delegates to the existing implementation so
the generated CSV/JSON artifacts and numerical checks remain unchanged.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    runpy.run_path(
        str(ROOT / "code" / "build_whole_reactor_dynamic_accountancy_summary.py"),
        run_name="__main__",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
