#!/usr/bin/env python3
"""Paper-facing entry point for Li2TiO3 outlet-only profile sensitivity.

The implementation lives in the older compatibility script so that existing
result paths remain stable.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("build_li2_outlet_identifiability_nuisance.py")),
        run_name="__main__",
    )
