#!/usr/bin/env python3
"""Protect accepted manuscript figures from accidental regeneration."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def require_locked_figure_regeneration_allowed(*outputs: Path) -> None:
    """Stop a figure script before overwriting accepted main-figure files."""
    existing = [path for path in outputs if path.exists()]
    if not existing:
        return
    if (
        os.environ.get("ALLOW_LOCKED_FIGURE_REGEN") == "1"
        and os.environ.get("CONFIRM_LOCKED_FIGURE_REGEN") == "YES"
    ):
        return
    rel = "\n  ".join(str(path) for path in existing)
    sys.exit(
        "Refusing to overwrite accepted manuscript figure(s):\n"
        f"  {rel}\n"
        "Set both ALLOW_LOCKED_FIGURE_REGEN=1 and "
        "CONFIRM_LOCKED_FIGURE_REGEN=YES only when the user explicitly asks "
        "to redraw locked figures. Routine manuscript builds must not "
        "overwrite the accepted figure style."
    )
