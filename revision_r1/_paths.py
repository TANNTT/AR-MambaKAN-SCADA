"""Resolve local data roots from environment variables (no machine defaults)."""

from __future__ import annotations

import os
from pathlib import Path


def require_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable {name}. "
            "See revision_r1/README.md for the list of variables."
        )
    return Path(value)


def env_or_none(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None
