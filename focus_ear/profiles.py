"""Enrolled speaker profiles (phase 3, not implemented yet).

Plan: 'n' in the TUI names the selected speaker and saves its centroid to
~/.focus-ear/speakers.json ({name: [192 floats]}). On startup the saved
profiles seed the clusterer, so when the lecturer's voice matches one they
are recognised and auto-selected.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DATA_DIR

PROFILES_PATH = DATA_DIR / "speakers.json"


def load_profiles(path: Path = PROFILES_PATH) -> dict[str, np.ndarray]:
    raise NotImplementedError("enrollment arrives in phase 3")


def save_profiles(profiles: dict[str, np.ndarray], path: Path = PROFILES_PATH) -> None:
    raise NotImplementedError("enrollment arrives in phase 3")
