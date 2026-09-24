"""Enrolled speakers: named voiceprints saved across sessions.

Stored in ~/.focus-ear/speakers.json. On startup they seed the clusterer, so
a known lecturer is recognised, and selected automatically, once they've
been matched a few times.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DATA_DIR
from .embeddings import EMBEDDING_DIM, MODEL_SOURCE

log = logging.getLogger(__name__)

PROFILES_PATH = DATA_DIR / "speakers.json"


def load_profiles(path: Path = PROFILES_PATH) -> list[tuple[str, np.ndarray]]:
    """Returns [(name, voiceprint)]. A missing file means no profiles; a damaged one is logged and ignored."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if data.get("model") != MODEL_SOURCE:
            log.warning("%s was made with %r, not %r; ignoring it", path, data.get("model"), MODEL_SOURCE)
            return []
        profiles = []
        for entry in data["speakers"]:
            v = np.asarray(entry["embedding"], dtype=np.float32)
            if v.shape != (EMBEDDING_DIM,):
                raise ValueError(f"{entry.get('name')!r} has a {v.shape} embedding")
            profiles.append((str(entry["name"]), v))
        return profiles
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("couldn't read speaker profiles from %s: %s", path, exc)
        return []


def save_profiles(profiles: list[tuple[str, np.ndarray]], path: Path = PROFILES_PATH) -> None:
    """Replace the file with ``profiles`` atomically, so a crash can't leave it half-written."""
    data = {
        "version": 1,
        "model": MODEL_SOURCE,
        "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "speakers": [{"name": name, "embedding": [round(float(x), 6) for x in v]} for name, v in profiles],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".speakers-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
