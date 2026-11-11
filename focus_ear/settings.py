"""Choices remembered between sessions: ~/.focus-ear/settings.json.

The output device used to be hard-coded ("Mewo"). Now the last device you
chose, in the app or with --output-device, is remembered here; "Mewo" is
only the first-run guess. A missing or damaged file means the defaults.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from .config import SESSIONS_DIR

log = logging.getLogger(__name__)

# Next to the sessions: ~/.focus-ear/settings.json, or beside FOCUS_EAR_SESSIONS (tests, demos).
SETTINGS_PATH = SESSIONS_DIR.parent / "settings.json"

DEFAULTS: dict = {
    "output_device": "Mewo",    # first-run guess: the Galaxy Buds' name
    "mode": "listen",           # "listen" or "full" (focus on a speaker)
    "courses": [],              # recently used course names, newest first
    "auto_process": True,       # run the after-class pass when a recording stops
    "auto_notes": True,         # then write lecture notes (transcript text to Claude)
    "notes_model": None,        # claude --model for notes; None = the CLI's default
}


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    settings = dict(DEFAULTS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            settings.update({k: v for k, v in data.items() if k in DEFAULTS})
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        log.warning("couldn't read %s (%s); using the defaults", path, exc)
    return settings


def write_json_atomic(path: Path, data, indent: int | None = 1) -> None:
    """Replace ``path`` with ``data`` as JSON; a crash leaves the old file, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def update_settings(path: Path = SETTINGS_PATH, **changes) -> dict:
    """Merge ``changes`` into the file and return the result. ``course`` moves a course to the front."""
    settings = load_settings(path)
    course = changes.pop("course", None)
    if course:
        settings["courses"] = [course] + [c for c in settings["courses"] if c != course][:19]
    settings.update({k: v for k, v in changes.items() if k in DEFAULTS})
    write_json_atomic(path, settings)
    return settings
