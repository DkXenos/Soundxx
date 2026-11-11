"""Sessions: one folder per lecture under ~/.focus-ear/sessions/.

    2026-11-12_0900_machine-learning/
      session.json      id, title, course, start/end, status, duration, devices, models, config, pauses, speakers
      audio.wav         the raw mic (RawRecorder), 16 kHz mono
      live.jsonl/.md    the live transcript exactly as produced (never edited: evaluation input)
      final.json        the after-class machine output (never edited: evaluation input)
      transcript.json   what the app shows and edits (a copy of final, or of live before processing)
      transcript.md     rendered from transcript.json
      notes.json/.md    summary, key points, chapters
      speakers.npz      this session's voice clusters
      engine.log        this session's log and health lines

``status`` goes recording -> recorded -> processing -> ready. A folder still
marked "recording" whose engine process is gone is an interrupted session:
its audio and live transcript are valid up to the last second or so
(both are fsynced as they're written), and recover() closes it.

Every write of session.json is atomic (write a temp file, fsync, rename).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

from .config import SESSIONS_DIR
from .settings import write_json_atomic

log = logging.getLogger(__name__)

STATUSES = ("recording", "recorded", "processing", "ready")
SESSION_FILE = "session.json"
AUDIO_FILE = "audio.wav"
LIVE_STEM = "live"          # live.md, live.jsonl
VERSION = 1

_ID = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}[A-Za-z0-9_-]*$")
_LEGACY = re.compile(r"^(?P<stem>[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}[A-Za-z0-9_-]*)\.(?P<ext>md|jsonl|wav)$")


class SessionError(RuntimeError):
    pass


def slugify(text: str, max_len: int = 40) -> str:
    """'Machine Learning (Kelas B)' -> 'machine-learning-kelas-b'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].rstrip("-")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except (OSError, ValueError):
        return False
    return True


def wav_seconds(path: Path) -> float:
    """Length of a WAV, also one whose header a crash left a little short (the data is what counts)."""
    try:
        with wave.open(str(path), "rb") as w:
            rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
            frames = w.getnframes()
        data_bytes = max(0, path.stat().st_size - 44)
        return max(frames, data_bytes // (width * channels)) / rate
    except (OSError, wave.Error, EOFError, ZeroDivisionError):
        return 0.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class SessionStore:
    """Creates, lists and updates session folders. Safe to use from several threads of one process."""

    def __init__(self, root: Path = SESSIONS_DIR):
        self.root = Path(root)
        self._lock = threading.RLock()

    # Paths -----------------------------------------------------------------
    def path(self, sid: str) -> Path:
        if not _ID.match(sid or ""):
            raise SessionError(f"not a session id: {sid!r}")
        return self.root / sid

    def exists(self, sid: str) -> bool:
        try:
            return (self.path(sid) / SESSION_FILE).exists()
        except SessionError:
            return False

    # Create / read / write -------------------------------------------------
    def create(self, title: str = "", course: str = "", started: datetime | None = None,
               status: str = "recording", **meta) -> dict:
        started = started or datetime.now()
        base = started.strftime("%Y-%m-%d_%H%M")
        slug = slugify(title or course)
        if slug:
            base += f"_{slug}"
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            n = 1
            while True:
                sid = base if n == 1 else f"{base}_{n}"
                try:
                    (self.root / sid).mkdir()
                    break
                except FileExistsError:
                    n += 1
            data = {
                "version": VERSION, "id": sid, "title": title.strip(), "course": course.strip(),
                "started": started.astimezone().isoformat(timespec="seconds"), "ended": None,
                "status": status, "duration_s": 0.0, "devices": {}, "models": {}, "config": {},
                "pauses": [], "speakers": [], "engine_pid": None, "interrupted": False,
                "notes_status": None, "processing": None,
            }
            data.update(meta)
            write_json_atomic(self.path(sid) / SESSION_FILE, data)
            return data

    def load(self, sid: str) -> dict:
        p = self.path(sid) / SESSION_FILE
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise SessionError(f"no such session: {sid}") from None
        except ValueError as exc:
            raise SessionError(f"{p} is damaged: {exc}") from None
        data["id"] = sid
        return data

    def update(self, sid: str, **changes) -> dict:
        with self._lock:
            data = self.load(sid)
            if "status" in changes and changes["status"] not in STATUSES:
                raise SessionError(f"status must be one of {STATUSES}")
            data.update(changes)
            write_json_atomic(self.path(sid) / SESSION_FILE, data)
            return data

    def edit(self, sid: str, fn) -> dict:
        """Read-modify-write under the store's lock: ``fn(data)`` changes data in place."""
        with self._lock:
            data = self.load(sid)
            fn(data)
            write_json_atomic(self.path(sid) / SESSION_FILE, data)
            return data

    def delete(self, sid: str) -> None:
        """Move a session to ~/.Trash (recoverable), never rm."""
        src = self.path(sid)
        trash = Path.home() / ".Trash"
        dest = trash / f"focus-ear {sid}"
        n = 2
        while dest.exists():
            dest = trash / f"focus-ear {sid} {n}"
            n += 1
        shutil.move(str(src), str(dest))

    # Listing ---------------------------------------------------------------
    def list(self) -> list[dict]:
        """Every session, newest first, each with a few derived fields for the session list."""
        if not self.root.exists():
            return []
        out = []
        for d in self.root.iterdir():
            if not d.is_dir() or not (d / SESSION_FILE).exists():
                continue
            try:
                data = self.load(d.name)
            except SessionError as exc:
                log.warning("%s", exc)
                continue
            out.append(self.describe(data))
        return sorted(out, key=lambda x: x.get("started") or "", reverse=True)

    def describe(self, data: dict) -> dict:
        d = self.path(data["id"])
        data = dict(data)
        data["has_audio"] = (d / AUDIO_FILE).exists()
        data["has_final"] = (d / "final.json").exists()
        data["has_notes"] = (d / "notes.json").exists()
        data["needs_recovery"] = self.is_interrupted(data)
        return data

    def is_interrupted(self, data: dict) -> bool:
        return data.get("status") == "recording" and not pid_alive(data.get("engine_pid"))

    def interrupted(self) -> list[str]:
        return [s["id"] for s in self.list() if s["needs_recovery"]]

    def recover(self, sid: str) -> dict:
        """Close a session whose engine died: it keeps what was written before."""
        with self._lock:
            data = self.load(sid)
            if not self.is_interrupted(data):
                return data
            d = self.path(sid)
            audio = d / AUDIO_FILE
            if audio.exists():
                repair_wav_header(audio)
            ended = datetime.fromtimestamp(max((p.stat().st_mtime for p in d.iterdir()), default=0)).astimezone()
            data.update(status="recorded" if audio.exists() else "ready", interrupted=True, engine_pid=None,
                        ended=ended.isoformat(timespec="seconds"),
                        duration_s=round(wav_seconds(audio), 1) if audio.exists() else data.get("duration_s", 0.0))
            write_json_atomic(d / SESSION_FILE, data)
            log.info("recovered interrupted session %s (%.0f s of audio)", sid, data["duration_s"])
            return data

    # The flat files of earlier versions ----------------------------------------
    def import_flat(self) -> list[str]:
        """Move ~/.focus-ear/sessions/<stem>.{md,jsonl,wav} into session folders (once; idempotent)."""
        if not self.root.exists():
            return []
        groups: dict[str, dict[str, Path]] = {}
        for p in self.root.iterdir():
            m = _LEGACY.match(p.name)
            if p.is_file() and m:
                groups.setdefault(m["stem"], {})[m["ext"]] = p
        imported = []
        for stem, files in sorted(groups.items()):
            try:
                started = datetime.strptime(stem[:15], "%Y-%m-%d_%H%M")
            except ValueError:
                continue
            title = stem[16:].replace("_", " ").strip() if len(stem) > 15 else ""
            data = self.create(title=title, started=started, status="ready", imported_from=stem)
            d = self.path(data["id"])
            names = {"md": f"{LIVE_STEM}.md", "jsonl": f"{LIVE_STEM}.jsonl", "wav": AUDIO_FILE}
            for ext, src in files.items():
                shutil.move(str(src), str(d / names[ext]))
            changes = {"status": "recorded" if "wav" in files else "ready", "ended": data["started"]}
            if "wav" in files:
                changes["duration_s"] = round(wav_seconds(d / AUDIO_FILE), 1)
            elif "jsonl" in files:
                changes["duration_s"] = _jsonl_end(d / f"{LIVE_STEM}.jsonl")
            self.update(data["id"], **changes)
            imported.append(data["id"])
            log.info("imported %s into %s", stem, d)
        return imported


def _jsonl_end(path: Path) -> float:
    end = 0.0
    for rec in read_jsonl(path):
        end = max(end, float(rec.get("end") or 0.0))
    return round(end, 1)


def read_jsonl(path: Path) -> list[dict]:
    """Every complete record; a last line cut off by a crash is skipped."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    log.debug("skipping a damaged line in %s", path)
    except FileNotFoundError:
        pass
    return out


def repair_wav_header(path: Path) -> None:
    """Make the header's lengths match the data a crash left behind (RawRecorder writes 16-bit PCM)."""
    try:
        with wave.open(str(path), "rb") as w:
            params = w.getparams()
            declared = w.getnframes()
        size = path.stat().st_size
        frames = (size - 44) // (params.sampwidth * params.nchannels)
        if frames <= declared or params.comptype != "NONE":
            return
        with open(path, "r+b") as f:
            data_bytes = frames * params.sampwidth * params.nchannels
            f.seek(4)
            f.write((36 + data_bytes).to_bytes(4, "little"))
            f.seek(40)
            f.write(data_bytes.to_bytes(4, "little"))
            f.truncate(44 + data_bytes)
    except (OSError, wave.Error, EOFError) as exc:
        log.warning("couldn't check the header of %s: %s", path, exc)
