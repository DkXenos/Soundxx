"""The app's side of the engine and job subprocesses.

EngineClient runs focus_ear.headless (one per live session, plus a standby
while the Start screen is open so Start is immediate) and turns its JSON
lines into state the server pushes over the WebSocket. JobRunner runs the
after-class pass (python -m focus_ear.post) and notes one at a time, never
while an engine exists: the live chain and Whisper large don't fit in 8 GB
together. Starting a recording stops a running pass; it carries on from
its last finished step afterwards.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from ..sessions import SessionStore, now_iso
from ..settings import load_settings

log = logging.getLogger("focus_ear.app")

ROOT = Path(__file__).resolve().parents[2]


class EngineClient:
    def __init__(self, store: SessionStore, publish: Callable[[dict], None], on_exit: Callable[[str | None], None]):
        self.store = store
        self.publish = publish
        self.on_exit = on_exit
        self.proc: subprocess.Popen | None = None
        self.ready: dict | None = None
        self.state: dict | None = None
        self.loading: str | None = None
        self.fatal: str | None = None
        self.session_id: str | None = None
        self.lines: list[dict] = []
        self.events: deque[dict] = deque(maxlen=50)
        self._lock = threading.Lock()
        self._stopped = threading.Event()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def recording(self) -> bool:
        return self.running and self.session_id is not None

    def summary(self) -> dict:
        return {"running": self.running, "ready": self.ready is not None, "loading": self.loading,
                "fatal": self.fatal, "session": self.session_id, "state": self.state,
                "pid": self.proc.pid if self.proc else None}

    def spawn(self, output: str | None, mode: str) -> None:
        with self._lock:
            if self.running:
                return
            self.ready = self.state = self.fatal = None
            self.session_id = None
            self.lines = []
            self.loading = "starting"
            self._stopped.clear()
            cmd = [sys.executable, "-m", "focus_ear.headless", "--mode", mode]
            if output:
                cmd += ["--output", output]
            env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
            threading.Thread(target=self._read, args=(self.proc,), name="engine-reader", daemon=True).start()
            log.info("engine started (pid %d)", self.proc.pid)

    def send(self, **cmd) -> bool:
        p = self.proc
        if p is None or p.poll() is not None:
            return False
        try:
            p.stdin.write(json.dumps(cmd) + "\n")
            p.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def start_session(self, sid: str) -> None:
        self.session_id = sid
        self.lines = []
        self.send(cmd="start", session_dir=str(self.store.path(sid)))
        # No idle sleep while this engine records; ends by itself when the engine exits. Only -i: closing
        # the lid still sleeps the Mac (the built-in mic is off with the lid closed anyway).
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(self.proc.pid)])
        except OSError:
            log.warning("caffeinate isn't available; the Mac may sleep during a long lecture")

    def stop(self, timeout: float = 60.0) -> None:
        """A clean stop: the transcript is finished and the session closed. Blocks until the process exits."""
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            self.send(cmd="stop")
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                log.warning("engine didn't stop in %.0f s; terminating it", timeout)
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
        self._stopped.wait(5)

    def release(self) -> None:
        """Close a standby engine (no session): frees its memory for after-class work."""
        if self.running and self.session_id is None:
            self.stop(timeout=15)

    def _read(self, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            t = msg.get("type")
            if t == "state":
                msg.pop("type")
                self.state = msg
                self.publish({"type": "state", "state": msg})
                continue
            if t == "loading":
                self.loading = msg["what"]
            elif t == "ready":
                self.ready, self.loading = msg, None
            elif t == "fatal":
                self.fatal, self.loading = msg["message"], None
            elif t == "line":
                self.lines.append(msg)
                del self.lines[:-2000]
            elif t == "event":
                msg["at"] = time.time()
                self.events.append(msg)
            self.publish(msg)
        code = proc.wait()
        sid = self.session_id
        log.info("engine exited (code %s)", code)
        if sid is not None:
            try:
                data = self.store.load(sid)
                if data.get("status") == "recording":  # it died: keep what it wrote
                    self.store.recover(sid)
                    self.publish({"type": "event", "level": "error", "at": time.time(),
                                  "message": "The engine stopped unexpectedly. The recording up to that moment "
                                             "is saved."})
            except Exception:  # noqa: BLE001
                log.exception("couldn't close session %s", sid)
        with self._lock:
            if self.proc is proc:
                self.proc = None
                self.session_id = None
                self.loading = None
        self._stopped.set()
        self.publish({"type": "engine_exit", "code": code, "session": sid})
        self.on_exit(sid)


class JobRunner:
    """One job at a time: {"kind": "process" | "notes", "session": id}."""

    def __init__(self, store: SessionStore, publish: Callable[[dict], None], engine_busy: Callable[[], bool]):
        self.store = store
        self.publish = publish
        self.engine_busy = engine_busy
        self.queue: deque[dict] = deque()
        self.current: dict | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._interrupted = False
        threading.Thread(target=self._run, name="jobs", daemon=True).start()

    def add(self, kind: str, sid: str, front: bool = False, **opts) -> None:
        with self._lock:
            if any(j.get("kind") == kind and j.get("session") == sid for j in list(self.queue) + [self.current or {}]):
                return
            job = {"kind": kind, "session": sid, "added": now_iso(), **opts}
            (self.queue.appendleft if front else self.queue.append)(job)
        if kind == "process":
            try:
                self.store.edit(sid, lambda d: d.__setitem__("processing", {**(d.get("processing") or {}),
                                                                             "queued": True, "error": None}))
            except Exception:  # noqa: BLE001
                pass
        self._wake.set()
        self.publish({"type": "jobs", **self.summary()})

    def summary(self) -> dict:
        cur = dict(self.current) if self.current else None
        if cur and cur["kind"] == "process":
            cur["progress"] = self._progress(cur["session"])
        return {"current": cur, "queue": list(self.queue), "waiting_for_engine": bool(self.queue) and self.engine_busy()}

    def _progress(self, sid: str) -> dict | None:
        try:
            p = self.store.path(sid) / "post" / "progress.json"
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def interrupt(self) -> None:
        """Stop the running pass (recording takes priority); it's requeued at the front."""
        with self._lock:
            proc, cur = self._proc, self.current
        if proc is not None and proc.poll() is None and cur is not None and cur["kind"] == "process":
            self._interrupted = True
            proc.send_signal(signal.SIGTERM)

    def shutdown(self) -> None:
        """App quitting: stop a running pass cleanly (it resumes from its last finished step next launch)."""
        with self._lock:
            proc = self._proc
            self.queue.clear()
        if proc is not None and proc.poll() is None:
            self._interrupted = True
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()

    def poke(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait(2.0)
            self._wake.clear()
            if self.current is None and self.queue:
                job = self.queue[0]
                if job["kind"] == "process" and self.engine_busy():
                    continue
                with self._lock:
                    self.current = self.queue.popleft()
                self.publish({"type": "jobs", **self.summary()})
                try:
                    self._do(self.current)
                except Exception:  # noqa: BLE001
                    log.exception("job %s failed", self.current)
                with self._lock:
                    done, self.current = self.current, None
                self.publish({"type": "jobs", **self.summary()})
                self.publish({"type": "session_changed", "session": done["session"]})
                self._wake.set()

    def _do(self, job: dict) -> None:
        sid = job["session"]
        if job["kind"] == "notes":
            from ..notes import generate_for_session
            model = load_settings().get("notes_model")
            status = generate_for_session(self.store, sid, model, force=bool(job.get("force")))
            log.info("notes for %s: %s", sid, status)
            return
        cmd = [sys.executable, "-m", "focus_ear.post", str(self.store.path(sid))]
        if job.get("from_step"):
            cmd += ["--from", job["from_step"]]
        elif job.get("force"):
            cmd += ["--force"]
        self._interrupted = False
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                env=dict(os.environ, TOKENIZERS_PARALLELISM="false"))
        with self._lock:
            self._proc = proc
        while proc.poll() is None:
            time.sleep(1.0)
            self.publish({"type": "jobs", **self.summary()})
        with self._lock:
            self._proc = None
        if self._interrupted:
            log.info("processing of %s paused for a recording; it resumes afterwards", sid)
            self.store.edit(sid, lambda d: d.update(status="recorded", processing={**(d.get("processing") or {}),
                                                                                    "queued": True, "step": None}))
            with self._lock:
                self.queue.appendleft({"kind": "process", "session": sid, "added": now_iso()})
            return
        try:
            data = self.store.load(sid)
        except Exception:  # noqa: BLE001
            return
        if proc.returncode == 0 and data.get("status") == "ready" and load_settings().get("auto_notes", True) \
                and data.get("notes_status") == "queued":
            self.add("notes", sid)
