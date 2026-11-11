"""The app's local HTTP + WebSocket server (FastAPI), on 127.0.0.1 only.

Every request must carry the launch token (header X-Token, or ?token= for
the audio element and the WebSocket) and a Host of 127.0.0.1:<port>, so no
web page and no other local program can drive the app or read a lecture.
This process stays light: no torch, no MLX. Audio work happens in the
engine and job subprocesses (engine_client.py).
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import secrets
import subprocess
import threading
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import preflight, review
from ..audio_io import probe_devices
from ..config import SESSIONS_DIR
from ..sessions import AUDIO_FILE, SessionError, SessionStore, pid_alive
from ..settings import load_settings, update_settings
from .engine_client import EngineClient, JobRunner

log = logging.getLogger("focus_ear.app")

STATIC = Path(__file__).parent / "static"


class Hub:
    """Fan-out of messages from worker threads to every open WebSocket."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queues: set[asyncio.Queue] = set()

    def publish(self, msg: dict) -> None:
        loop = self.loop
        if loop is None:
            return
        for q in list(self.queues):
            loop.call_soon_threadsafe(self._put, q, msg)

    @staticmethod
    def _put(q: asyncio.Queue, msg: dict) -> None:
        if q.qsize() > 200 and msg.get("type") == "state":
            return  # a slow client skips state frames, never lines or events
        q.put_nowait(msg)


def create_app(store: SessionStore | None = None, token: str | None = None, port: int = 0,
               spawn_engine: bool = True) -> FastAPI:
    store = store or SessionStore(SESSIONS_DIR)
    token = token or secrets.token_urlsafe(24)
    hub = Hub()
    app = FastAPI(title="Soundxx", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.token, app.state.port, app.state.store, app.state.hub = token, port, store, hub

    def after_engine(sid: str | None) -> None:
        jobs.poke()
        if sid is not None and load_settings().get("auto_process", True):
            try:
                data = store.load(sid)
                if data.get("status") == "recorded" and data.get("duration_s", 0) >= 5:
                    jobs.add("process", sid)
            except SessionError:
                pass

    engine = EngineClient(store, hub.publish, after_engine)
    jobs = JobRunner(store, hub.publish, lambda: engine.running)
    app.state.engine, app.state.jobs = engine, jobs

    # ------------------------------------------------------------------
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        allowed = {f"127.0.0.1:{app.state.port}", f"localhost:{app.state.port}", "testserver"}
        if host not in allowed:
            return PlainTextResponse("forbidden host", status_code=403)
        path = request.url.path
        if path.startswith("/api/"):
            given = request.headers.get("x-token") or request.query_params.get("token")
            if not given or not secrets.compare_digest(given, app.state.token):
                return PlainTextResponse("missing or wrong token", status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    async def startup() -> None:
        hub.loop = asyncio.get_running_loop()
        imported = await asyncio.to_thread(store.import_flat)
        if imported:
            log.info("imported %d earlier session(s)", len(imported))
        # Notes left queued (offline last time) and passes that never finished.
        for s in await asyncio.to_thread(store.list):
            if s.get("notes_status") == "queued" and s["status"] == "ready":
                jobs.add("notes", s["id"])
            elif s["status"] == "processing" or (s.get("processing") or {}).get("queued"):
                if pid_alive((s.get("processing") or {}).get("pid")):
                    continue  # a pass from an earlier app run is still working on it: let it finish
                if s["status"] == "processing":
                    store.update(s["id"], status="recorded")
                jobs.add("process", s["id"])

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        await startup()
        yield
        await asyncio.to_thread(engine.stop, 60)
        await asyncio.to_thread(jobs.shutdown)

    app.router.lifespan_context = lifespan

    def sid_or_404(sid: str) -> dict:
        try:
            return store.load(sid)
        except SessionError as exc:
            raise HTTPException(404, str(exc)) from None

    # Status, devices, preflight ------------------------------------------
    @app.get("/api/status")
    def status():
        return {"engine": engine.summary(), "jobs": jobs.summary(), "settings": load_settings(),
                "events": list(engine.events)[-10:]}

    @app.get("/api/devices")
    def devices():
        devs = probe_devices()
        if devs is None:
            return {"ok": False, "inputs": [], "outputs": []}
        outputs = [{"name": d["name"], "loudspeaker": preflight.is_loudspeaker(d["name"])}
                   for d in devs if d["outputs"] > 0]
        return {"ok": True, "outputs": outputs, "inputs": [d["name"] for d in devs if d["inputs"] > 0]}

    @app.get("/api/preflight")
    def run_preflight(output: str | None = None):
        devs = probe_devices()
        st = engine.state or {}
        out = output if output is not None else load_settings()["output_device"]
        checks = preflight.run_preflight(
            [{"name": d["name"], "inputs": d["inputs"], "outputs": d["outputs"], "samplerate": 0} for d in devs]
            if devs is not None else None,
            out or None, st.get("mic_db") if engine.ready else None, st.get("mic_silent"), store.interrupted())
        if engine.fatal:
            checks.insert(0, {"id": "engine", "level": "blocker", "title": "Audio engine",
                              "message": engine.fatal, "fix": "Check the earbuds and the mic, then press Try again."})
        return {"checks": checks, "can_start": preflight.can_start(checks)}

    # Engine ---------------------------------------------------------------
    @app.post("/api/engine/warm")
    def warm(body: dict | None = None):
        body = body or {}
        settings = load_settings()
        output = body.get("output", settings["output_device"])
        if not engine.running:
            jobs.interrupt()  # recording comes first; the pass resumes afterwards
            engine.spawn(output or None, body.get("mode") or settings["mode"])
        return engine.summary()

    @app.post("/api/engine/release")
    def release():
        threading.Thread(target=engine.release, daemon=True).start()
        return {"ok": True}

    @app.post("/api/live/start")
    def live_start(body: dict):
        if not engine.ready or not engine.running:
            raise HTTPException(409, "the engine isn't ready yet")
        if engine.recording:
            raise HTTPException(409, "already recording")
        title, course = str(body.get("title", "")).strip(), str(body.get("course", "")).strip()
        info = store.create(title=title, course=course, engine_pid=engine.proc.pid)
        if course:
            update_settings(course=course)
        engine.start_session(info["id"])
        return {"session": info["id"]}

    @app.post("/api/live/command")
    def live_command(body: dict):
        cmd = body.get("cmd")
        if cmd not in ("pause", "resume", "select", "rename", "reset", "set_mode", "set_output"):
            raise HTTPException(400, f"unknown command {cmd!r}")
        if cmd == "set_output":
            name = body.get("device")
            if name and preflight.is_loudspeaker(name):
                raise HTTPException(400, "loudspeakers would feed back into the mic")
        if not engine.send(**body):
            raise HTTPException(409, "the engine isn't running")
        return {"ok": True}

    @app.post("/api/live/stop")
    def live_stop():
        sid = engine.session_id
        threading.Thread(target=engine.stop, daemon=True).start()
        return {"session": sid}

    @app.get("/api/live/lines")
    def live_lines():
        return {"session": engine.session_id, "lines": engine.lines}

    # Sessions ---------------------------------------------------------------
    @app.get("/api/sessions")
    def sessions():
        return {"sessions": store.list(), "courses": load_settings()["courses"]}

    @app.get("/api/sessions/{sid}")
    def session(sid: str):
        data = store.describe(sid_or_404(sid))
        folder = store.path(sid)
        doc = None
        if (folder / "final.json").exists() or (folder / "live.jsonl").exists() or (folder / review.DOC_FILE).exists():
            doc = review.load_or_build(folder, data)
            doc = {k: v for k, v in doc.items() if k != "history"} | {
                "can_undo": bool(doc.get("history")),
                "last_edit": doc["history"][-1]["op"] if doc.get("history") else None}
        notes = None
        if (folder / "notes.json").exists():
            notes = json.loads((folder / "notes.json").read_text(encoding="utf-8"))
        progress = jobs._progress(sid) if data["status"] == "processing" else None
        live = sid == engine.session_id
        return {"session": data, "transcript": doc, "notes": notes, "progress": progress, "live": live}

    @app.patch("/api/sessions/{sid}")
    def edit_session(sid: str, body: dict):
        sid_or_404(sid)
        changes = {k: str(v).strip() for k, v in body.items() if k in ("title", "course")}
        data = store.update(sid, **changes)
        if changes.get("course"):
            update_settings(course=changes["course"])
        folder = store.path(sid)
        if (folder / review.DOC_FILE).exists():
            review.save(folder, review.load_doc(folder), data)
        return data

    @app.post("/api/sessions/{sid}/recover")
    def recover(sid: str):
        sid_or_404(sid)
        data = store.recover(sid)
        if data["status"] == "recorded" and load_settings().get("auto_process", True):
            jobs.add("process", sid)
        return data

    @app.post("/api/sessions/{sid}/process")
    def process(sid: str, body: dict | None = None):
        data = sid_or_404(sid)
        if data["status"] == "recording":
            raise HTTPException(409, "still recording")
        if not (store.path(sid) / AUDIO_FILE).exists():
            raise HTTPException(409, "this session has no recording to process")
        body = body or {}
        jobs.add("process", sid, front=True, force=bool(body.get("force")), from_step=body.get("from_step"))
        return {"queued": True, "waiting_for_engine": engine.running}

    @app.post("/api/sessions/{sid}/notes")
    def notes(sid: str):
        sid_or_404(sid)
        store.update(sid, notes_status="queued", notes_error=None)
        jobs.add("notes", sid, front=True, force=True)  # asked for by hand: write them whatever the length
        return {"queued": True}

    def _doc(sid: str) -> tuple[Path, dict, dict]:
        data = sid_or_404(sid)
        folder = store.path(sid)
        return folder, review.load_or_build(folder, data), data

    @app.post("/api/sessions/{sid}/edit")
    def edit(sid: str, body: dict):
        folder, doc, data = _doc(sid)
        try:
            review.apply(doc, body)
        except (review.EditError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from None
        review.save(folder, doc, data)
        return {"ok": True}

    @app.post("/api/sessions/{sid}/undo")
    def undo(sid: str):
        folder, doc, data = _doc(sid)
        op = review.undo(doc)
        review.save(folder, doc, data)
        return {"undone": op}

    @app.post("/api/sessions/{sid}/remember")
    def remember(sid: str, body: dict):
        """Save a speaker's voice from this session, so they're recognised (named) next time."""
        import numpy as np

        from ..profiles import load_profiles, save_profiles
        folder, doc, data = _doc(sid)
        speaker, name = body.get("speaker"), str(body.get("name", "")).strip()
        if not name or speaker not in doc["speakers"]:
            raise HTTPException(400, "pick a speaker and give a name")
        npz = folder / "speakers.npz"
        if not npz.exists():
            raise HTTPException(409, "this session has no voiceprints (process it first)")
        z = np.load(npz)
        ids = [str(i) if str(i).startswith("S") else f"S{i}" for i in z["ids"].tolist()]
        # A merged speaker's lines may come from several clusters: average their voiceprints.
        sources = {speaker} | {h["op"]["source"] for h in doc.get("history", [])
                               if h["op"].get("op") == "merge_speakers" and h["op"].get("target") == speaker}
        rows = [i for i, x in enumerate(ids) if x in sources]
        if not rows:
            raise HTTPException(409, "no voiceprint for this speaker in this session")
        v = z["centroids"][rows].mean(axis=0)
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        profiles = [(n, p) for n, p in load_profiles() if n != name] + [(name, v)]
        save_profiles(profiles)
        review.apply(doc, {"op": "mark_remembered", "speaker": speaker, "name": name})
        review.save(folder, doc, data)
        return {"ok": True, "profiles": [n for n, _ in profiles]}

    @app.get("/api/profiles")
    def profiles():
        from ..profiles import load_profiles
        return {"profiles": [n for n, _ in load_profiles()]}

    @app.post("/api/profiles/forget")
    def forget(body: dict):
        from ..profiles import load_profiles, save_profiles
        name = body.get("name")
        save_profiles([(n, v) for n, v in load_profiles() if n != name])
        return {"ok": True}

    @app.get("/api/sessions/{sid}/audio")
    def audio(sid: str):
        sid_or_404(sid)
        p = store.path(sid) / AUDIO_FILE
        if not p.exists():
            raise HTTPException(404, "no recording")
        return FileResponse(p, media_type="audio/wav")

    @app.get("/api/sessions/{sid}/export/{kind}")
    def export(sid: str, kind: str):
        data = sid_or_404(sid)
        folder = store.path(sid)
        name = data["id"]
        if kind == "transcript.md":
            review.load_or_build(folder, data)
            return Response((folder / review.MD_FILE).read_text(encoding="utf-8"), media_type="text/markdown",
                            headers={"Content-Disposition": f'attachment; filename="{name} transcript.md"'})
        if kind == "notes.md":
            p = folder / "notes.md"
            if not p.exists():
                raise HTTPException(404, "no notes yet")
            return Response(p.read_text(encoding="utf-8"), media_type="text/markdown",
                            headers={"Content-Disposition": f'attachment; filename="{name} notes.md"'})
        if kind == "reference.json":
            doc = review.load_or_build(folder, data)
            ref = {"session": name, "corrected": bool(doc.get("history")), "lines": review.text_only(doc)}
            return JSONResponse(ref, headers={"Content-Disposition": f'attachment; filename="{name} reference.json"'})
        if kind == "bundle.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in ("transcript.md", "notes.md", "session.json", "transcript.json", "notes.json"):
                    if (folder / f).exists():
                        z.write(folder / f, f)
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})
        raise HTTPException(404, "unknown export")

    @app.post("/api/sessions/{sid}/reveal")
    def reveal(sid: str):
        sid_or_404(sid)
        folder = store.path(sid)
        target = folder / "transcript.md" if (folder / "transcript.md").exists() else folder
        subprocess.Popen(["open", "-R", str(target)])
        return {"ok": True}

    @app.post("/api/sessions/{sid}/delete")
    def delete(sid: str):
        sid_or_404(sid)
        if sid == engine.session_id:
            raise HTTPException(409, "stop the recording first")
        store.delete(sid)
        return {"ok": True, "moved_to": "Trash"}

    @app.post("/api/import")
    def import_audio(body: dict):
        from ..post.importer import import_recording
        path = Path(str(body.get("path", ""))).expanduser()
        if not path.is_file():
            raise HTTPException(400, f"no such file: {path}")
        try:
            data = import_recording(store, path, title=body.get("title") or path.stem, course=body.get("course", ""))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"couldn't read {path.name}: {exc}") from None
        if load_settings().get("auto_process", True):
            jobs.add("process", data["id"])
        return data

    @app.patch("/api/settings")
    def settings(body: dict):
        return update_settings(**{k: v for k, v in body.items() if k in ("auto_process", "auto_notes", "notes_model")})

    # Live push ---------------------------------------------------------------
    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        host = socket.headers.get("host", "")
        if socket.query_params.get("token") != app.state.token or host not in (
                f"127.0.0.1:{app.state.port}", f"localhost:{app.state.port}", "testserver"):
            await socket.close(code=4401)
            return
        await socket.accept()
        q: asyncio.Queue = asyncio.Queue()
        hub.queues.add(q)
        try:
            await socket.send_json({"type": "hello", "engine": engine.summary(), "jobs": jobs.summary()})
            while True:
                msg = await q.get()
                await socket.send_json(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.queues.discard(q)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
