from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from ..config import SESSIONS_DIR
from ..sessions import SessionError, SessionStore, now_iso
from ..settings import write_json_atomic
from .common import OUTPUTS, POST_DIR, STEPS, PeakMemory, Progress, load_audio, read_json, step_path, write_step

log = logging.getLogger("focus_ear.post")


def resolve(target: str, root: Path = SESSIONS_DIR) -> tuple[Path, SessionStore | None, str | None]:
    p = Path(target).expanduser()
    if p.is_dir():
        folder = p.resolve()
    else:
        folder = root / target
    if not (folder / "audio.wav").exists():
        raise SystemExit(f"no audio.wav in {folder}")
    store = SessionStore(folder.parent) if (folder / "session.json").exists() else None
    return folder, store, folder.name if store else None


def run_step(folder: Path, step: str, opts: dict) -> dict:
    """One step, in this process. Returns its timing record."""
    progress = Progress(folder, step)
    t0 = time.perf_counter()
    with PeakMemory() as mem:
        if step == "regions":
            from . import regions
            data = regions.run(folder, load_audio(folder), progress, threshold=opts.get("vad_threshold", 0.5))
        elif step == "diarize":
            from . import diarize
            reg = read_json(step_path(folder, "regions"))
            data = diarize.run(folder, load_audio(folder), reg["regions"], progress,
                               threshold=opts.get("cluster_threshold", diarize_default()))
        elif step == "asr":
            from . import final_asr
            reg = read_json(step_path(folder, "regions"))
            diar = read_json(step_path(folder, "diarize"))
            data = final_asr.run(folder, load_audio(folder), diar["turns"], reg["regions"], reg.get("probs_100ms"),
                                 progress, model=opts.get("asr_model", final_asr.MODEL),
                                 en_threshold=opts.get("en_threshold", 0.6),
                                 **({"prompt": opts["prompt"] or None} if "prompt" in opts else {}))
        elif step == "merge":
            from . import merge
            data = merge.run(read_json(step_path(folder, "asr")), read_json(step_path(folder, "diarize")),
                             read_json(step_path(folder, "regions")))
        else:
            raise ValueError(step)
        write_step(folder, step, data)
        if step == "merge":
            from ..review import install_final
            session = _session(folder)
            install_final(folder, data, session)
    timing = {"step": step, "seconds": round(time.perf_counter() - t0, 1), "peak_mb": round(mem.peak_mb)}
    write_json_atomic(folder / POST_DIR / f"timing-{step}.json", timing, indent=None)
    progress(1.0, "done", force=True)
    return timing


def diarize_default() -> float:
    from .diarize import DEFAULT_THRESHOLD
    return DEFAULT_THRESHOLD


def _session(folder: Path) -> dict | None:
    try:
        return json.loads((folder / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


_CHILD: list = [None]
_STOPPING = [False]


def _on_term(signum, frame) -> None:
    """SIGTERM (the app needs the memory for a recording): stop the step's process too."""
    _STOPPING[0] = True
    child = _CHILD[0]
    if child is not None and child.poll() is None:
        child.terminate()


def run_all(folder: Path, store: SessionStore | None, sid: str | None, force: bool = False,
            opts: dict | None = None, from_step: str | None = None) -> int:
    opts = opts or {}
    if store is not None:
        before = store.load(sid)["status"]
        store.update(sid, status="processing", processing={"started": now_iso(), "step": None, "error": None,
                                                           "pid": os.getpid()})
    else:
        before = None
    timings = []
    redo = force
    start = STEPS.index(from_step) if from_step else 0
    try:
        for i, step in enumerate(STEPS):
            if i >= start and from_step:
                redo = True
            if step_path(folder, step).exists() and not redo:
                log.info("%s: already done", step)
                continue
            redo = True  # everything after a rerun step is stale
            if store is not None:
                store.edit(sid, lambda d: d.__setitem__("processing", {**(d.get("processing") or {}), "step": step}))
            log.info("%s…", step)
            cmd = [sys.executable, "-m", "focus_ear.post", str(folder), "--step", step, "--force",
                   "--opts", json.dumps(opts)]
            proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[2])
            _CHILD[0] = proc
            proc.wait()
            _CHILD[0] = None
            if _STOPPING[0]:
                raise RuntimeError("stopped before the end (it resumes from this step)")
            if proc.returncode != 0:
                raise RuntimeError(f"step {step} failed (exit {proc.returncode}); see {folder / POST_DIR / 'post.log'}")
            timing = read_json(folder / POST_DIR / f"timing-{step}.json")
            timings.append(timing)
            log.info("%s: %.1f s, peak %d MB", step, timing["seconds"], timing["peak_mb"])
    except Exception as exc:  # noqa: BLE001
        log.error("%s", exc)
        if store is not None:
            store.edit(sid, lambda d: d.update(status=before if before in ("recorded", "ready") else "recorded",
                                               processing={**(d.get("processing") or {}),
                                                           "error": None if _STOPPING[0] else str(exc),
                                                           "ended": now_iso(), "pid": None,
                                                           "queued": _STOPPING[0]}))
        return 1
    if store is not None:
        def done(d):
            d["status"] = "ready"
            d["processing"] = {**(d.get("processing") or {}), "step": None, "error": None, "ended": now_iso(), "pid": None,
                               "timings": timings or (d.get("processing") or {}).get("timings")}
            if not (folder / "notes.json").exists() and d.get("notes_status") in (None, "failed"):
                d["notes_status"] = "queued"
        store.edit(sid, done)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="focus_ear.post", description="the after-class pass over a session")
    p.add_argument("session", help="session id, or a folder with audio.wav")
    p.add_argument("--step", choices=STEPS, help="run only this step, in this process")
    p.add_argument("--from", dest="from_step", choices=STEPS, help="rerun from this step on")
    p.add_argument("--force", action="store_true", help="rerun steps whose output exists")
    p.add_argument("--opts", default="{}", help=argparse.SUPPRESS)
    p.add_argument("--cluster-threshold", type=float, help="offline clustering: cosine distance to cut at")
    p.add_argument("--en-threshold", type=float, help="English if Whisper is at least this sure (1 = never)")
    p.add_argument("--asr-model", help="final transcription model (default whisper-large-v3-turbo)")
    args = p.parse_args(argv)
    try:
        folder, store, sid = resolve(args.session)
    except SessionError as exc:
        raise SystemExit(str(exc)) from None
    (folder / POST_DIR).mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    root = logging.getLogger("focus_ear")
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(folder / POST_DIR / "post.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    opts = json.loads(args.opts)
    for key, value in (("cluster_threshold", args.cluster_threshold), ("en_threshold", args.en_threshold),
                       ("asr_model", args.asr_model)):
        if value is not None:
            opts[key] = value
    if args.step:
        if step_path(folder, args.step).exists() and not args.force:
            log.info("%s: already done (--force to rerun)", args.step)
            return 0
        try:
            run_step(folder, args.step, opts)
        except Exception:
            log.exception("%s failed", args.step)
            return 1
        return 0
    import signal
    signal.signal(signal.SIGTERM, _on_term)
    return run_all(folder, store, sid, args.force, opts, args.from_step)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
