"""The live engine as a subprocess of the app, driven by JSON lines.

    python -m focus_ear.headless [--output NAME] [--mode listen|full] [--debug]

stdout carries only the protocol, one JSON object per line:

    {"type": "loading", "what": "speech recognition"}
    {"type": "ready", "input": ..., "output": ..., "models": {...}}     standby: mic meter live, nothing saved
    {"type": "state", ...}                                              5 times a second
    {"type": "line", "start": 12.3, "end": 15.1, "speaker_id": 2, "speaker": "Speaker 2", "text": ...}
    {"type": "event", "level": "info|warning|error", "message": ...}
    {"type": "started", "session": "<id>"} / {"type": "stopped", "session": ..., "duration_s": ...}
    {"type": "fatal", "message": ...}                                   then the process exits

stdin takes commands, one JSON object per line:

    {"cmd": "start", "session_dir": "/…/sessions/<id>"}   begin recording into an existing session folder
    {"cmd": "pause"} / {"cmd": "resume"}
    {"cmd": "select", "speaker": 3 | null}
    {"cmd": "rename", "speaker": 3, "name": "Pak Budi", "remember": true}
    {"cmd": "reset"}
    {"cmd": "set_mode", "mode": "listen" | "full"}
    {"cmd": "set_output", "device": "Mewo" | null}
    {"cmd": "stop"}                                         finish the transcript, close the session, exit

stdin closing (the app died) or SIGTERM is a "stop": the recording is
closed properly either way. Everything else goes to stderr and, once a
session has started, to its engine.log.

Speakers are identified in both modes, so the live transcript is labelled.
"listen" only leaves out the speaker gain and its 250 ms lookahead;
switching mode swaps the playback stages between two blocks.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

from .audio_io import AudioEngine, DeviceError
from .config import MODES, Config
from .health import health_line, memory_mb
from .main import (ASR_DRAIN_TIMEOUT_S, HEALTH_INTERVAL_S, Interrupts, Session, _hard_exit_if_needed, _start_processing,
                   build_stages, finish_transcription, load_denoise_model, load_speaker_models, load_transcriber,
                   make_segmenter, open_transcript, save_speakers_npz, session_config, session_models, setup_logging)
from .pipeline import MIC_SILENT_HINT
from .profiles import save_profiles
from .recorder import RawRecorder
from .sessions import AUDIO_FILE, LIVE_STEM, SessionStore, now_iso
from .settings import update_settings
from .vad import VadAnalyzer

log = logging.getLogger("focus_ear")

STATE_INTERVAL_S = 0.2


def app_config(mode: str, output: str | None, debug: bool = False) -> Config:
    """The app's live chain: speakers always identified (labels), gain per mode, one output buffer for both."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    # 64 ms covers both modes: the buffer is fixed for the session, the mode isn't.
    return Config.for_mode("full", output_device=output, debug=debug, tui=False, buffer_ms=64.0)


def mode_overrides(cfg: Config, mode: str) -> Config:
    """``cfg`` with the per-mode stage settings of ``mode`` (gate depth, lookahead)."""
    import dataclasses
    stage_keys = ("gate_attenuation_db", "lookahead_ms")
    changes = {k: v for k, v in Config.for_mode(mode).__dict__.items() if k in stage_keys}
    return dataclasses.replace(cfg, mode=mode, **changes)


class Protocol:
    """Writes protocol lines to the real stdout; library prints go to stderr instead."""

    def __init__(self) -> None:
        fd = os.dup(1)
        os.dup2(2, 1)  # anything printing to fd 1 (tqdm, loguru, print) now lands on stderr
        sys.stdout = sys.stderr
        self._out = os.fdopen(fd, "w", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def send(self, type_: str, **fields) -> None:
        line = json.dumps({"type": type_, **fields}, ensure_ascii=False, default=_jsonable)
        with self._lock:
            try:
                self._out.write(line + "\n")
                self._out.flush()
            except (BrokenPipeError, ValueError):
                pass  # the app is gone; the stop path still closes the session


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:
        pass
    return str(x)


class EventHandler(logging.Handler):
    """Warnings and errors from anywhere in the engine reach the app as events."""

    def __init__(self, proto: Protocol):
        super().__init__(logging.WARNING)
        self.proto = proto

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.proto.send("event", level=record.levelname.lower(), message=record.getMessage())
        except Exception:  # noqa: BLE001
            pass


class Engine:
    def __init__(self, proto: Protocol, mode: str, output: str | None, debug: bool):
        self.proto = proto
        self.mode = mode
        self.cfg = app_config(mode, output, debug)
        self.session: Session | None = None
        self.store: SessionStore | None = None
        self.sid: str | None = None
        self.phase = "loading"
        self.commands: queue.Queue = queue.Queue()
        self._clip_count = 0
        self._clip_until = 0.0
        self._next_health = 0.0
        self._next_metrics = 0.0
        self._pause_started: float | None = None
        self._log_handler: logging.Handler | None = None

    # Startup -------------------------------------------------------------
    def load(self) -> None:
        cfg = self.cfg
        send = self.proto.send
        send("loading", what="voice activity")
        vad = VadAnalyzer(cfg.vad_threshold)
        send("loading", what="speaker model")
        tracker, analyzer, device = load_speaker_models(cfg)
        send("loading", what="speech recognition")
        transcriber, asr_error = load_transcriber(cfg, tracker)
        send("loading", what="noise suppression")
        denoise_model = load_denoise_model(cfg)
        send("loading", what="audio devices")
        engine = AudioEngine(cfg)
        try:
            engine.start()
        except BaseException:
            if transcriber is not None:
                transcriber.close(timeout=2)
            raise
        segmenter = make_segmenter(cfg, transcriber) if transcriber is not None else None
        stage_cfg = mode_overrides(cfg, self.mode)
        self.session = _start_processing(stage_cfg, engine, vad, tracker, analyzer, device, segmenter,
                                         denoise_model, recorder=None,
                                         gain_tracker=tracker if self.mode == "full" else None,
                                         ab=False, paused=True)
        self.session.transcriber, self.session.asr_error = transcriber, asr_error
        self.models = session_models(cfg, transcriber, device)
        self.phase = "standby"
        send("ready", input=engine.input_name, output=engine.output_name, models=self.models,
             asr_error=asr_error, mode=self.mode, pid=os.getpid())

    # Commands ------------------------------------------------------------
    def handle(self, msg: dict) -> bool:
        """Apply one command; False means stop."""
        cmd = msg.get("cmd")
        s = self.session
        try:
            if cmd == "stop":
                return False
            if cmd == "start":
                self.start(Path(msg["session_dir"]))
            elif cmd == "pause":
                self.pause(True)
            elif cmd == "resume":
                self.pause(False)
            elif cmd == "select":
                if s.tracker is not None:
                    sid = msg.get("speaker")
                    self.event("info", s.tracker.select_id(None if sid is None else int(sid)))
            elif cmd == "rename":
                self.rename(int(msg["speaker"]), str(msg.get("name", "")), bool(msg.get("remember", True)))
            elif cmd == "reset":
                if s.tracker is not None:
                    s.tracker.reset()
                    self.event("info", "Speakers reset (saved voices kept)")
            elif cmd == "set_mode":
                self.set_mode(str(msg["mode"]))
            elif cmd == "set_output":
                device = msg.get("device")
                s.engine.cfg.output_device = device
                s.engine.set_output(device)
                update_settings(output_device=device)
            else:
                self.event("warning", f"unknown command {cmd!r}")
        except Exception as exc:  # noqa: BLE001 - a bad command must not end the lecture
            log.exception("command %r failed", cmd)
            self.event("error", f"{cmd} failed: {type(exc).__name__}: {exc}")
        return True

    def event(self, level: str, message: str) -> None:
        self.proto.send("event", level=level, message=message)

    def start(self, folder: Path) -> None:
        s = self.session
        if self.phase != "standby":
            self.event("warning", "already recording")
            return
        self.store = SessionStore(folder.parent)
        self.sid = folder.name
        self._attach_log(folder / "engine.log")
        recorder = RawRecorder(folder / AUDIO_FILE, s.engine.samplerate, self.cfg.record_rate)
        s.recorder = recorder
        s.folder = folder
        if s.transcriber is not None:
            open_transcript(self.cfg, s.transcriber, f"live, {s.engine.input_name}", folder / LIVE_STEM,
                            recording=recorder.path)
        self.store.update(self.sid, status="recording", engine_pid=os.getpid(), started=now_iso(),
                          devices={"input": s.engine.input_name, "output": s.engine.output_name},
                          models=self.models, config=session_config(mode_overrides(self.cfg, self.mode)))
        s.pipeline.raw_taps.append(recorder)
        s.pipeline.set_paused(False)
        s.started = time.monotonic()
        self.phase = "recording"
        log.info("recording session %s into %s", self.sid, folder)
        self.proto.send("started", session=self.sid)

    def pause(self, on: bool) -> None:
        s = self.session
        if self.phase not in ("recording", "paused") or (self.phase == "paused") == on:
            return
        s.pipeline.set_paused(on)
        at = round(s.pipeline.media_time, 2)
        if on:
            self._pause_started = time.monotonic()
            self.phase = "paused"
            self.store.edit(self.sid, lambda d: d["pauses"].append({"at_s": at, "paused": now_iso(), "resumed": None}))
            self.event("info", "Paused: nothing is recorded, transcribed or played")
        else:
            length = time.monotonic() - (self._pause_started or time.monotonic())

            def resumed(d):
                if d["pauses"] and d["pauses"][-1].get("resumed") is None:
                    d["pauses"][-1].update(resumed=now_iso(), length_s=round(length, 1))
            self.store.edit(self.sid, resumed)
            self.phase = "recording"
            self.event("info", "Recording again")

    def rename(self, speaker_id: int, name: str, remember: bool) -> None:
        tracker = self.session.tracker
        if tracker is None:
            return
        profiles = tracker.name_speaker(speaker_id, name, remember)
        if profiles is None:
            self.event("warning", "That speaker is no longer listed")
            return
        if remember:
            save_profiles(profiles)
            self.event("info", f"Saved {name.strip()}: recognised automatically next time")
        else:
            self.event("info", f"Named {name.strip()} for this session")

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self.mode:
            return
        s = self.session
        cfg = mode_overrides(self.cfg, mode)
        built = build_stages(cfg, s.engine.samplerate, s.engine.blocksize, s.tracker if mode == "full" else None)
        s.pipeline.set_stages(built.stages)
        s.gate, s.speaker_gain, s.agc, s.limiter = built.gate, built.speaker_gain, built.agc, built.limiter
        self.mode = mode
        log.info("mode: %s", mode)
        if self.sid is not None:
            self.store.update(self.sid, config={**session_config(cfg)})
        update_settings(mode=mode)
        self.event("info", "Focus mode: the selected speaker is louder (about 250 ms more delay)" if mode == "full"
                   else "Listen mode: everyone, lowest delay")

    # State ---------------------------------------------------------------
    def state(self) -> dict:
        s = self.session
        engine, pipe = s.engine, s.pipeline
        now = time.monotonic()
        clips = pipe.clips["mic"].count
        if clips > self._clip_count:
            self._clip_until = now + 2.0
        self._clip_count = clips
        speakers, selected = [], None
        if s.tracker is not None:
            snap = s.tracker.snapshot(now)
            selected = next((v.id for v in snap.speakers if v.selected), None)
            speakers = [{"id": v.id, "key": v.key, "label": v.label, "level_db": round(v.level_db, 1),
                         "talking": v.talking, "selected": v.selected, "enrolled": v.enrolled,
                         "speaking_s": round(v.speaking_s, 1)} for v in snap.speakers]
        tr, rec = s.transcriber, s.recorder
        snap_e2e = getattr(self, "_last_e2e", None)
        return {
            "phase": self.phase, "mode": self.mode, "input": engine.input_name,
            "output": engine.output_name, "wanted_output": engine.cfg.output_device,
            "output_state": engine.state, "mic_db": round(max(pipe.in_db, -120.0), 1),
            "mic_silent": pipe.mic_silent, "clipping": now < self._clip_until,
            "speech": bool(pipe.ctx is not None and pipe.ctx.is_speech and not pipe.paused),
            "speakers": speakers, "selected": selected,
            "recording_s": round(pipe.media_time, 1), "asr_pending": tr.pending if tr else 0,
            "asr_dropped": tr.dropped if tr else 0, "asr_on": tr is not None, "asr_error": s.asr_error,
            "latency_ms": snap_e2e, "errors": pipe.errors, "drops": pipe.drop_events,
            "recorder_error": rec.error if rec else None,
            "recorder_lost_s": round(rec.dropped / rec.rate, 1) if rec else 0.0,
            "denoise_skips": pipe.denoise_skips, "engine_error": str(engine.error) if engine.error else None,
        }

    def tick(self) -> None:
        s = self.session
        now = time.monotonic()
        if now >= self._next_metrics:
            self._next_metrics = now + 1.0
            snap = s.metrics.snapshot()
            if snap.e2e_ms is not None:
                self._last_e2e = round(snap.e2e_ms)
            s.log_metrics(snap)
            if now >= self._next_health:
                self._next_health = now + HEALTH_INTERVAL_S
                try:
                    log.info("health: %s", health_line(s))
                except Exception as exc:  # noqa: BLE001
                    log.debug("health line failed: %s", exc)
        if s.tracker is not None:
            for message in s.tracker.drain_events():
                self.event("info", message)
        if s.transcriber is not None:
            for line in s.transcriber.drain_lines():
                self.proto.send("line", start=round(line.start, 2), end=round(line.end, 2),
                                speaker_id=line.speaker_id, speaker=line.speaker, text=line.display_text,
                                garbled=line.garbled)
        self.proto.send("state", **self.state())

    # Shutdown ------------------------------------------------------------
    def stop(self) -> int:
        s = self.session
        if s is None:
            return 0
        self.phase = "stopping"
        self.proto.send("state", **self.state())
        with Interrupts() as hurry:
            s.engine.stop()
            s.pipeline.stop()  # flushes the open utterance, closes the recording
            finish_transcription(s.transcriber, ASR_DRAIN_TIMEOUT_S, hurry)
            if s.transcriber is not None:
                for line in s.transcriber.drain_lines():
                    self.proto.send("line", start=round(line.start, 2), end=round(line.end, 2),
                                    speaker_id=line.speaker_id, speaker=line.speaker, text=line.display_text,
                                    garbled=line.garbled)
        duration = 0.0
        if self.sid is not None:
            speakers = s.tracker.export() if s.tracker is not None else []
            try:
                if speakers:
                    save_speakers_npz(s.folder, speakers)
            except Exception:  # noqa: BLE001
                log.exception("couldn't save speakers.npz")
            duration = s.recorder.seconds if s.recorder is not None else s.pipeline.media_time

            def close(d):
                if d["pauses"] and d["pauses"][-1].get("resumed") is None:
                    d["pauses"][-1]["resumed"] = now_iso()
                d.update(status="recorded", ended=now_iso(), duration_s=round(duration, 1), engine_pid=None,
                         speakers=[{k: v for k, v in x.items() if k != "centroid"} for x in speakers],
                         asr_dropped=s.transcriber.dropped if s.transcriber is not None else 0,
                         recorder_lost_s=round(s.recorder.dropped / s.recorder.rate, 1) if s.recorder else 0.0)
            self.store.edit(self.sid, close)
            log.info("session %s closed: %.0f s recorded; peak memory %.0f MB", self.sid, duration, memory_mb())
        self.proto.send("stopped", session=self.sid, duration_s=round(duration, 1))
        return 0

    def _attach_log(self, path: Path) -> None:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logging.getLogger("focus_ear").addHandler(handler)
        self._log_handler = handler


def _read_commands(q: queue.Queue) -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            q.put(json.loads(raw))
        except ValueError:
            log.warning("ignoring a malformed command: %r", raw[:200])
    q.put({"cmd": "stop", "why": "stdin closed"})


def run_headless(mode: str = "listen", output: str | None = None, debug: bool = False) -> int:
    proto = Protocol()
    setup_logging(debug)
    root = logging.getLogger("focus_ear")
    root.addHandler(EventHandler(proto))
    sys.setswitchinterval(0.001)  # as in main.run: audio callbacks need the GIL often
    engine = Engine(proto, mode, output, debug)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, lambda *_: engine.commands.put({"cmd": "stop", "why": "signal"}))
    try:
        engine.load()
    except DeviceError as exc:
        proto.send("fatal", message=str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001
        log.exception("engine failed to start")
        proto.send("fatal", message=f"{type(exc).__name__}: {exc}")
        return 2
    threading.Thread(target=_read_commands, args=(engine.commands,), name="commands", daemon=True).start()
    warned_silent = False
    code = 0
    try:
        while True:
            try:
                msg = engine.commands.get(timeout=STATE_INTERVAL_S)
                if not engine.handle(msg):
                    break
                continue  # drain commands before the next state
            except queue.Empty:
                pass
            if engine.session.engine.error is not None:
                proto.send("event", level="error", message=str(engine.session.engine.error))
                code = 2
                break
            if engine.session.pipeline.mic_silent and not warned_silent:
                warned_silent = True
                proto.send("event", level="error", message=MIC_SILENT_HINT)
            engine.tick()
    finally:
        engine.stop()
        _hard_exit_if_needed(code, engine.session.engine.wedged, engine.session.transcriber)
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="focus_ear.headless", description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=MODES, default="listen")
    p.add_argument("--output", default=None, help="output device name; omit for none (record only)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)
    return run_headless(args.mode, args.output, args.debug)


if __name__ == "__main__":
    sys.exit(main())
