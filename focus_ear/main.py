"""Command-line entry point: python -m focus_ear [options]."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

from .audio_io import AudioEngine, DeviceError, find_device, list_devices
from .clustering import OnlineClusterer, SpeakerTracker
from .config import DATA_DIR, Config
from .gain import NoiseGate, SpeakerGain, db_to_gain
from .pipeline import MIC_SILENT_HINT, MetricsCollector, MetricsSnapshot, Pipeline
from .profiles import load_profiles, save_profiles
from .vad import VadAnalyzer

log = logging.getLogger("focus_ear")

LOG_FILE = DATA_DIR / "focus-ear.log"


def _latency(value: str) -> str | float:
    if value in ("low", "high"):
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected 'low', 'high' or seconds") from None


def _max_speakers(value: str) -> int:
    n = int(value)
    if not 1 <= n <= 9:
        raise argparse.ArgumentTypeError("1-9 (one per number key)")
    return n


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(
        prog="focus-ear",
        description="Boost one speaker in the room, attenuate everyone else, and play it to your earbuds.",
    )
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--input-device", metavar="NAME|INDEX", default=d.input_device,
                   help="mic to record from (default: the built-in MacBook microphone)")
    p.add_argument("--output-device", metavar="NAME|INDEX", default=d.output_device,
                   help=f"where to play the result, by name substring (default: {d.output_device!r})")
    p.add_argument("--passthrough", action="store_true",
                   help="no audio processing at all (speech and speakers are still shown)")
    p.add_argument("--no-tui", dest="tui", action="store_false",
                   help="plain one-line status instead of the interactive UI (no speaker selection)")

    g = p.add_argument_group("speech detection and noise gate")
    g.add_argument("--vad-threshold", type=float, default=d.vad_threshold, metavar="P",
                   help=f"speech probability that counts as speech (default: {d.vad_threshold:g})")
    g.add_argument("--no-gate", dest="gate", action="store_false",
                   help="don't attenuate non-speech (A/B comparison)")
    g.add_argument("--gate-attenuation", type=float, default=d.gate_attenuation_db, metavar="DB",
                   help=f"gain applied when nobody is speaking (default: {d.gate_attenuation_db:g} dB)")
    g.add_argument("--gate-hangover-ms", type=float, default=d.gate_hangover_ms, metavar="MS",
                   help=f"keep the gate open this long after speech ends (default: {d.gate_hangover_ms:g})")
    g.add_argument("--gate-lookahead-ms", type=float, default=d.gate_lookahead_ms, metavar="MS",
                   help="delay the audio so the gate opens before a word starts; adds this much "
                        f"latency (default: {d.gate_lookahead_ms:g})")

    s = p.add_argument_group("speaker identification")
    s.add_argument("--no-speakers", dest="speakers", action="store_false",
                   help="skip speaker identification entirely (no model load, no lookahead delay)")
    s.add_argument("--device", choices=("auto", "mps", "cpu"), default=d.device,
                   help="where the speaker model runs (default: auto = Apple GPU if available)")
    s.add_argument("--cluster-threshold", type=float, default=d.cluster_threshold, metavar="COS",
                   help="cosine similarity needed to count as a known speaker; raise it if different "
                        f"people get merged, lower it if one person splits (default: {d.cluster_threshold:g})")
    s.add_argument("--merge-threshold", type=float, default=d.merge_threshold, metavar="COS",
                   help=f"merge two speakers whose voiceprints converge this far (default: {d.merge_threshold:g})")
    s.add_argument("--embed-window", type=float, default=d.embed_window_s, metavar="SEC",
                   help=f"seconds of speech per voiceprint (default: {d.embed_window_s:g})")
    s.add_argument("--embed-hop", type=float, default=d.embed_hop_s, metavar="SEC",
                   help=f"how often to take a voiceprint (default: {d.embed_hop_s:g})")
    s.add_argument("--max-speakers", type=_max_speakers, default=d.max_speakers, metavar="N",
                   help=f"speakers tracked at once, 1-9 (default: {d.max_speakers})")
    s.add_argument("--speaker-timeout", type=float, default=d.speaker_timeout_s, metavar="SEC",
                   help=f"forget unnamed speakers not heard for this long (default: {d.speaker_timeout_s:g})")
    s.add_argument("--min-sightings", type=int, default=d.min_sightings, metavar="N",
                   help=f"voiceprints needed before a speaker is shown (default: {d.min_sightings})")

    k = p.add_argument_group("per-speaker gain")
    k.add_argument("--boost", type=float, default=d.boost_db, metavar="DB",
                   help=f"gain for the selected speaker (default: {d.boost_db:g} dB)")
    k.add_argument("--attenuation", type=float, default=d.attenuation_db, metavar="DB",
                   help=f"gain for everyone else (default: {d.attenuation_db:g} dB)")
    k.add_argument("--lookahead", type=float, default=d.lookahead_ms, metavar="MS",
                   help="delay the audio so speaker decisions line up with it: more = fewer wrong "
                        f"cuts at a change of speaker, but that much more latency (default: {d.lookahead_ms:g})")

    p.add_argument("--buffer-ms", type=float, default=d.buffer_ms, metavar="MS",
                   help=f"output jitter buffer; raise it if you hear dropouts (default: {d.buffer_ms:g})")
    p.add_argument("--latency", type=_latency, default=d.latency, metavar="low|high|SEC",
                   help="PortAudio suggested device latency (default: low)")
    p.add_argument("--allow-speakers", action="store_true",
                   help="permit loudspeaker output (the mic will pick it up and feed back)")
    p.add_argument("--debug", action="store_true",
                   help=f"show the debug panel and log timing every second to {LOG_FILE}")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        input_device=args.input_device, output_device=args.output_device, allow_speakers=args.allow_speakers,
        latency=args.latency, buffer_ms=args.buffer_ms, passthrough=args.passthrough, tui=args.tui,
        vad_threshold=args.vad_threshold, gate=args.gate, gate_attenuation_db=args.gate_attenuation,
        gate_hangover_ms=args.gate_hangover_ms, gate_lookahead_ms=args.gate_lookahead_ms,
        speakers=args.speakers, device=args.device, cluster_threshold=args.cluster_threshold,
        merge_threshold=args.merge_threshold, embed_window_s=args.embed_window, embed_hop_s=args.embed_hop,
        max_speakers=args.max_speakers, speaker_timeout_s=args.speaker_timeout,
        min_sightings=args.min_sightings, boost_db=args.boost, attenuation_db=args.attenuation,
        lookahead_ms=args.lookahead, debug=args.debug,
    )


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

class _StatusAwareHandler(logging.StreamHandler):
    """Clears the live status line before printing, so log lines don't collide with it."""

    def emit(self, record: logging.LogRecord) -> None:
        if self.stream.isatty():
            self.stream.write("\r\x1b[2K")
        super().emit(record)


def setup_logging(debug: bool) -> logging.Handler:
    """Logs to the console and the log file; returns the console handler (the TUI removes it)."""
    root = logging.getLogger("focus_ear")
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    console = _StatusAwareHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(LOG_FILE)
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(file)
    return console


def print_devices(cfg: Config) -> None:
    devices = list_devices()

    def pick(query: str | None, kind: str) -> int | None:
        try:
            return find_device(devices, query, kind)
        except DeviceError:
            return None

    in_idx = pick(cfg.input_device, "input")
    out_idx = pick(cfg.output_device, "output")
    print(f"{'#':>3}  {'in':>3} {'out':>3}  {'rate':>6}  name")
    for i, d in enumerate(devices):
        mark = ""
        if i == in_idx:
            mark += "  <- input"
        if i == out_idx:
            mark += "  <- output"
        print(f"{i:>3}  {d['inputs']:>3} {d['outputs']:>3}  {d['samplerate']:>6.0f}  {d['name']}{mark}")
    if out_idx is None:
        print(f"\nNo output matches {cfg.output_device!r}. Are the buds connected? "
              "Pass --output-device with a name from the list.")


@dataclass
class Session:
    """Everything the UI needs, wired up and running."""
    cfg: Config
    engine: AudioEngine
    pipeline: Pipeline
    metrics: MetricsCollector
    gate: NoiseGate | None = None
    speaker_gain: SpeakerGain | None = None
    tracker: SpeakerTracker | None = None
    analyzer: object | None = None          # embeddings.SpeakerAnalyzer
    embed_device: str | None = None
    save_profiles: Callable = field(default=save_profiles)

    def log_metrics(self, snap: MetricsSnapshot) -> None:
        if self.cfg.debug:
            log.debug("[%s] %s", self.engine.state, format_debug(self, snap))


def _meter(db: float, width: int = 12, floor: float = -70.0) -> str:
    filled = round(width * min(1.0, max(0.0, (db - floor) / -floor)))
    return "█" * filled + "·" * (width - filled)


def _gate_label(gate: NoiseGate | None) -> str:
    if gate is None:
        return "gate off"
    return "gate open" if gate.gain_db > -0.5 else f"gate {gate.gain_db:.0f} dB"


def format_status(session: Session, snap: MetricsSnapshot | None) -> str:
    engine, pipeline = session.engine, session.pipeline
    mic = f"mic {_meter(pipeline.in_db)} {max(pipeline.in_db, -99):5.0f} dB"
    if pipeline.mic_silent:
        return f"⚠ {MIC_SILENT_HINT}"
    speech = "SPEECH " if pipeline.ctx is not None and pipeline.ctx.is_speech else "silence"
    if engine.state != "running":
        return f"○ waiting for {engine.cfg.output_device!r} (asleep or disconnected)  {mic}  {speech}"
    parts = [f"● {engine.output_name}", mic, speech, _gate_label(session.gate)]
    if session.tracker is not None:
        t = session.tracker.snapshot()
        talking = next((v.label for v in t.speakers if v.talking), None)
        parts.append(f"talking: {talking or '-'}  focus: {t.selected_label or 'everyone'}")
    if snap is not None:
        if snap.e2e_ms is not None:
            parts.append(f"latency {snap.e2e_ms:4.0f} ms")
        glitches = snap.out_underruns + snap.out_underflows + snap.in_overflows + snap.worker_drop_events
        if glitches:
            parts.append(f"{glitches} glitch{'es' if glitches > 1 else ''}/s")
    return "  ".join(parts)


def format_debug(session: Session, snap: MetricsSnapshot) -> str:
    pipeline = session.pipeline
    e2e = f"{snap.e2e_ms:.0f}" if snap.e2e_ms is not None else "?"
    stages = " ".join(f"{name} {mean:.2f}/{mx:.2f}" for name, (mean, mx) in snap.stages.items())
    vad_now = pipeline.ctx.vad_prob if pipeline.ctx is not None else 0.0
    line = (
        f"e2e {e2e} ms ≈ in-dev {snap.in_device_ms:.0f} + in-queue {snap.in_queue_ms:.0f}"
        f" + proc {snap.proc_ms:.1f} + lookahead {snap.delay_ms:.0f} + out-buf {snap.out_buffer_ms:.0f}"
        f" + out-dev {snap.out_device_ms:.0f}"
        f" | worker ms/block mean/max (budget {snap.block_ms:.0f}): {stages or '-'} | RTF {snap.rtf:.3f}"
        f" | vad p={vad_now:.2f} mean {snap.vad_prob_mean:.2f} speech {snap.speech_pct:.0f}%"
        f" | {_gate_label(session.gate)} | mic {pipeline.in_db:.0f} dB"
        f" | in-ovf {snap.in_overflows} in-drop {snap.in_dropped_ms:.0f}ms"
        f" out-underrun {snap.out_underruns} out-underflow {snap.out_underflows}"
        f" out-skip {snap.out_skipped_ms:.0f}ms"
        f" worker-drop {snap.worker_drop_events}x/{snap.worker_dropped_ms:.0f}ms"
    )
    a = session.analyzer
    if a is not None:
        line += (f" | embed {a.last_ms:.1f}ms done {a.embedded} dropped {a.dropped}"
                 f" silent-skip {a.skipped_silent}")
    return line


# --------------------------------------------------------------------------

def build_session(cfg: Config) -> Session:
    """Load the models, start audio and the worker. Raises DeviceError if the devices can't work."""
    vad = VadAnalyzer(cfg.vad_threshold)
    tracker = analyzer = None
    device = None
    if cfg.speakers:
        from .embeddings import EcapaEmbedder, SpeakerAnalyzer  # imports torch: only when needed

        log.info("loading the speaker model…")
        embedder = EcapaEmbedder(cfg.device)
        device = embedder.device.upper()
        clusterer = OnlineClusterer(
            threshold=cfg.cluster_threshold, merge_threshold=cfg.merge_threshold, alpha=cfg.cluster_alpha,
            max_speakers=cfg.max_speakers, timeout_s=cfg.speaker_timeout_s, min_count=cfg.min_sightings)
        profiles = load_profiles()
        for name, voiceprint in profiles:
            clusterer.add_enrolled(name, voiceprint)
        if profiles:
            log.info("saved speakers: %s", ", ".join(name for name, _ in profiles))
        tracker = SpeakerTracker(clusterer, db_to_gain(cfg.boost_db), db_to_gain(cfg.attenuation_db))
        analyzer = SpeakerAnalyzer(embedder, tracker, cfg.embed_window_s, cfg.embed_hop_s)

    engine = AudioEngine(cfg)
    engine.start()
    try:
        return _start_processing(cfg, engine, vad, tracker, analyzer, device)
    except BaseException:
        engine.stop()
        raise


def _start_processing(cfg: Config, engine: AudioEngine, vad: VadAnalyzer, tracker: SpeakerTracker | None,
                      analyzer: object | None, device: str | None) -> Session:
    stages = []
    gate = speaker_gain = None
    if not cfg.passthrough:
        if cfg.gate:
            gate = NoiseGate(engine.samplerate, cfg.gate_attenuation_db, cfg.gate_hangover_ms,
                             cfg.ramp_ms, cfg.gate_lookahead_ms)
            stages.append(gate)
        if tracker is not None:
            speaker_gain = SpeakerGain(engine.samplerate, tracker.gain_target, cfg.lookahead_ms, cfg.ramp_ms)
            stages.append(speaker_gain)
    log.info("noise gate %s; speaker gain %s", "on" if gate else "off",
             f"on ({cfg.lookahead_ms:g} ms lookahead)" if speaker_gain else "off")

    analyzers = [vad] + ([analyzer] if analyzer is not None else [])
    pipeline = Pipeline(engine, cfg, analyzers=analyzers, stages=stages)
    pipeline.start()
    return Session(cfg=cfg, engine=engine, pipeline=pipeline, metrics=MetricsCollector(engine, pipeline),
                   gate=gate, speaker_gain=speaker_gain, tracker=tracker, analyzer=analyzer,
                   embed_device=device)


def run_plain(session: Session) -> int:
    """One-line status (or per-second debug lines) until Ctrl-C."""
    cfg, engine, pipeline = session.cfg, session.engine, session.pipeline
    live_status = sys.stdout.isatty() and not cfg.debug
    snap: MetricsSnapshot | None = None
    next_report = time.monotonic() + 1.0
    warned_silent = False
    log.info("running, Ctrl-C to quit")
    try:
        while True:
            time.sleep(0.1)
            if engine.error is not None:
                log.error("%s", engine.error)
                return 2
            if pipeline.mic_silent and not warned_silent:
                warned_silent = True
                log.warning(MIC_SILENT_HINT)
            if session.tracker is not None:
                for event in session.tracker.drain_events():
                    log.info("%s", event)
            if time.monotonic() >= next_report:
                next_report += 1.0
                snap = session.metrics.snapshot()
                session.log_metrics(snap)
            if live_status:
                sys.stdout.write("\r\x1b[2K" + format_status(session, snap))
                sys.stdout.flush()
    except KeyboardInterrupt:
        return 0
    finally:
        if live_status:
            sys.stdout.write("\n")


def run(cfg: Config, console: logging.Handler | None = None) -> int:
    # Audio callbacks need the GIL too. Hand it over more often than the 5 ms
    # default so a busy worker can't starve them into dropouts.
    sys.setswitchinterval(0.001)
    try:
        session = build_session(cfg)
    except DeviceError as exc:
        log.error("%s", exc)
        return 2

    use_tui = cfg.tui and sys.stdin.isatty() and sys.stdout.isatty()
    try:
        if use_tui:
            from .tui import run_tui

            if console is not None:
                logging.getLogger("focus_ear").removeHandler(console)  # stderr would corrupt the screen
            try:
                code = run_tui(session)
            finally:
                if console is not None:
                    logging.getLogger("focus_ear").addHandler(console)
        else:
            code = run_plain(session)
    finally:
        session.engine.stop()
        session.pipeline.stop()
        log.info("stopped")
        if session.engine.wedged:
            # PortAudio's atexit handler would block on the stream that hung.
            logging.shutdown()
            os._exit(3)
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.list_devices:
        print_devices(cfg)
        return 0
    console = setup_logging(cfg.debug)
    return run(cfg, console)
