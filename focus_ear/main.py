"""Command-line entry point: python -m focus_ear [options]."""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_io import AudioEngine, DeviceError, find_device, list_devices
from .clustering import OnlineClusterer, SpeakerTracker
from .config import DATA_DIR, SESSIONS_DIR, Config
from .dsp import Agc, HighPass, Limiter
from .gain import NoiseGate, SpeakerGain, db_to_gain
from .health import format_conditioning, format_uptime, health_line
from .pipeline import MIC_SILENT_HINT, MetricsCollector, MetricsSnapshot, Pipeline
from .profiles import load_profiles, save_profiles
from .transcript import AsrUnavailable, Transcriber, TranscriptLine, TranscriptWriter, UtteranceSegmenter, clock
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


def _fraction(value: str) -> float:
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError("0.0-1.0")
    return x


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

    c = p.add_argument_group("input conditioning (before everything else)")
    c.add_argument("--highpass", type=float, default=d.highpass_hz, metavar="HZ",
                   help=f"high-pass cutoff against rumble and handling noise; 0 = off (default: {d.highpass_hz:g})")
    c.add_argument("--denoise", choices=("on", "off"), default="on" if d.denoise else "off",
                   help="DeepFilterNet3 noise suppression (default: on)")
    c.add_argument("--denoise-mix", type=_fraction, default=d.denoise_mix, metavar="0-1",
                   help="1 = fully denoised, lower blends the original back in, which can sound more "
                        f"natural (default: {d.denoise_mix:g})")
    c.add_argument("--denoise-scope", choices=("all", "playback", "models"), default=d.denoise_scope,
                   help="who hears the denoised audio: all; playback (your ears only; VAD, embeddings and "
                        "Whisper get the raw mic); models (VAD, embeddings and Whisper only; you hear the "
                        f"raw mic) (default: {d.denoise_scope})")
    c.add_argument("--latency-budget", type=float, default=d.latency_budget_ms, metavar="MS",
                   help="warn at startup if denoising pushes the estimated mic-to-ear latency past this "
                        f"(default: {d.latency_budget_ms:g})")

    o = p.add_argument_group("output level")
    o.add_argument("--no-agc", dest="agc", action="store_false",
                   help="don't level the focused speech (automatic gain control)")
    o.add_argument("--agc-target", type=float, default=d.agc_target_db, metavar="DBFS",
                   help=f"speech level the AGC aims for, RMS (default: {d.agc_target_db:g})")
    o.add_argument("--no-limiter", dest="limiter", action="store_false",
                   help="no peak limiter (loud sounds may then clip)")
    o.add_argument("--limiter-ceiling", type=float, default=d.limiter_ceiling_db, metavar="DBFS",
                   help="no output sample goes above this: caps a cough or a slammed door "
                        f"(default: {d.limiter_ceiling_db:g})")

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

    t = p.add_argument_group("transcription")
    t.add_argument("--no-transcribe", dest="transcribe", action="store_false",
                   help="don't transcribe (no Whisper model load)")
    t.add_argument("--asr-model", default=d.asr_model, metavar="REPO|PATH",
                   help="MLX Whisper model: a Hugging Face repo such as mlx-community/whisper-medium-mlx "
                        f"or mlx-community/whisper-large-v3-turbo, or a local folder (default: {d.asr_model})")
    t.add_argument("--language", choices=("id", "en", "auto"), default=d.language,
                   help=f"language to transcribe as; auto detects it per utterance (default: {d.language})")
    t.add_argument("--initial-prompt", default=d.initial_prompt, metavar="TEXT",
                   help="context given to Whisper before every utterance; biases spelling and vocabulary. "
                        "'' for none (default: a short code-mixed Indonesian lecture sentence)")
    t.add_argument("--utterance-gap", type=float, default=d.utterance_gap_ms, metavar="MS",
                   help=f"silence that ends an utterance (default: {d.utterance_gap_ms:g})")
    t.add_argument("--max-utterance", type=float, default=d.max_utterance_s, metavar="SEC",
                   help=f"longest utterance before it's cut and transcribed (default: {d.max_utterance_s:g})")
    t.add_argument("--no-save", dest="save", action="store_false",
                   help=f"don't write the transcript to {SESSIONS_DIR}")
    t.add_argument("--file", metavar="WAV",
                   help="transcribe an audio file instead of the mic (no audio devices, no UI); "
                        "writes the same transcript files")

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
        lookahead_ms=args.lookahead, transcribe=args.transcribe, asr_model=args.asr_model,
        language=args.language, initial_prompt=args.initial_prompt, utterance_gap_ms=args.utterance_gap,
        max_utterance_s=args.max_utterance, save=args.save, debug=args.debug,
        highpass_hz=args.highpass, denoise=args.denoise == "on", denoise_mix=args.denoise_mix,
        denoise_scope=args.denoise_scope, latency_budget_ms=args.latency_budget, agc=args.agc,
        agc_target_db=args.agc_target, limiter=args.limiter, limiter_ceiling_db=args.limiter_ceiling,
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
    # Bounded: a 90-minute --debug session writes a few MB; keep at most 4 x 5 MB.
    file = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5 * 2**20, backupCount=3)
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
    transcriber: Transcriber | None = None
    segmenter: UtteranceSegmenter | None = None
    asr_error: str | None = None            # why transcription is off despite being asked for
    denoiser: object | None = None          # denoise.Denoiser
    agc: Agc | None = None
    limiter: Limiter | None = None
    started: float = field(default_factory=time.monotonic)
    save_profiles: Callable = field(default=save_profiles)
    _next_health: float = 0.0
    _clips: dict = field(default_factory=dict)      # accumulated since the last warning
    _next_clip_warning: float = 0.0

    def log_metrics(self, snap: MetricsSnapshot) -> None:
        """Called once a second by the UI or the plain loop (never by the worker)."""
        now = time.monotonic()
        for name, n in snap.clips.items():
            self._clips[name] = self._clips.get(name, 0) + n
        if self._clips and now >= self._next_clip_warning:
            self._next_clip_warning = now + CLIP_WARNING_INTERVAL_S
            report_clips(self._clips)
            self._clips = {}
        if self.cfg.debug:
            log.debug("[%s] %s", self.engine.state, format_debug(self, snap))
            if now >= self._next_health:
                self._next_health = now + HEALTH_INTERVAL_S
                log.debug("health: %s", health_line(self))


HEALTH_INTERVAL_S = 30.0
CLIP_WARNING_INTERVAL_S = 10.0


def report_clips(clips: dict[str, int]) -> None:
    """Warn about full-scale samples; ``clips`` maps chain point -> sample count."""
    if clips.get("output"):
        log.warning("output clipped: %d samples reached full scale%s", clips["output"],
                    "" if clips.get("mic") else "; the limiter is off or its ceiling is 0 dBFS")
    if clips.get("mic"):
        log.warning("the mic is clipping (%d samples at full scale): it's too close to something loud, "
                    "or the input level in Audio MIDI Setup is too high", clips["mic"])
    if clips.get("denoise"):
        log.debug("denoiser output reached full scale on %d samples (the limiter catches it)", clips["denoise"])


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
    if session.transcriber is not None and session.transcriber.pending:
        parts.append(f"transcribing {session.transcriber.pending}")
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
    line += format_conditioning(session)
    tr, seg = session.transcriber, session.segmenter
    if tr is not None and seg is not None:
        line += (f" | asr pending {tr.pending} done {tr.done} dropped {tr.dropped} failed {tr.failed}"
                 f" last {tr.last_processing_s:.2f}s/{tr.last_audio_s:.1f}s"
                 f" discarded short {seg.too_short} other-speaker {seg.other_speaker}")
    return line


def format_line(line: TranscriptLine) -> str:
    who = f"{line.speaker}: " if line.speaker else ""
    return f"[{clock(line.start)}] {who}{line.display_text}"


# --------------------------------------------------------------------------

def load_speaker_models(cfg: Config, background: bool = True
                        ) -> tuple[SpeakerTracker | None, object | None, str | None]:
    """(tracker, SpeakerAnalyzer, device), or Nones with --no-speakers."""
    if not cfg.speakers:
        return None, None, None
    from .embeddings import EcapaEmbedder, SpeakerAnalyzer  # imports torch: only when needed

    log.info("loading the speaker model…")
    embedder = EcapaEmbedder(cfg.device)
    clusterer = OnlineClusterer(
        threshold=cfg.cluster_threshold, merge_threshold=cfg.merge_threshold, alpha=cfg.cluster_alpha,
        max_speakers=cfg.max_speakers, timeout_s=cfg.speaker_timeout_s, min_count=cfg.min_sightings)
    profiles = load_profiles()
    for name, voiceprint in profiles:
        clusterer.add_enrolled(name, voiceprint)
    if profiles:
        log.info("saved speakers: %s", ", ".join(name for name, _ in profiles))
    tracker = SpeakerTracker(clusterer, db_to_gain(cfg.boost_db), db_to_gain(cfg.attenuation_db))
    analyzer = SpeakerAnalyzer(embedder, tracker, cfg.embed_window_s, cfg.embed_hop_s, background=background)
    return tracker, analyzer, embedder.device.upper()


def load_transcriber(cfg: Config, tracker: SpeakerTracker | None, live: bool = True
                     ) -> tuple[Transcriber | None, str | None]:
    """(transcriber, None), or (None, why not). A failed model load doesn't stop the audio."""
    if not cfg.transcribe:
        return None, None
    log.info("loading the speech recognition model…")
    try:
        transcriber = Transcriber(
            cfg.asr_model, language=None if cfg.language == "auto" else cfg.language,
            initial_prompt=cfg.initial_prompt, queue_size=cfg.asr_queue, drop_oldest=live,
            detect_language=cfg.debug, label_of=tracker.label if tracker is not None else lambda _: None)
    except AsrUnavailable as exc:
        log.error("transcription is off: no speech recognition model could be loaded (%s)", exc)
        return None, str(exc)
    return transcriber, None


def start_transcript(cfg: Config, transcriber: Transcriber | None, tracker: SpeakerTracker | None,
                     source: str, suffix: str = "") -> UtteranceSegmenter | None:
    """Open the transcript files (unless --no-save) and make the segmenter that feeds the transcriber."""
    if transcriber is None:
        return None
    if cfg.save:
        transcriber.writer = TranscriptWriter(SESSIONS_DIR, datetime.now(), source, transcriber.description,
                                              cfg.language, suffix)
        log.info("transcript: %s", transcriber.writer.md_path)
    return UtteranceSegmenter(
        transcriber.submit, selected=(lambda: tracker.selected_id) if tracker is not None else (lambda: None),
        gap_ms=cfg.utterance_gap_ms, max_s=cfg.max_utterance_s, preroll_ms=cfg.preroll_ms,
        min_ms=cfg.min_utterance_ms)


def load_denoise_model(cfg: Config):
    """(model, params), or None if denoising is off or DeepFilterNet can't load (audio carries on)."""
    if not cfg.denoise:
        return None
    log.info("loading the noise suppression model…")
    try:
        from .denoise import load_model
        return load_model()
    except Exception as exc:  # noqa: BLE001 - ImportError when not installed, download errors, ...
        log.warning("noise suppression is off: DeepFilterNet couldn't load (%s: %s). Install it with "
                    "pip install --no-deps -r requirements-denoise.txt, or pass --denoise off.",
                    type(exc).__name__, exc)
        return None


def make_conditioning(cfg: Config, rate: int, blocksize: int, denoise_model) -> tuple[HighPass | None, object | None]:
    """The input high-pass and denoiser for this stream rate and block size."""
    highpass = HighPass(rate, cfg.highpass_hz) if cfg.highpass_hz > 0 else None
    denoiser = None
    if denoise_model is not None:
        from .denoise import Denoiser
        denoiser = Denoiser(rate, blocksize, *denoise_model, mix=cfg.denoise_mix)
        log.info("noise suppression: DeepFilterNet3, scope %s, mix %g, delay %.0f ms, %.1f ms per %.0f ms block",
                 cfg.denoise_scope, cfg.denoise_mix, denoiser.delay_ms, denoiser.warmup_ms, cfg.block_ms)
    return highpass, denoiser


def check_latency_budget(cfg: Config, engine: AudioEngine, pipeline: Pipeline, denoiser) -> None:
    """Estimate mic-to-ear latency from the stream latencies and delays; warn if denoising breaks the budget.

    Components: input device + one block + output buffer + output device +
    the playback path's delays. Denoising adds its delay and its processing
    time. Bluetooth output latency is only known once the Buds are connected.
    """
    fs = engine.samplerate
    st = engine.stats
    out_ms = 1e3 * st.out_latency if engine.state == "running" else None
    den_ms = denoiser.delay_ms + denoiser.warmup_ms if denoiser is not None else 0.0
    other_delay_ms = 1e3 * pipeline.delay_samples / fs - (denoiser.delay_ms if denoiser is not None else 0.0)
    base = 1e3 * st.in_latency + cfg.block_ms + cfg.buffer_ms + (out_ms or 0.0) + other_delay_ms
    total = base + den_ms
    unknown = "" if out_ms is not None else " (output not connected yet: its latency isn't included)"
    log.info("estimated latency %.0f ms%s: input %.0f + block %.0f + output buffer %.0f + output device %s "
             "+ lookahead/limiter %.0f%s", total, unknown, 1e3 * st.in_latency, cfg.block_ms, cfg.buffer_ms,
             f"{out_ms:.0f}" if out_ms is not None else "?", other_delay_ms,
             f" + denoise {den_ms:.0f}" if denoiser is not None else "")
    if denoiser is None or total <= cfg.latency_budget_ms:
        return
    if base <= cfg.latency_budget_ms:
        log.warning("denoising pushes the estimated latency from %.0f to %.0f ms, past --latency-budget %.0f ms. "
                    "If the delay bothers you, try --denoise off.", base, total, cfg.latency_budget_ms)
    else:
        log.info("the estimated latency is over --latency-budget %.0f ms even without denoising (%.0f ms); "
                 "denoising adds %.0f ms of it (--denoise off). --lookahead is the bigger lever.",
                 cfg.latency_budget_ms, base, den_ms)


def output_stages(cfg: Config, rate: int, blocksize: int, tracker: SpeakerTracker | None
                  ) -> tuple[Agc | None, Limiter | None]:
    agc = limiter = None
    if cfg.agc:
        # Level only speech that plays at full volume: attenuated students mustn't pull the gain around.
        full = (lambda ctx: tracker.gain_target(ctx.speaker_id) > tracker.attenuation_gain) if tracker else (
            lambda ctx: True)
        agc = Agc(rate, blocksize, cfg.agc_target_db, adapt=full)
    if cfg.limiter:
        limiter = Limiter(rate, cfg.limiter_ceiling_db)
    return agc, limiter


def build_session(cfg: Config) -> Session:
    """Load the models, start audio and the worker. Raises DeviceError if the devices can't work."""
    vad = VadAnalyzer(cfg.vad_threshold)
    tracker, analyzer, device = load_speaker_models(cfg)
    transcriber, asr_error = load_transcriber(cfg, tracker)
    denoise_model = load_denoise_model(cfg)

    engine = AudioEngine(cfg)
    try:
        engine.start()
        segmenter = start_transcript(cfg, transcriber, tracker, f"live, {engine.input_name}")
        session = _start_processing(cfg, engine, vad, tracker, analyzer, device, segmenter, denoise_model)
    except BaseException:
        engine.stop()
        if transcriber is not None:
            transcriber.close(timeout=2)
        raise
    session.transcriber, session.asr_error = transcriber, asr_error
    return session


def _start_processing(cfg: Config, engine: AudioEngine, vad: VadAnalyzer, tracker: SpeakerTracker | None,
                      analyzer: object | None, device: str | None,
                      segmenter: UtteranceSegmenter | None = None, denoise_model=None) -> Session:
    fs, bs = engine.samplerate, engine.blocksize
    stages = []
    gate = speaker_gain = agc = limiter = None
    if not cfg.passthrough:
        if cfg.gate:
            gate = NoiseGate(fs, cfg.gate_attenuation_db, cfg.gate_hangover_ms, cfg.ramp_ms, cfg.gate_lookahead_ms)
            stages.append(gate)
        if tracker is not None:
            speaker_gain = SpeakerGain(fs, tracker.gain_target, cfg.lookahead_ms, cfg.ramp_ms)
            stages.append(speaker_gain)
        agc, limiter = output_stages(cfg, fs, bs, tracker)
        stages += [x for x in (agc, limiter) if x is not None]
    log.info("noise gate %s; speaker gain %s; AGC %s; limiter %s", "on" if gate else "off",
             f"on ({cfg.lookahead_ms:g} ms lookahead)" if speaker_gain else "off",
             f"on (target {cfg.agc_target_db:g} dBFS)" if agc else "off",
             f"on (ceiling {cfg.limiter_ceiling_db:g} dBFS)" if limiter else "off")
    # --passthrough is the A/B reference: nothing touches the audio, not even the high-pass.
    highpass, denoiser = make_conditioning(cfg, fs, bs, None if cfg.passthrough else denoise_model)
    if cfg.passthrough:
        highpass = None

    # Order matters: the segmenter reads what the VAD and speaker analyzer wrote for the same frame.
    analyzers = [a for a in (vad, analyzer, segmenter) if a is not None]
    pipeline = Pipeline(engine, cfg, analyzers=analyzers, stages=stages, highpass=highpass,
                        denoiser=denoiser, denoise_scope=cfg.denoise_scope)
    check_latency_budget(cfg, engine, pipeline, denoiser)
    # The mic has been recording while the models were set up. That backlog
    # isn't the worker falling behind: start from now instead of counting a drop.
    engine.skip_input(engine.input_backlog())
    pipeline.start()
    return Session(cfg=cfg, engine=engine, pipeline=pipeline, metrics=MetricsCollector(engine, pipeline),
                   gate=gate, speaker_gain=speaker_gain, tracker=tracker, analyzer=analyzer,
                   embed_device=device, segmenter=segmenter, denoiser=denoiser, agc=agc, limiter=limiter)


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
            if session.transcriber is not None:
                for line in session.transcriber.drain_lines():
                    log.info("%s", format_line(line))
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


ASR_DRAIN_TIMEOUT_S = 30.0  # at quit, wait at most this long for queued utterances
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class Interrupts:
    """While active, Ctrl-C (and SIGTERM/SIGHUP) are counted instead of raising KeyboardInterrupt.

    Used around shutdown, so a second Ctrl-C can't cut the transcript flush
    off halfway (engine stopped, utterance not yet flushed, files not
    closed). It only means "stop waiting for Whisper". Also used by --file,
    where the first Ctrl-C stops reading after the current block.
    """

    def __init__(self, message: str = ""):
        self.count = 0
        self.message = message
        self._old: dict[int, object] = {}

    def _on_signal(self, signum, frame) -> None:
        self.count += 1
        if self.message:
            os.write(2, f"\n{self.message}\n".encode())

    def __enter__(self) -> Interrupts:
        for sig in STOP_SIGNALS:
            self._old[sig] = signal.signal(sig, self._on_signal)
        return self

    def __exit__(self, *exc) -> None:
        for sig, handler in self._old.items():
            signal.signal(sig, handler)


def finish_transcription(transcriber: Transcriber | None, timeout: float | None,
                         interrupts: Interrupts | None = None) -> None:
    """Transcribe what's left, then close the files. An interrupt or the timeout skips the waiting only."""
    if transcriber is None:
        return
    if transcriber.pending:
        log.info("transcribing the last %d utterance(s)… (Ctrl-C to skip)", transcriber.pending)
    deadline = None if timeout is None else time.monotonic() + timeout
    while not transcriber.close(0.2):
        if (interrupts is not None and interrupts.count) or (deadline is not None and time.monotonic() > deadline):
            lost = transcriber.abandon()
            log.warning("stopped waiting for transcription; %d utterance(s) not transcribed", lost)
            break
    if transcriber.dropped:
        log.warning("%d utterance(s) were not transcribed (transcription fell behind, or skipped at quit)",
                    transcriber.dropped)
    if transcriber.writer is not None:
        log.info("transcript saved: %s (%d lines) and %s", transcriber.writer.md_path, transcriber.writer.lines,
                 transcriber.writer.jsonl_path.name)


def _hard_exit_if_needed(code: int, engine_wedged: bool, transcriber: Transcriber | None) -> None:
    """Skip interpreter teardown when a native thread can't be stopped cleanly.

    PortAudio's atexit handler would block on a stream that hung; tearing the
    interpreter down under an MLX inference still running on the ASR thread
    can crash. The transcript is already closed by now.
    """
    if engine_wedged or (transcriber is not None and transcriber.alive):
        logging.shutdown()
        os._exit(3 if engine_wedged else code)


def run_file(cfg: Config, path: str) -> int:
    """Run the analysis chain and transcription over an audio file, as fast as it goes."""
    import soundfile as sf

    try:
        info = sf.info(path)
    except (RuntimeError, sf.LibsndfileError) as exc:
        log.error("can't read %s: %s", path, exc)
        return 2
    fs, total = info.samplerate, info.frames
    blocksize = round(fs * cfg.block_ms / 1000)
    log.info("file: %s (%s, %d Hz, %d ch)", path, clock(total / fs), fs, info.channels)

    vad = VadAnalyzer(cfg.vad_threshold)
    # Inline embeddings: nothing is live, so every window gets identified and runs are repeatable.
    tracker, analyzer, _ = load_speaker_models(cfg, background=False)
    transcriber, _ = load_transcriber(cfg, tracker, live=False)
    if transcriber is None and cfg.transcribe:
        return 2
    # Only the models listen in --file mode, so "playback" scope means they get the raw audio.
    denoise_model = load_denoise_model(cfg) if cfg.denoise_scope != "playback" else None
    highpass, denoiser = make_conditioning(cfg, fs, blocksize, denoise_model)
    segmenter = start_transcript(cfg, transcriber, tracker, f"file, {Path(path).resolve()}",
                                 suffix=f"_{Path(path).stem}")
    engine = types.SimpleNamespace(samplerate=fs, blocksize=blocksize)  # what process_block needs
    pipeline = Pipeline(engine, cfg, analyzers=[a for a in (vad, analyzer, segmenter) if a is not None],
                        highpass=highpass, denoiser=denoiser,
                        denoise_scope="all" if denoiser is not None else cfg.denoise_scope)
    logprobs: list[float] = []

    def show_lines() -> None:
        if transcriber is not None:
            for line in transcriber.drain_lines():
                print(format_line(line), flush=True)
                if line.avg_logprob is not None:
                    logprobs.append(line.avg_logprob)

    t0 = time.perf_counter()
    pos, next_progress = 0, 0.1
    with Interrupts("stopping after this block, then finishing the transcript (Ctrl-C again to skip)") as stop:
        for block in sf.blocks(path, blocksize=blocksize, dtype="float32", always_2d=True):
            if stop.count:
                log.warning("interrupted at %s", clock(pos / fs))
                break
            mono = np.ascontiguousarray(block.mean(axis=1) if block.shape[1] > 1 else block[:, 0])
            pipeline.process_block(mono, adc_time=pos / fs)
            pos += len(mono)
            show_lines()
            if total and pos / total >= next_progress:
                log.info("%3.0f%% (%s)", 100 * pos / total, clock(pos / fs))
                next_progress += 0.1
        interrupted = stop.count > 0
        if segmenter is not None:
            segmenter.flush()
        stop.count = 0
        finish_transcription(transcriber, timeout=ASR_DRAIN_TIMEOUT_S if interrupted else None, interrupts=stop)
        show_lines()
    elapsed = time.perf_counter() - t0
    log.info("%s of audio in %.1f s (%.0fx real time)", clock(pos / fs), elapsed, pos / fs / max(elapsed, 1e-9))
    log_file_summary(pipeline, tracker, segmenter, denoiser, logprobs)
    code = 130 if interrupted else 0
    _hard_exit_if_needed(code, False, transcriber)
    return code


def log_file_summary(pipeline: Pipeline, tracker: SpeakerTracker | None, segmenter: UtteranceSegmenter | None,
                     denoiser, logprobs: list[float]) -> None:
    """What the models made of the file: comparable across --denoise-scope and --denoise settings."""
    cfg = pipeline.cfg
    scope = f"denoise {'on, scope ' + cfg.denoise_scope if denoiser is not None else 'off'}"
    log.info("summary (%s, high-pass %s):", scope, f"{cfg.highpass_hz:g} Hz" if pipeline.highpass else "off")
    if pipeline.blocks:
        log.info("  speech (VAD): %.1f%% of blocks", 100 * pipeline.speech_blocks / pipeline.blocks)
    if tracker is not None:
        snap = tracker.snapshot()
        mean_sim = tracker.match_sim_sum / tracker.matches if tracker.matches else float("nan")
        log.info("  speakers: %d shown, %d clusters founded; voiceprints matched their speaker at "
                 "cosine %.3f on average (%d matches)", len(snap.speakers), tracker.clusters_founded,
                 mean_sim, tracker.matches)
    if segmenter is not None:
        mean_lp = sum(logprobs) / len(logprobs) if logprobs else float("nan")
        log.info("  utterances: %d transcribed (mean avg_logprob %.3f), %d too short, %d other speakers",
                 segmenter.utterances, mean_lp, segmenter.too_short, segmenter.other_speaker)
    if denoiser is not None:
        mean_ms = pipeline.timer.drain().get("denoise", (0.0, 0.0))[0]
        log.info("  denoise: %.2f ms per %.0f ms block, delay %.0f ms", mean_ms, cfg.block_ms, denoiser.delay_ms)
    clips = {name: m.count for name, m in pipeline.clips.items() if m.count}
    if clips:
        report_clips(clips)


def run(cfg: Config, console: logging.Handler | None = None) -> int:
    # Audio callbacks need the GIL too. Hand it over more often than the 5 ms
    # default so a busy worker can't starve them into dropouts.
    sys.setswitchinterval(0.001)
    # Closing the terminal (SIGHUP) or `kill` (SIGTERM) quits like Ctrl-C, so the transcript is closed properly.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.default_int_handler)
    try:
        session = build_session(cfg)
    except DeviceError as exc:
        log.error("%s", exc)
        return 2

    use_tui = cfg.tui and sys.stdin.isatty() and sys.stdout.isatty()
    code = 0
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
    except KeyboardInterrupt:
        code = 0
    finally:
        # From here on Ctrl-C can't interrupt the cleanup, only skip waiting for Whisper.
        with Interrupts("finishing the transcript… (Ctrl-C again to skip what Whisper hasn't done yet)") as hurry:
            session.engine.stop()
            session.pipeline.stop()  # also closes the utterance being spoken, so it gets transcribed
            finish_transcription(session.transcriber, ASR_DRAIN_TIMEOUT_S, hurry)
            if session.transcriber is not None:
                for line in session.transcriber.drain_lines():  # the last words, finished after the UI closed
                    log.info("%s", format_line(line))
            log.info("stopped after %s", format_uptime(time.monotonic() - session.started))
        _hard_exit_if_needed(code, session.engine.wedged, session.transcriber)
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.list_devices:
        print_devices(cfg)
        return 0
    console = setup_logging(cfg.debug)
    if args.file:
        return run_file(cfg, args.file)
    return run(cfg, console)
