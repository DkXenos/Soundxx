"""Command-line entry point: python -m focus_ear [options]."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from .audio_io import AudioEngine, DeviceError, find_device, list_devices
from .config import DATA_DIR, Config
from .pipeline import MetricsCollector, MetricsSnapshot, Pipeline

log = logging.getLogger("focus_ear")

LOG_FILE = DATA_DIR / "focus-ear.log"
MIC_PERMISSION_HINT = ("the mic is delivering pure digital silence: give your terminal app microphone "
                       "access (System Settings > Privacy & Security > Microphone), then restart it")


def _latency(value: str) -> str | float:
    if value in ("low", "high"):
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected 'low', 'high' or seconds") from None


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
    p.add_argument("--passthrough", action="store_true", help="no filtering; just mic -> buds")
    p.add_argument("--attenuation", type=float, default=d.attenuation_db, metavar="DB",
                   help=f"gain for non-selected speakers (default: {d.attenuation_db:g} dB)")
    p.add_argument("--boost", type=float, default=d.boost_db, metavar="DB",
                   help=f"gain for the selected speaker (default: {d.boost_db:g} dB)")
    p.add_argument("--threshold", type=float, default=d.threshold, metavar="COS",
                   help=f"cosine similarity to join an existing speaker (default: {d.threshold:g})")
    p.add_argument("--buffer-ms", type=float, default=d.buffer_ms, metavar="MS",
                   help=f"output jitter buffer; raise it if you hear dropouts (default: {d.buffer_ms:g})")
    p.add_argument("--latency", type=_latency, default=d.latency, metavar="low|high|SEC",
                   help="PortAudio suggested device latency (default: low)")
    p.add_argument("--allow-speakers", action="store_true",
                   help="permit loudspeaker output (the mic will pick it up and feed back)")
    p.add_argument("--debug", action="store_true",
                   help=f"log per-stage timing and end-to-end latency every second (also to {LOG_FILE})")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        input_device=args.input_device,
        output_device=args.output_device,
        allow_speakers=args.allow_speakers,
        latency=args.latency,
        buffer_ms=args.buffer_ms,
        passthrough=args.passthrough,
        attenuation_db=args.attenuation,
        boost_db=args.boost,
        threshold=args.threshold,
        debug=args.debug,
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


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger("focus_ear")
    root.setLevel(level)

    console = _StatusAwareHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(LOG_FILE)
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(file)


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


def _meter(db: float, width: int = 12, floor: float = -70.0) -> str:
    filled = round(width * min(1.0, max(0.0, (db - floor) / -floor)))
    return "█" * filled + "·" * (width - filled)


def format_status(engine: AudioEngine, pipeline: Pipeline, snap: MetricsSnapshot | None) -> str:
    mic = f"mic {_meter(pipeline.in_db)} {max(pipeline.in_db, -99):5.0f} dB"
    if pipeline.mic_silent:
        return f"⚠ {MIC_PERMISSION_HINT}"
    if engine.state != "running":
        return f"○ waiting for {engine.cfg.output_device!r} (asleep or disconnected)  {mic}"
    parts = [f"● {engine.output_name}", mic]
    if snap is not None:
        if snap.e2e_ms is not None:
            parts.append(f"latency {snap.e2e_ms:4.0f} ms")
        glitches = snap.out_underruns + snap.out_underflows + snap.in_overflows
        if glitches:
            parts.append(f"{glitches} glitch{'es' if glitches > 1 else ''}/s")
    return "  ".join(parts)


def format_debug(snap: MetricsSnapshot, pipeline: Pipeline) -> str:
    e2e = f"{snap.e2e_ms:.0f}" if snap.e2e_ms is not None else "?"
    stages = " ".join(f"{name} {mean:.2f}/{mx:.2f}" for name, (mean, mx) in snap.stages.items())
    return (
        f"e2e {e2e} ms ≈ in-dev {snap.in_device_ms:.0f} + in-queue {snap.in_queue_ms:.0f}"
        f" + proc {snap.proc_ms:.1f} + out-buf {snap.out_buffer_ms:.0f} + out-dev {snap.out_device_ms:.0f}"
        f" | stage ms mean/max: {stages or '-'} | RTF {snap.rtf:.3f} | mic {pipeline.in_db:.0f} dB"
        f" | in-ovf {snap.in_overflows} in-drop {snap.in_dropped_ms:.0f}ms"
        f" out-underrun {snap.out_underruns} out-underflow {snap.out_underflows}"
        f" out-skip {snap.out_skipped_ms:.0f}ms worker-skip {snap.worker_skipped_ms:.0f}ms"
    )


# --------------------------------------------------------------------------

def run(cfg: Config) -> int:
    # Audio callbacks need the GIL too. Hand it over more often than the 5 ms
    # default so a busy worker can't starve them into dropouts.
    sys.setswitchinterval(0.001)

    if not cfg.passthrough:
        log.info("speaker gating isn't implemented yet; running in passthrough mode")
    engine = AudioEngine(cfg)
    pipeline = Pipeline(engine, cfg, stages=[])
    try:
        engine.start()
    except DeviceError as exc:
        log.error("%s", exc)
        return 2
    pipeline.start()
    metrics = MetricsCollector(engine, pipeline)
    live_status = sys.stdout.isatty() and not cfg.debug
    snap: MetricsSnapshot | None = None
    next_report = time.monotonic() + 1.0
    warned_silent = False
    log.info("running, Ctrl-C to quit")

    try:
        while True:
            time.sleep(0.25)
            if engine.error is not None:
                log.error("%s", engine.error)
                return 2
            if pipeline.mic_silent and not warned_silent:
                warned_silent = True
                log.warning(MIC_PERMISSION_HINT)
            if time.monotonic() >= next_report:
                next_report += 1.0
                snap = metrics.snapshot()
                if cfg.debug:
                    log.debug("[%s] %s", engine.state, format_debug(snap, pipeline))
            if live_status:
                sys.stdout.write("\r\x1b[2K" + format_status(engine, pipeline, snap))
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if live_status:
            sys.stdout.write("\n")
        engine.stop()
        pipeline.stop()
        log.info("stopped")
        if engine.wedged:
            # PortAudio's atexit handler would block on the stream that hung.
            logging.shutdown()
            os._exit(3)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.list_devices:
        print_devices(cfg)
        return 0
    setup_logging(cfg.debug)
    return run(cfg)
