"""Long-session health: memory, uptime, drops and queue depths, for the debug log and panel."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import resource
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import Session

_RUSAGE_INFO_V2 = 2
_libproc = None


def memory_mb() -> float:
    """This process's memory footprint in MB: the "Memory" column of Activity Monitor.

    Falls back to peak resident size where proc_pid_rusage isn't available.
    """
    global _libproc
    if sys.platform == "darwin":
        try:
            if _libproc is None:
                _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib")
            # struct rusage_info_v2: a 16-byte uuid, then uint64 fields; ri_phys_footprint is the 8th.
            buf = (ctypes.c_uint64 * 32)()
            if _libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V2, ctypes.byref(buf)) == 0:
                return buf[2 + 7] / 2**20
        except OSError:
            pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**20 if sys.platform == "darwin" else 2**10)


def gpu_memory_mb() -> dict[str, float]:
    """GPU memory held by MLX (Whisper) and torch MPS (speaker model), if they're loaded."""
    out = {}
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            out["mlx"] = (mx.get_active_memory() + mx.get_cache_memory()) / 2**20
        except Exception:  # noqa: BLE001 - older MLX spells these mx.metal.*
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                # torch's own tensors; driver_allocated_memory() would count MLX's buffers too.
                out["mps"] = torch.mps.current_allocated_memory() / 2**20
        except Exception:  # noqa: BLE001
            pass
    return out


def format_uptime(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def health_line(session: Session) -> str:
    """One line: uptime, memory, cumulative drops and clips, current queue depths."""
    s = session
    engine, pipe = s.engine, s.pipeline
    st = engine.stats
    fs = engine.samplerate or 1
    gpu = " ".join(f"{k} {v:.0f}" for k, v in gpu_memory_mb().items())
    parts = [
        f"up {format_uptime(time.monotonic() - s.started)}",
        f"mem {memory_mb():.0f} MB" + (f" ({gpu})" if gpu else ""),
        f"drops: worker {pipe.drop_events} in-ovf {st.in_overflows} underrun {st.out_underruns}"
        + (f" embed {s.analyzer.dropped}" if s.analyzer is not None else "")
        + (f" asr {s.transcriber.dropped}" if s.transcriber is not None else ""),
        f"queues: in {1e3 * engine.input_backlog() / fs:.0f} ms out {1e3 * engine.out_ring.available() / fs:.0f} ms"
        + (f" asr {s.transcriber.pending}" if s.transcriber is not None else ""),
        "clips: " + " ".join(f"{name} {m.count}" for name, m in pipe.clips.items()),
        f"reconnects {st.reconnects}",
    ]
    if pipe.errors:
        parts.append(f"processing errors {pipe.errors} (last: {pipe.last_error})")
    return " | ".join(parts)


def format_conditioning(session: Session) -> str:
    """Denoise cost per 10 ms frame, AGC and limiter state: " | "-separated, for the debug line and panel."""
    parts = []
    d = getattr(session, "denoiser", None)
    if d is not None:
        frames_per_block = session.engine.blocksize / session.engine.samplerate / 0.010
        parts.append(f"denoise ({session.pipeline.denoise_scope}) {d.last_ms:.2f} ms/block = "
                     f"{d.last_ms / frames_per_block:.2f} ms per 10 ms frame, delay {d.delay_ms:.0f} ms, "
                     f"SNR est {d.lsnr:.0f} dB")
    agc = getattr(session, "agc", None)
    if agc is not None:
        parts.append(f"agc {agc.gain_db:+.1f} dB (speech {agc.level_db:.0f} dBFS)")
    limiter = getattr(session, "limiter", None)
    if limiter is not None:
        parts.append(f"limiter {limiter.gain_db:.1f} dB ({limiter.limited / session.engine.samplerate:.1f} s "
                     "cut by more than 1 dB)")
    return "".join(f" | {x}" for x in parts)
