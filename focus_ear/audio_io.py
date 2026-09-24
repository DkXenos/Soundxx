"""Audio I/O: device selection, lock-free ring buffers and self-healing streams.

Input and output are two independent PortAudio streams rather than one duplex
stream. The built-in mic and the Bluetooth buds run on separate hardware
clocks, and when the buds go to sleep only the output side should die: the mic
keeps feeding the analysis while we wait for the buds to come back.

    input callback --in_ring--> worker (pipeline.py) --out_ring--> output callback

The callbacks only copy samples and bump counters: no locks, no logging, no
allocation in the steady state. After start(), every PortAudio open, close and
re-initialisation happens on the supervisor thread.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .config import Config

log = logging.getLogger(__name__)


class RingBuffer:
    """Single-producer / single-consumer float32 ring buffer.

    Lock-free: only the producer advances ``_w`` and only the consumer advances
    ``_r``. Both are ever-increasing Python ints (atomic to assign under the
    GIL), and each side publishes its index only after its copy is complete,
    so neither side can observe a half-written block.
    """

    def __init__(self, min_capacity: int):
        size = 1 << (max(min_capacity, 2) - 1).bit_length()
        self._buf = np.zeros(size, dtype=np.float32)
        self._mask = size - 1
        self._w = 0
        self._r = 0

    @property
    def capacity(self) -> int:
        return len(self._buf)

    @property
    def write_index(self) -> int:
        return self._w

    @property
    def read_index(self) -> int:
        return self._r

    def available(self) -> int:
        return self._w - self._r

    # Producer side -------------------------------------------------------
    def write(self, data: np.ndarray) -> int:
        """Copy as much of ``data`` as fits; returns the number of samples written."""
        n = min(len(data), len(self._buf) - (self._w - self._r))
        if n <= 0:
            return 0
        start = self._w & self._mask
        first = min(n, len(self._buf) - start)
        self._buf[start:start + first] = data[:first]
        if n > first:
            self._buf[:n - first] = data[first:n]
        self._w += n
        return n

    # Consumer side -------------------------------------------------------
    def read_into(self, out: np.ndarray) -> int:
        """Fill ``out`` from the buffer; returns the number of samples read."""
        n = min(len(out), self._w - self._r)
        if n <= 0:
            return 0
        start = self._r & self._mask
        first = min(n, len(self._buf) - start)
        out[:first] = self._buf[start:start + first]
        if n > first:
            out[first:n] = self._buf[:n - first]
        self._r += n
        return n

    def skip(self, n: int) -> int:
        """Discard up to ``n`` of the oldest samples."""
        n = max(0, min(n, self._w - self._r))
        self._r += n
        return n


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------

class DeviceError(RuntimeError):
    """A device choice that can't work. Fatal: the user has to change a flag."""


class AudioSystemWedged(RuntimeError):
    """CoreAudio hung while closing a stream. Fatal: the process has to restart."""


_BUILTIN_MIC = re.compile(r"macbook.*microphone|built-in microphone", re.IGNORECASE)

# Runs in a throwaway interpreter; see probe_devices().
_PROBE_SCRIPT = (
    "import json, sounddevice as sd; "
    "print(json.dumps([{'name': d['name'], 'inputs': d['max_input_channels'], "
    "'outputs': d['max_output_channels']} for d in sd.query_devices()]))"
)


def list_devices() -> list[dict]:
    """PortAudio's device list as seen by this process; list position == device index."""
    return [
        {
            "name": d["name"],
            "inputs": d["max_input_channels"],
            "outputs": d["max_output_channels"],
            "samplerate": d["default_samplerate"],
        }
        for d in sd.query_devices()
    ]


def probe_devices(timeout: float = 5.0) -> list[dict] | None:
    """Enumerate devices in a fresh process.

    PortAudio enumerates devices only when it initialises, so this process
    can't see the buds reappear without tearing down every open stream,
    including the mic. A throwaway interpreter gets a fresh list without
    disturbing anything. Returns None if the probe itself failed.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        return json.loads(proc.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        log.debug("device probe failed: %s", exc)
        return None


def find_device(devices: list[dict], query: str | None, kind: str) -> int:
    """Resolve ``query`` (index, exact name or case-insensitive substring) to an index.

    With ``query=None`` for input, picks the built-in mic. Never the system
    default input: if that happens to be the buds' mic, opening it would drop
    the buds into the Bluetooth hands-free profile (low-quality call audio).
    """
    key = "inputs" if kind == "input" else "outputs"
    candidates = [i for i, d in enumerate(devices) if d[key] > 0]

    if query is None:
        if kind == "input":
            for i in candidates:
                if _BUILTIN_MIC.search(devices[i]["name"]):
                    return i
            raise DeviceError("no built-in microphone found; pass --input-device")
        raise DeviceError("no output device given; pass --output-device")

    query = str(query).strip()
    if query.isdigit():
        if int(query) in candidates:
            return int(query)
        raise DeviceError(f"device #{query} is not an {kind} device (see --list-devices)")

    q = query.lower()
    exact = [i for i in candidates if devices[i]["name"].lower() == q]
    if exact:
        return exact[0]
    hits = [i for i in candidates if q in devices[i]["name"].lower()]
    if not hits:
        names = ", ".join(repr(devices[i]["name"]) for i in candidates) or "none"
        raise DeviceError(f"no {kind} device matches {query!r} (available: {names})")
    if len(hits) > 1:
        log.warning("%r matches several %s devices; using %r", query, kind, devices[hits[0]]["name"])
    return hits[0]


def check_device_pair(input_name: str, output_name: str, allow_speakers: bool) -> None:
    if input_name == output_name:
        raise DeviceError(
            f"{input_name!r} is both input and output. Using a Bluetooth headset's mic "
            "switches it to hands-free mode (mono, narrowband); record from the "
            "built-in mic instead."
        )
    if "speaker" in output_name.lower() and not allow_speakers:
        raise DeviceError(
            f"output {output_name!r} looks like loudspeakers: playing the mic out loud "
            "will howl with feedback. Pass --allow-speakers if you really mean it."
        )


def _reinit_portaudio() -> None:
    # sounddevice has no public way to refresh the device list. Pa_Terminate()
    # followed by Pa_Initialize() is the only way to pick up hot-plugged
    # devices, and it invalidates every open stream, so close them first.
    sd._terminate()
    sd._initialize()


def _close_quietly(stream: sd._StreamBase | None, timeout: float = 2.0) -> bool:
    """Close a stream without letting a wedged CoreAudio device hang the caller.

    Returns False if the close hung. CoreAudio can occasionally deadlock inside
    AudioOutputUnitStop (seen once in testing, not reproducible); after that,
    PortAudio in this process can't be trusted.
    """
    if stream is None:
        return True
    t = threading.Thread(target=stream.close, kwargs={"ignore_errors": True}, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        log.error("closing an audio stream hung inside CoreAudio; abandoning it")
        return False
    return True


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

@dataclass
class AudioStats:
    """Written by the audio callbacks, read by everyone else.

    Counters only ever increase; readers diff two snapshots instead of
    resetting, so there is never a write-write race with a callback.
    """
    in_heartbeat: float = 0.0     # time.monotonic() of the last callback
    out_heartbeat: float = 0.0
    in_latency: float = 0.0       # PortAudio-reported stream latency (s); over-reported at 16 kHz
    in_device_latency: float = 0.0  # measured age of a block when its callback runs
    out_latency: float = 0.0
    in_overflows: int = 0         # device-reported input overflows
    in_dropped: int = 0           # samples lost because the input ring was full
    out_underflows: int = 0       # device-reported output underflows
    out_underruns: int = 0        # callbacks that found our output ring short
    out_skipped: int = 0          # samples discarded to cap buffering latency
    out_ring_full: int = 0        # samples the worker couldn't queue (output gone)
    out_level_sum: int = 0        # output ring fill at each callback, for the mean
    out_level_count: int = 0
    out_device_latency: float = 0.0  # DAC time minus callback time, last seen
    e2e_sum: float = 0.0          # mic ADC to buds DAC, summed per callback
    e2e_count: int = 0


class AudioEngine:
    STALE_S = 1.0          # this long without a callback means the stream is dead
    OPEN_GRACE_S = 3.0     # Bluetooth links can take a while to start pulling audio
    PROBE_INTERVAL_S = 2.0
    POLL_S = 0.25

    def __init__(self, cfg: Config):
        self.cfg = cfg
        fs = cfg.samplerate
        self.in_ring = RingBuffer(2 * fs)
        self.out_ring = RingBuffer(2 * fs)
        self.stats = AudioStats()

        # "stopped" | "running" | "waiting" (a device is missing) | "failed"
        self.state = "stopped"
        self.error: Exception | None = None
        self.wedged = False   # a stream close hung; PortAudio must not be touched again
        self.input_name: str | None = None
        self.output_name: str | None = None
        self._in_query = cfg.input_device
        self._out_query = cfg.output_device

        self._prefill = int(cfg.buffer_ms * fs / 1000)
        self._drift_slack = 3 * cfg.blocksize
        self._out_primed = False
        self._out_scratch = np.zeros(8192, dtype=np.float32)
        # (ring write index, ADC time of that sample): lets the output side
        # work out when any sample it plays was captured.
        self._in_anchor = (0, 0.0)
        self._out_anchor: tuple[int, float] | None = None

        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        self._stop = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._grace_until = 0.0
        self._next_probe = 0.0

    # Lifecycle -----------------------------------------------------------
    def start(self) -> None:
        """Open the streams; raises DeviceError if the configuration can't work.

        A missing output device is not an error: the engine waits for it.
        """
        self._stop.clear()
        try:
            self._open_streams()
        except BaseException:
            self.stop()
            raise
        if self._out_stream is None:
            log.warning("no output device matches %r yet; waiting for it", self._out_query)
        self._supervisor = threading.Thread(target=self._supervise, name="audio-supervisor", daemon=True)
        self._supervisor.start()

    def stop(self) -> None:
        self._stop.set()
        if self._supervisor is not None:
            self._supervisor.join()
            self._supervisor = None
        self._close_streams()
        self.state = "stopped"

    def _close_streams(self) -> None:
        for stream in (self._in_stream, self._out_stream):
            if not _close_quietly(stream):
                self.wedged = True
        self._in_stream = self._out_stream = None

    def _close_dead(self, stream: sd._StreamBase) -> None:
        if not _close_quietly(stream):
            self.wedged = True

    # Worker-facing API (called from the pipeline thread only) -------------
    def input_backlog(self) -> int:
        return self.in_ring.available()

    def read_input(self, out: np.ndarray) -> float:
        """Fill ``out`` from the mic; returns the ADC time of ``out[0]``.

        The caller checks input_backlog() first, so a full block is available.
        """
        k = self.in_ring.read_index
        self.in_ring.read_into(out)
        w0, t0 = self._in_anchor
        return t0 + (k - w0) / self.cfg.samplerate

    def skip_input(self, n: int) -> int:
        return self.in_ring.skip(n)

    def write_output(self, block: np.ndarray, adc_time: float) -> None:
        w = self.out_ring.write_index
        written = self.out_ring.write(block)
        self._out_anchor = (w, adc_time)
        if written < len(block):
            self.stats.out_ring_full += len(block) - written

    # Stream management (main thread before start(), supervisor after) -----
    def _open_streams(self) -> None:
        """Resolve devices against PortAudio's current list and open whatever is missing."""
        devices = list_devices()
        in_idx = find_device(devices, self._in_query, "input")
        try:
            out_idx = find_device(devices, self._out_query, "output")
        except DeviceError:
            out_idx = None
        if out_idx is not None:
            check_device_pair(devices[in_idx]["name"], devices[out_idx]["name"], self.cfg.allow_speakers)

        # Pin queries to names: indices shift when PortAudio re-enumerates.
        if self._in_stream is None:
            self._in_stream = self._open_input(in_idx)
            self.input_name = self._in_query = devices[in_idx]["name"]
            log.info("input:  #%d %s", in_idx, self.input_name)
        if out_idx is None:
            self.state = "waiting"
            return
        if self._out_stream is None:
            self._out_stream = self._open_output(out_idx, devices[out_idx]["outputs"])
            self.output_name = self._out_query = devices[out_idx]["name"]
            log.info("output: #%d %s", out_idx, self.output_name)
        self.state = "running"

    def _open_input(self, idx: int) -> sd.InputStream:
        stream = sd.InputStream(
            device=idx, samplerate=self.cfg.samplerate, blocksize=self.cfg.blocksize,
            channels=1, dtype="float32", latency=self.cfg.latency, callback=self._on_input,
        )
        stream.start()
        self.stats.in_latency = stream.latency
        self.stats.in_heartbeat = time.monotonic()
        self._grace_until = time.monotonic() + self.OPEN_GRACE_S
        return stream

    def _open_output(self, idx: int, max_channels: int) -> sd.OutputStream:
        self._out_primed = False
        stream = sd.OutputStream(
            # blocksize=0 lets CoreAudio pick its own buffer size (Bluetooth
            # devices are picky); the ring absorbs the size mismatch.
            device=idx, samplerate=self.cfg.samplerate, blocksize=0,
            channels=min(2, max_channels), dtype="float32", latency=self.cfg.latency,
            callback=self._on_output,
        )
        stream.start()
        self.stats.out_latency = stream.latency
        self.stats.out_heartbeat = time.monotonic()
        self._grace_until = time.monotonic() + self.OPEN_GRACE_S
        return stream

    def _supervise(self) -> None:
        while not self._stop.wait(self.POLL_S):
            try:
                self._reap_dead_streams()
                if self.state == "waiting" and time.monotonic() >= self._next_probe:
                    self._next_probe = time.monotonic() + self.PROBE_INTERVAL_S
                    self._try_reconnect()
            except (DeviceError, AudioSystemWedged) as exc:
                self.error, self.state = exc, "failed"
                return
            except Exception:
                log.exception("audio supervisor error")

    def _reap_dead_streams(self) -> None:
        now = time.monotonic()
        if now < self._grace_until:
            return
        st = self.stats
        if self._out_stream is not None and (
                not self._out_stream.active or now - st.out_heartbeat > self.STALE_S):
            log.warning("output %r stopped (asleep or disconnected); waiting for it", self.output_name)
            self._close_dead(self._out_stream)
            self._out_stream = None
            self.state = "waiting"
        if self._in_stream is not None and (
                not self._in_stream.active or now - st.in_heartbeat > self.STALE_S):
            log.warning("input %r stopped; waiting for it", self.input_name)
            self._close_dead(self._in_stream)
            self._in_stream = None
            self.state = "waiting"

    def _try_reconnect(self) -> None:
        devices = probe_devices()
        if devices is None:
            return
        try:
            find_device(devices, self._in_query, "input")
            find_device(devices, self._out_query, "output")
        except DeviceError:
            return  # not back yet

        log.info("devices available again; re-initialising PortAudio")
        self._close_streams()
        if self.wedged:
            # Pa_Terminate would block on the stream that failed to close.
            raise AudioSystemWedged("CoreAudio stopped responding while closing a stream; restart focus-ear")
        _reinit_portaudio()
        try:
            self._open_streams()
        except sd.PortAudioError as exc:
            # The device can be listed before its Bluetooth link is ready.
            # Keep whatever did open; the next probe starts over.
            log.warning("reopening streams failed (%s); will retry", exc)
            self.state = "waiting"

    # Audio callbacks (PortAudio threads) -----------------------------------
    def _on_input(self, indata, frames, time_info, status) -> None:
        st = self.stats
        now = time.monotonic()
        st.in_heartbeat = now
        if status.input_overflow:
            st.in_overflows += 1
        if time_info.inputBufferAdcTime and time_info.currentTime:
            adc = time_info.inputBufferAdcTime + (now - time_info.currentTime)
        else:
            adc = now - st.in_latency
        st.in_device_latency = now - adc
        self._in_anchor = (self.in_ring.write_index, adc)
        written = self.in_ring.write(indata[:, 0])
        if written < frames:
            st.in_dropped += frames - written

    def _on_output(self, outdata, frames, time_info, status) -> None:
        st = self.stats
        now = time.monotonic()
        st.out_heartbeat = now
        if status.output_underflow:
            st.out_underflows += 1
        ring = self.out_ring
        avail = ring.available()
        st.out_level_sum += avail
        st.out_level_count += 1

        # Hold `prefill` samples in reserve so worker jitter doesn't cause
        # dropouts. Until that much has built up (at startup, after an underrun,
        # after a reconnect) play silence. Anything beyond the reserve plus some
        # slack is stale (clock drift between mic and buds, or audio queued
        # while the buds were away) and gets dropped to keep latency bounded.
        if not self._out_primed:
            if avail < self._prefill + frames:
                outdata.fill(0)
                return
            self._out_primed = True
            slack = 0
        else:
            slack = self._drift_slack
        excess = avail - (self._prefill + frames)
        if excess > slack:
            st.out_skipped += ring.skip(excess)

        if frames > len(self._out_scratch):
            self._out_scratch = np.zeros(2 * frames, dtype=np.float32)
        mono = self._out_scratch[:frames]
        r = ring.read_index
        n = ring.read_into(mono)
        if n < frames:
            mono[n:] = 0.0
            self._out_primed = False
            st.out_underruns += 1
        outdata[:] = mono[:, None]

        anchor = self._out_anchor
        if n and anchor is not None:
            if time_info.outputBufferDacTime and time_info.currentTime:
                dac = time_info.outputBufferDacTime + (now - time_info.currentTime)
            else:
                dac = now + st.out_latency
            w0, t0 = anchor
            st.e2e_sum += dac - (t0 + (r - w0) / self.cfg.samplerate)
            st.e2e_count += 1
            st.out_device_latency = dac - now
