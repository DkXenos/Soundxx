"""Audio quality: input high-pass, output AGC and limiter, and clip detection.

All of it runs on the pipeline worker thread, block by block, vectorised
with numpy/scipy (the limiter's release is the one short Python loop, over
~44 chunks per block rather than 1411 samples).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.signal import butter, sosfilt

from .gain import DelayLine, SmoothGain, db_to_gain, gain_to_db

if TYPE_CHECKING:
    from .pipeline import BlockContext


class HighPass:
    """4th-order Butterworth high-pass: removes AC rumble (50/60 Hz) and handling noise.

    At the default 80 Hz it takes 50 Hz down ~16 dB and 60 Hz ~10 dB, and
    leaves speech (fundamentals from ~85 Hz up) essentially untouched above
    ~150 Hz. Streaming: the filter state carries over between blocks.
    """

    name = "highpass"

    def __init__(self, rate: int, cutoff_hz: float = 80.0, order: int = 4):
        self.sos = butter(order, cutoff_hz, btype="highpass", fs=rate, output="sos")
        self._zi = np.zeros((self.sos.shape[0], 2))

    def process(self, block: np.ndarray, ctx: BlockContext | None = None) -> np.ndarray:
        y, self._zi = sosfilt(self.sos, block, zi=self._zi)
        return y.astype(np.float32)

    def reset(self) -> None:
        self._zi[:] = 0.0


class Agc:
    """Output AudioStage: slowly levels the speech you're listening to towards ``target_db``.

    Adapts only during speech that plays at full volume (``adapt(ctx)``):
    attenuated speakers and silence never pull the gain around, so the room
    tone isn't pumped up in pauses. The level is a 1 s power average; the gain
    follows ``target - level`` within [min_gain_db, max_gain_db], moving at
    most ``up_db_per_s`` upwards and ``down_db_per_s`` downwards, ramped
    across each block. Sudden peaks (a cough) are the limiter's job.
    """

    name = "agc"

    def __init__(self, rate: int, blocksize: int, target_db: float = -23.0, max_gain_db: float = 24.0,
                 min_gain_db: float = -12.0, tau_s: float = 1.0, up_db_per_s: float = 6.0,
                 down_db_per_s: float = 20.0, adapt: Callable[[BlockContext], bool] = lambda ctx: True):
        self.rate = rate
        self.target_db = target_db
        self.max_gain_db, self.min_gain_db = max_gain_db, min_gain_db
        self.tau_s = tau_s
        self.up, self.down = up_db_per_s, down_db_per_s
        self.adapt = adapt
        self.gain = SmoothGain(rate, 1e3 * blocksize / rate, gain=1.0)
        self.reset()

    @property
    def gain_db(self) -> float:
        return gain_to_db(self.gain.current)

    def process(self, block: np.ndarray, ctx: BlockContext) -> np.ndarray:
        dt = len(block) / self.rate
        if ctx.is_speech and self.adapt(ctx):
            power = float(np.mean(block * block))
            a = math.exp(-dt / self.tau_s)
            self._power = power if self._power is None else a * self._power + (1 - a) * power
            level_db = 10 * math.log10(max(self._power, 1e-12))
            self.level_db = level_db
            want = min(self.max_gain_db, max(self.min_gain_db, self.target_db - level_db))
            step = want - self._gain_db
            self._gain_db += max(-self.down * dt, min(self.up * dt, step))
            self.gain.set_target(db_to_gain(self._gain_db))
        return self.gain.process(block)

    def reset(self) -> None:
        self._power: float | None = None
        self._gain_db = 0.0
        self.level_db = -200.0
        self.gain.reset(1.0)


class Limiter:
    """Output AudioStage: lookahead peak limiter. No sample leaves above ``ceiling_db``.

    For each output sample the gain is the smallest needed anywhere in the
    next ``lookahead_ms`` (the audio is delayed by that much), smoothed by a
    moving average of the same length. That construction never exceeds the
    gain a peak needs, so the ceiling holds by design. The gain then comes
    back up over ``release_ms``. Below the ceiling the gain is exactly 1: the
    signal passes bit for bit, just delayed.
    """

    name = "limiter"
    CHUNK = 32  # release envelope resolution (samples); steps are < 0.1 dB
    ONE_DB = 10 ** (-1 / 20)

    def __init__(self, rate: int, ceiling_db: float = -6.0, lookahead_ms: float = 2.0, release_ms: float = 80.0):
        self.ceiling = db_to_gain(ceiling_db)
        self.L = max(1, round(rate * lookahead_ms / 1000))
        self.delay = DelayLine(self.L)
        self._a = math.exp(-self.CHUNK / (rate * release_ms / 1000))
        self.reset()
        self.limited = 0   # samples turned down by more than 1 dB (not counting release tails); only increases

    @property
    def delay_samples(self) -> int:
        return self.L

    @property
    def gain_db(self) -> float:
        return gain_to_db(self._env)

    def process(self, block: np.ndarray, ctx: BlockContext | None = None) -> np.ndarray:
        n, L = len(block), self.L
        req = np.minimum(1.0, self.ceiling / np.maximum(np.abs(block), 1e-12))
        req_all = np.concatenate((self._req, req))                     # input samples n0-L .. n0+n-1
        self._req = req_all[-L:]
        # h[k]: least gain needed by any of the input samples k-L .. k (sliding min over L+1).
        h = np.lib.stride_tricks.sliding_window_view(req_all, L + 1).min(axis=1)  # n values
        h_all = np.concatenate((self._h, h))
        self._h = h_all[-(L - 1):] if L > 1 else h_all[:0]
        # Mean of h over the last L: n values. Window sums rather than a running
        # cumsum, whose rounding left the gain at 0.9999999 forever after the first peak.
        s = np.lib.stride_tricks.sliding_window_view(h_all, L).sum(axis=1) / L

        if s.min() < 1.0 or self._env < 1.0:
            g = np.empty(n)
            env, a = self._env, self._a
            for i in range(0, n, self.CHUNK):                           # attack: follow s; release: slowly
                seg = s[i:i + self.CHUNK]
                env = min(float(seg.min()), 1.0 - (1.0 - env) * a)
                if env > 1.0 - 1e-4:  # -0.0009 dB: snap to exactly 1 so the fast path resumes
                    env = 1.0
                g[i:i + self.CHUNK] = np.minimum(seg, env)
            self._env = env
            self.limited += int(np.count_nonzero(g < self.ONE_DB))
            out = (self.delay.process(block) * g).astype(np.float32)
        else:
            out = self.delay.process(block)
        return out

    def reset(self) -> None:
        self._req = np.ones(self.L)
        self._h = np.ones(max(0, self.L - 1))
        self._env = 1.0
        self.delay.reset()


class ClipMeter:
    """Counts samples at or beyond ``limit`` at one point in the chain."""

    def __init__(self, name: str, limit: float = 1.0):
        self.name = name
        self.limit = limit
        self.count = 0      # only increases
        self.peak = 0.0     # largest |sample| seen

    def check(self, x: np.ndarray) -> int:
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if peak > self.peak:
            self.peak = peak
        if peak < self.limit:
            return 0
        n = int(np.count_nonzero(np.abs(x) >= self.limit))
        self.count += n
        return n
