"""Click-free gain control: the noise gate (Stage 2) and per-speaker gain (Stage 4).

Both are built on SmoothGain. Everything here runs on the pipeline worker thread.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .pipeline import BlockContext


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def gain_to_db(gain: float) -> float:
    return 20.0 * math.log10(gain) if gain > 1e-10 else -200.0


class SmoothGain:
    """A gain that never jumps: every change of target is a linear ramp over ``ramp_ms``.

    Retargeting mid-ramp starts a fresh ramp from wherever the gain currently
    is, so the gain curve stays continuous whatever the caller does.
    """

    def __init__(self, samplerate: int, ramp_ms: float = 50.0, gain: float = 1.0):
        self.ramp_samples = max(1, round(samplerate * ramp_ms / 1000))
        self._current = self._target = float(gain)
        self._step = 0.0
        self._remaining = 0

    @property
    def current(self) -> float:
        return self._current

    @property
    def target(self) -> float:
        return self._target

    def set_target(self, gain: float) -> None:
        if gain == self._target:
            return
        self._target = float(gain)
        self._step = (self._target - self._current) / self.ramp_samples
        self._remaining = self.ramp_samples

    def set_target_db(self, db: float) -> None:
        self.set_target(db_to_gain(db))

    def reset(self, gain: float | None = None) -> None:
        """Jump to ``gain`` (default: the current target) with no ramp."""
        self._current = self._target = self._target if gain is None else float(gain)
        self._remaining = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        """Apply the gain to ``block`` in place and return it."""
        n = len(block)
        k = min(n, self._remaining)
        if k:
            ramp = self._current + self._step * np.arange(1, k + 1)
            self._remaining -= k
            if self._remaining == 0:
                ramp[-1] = self._target  # land exactly: no float residue, so unity stays bit-exact
            block[:k] *= ramp
            self._current = float(ramp[-1])
        if k < n and self._current != 1.0:
            block[k:] *= self._current
        return block


class DelayLine:
    """Fixed delay. Used as lookahead: a gain change decided from the newest
    audio takes effect before that audio reaches the output."""

    def __init__(self, samples: int):
        self._buf = np.zeros(max(0, samples), dtype=np.float32)

    @property
    def samples(self) -> int:
        return len(self._buf)

    def process(self, block: np.ndarray) -> np.ndarray:
        if not len(self._buf):
            return block
        joined = np.concatenate((self._buf, block))
        self._buf = joined[len(block):]
        return joined[:len(block)]

    def reset(self) -> None:
        self._buf[:] = 0.0


class NoiseGate:
    """Pipeline AudioStage: unity gain during speech, ``attenuation_db`` otherwise.

    Opens as soon as the VAD reports speech and stays open for
    ``hangover_ms`` after it stops, so word and sentence endings aren't
    clipped. Every open/close goes through a SmoothGain ramp.
    ``lookahead_ms`` delays the audio so the gate can open before a word's
    onset arrives, at the cost of that much extra latency.
    """

    name = "gate"

    def __init__(self, samplerate: int, attenuation_db: float = -18.0, hangover_ms: float = 200.0,
                 ramp_ms: float = 50.0, lookahead_ms: float = 0.0):
        self.closed_gain = db_to_gain(attenuation_db)
        self.hangover = round(samplerate * hangover_ms / 1000)
        self.gain = SmoothGain(samplerate, ramp_ms, gain=self.closed_gain)
        self.delay = DelayLine(round(samplerate * lookahead_ms / 1000))
        self._hold = 0  # samples of hangover left

    @property
    def delay_samples(self) -> int:
        return self.delay.samples

    @property
    def is_open(self) -> bool:
        return self._hold > 0

    @property
    def gain_db(self) -> float:
        return gain_to_db(self.gain.current)

    def process(self, block: np.ndarray, ctx: BlockContext) -> np.ndarray:
        if ctx.is_speech:
            self._hold = self.hangover
        else:
            self._hold = max(0, self._hold - len(block))
        self.gain.set_target(1.0 if self._hold > 0 else self.closed_gain)
        return self.gain.process(self.delay.process(block))

    def reset(self) -> None:
        self._hold = 0
        self.gain.reset(self.closed_gain)
        self.delay.reset()


class SpeakerGain:
    """Pipeline AudioStage (Stage 4): full volume for the selected speaker, the rest turned down.

    The gain for each block comes from ``target_for(ctx.speaker_id)``, which
    returns a linear gain, or None to keep the current one. Changes ramp through a SmoothGain like the gate's.

    Speaker identity arrives late: a window's embedding exists only after the
    window has been heard, and a new voice has to fill most of the window
    before it wins. ``lookahead_ms`` delays the audio by that much so each
    decision lands closer to the audio it was computed from. Every
    millisecond of it is added latency. ~250 ms narrows the lag at a change
    of speaker; ~900 ms (for a 1.5 s window) all but closes it.
    """

    name = "speaker-gain"

    def __init__(self, samplerate: int, target_for, lookahead_ms: float = 250.0, ramp_ms: float = 50.0):
        self.target_for = target_for
        self.delay = DelayLine(round(samplerate * lookahead_ms / 1000))
        self.gain = SmoothGain(samplerate, ramp_ms, gain=1.0)

    @property
    def delay_samples(self) -> int:
        return self.delay.samples

    @property
    def gain_db(self) -> float:
        return gain_to_db(self.gain.current)

    def process(self, block: np.ndarray, ctx: BlockContext) -> np.ndarray:
        target = self.target_for(ctx.speaker_id)
        if target is not None:
            self.gain.set_target(target)
        return self.gain.process(self.delay.process(block))

    def reset(self) -> None:
        self.gain.reset(1.0)
        self.delay.reset()
