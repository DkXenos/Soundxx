"""Gain control (phase 2, not implemented yet).

Plan: per-sample gain that ramps linearly (in dB) to its target over ~50 ms,
so a speaker change never switches gain abruptly (abrupt steps click).
Targets: selected speaker -> boost_db, other speakers -> attenuation_db,
non-speech -> attenuation_db, passthrough -> 0 dB. A soft limiter after the
boost keeps the output below 0 dBFS.
"""
from __future__ import annotations

import numpy as np


class GainController:
    def __init__(self, samplerate: int = 16_000, ramp_ms: float = 50.0):
        self.samplerate = samplerate
        self.ramp_ms = ramp_ms

    def set_target_db(self, db: float) -> None:
        raise NotImplementedError("gain arrives in phase 2")

    def process(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError("gain arrives in phase 2")
