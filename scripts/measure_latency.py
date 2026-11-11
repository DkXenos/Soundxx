"""Measure the real mic-to-ear latency, Bluetooth included, by ear-to-mic loopback.

    1. Start a recording in Soundxx (or scripts/record.command), playing into the Buds, listen mode.
    2. Hold one Bud against the MacBook's microphone (top left of the keyboard, by the left speaker grille).
    3. venv/bin/python scripts/measure_latency.py [--clicks 8]

The script plays short clicks from the MacBook speakers and records the mic at the same time. Each
click reaches the mic twice: straight from the speaker, then again out of the Bud after going through
focus-ear and Bluetooth. The gap between the two is the latency you hear. (CoreAudio can't report
Bluetooth's own delay, so the app's "measured latency" stops at the Mac.)

The speaker is used only for these clicks (about one second in total, quiet).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sounddevice as sd

RATE = 48_000


def click(n: int = 96) -> np.ndarray:
    t = np.arange(n) / RATE
    return (np.sin(2 * np.pi * 3000 * t) * np.hanning(n)).astype(np.float32)


def onsets(x: np.ndarray, start: int, window: int, k: int = 2) -> list[int]:
    """The ``k`` strongest separated peaks of the envelope in x[start:start+window]."""
    seg = np.abs(x[start:start + window])
    env = np.convolve(seg, np.ones(48) / 48, mode="same")
    found = []
    env = env.copy()
    for _ in range(k):
        i = int(np.argmax(env))
        if env[i] <= 0:
            break
        found.append(start + i)
        env[max(0, i - RATE // 20):i + RATE // 20] = 0  # 50 ms exclusion around a peak
    return sorted(found)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clicks", type=int, default=8)
    p.add_argument("--gap", type=float, default=1.5, help="seconds between clicks")
    p.add_argument("--volume", type=float, default=0.2)
    p.add_argument("--speaker", default="MacBook Pro Speakers")
    p.add_argument("--mic", default="MacBook Pro Microphone")
    args = p.parse_args(argv)
    n = int((args.clicks * args.gap + 2.0) * RATE)
    play = np.zeros(n, np.float32)
    at = [int((0.5 + i * args.gap) * RATE) for i in range(args.clicks)]
    c = click() * args.volume
    for a in at:
        play[a:a + len(c)] = c
    print("Playing clicks; keep a Bud on the mic and the room quiet…")
    rec = sd.playrec(play[:, None], samplerate=RATE, channels=1, device=(args.mic, args.speaker), dtype="float32")
    sd.wait()
    x = rec[:, 0]
    lags = []
    for a in at:
        peaks = onsets(x, a, int(min(args.gap, 1.2) * RATE))
        if len(peaks) == 2:
            lags.append((peaks[1] - peaks[0]) / RATE * 1000)
    if len(lags) < max(2, args.clicks // 2):
        print(f"Found the echo for only {len(lags)} of {args.clicks} clicks. Is focus-ear running and playing "
              "into the Bud, and is the Bud right on the mic?")
        return 1
    lags = np.array(lags)
    med = float(np.median(lags))
    good = lags[np.abs(lags - med) < 50]
    print(f"Mic to ear, Bluetooth included: {np.median(good):.0f} ms (range {good.min():.0f}–{good.max():.0f} ms, "
          f"{len(good)} of {args.clicks} clicks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
