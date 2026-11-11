"""A synthetic code-mixed lecture with exact ground truth, for when there's no real one.

    python -m focus_ear.eval.synth OUT_DIR [--snr 10] [--repeat 1] [--seed 0]

macOS text-to-speech voices: Damayanti (id_ID) is the lecturer, far away in
a reverberant hall; Amira (ms_MY) asks in Indonesian and Daniel (en_GB) in
English, both close to the mic. Room noise (pink noise, a 50 Hz hum, fan
rumble) is added at ``--snr`` dB relative to the lecturer.

Writes OUT_DIR/audio.wav (16 kHz mono) and OUT_DIR/reference.json:
    {"turns": [{"start", "end", "speaker"}], "lines": [{"start", "end", "speaker", "text"}],
     "terms": [...English technical terms...], "speakers": {...}, "snr_db": ...}

Every number this produces is about TTS voices, not people: it checks that
the machinery works and compares settings, it doesn't predict a real hall.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

RATE = 16_000

VOICES = {"Dosen": "Damayanti", "Mahasiswa 1": "Amira", "Mahasiswa 2": "Daniel"}

SCRIPT = [
    ("Dosen", "Selamat pagi semuanya. Hari ini kita lanjut materi struktur data, khususnya linked list dan array."),
    ("Dosen", "Jadi kalau kita pakai array, kita bisa akses elemen dengan index secara langsung."),
    ("Dosen", "Kompleksitas waktunya O of one, karena alamat memorinya bisa dihitung."),
    ("Dosen", "Tapi kalau kita mau insert di tengah, kita harus shift semua elemen ke kanan."),
    ("Mahasiswa 1", "Pak, kalau linked list itu bedanya apa dengan array?"),
    ("Dosen", "Pertanyaan bagus. Linked list itu terdiri dari node, dan setiap node punya pointer ke node berikutnya."),
    ("Dosen", "Jadi insert dan delete bisa O of one, asal kita sudah pegang pointer ke posisinya."),
    ("Dosen", "Kekurangannya, untuk akses elemen ke n, kita harus traverse dari head satu per satu."),
    ("Mahasiswa 2", "Excuse me, does that mean searching a linked list is always linear time?"),
    ("Dosen", "Yes, betul. Searching di linked list itu O of n, karena tidak ada random access."),
    ("Dosen", "Sekarang kita lihat contoh code-nya. Kita define class Node dulu, dengan atribut data dan next."),
    ("Dosen", "Lalu kita buat function insert, yang menerima head dan value baru, terus return head yang baru."),
    ("Mahasiswa 1", "Kalau head-nya kosong, function-nya return apa, Pak?"),
    ("Dosen", "Kalau head-nya None, kita langsung return node baru itu sebagai head."),
    ("Dosen", "Nah, untuk doubly linked list, setiap node punya dua pointer, yaitu prev dan next."),
    ("Dosen", "Ini memudahkan kita untuk traverse ke belakang, tapi butuh memory lebih banyak."),
    ("Mahasiswa 2", "Could you explain when we should use a stack instead of a queue?"),
    ("Dosen", "Stack itu last in first out, cocok untuk undo atau recursion. Queue itu first in first out."),
    ("Dosen", "Contohnya, breadth first search pakai queue, sedangkan depth first search bisa pakai stack."),
    ("Dosen", "Oke, untuk tugas minggu depan, implementasikan stack menggunakan linked list di Python."),
    ("Mahasiswa 1", "Dikumpulkan lewat elearning ya, Pak?"),
    ("Dosen", "Iya, lewat elearning, paling lambat hari Jumat jam dua belas malam. Terima kasih semuanya."),
]

TERMS = ["linked list", "array", "index", "insert", "shift", "node", "pointer", "delete", "traverse", "head",
         "searching", "random access", "code", "define", "class", "function", "return", "value", "None",
         "doubly linked list", "prev", "next", "memory", "stack", "queue", "recursion", "breadth first search",
         "depth first search", "Python", "elearning", "undo"]


def tts(text: str, voice: str, tmp: Path) -> np.ndarray:
    aiff = tmp / "say.aiff"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    x, fs = sf.read(aiff, dtype="float32", always_2d=True)
    x = soxr.resample(x[:, 0], fs, RATE).astype(np.float32)
    return trim(x)


def trim(x: np.ndarray, thresh_db: float = -45.0) -> np.ndarray:
    frame = 160
    n = len(x) // frame
    e = 20 * np.log10(np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9)
    loud = np.flatnonzero(e > e.max() + thresh_db)
    if not len(loud):
        return x
    return x[loud[0] * frame:(loud[-1] + 1) * frame]


def rir(rt60: float, rng: np.random.Generator, length_s: float = 0.8, direct: float = 1.0) -> np.ndarray:
    """A synthetic room impulse response: a direct path, then exponentially decaying noise."""
    n = int(length_s * RATE)
    t = np.arange(n) / RATE
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * 0.25
    h[:int(0.003 * RATE)] = 0
    h[0] = direct
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def room_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec /= np.sqrt(np.maximum(f, 20.0))  # pink
    pink = np.fft.irfft(spec, n)
    t = np.arange(n) / RATE
    hum = 0.3 * np.sin(2 * np.pi * 50 * t) + 0.15 * np.sin(2 * np.pi * 100 * t)
    rumble = np.convolve(rng.standard_normal(n), np.ones(400) / 400, mode="same") * 3
    x = pink / pink.std() + hum + rumble
    return (x / x.std()).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def build(out: Path, snr_db: float = 10.0, repeat: int = 1, seed: int = 0, gap_s: float = 0.6) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    rooms = {"Dosen": rir(0.7, rng, direct=0.6), "Mahasiswa 1": rir(0.35, rng, direct=2.0),
             "Mahasiswa 2": rir(0.35, rng, direct=2.0)}
    gains = {"Dosen": 0.5, "Mahasiswa 1": 0.9, "Mahasiswa 2": 0.8}  # the lecturer is far, students near
    pieces, lines, turns = [], [], []
    pos = int(1.0 * RATE)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for r in range(repeat):
            for speaker, text in SCRIPT:
                dry = tts(text, VOICES[speaker], tmp)
                wet = np.convolve(dry, rooms[speaker])[:len(dry) + int(0.3 * RATE)] * gains[speaker]
                pieces.append((pos, wet.astype(np.float32)))
                start, end = pos / RATE, (pos + len(dry)) / RATE
                lines.append({"start": round(start, 3), "end": round(end, 3), "speaker": speaker, "text": text})
                if turns and turns[-1]["speaker"] == speaker and start - turns[-1]["end"] < 1.0:
                    turns[-1]["end"] = round(end, 3)
                else:
                    turns.append({"start": round(start, 3), "end": round(end, 3), "speaker": speaker})
                pause = gap_s + (rng.uniform(0.2, 0.8) if speaker != "Dosen" else rng.uniform(0.0, 0.4))
                pos += len(dry) + int(pause * RATE)
    total = pos + int(1.0 * RATE)
    clean = np.zeros(total, np.float32)
    for at, x in pieces:
        clean[at:at + len(x)] += x[:total - at]
    lecturer = np.concatenate([x for (at, x), ln in zip(pieces, lines) if ln["speaker"] == "Dosen"])
    noise = room_noise(total, rng) * rms(lecturer) / (10 ** (snr_db / 20))
    mix = clean + noise
    mix *= 0.25 / max(np.abs(mix).max(), 1e-9)
    sf.write(out / "audio.wav", mix, RATE, subtype="PCM_16")
    ref = {"turns": turns, "lines": lines, "terms": TERMS, "speakers": {k: f"{v} (macOS TTS)" for k, v in VOICES.items()},
           "snr_db": snr_db, "duration_s": round(total / RATE, 2), "synthetic": True}
    (out / "reference.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    return ref


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="focus_ear.eval.synth", description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--snr", type=float, default=10.0, help="dB, lecturer speech over room noise (default 10)")
    p.add_argument("--repeat", type=int, default=1, help="repeat the script to make it longer")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    ref = build(args.out, args.snr, args.repeat, args.seed)
    print(f"{args.out / 'audio.wav'}: {ref['duration_s']:.0f} s, {len(ref['lines'])} lines, {len(ref['turns'])} turns, "
          f"SNR {args.snr:g} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
