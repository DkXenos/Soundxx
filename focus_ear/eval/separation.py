"""Stage 7 experiment: can a pretrained separation model pull the lecturer out of overlapping speech?

    python -m focus_ear.eval.separation [--out eval/results/separation] [--model speechbrain/sepformer-whamr16k]

Test set: the lecturer (Damayanti, reverberant, far) overlapped by a student (Amira or Daniel, near) at
signal-to-interference ratios of +5, 0 and -5 dB, in room noise. The separated output closest to the
lecturer's voiceprint (ECAPA, from a clean stretch of the lecturer, as an enrolled profile would be) is
taken as "the lecturer".

Measured against doing nothing (the mixture):
  * SI-SDR improvement of the lecturer (dB)
  * WER of the lecturer's words (whisper-small, temperature 0, the live model)
  * real-time factor and peak memory of the separation model, on this Mac

Decision rule, fixed before the run: separation becomes a live AudioStage behind a switch only if
  RTF <= 0.5 on 1 s chunks, added latency <= 200 ms, peak memory <= 1 GB, and SI-SDRi >= 6 dB at 0 dB SIR;
otherwise it stays an offline result. The rule is applied in the report.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from .metrics import wer
from .synth import RATE, VOICES, rir, room_noise, rms, tts

ROOT = Path(__file__).resolve().parents[2]
MODEL = "speechbrain/sepformer-whamr16k"

PAIRS = [
    ("Jadi kalau kita pakai array, kita bisa akses elemen dengan index secara langsung.",
     "Mahasiswa 1", "Pak, kalau linked list itu bedanya apa dengan array?"),
    ("Linked list itu terdiri dari node, dan setiap node punya pointer ke node berikutnya.",
     "Mahasiswa 2", "Excuse me, does that mean searching a linked list is always linear time?"),
    ("Stack itu last in first out, cocok untuk undo atau recursion.",
     "Mahasiswa 1", "Kalau head-nya kosong, function-nya return apa, Pak?"),
    ("Kalau head-nya None, kita langsung return node baru itu sebagai head.",
     "Mahasiswa 2", "Could you explain when we should use a stack instead of a queue?"),
]
ENROLL = "Selamat pagi semuanya. Hari ini kita lanjut materi struktur data, khususnya linked list dan array."


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    alpha = float(est @ ref) / max(float(ref @ ref), 1e-12)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10(max(float(target @ target), 1e-12) / max(float(noise @ noise), 1e-12))


def build_mixtures(rng: np.random.Generator, tmp: Path) -> tuple[list[dict], np.ndarray]:
    room_far, room_near = rir(0.7, rng, direct=0.6), rir(0.35, rng, direct=2.0)
    enroll = np.convolve(tts(ENROLL, VOICES["Dosen"], tmp), room_far)
    mixes = []
    for lect_text, who, stud_text in PAIRS:
        lect = np.convolve(tts(lect_text, VOICES["Dosen"], tmp), room_far).astype(np.float32)
        stud = np.convolve(tts(stud_text, VOICES[who], tmp), room_near).astype(np.float32)
        n = max(len(lect), len(stud) + RATE // 2)
        lect = np.pad(lect, (0, n - len(lect)))
        stud = np.pad(stud, (RATE // 2, n - len(stud) - RATE // 2))  # the student starts half a second in
        for sir in (5.0, 0.0, -5.0):
            s = stud * rms(lect) / max(rms(stud), 1e-9) / (10 ** (sir / 20))
            noise = room_noise(n, rng) * rms(lect) / (10 ** (15 / 20))  # 15 dB SNR room
            mix = lect + s + noise
            peak = max(np.abs(mix).max(), 1e-9)
            mixes.append({"sir": sir, "student": who, "lect_text": lect_text, "mix": (mix / peak * 0.5).astype(np.float32),
                          "lect": (lect / peak * 0.5).astype(np.float32)})
    return mixes, enroll.astype(np.float32)


class Separator:
    def __init__(self, model: str, device: str):
        import torch
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy

        self.torch = torch
        save = Path.home() / ".focus-ear" / "models" / model.split("/")[-1]
        self.model = SepformerSeparation.from_hparams(source=model, savedir=str(save), run_opts={"device": device},
                                                      local_strategy=LocalStrategy.COPY)
        self.model.eval()
        self.device = device

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            t = self.torch.from_numpy(x).unsqueeze(0).to(self.device)
            est = self.model.separate_batch(t)  # [1, time, sources]
        return est[0].cpu().numpy().T  # [sources, time]


def main(argv: list[str] | None = None) -> int:
    from ..embeddings import EcapaEmbedder
    from ..health import memory_mb
    from ..post.common import PeakMemory

    p = argparse.ArgumentParser(prog="focus_ear.eval.separation")
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "separation")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", default="cpu", help="cpu or mps")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    with tempfile.TemporaryDirectory() as tmp:
        mixes, enroll = build_mixtures(rng, Path(tmp))
    ecapa = EcapaEmbedder("cpu")
    target = ecapa(enroll)

    import mlx_whisper  # the live model, at temperature 0
    from ..asr import _local_or_download
    from ..config import DEFAULT_PROMPT
    whisper = _local_or_download("mlx-community/whisper-small-mlx")

    def transcribe(x: np.ndarray) -> str:
        return mlx_whisper.transcribe(x, path_or_hf_repo=whisper, language="id", temperature=0.0,
                                      initial_prompt=DEFAULT_PROMPT, condition_on_previous_text=False)["text"]

    base_mem = memory_mb()
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        sep = Separator(args.model, args.device)
        load_s = time.perf_counter() - t0
        # Streaming cost: 1 s chunks, as a live stage would see them.
        chunk = np.ascontiguousarray(mixes[0]["mix"][:RATE])
        sep(chunk)
        t0 = time.perf_counter()
        for _ in range(5):
            sep(chunk)
        rtf_1s = (time.perf_counter() - t0) / 5
        rows = []
        for m in mixes:
            t0 = time.perf_counter()
            outs = sep(m["mix"])
            secs = time.perf_counter() - t0
            sims = [float(ecapa(o.astype(np.float32)) @ target) for o in outs]
            est = outs[int(np.argmax(sims))]
            est = est / max(np.abs(est).max(), 1e-9) * 0.5
            rows.append({
                "sir_db": m["sir"], "student": m["student"],
                "si_sdr_mix": si_sdr(m["mix"], m["lect"]), "si_sdr_sep": si_sdr(est, m["lect"]),
                "picked_similarity": max(sims), "other_similarity": min(sims),
                "wer_mix": wer(m["lect_text"], transcribe(m["mix"])).rate,
                "wer_sep": wer(m["lect_text"], transcribe(est.astype(np.float32))).rate,
                "rtf": secs / (len(m["mix"]) / RATE)})
    peak_extra = mem.peak_mb - base_mem

    by_sir = {}
    for sir in (5.0, 0.0, -5.0):
        rs = [r for r in rows if r["sir_db"] == sir]
        by_sir[sir] = {k: float(np.mean([r[k] for r in rs])) for k in rows[0] if k not in ("sir_db", "student")}
    sisdri_0 = by_sir[0.0]["si_sdr_sep"] - by_sir[0.0]["si_sdr_mix"]
    latency_ms = 1000 * (1.0 + rtf_1s)  # a 1 s chunk must be collected, then separated
    rule = {"rtf_1s_chunks <= 0.5": rtf_1s <= 0.5, "added latency <= 200 ms": latency_ms <= 200,
            "peak memory <= 1 GB": peak_extra <= 1024, "SI-SDRi >= 6 dB at 0 dB SIR": sisdri_0 >= 6}
    verdict = "live stage (behind a switch)" if all(rule.values()) else "offline only"
    result = {"model": args.model, "device": args.device, "load_s": load_s, "rtf_1s_chunk": rtf_1s,
              "latency_ms_estimate": latency_ms, "peak_extra_mb": peak_extra, "by_sir": by_sir, "rows": rows,
              "rule": rule, "verdict": verdict, "synthetic": True}
    (args.out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    md = [f"# Overlapping speech: {args.model} on {args.device}", "",
          "**Synthetic test set** (TTS voices, simulated rooms): the lecturer overlapped by a student. "
          "Numbers compare against doing nothing; they don't predict a real hall.", "",
          "| Student louder by | SI-SDR mixture | SI-SDR separated | Improvement | Lecturer WER mixture | Lecturer WER separated |",
          "|---|---|---|---|---|---|"]
    for sir, v in by_sir.items():
        md.append(f"| {-sir:+g} dB | {v['si_sdr_mix']:.1f} dB | {v['si_sdr_sep']:.1f} dB | "
                  f"{v['si_sdr_sep'] - v['si_sdr_mix']:+.1f} dB | {100 * v['wer_mix']:.0f}% | {100 * v['wer_sep']:.0f}% |")
    md += ["", f"Speed on this Mac: {rtf_1s * 1000:.0f} ms to separate a 1 s chunk (real-time factor {rtf_1s:.2f}); "
               f"whole clips at RTF {np.mean([r['rtf'] for r in rows]):.2f}. Model load {load_s:.1f} s. "
               f"Extra memory while running: {peak_extra:.0f} MB.", "",
           "Decision rule (fixed before the run):", ""]
    md += [f"- {'✓' if ok else '✗'} {name}" for name, ok in rule.items()]
    md += ["", f"**Verdict: {verdict}.**", ""]
    (args.out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
