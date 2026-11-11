"""Evaluation: how good are the transcripts and speaker labels, and do the settings help?

    python -m focus_ear.eval run CASE [CASE ...] [--out eval/results/NAME] [--live-runs 3] [--quick]
    python -m focus_ear.eval reference SESSION [--from SEC] [--to SEC]
    python -m focus_ear.eval stability SESSION [SESSION ...]

A CASE is a folder with audio.wav and reference.json:
    {"lines": [{"start", "end", "speaker", "text"}], "turns": [...optional...], "terms": [...optional...],
     "range": [start, end] (optional: score only this stretch), "synthetic": bool}
focus_ear.eval.synth makes synthetic ones; ``reference`` makes one from a session you corrected in the app
(its corrected transcript.json; the turns are the machine's line times with your labels, so DER there
measures labelling, not boundaries).

Per case it compares, on the same audio:

  live            the live chain replayed over the file (whisper-small, online clustering, as in class);
                  it samples when greedy decoding fails, so it runs --live-runs times: mean and range
  final           the after-class pass as shipped (temperature 0: one run is the answer)
  final, no prompt               without the code-mixed prompt sentence
  final, Indonesian only         English detection off (every chunk decoded as Indonesian)
  final, denoised                DeepFilterNet over the audio first (what --denoise-scope all/models feeds)

plus sweeps that reuse saved model outputs (no re-inference): offline clustering threshold, VAD threshold.
Writes results.json and report.md (tables) to --out.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .metrics import cer, der, lines_to_turns, term_recall, wer

log = logging.getLogger("focus_ear.eval")
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- cases

def load_case(folder: Path) -> dict:
    ref = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
    ref.setdefault("turns", lines_to_turns(ref["lines"]))
    ref.setdefault("terms", [])
    return ref


def clip(items: list[dict], rng: list[float] | None) -> list[dict]:
    if not rng:
        return items
    a, b = rng
    return [x for x in items if a <= (x["start"] + x["end"]) / 2 <= b]


def score(ref: dict, lines: list[dict]) -> dict:
    rng = ref.get("range")
    r_lines, h_lines = clip(ref["lines"], rng), clip(lines, rng)
    R = " ".join(x["text"] for x in sorted(r_lines, key=lambda x: x["start"]))
    H = " ".join(x["text"] for x in sorted(h_lines, key=lambda x: x["start"]))
    d = der(clip(ref["turns"], rng), lines_to_turns(h_lines))
    lecturer = max(d["per_speaker"].items(), key=lambda kv: kv[1]["speech_s"])[1] if d["per_speaker"] else {}
    return {"wer": wer(R, H).rate, "cer": cer(R, H).rate, "terms": term_recall(ref["terms"], R, H)["recall"],
            "der": d["der"], "speakers": d["speakers_hyp"], "speakers_ref": d["speakers_ref"],
            "lecturer_coverage": lecturer.get("coverage"), "lecturer_purity": lecturer.get("purity"),
            "lecturer_split": lecturer.get("split_into"), "lines": len(h_lines)}


# ---------------------------------------------------------------- systems

def live_replay(audio_path: Path) -> list[dict]:
    """The live chain over a file, as main.run_file runs it, returning lines instead of writing files."""
    import types

    import soundfile as sf

    from ..config import Config
    from ..main import load_speaker_models, load_transcriber, make_segmenter
    from ..pipeline import Pipeline
    from ..vad import VadAnalyzer

    cfg = Config.for_mode("full", save=False)
    info = sf.info(str(audio_path))
    fs = info.samplerate
    bs = round(fs * cfg.block_ms / 1000)
    tracker, analyzer, _ = load_speaker_models(cfg, background=False)
    tr, err = load_transcriber(cfg, tracker, live=False)
    if tr is None:
        raise RuntimeError(err)
    seg = make_segmenter(cfg, tr)
    pipe = Pipeline(types.SimpleNamespace(samplerate=fs, blocksize=bs), cfg,
                    analyzers=[a for a in (VadAnalyzer(cfg.vad_threshold), analyzer, seg) if a is not None])
    out, pos = [], 0
    for block in sf.blocks(str(audio_path), blocksize=bs, dtype="float32", always_2d=True):
        mono = np.ascontiguousarray(block.mean(axis=1))
        pipe.process_block(mono, adc_time=pos / fs)
        pos += len(mono)
    seg.flush()
    tr.close()
    for ln in tr.drain_lines():
        out.append({"start": ln.start, "end": ln.end, "speaker": ln.speaker_id if ln.speaker_id is not None else "?",
                    "text": ln.display_text})
    return out


def run_post(work: Path, opts: dict, from_step: str | None = None) -> dict:
    cmd = [sys.executable, "-m", "focus_ear.post", str(work), "--opts", json.dumps(opts)]
    if from_step:
        cmd += ["--from", from_step]
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    final = json.loads((work / "final.json").read_text(encoding="utf-8"))
    timings = [json.loads(p.read_text()) for p in sorted((work / "post").glob("timing-*.json"))]
    return {"lines": final["lines"], "seconds": round(time.perf_counter() - t0, 1),
            "peak_mb": max((t["peak_mb"] for t in timings), default=None)}


def denoise_file(src: Path, dst: Path) -> None:
    """The audio as the models hear it with --denoise-scope all: the live chain's DeepFilterNet, offline."""
    import soundfile as sf

    from ..denoise import Denoiser, load_model

    x, fs = sf.read(src, dtype="float32")
    model, params = load_model()
    bs = 512
    d = Denoiser(fs, bs, model, params, threaded=False)
    pad = np.concatenate([x, np.zeros(d.delay_samples + bs, np.float32)])
    out = np.concatenate([d.process(pad[i:i + bs].copy()) for i in range(0, len(pad) - bs + 1, bs)])
    sf.write(dst, out[d.delay_samples:d.delay_samples + len(x)], fs, subtype="PCM_16")


def fresh_work(case: Path, base: Path, name: str, audio: Path | None = None, keep_steps=()) -> Path:
    work = base / name
    if work.exists():
        shutil.rmtree(work)
    (work / "post").mkdir(parents=True)
    shutil.copy(audio or case / "audio.wav", work / "audio.wav")
    src = base / "final"
    for step_file in keep_steps:
        shutil.copy(src / step_file, work / step_file)
    return work


# ---------------------------------------------------------------- sweeps

def sweep_clustering(work: Path, ref: dict, thresholds=(0.7, 0.75, 0.8, 0.85, 0.9, 0.95)) -> list[dict]:
    from ..post import diarize as D
    z = np.load(work / "post" / "embeddings.npz")
    wins = [tuple(map(float, w)) for w in z["windows"]]
    emb = z["embeddings"].astype(np.float32)
    regions = [tuple(r) for r in json.loads((work / "post" / "regions.json").read_text())["regions"]]
    out = []
    for th in thresholds:
        labels = D.cluster(emb, th, max(1, int(round(2.0 / D.HOP_S))))
        turns = D.turns_from_windows(regions, wins, labels)
        for t in turns:
            t["speaker"] = t.pop("cluster")
        d = der(clip(ref["turns"], ref.get("range")), clip(turns, ref.get("range")))
        lect = max(d["per_speaker"].values(), key=lambda v: v["speech_s"]) if d["per_speaker"] else {}
        out.append({"threshold": th, "der": d["der"], "speakers": d["speakers_hyp"], "speakers_ref": d["speakers_ref"],
                    "lecturer_coverage": lect.get("coverage"), "lecturer_split": lect.get("split_into")})
    return out


def sweep_vad(work: Path, ref: dict, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> list[dict]:
    """Speech found vs the reference turns, from the saved 100 ms probabilities."""
    reg = json.loads((work / "post" / "regions.json").read_text())
    probs = np.array(reg["probs_100ms"], np.float32)
    step = 3 * reg["frame_s"]
    t = np.arange(len(probs)) * step
    truth = np.zeros(len(probs), bool)
    for turn in clip(ref["turns"], ref.get("range")):
        truth[(t >= turn["start"]) & (t < turn["end"])] = True
    out = []
    for th in thresholds:
        hyp = probs >= th
        out.append({"threshold": th, "missed": float((truth & ~hyp).sum() / max(truth.sum(), 1)),
                    "false_alarm": float((~truth & hyp).sum() / max(truth.sum(), 1))})
    return out


# ---------------------------------------------------------------- run

def run_case(case: Path, out: Path, live_runs: int, quick: bool) -> dict:
    ref = load_case(case)
    res: dict = {"case": case.name, "synthetic": bool(ref.get("synthetic")), "duration_s": ref.get("duration_s"),
                 "snr_db": ref.get("snr_db"), "range": ref.get("range"), "systems": {}}
    base = out / "work" / case.name
    base.mkdir(parents=True, exist_ok=True)

    log.info("[%s] final (as shipped)", case.name)
    work = fresh_work(case, base, "final")
    r = run_post(work, {})
    res["systems"]["final"] = {**score(ref, r["lines"]), "seconds": r["seconds"], "peak_mb": r["peak_mb"]}
    res["sweep_clustering"] = sweep_clustering(work, ref)
    res["sweep_vad"] = sweep_vad(work, ref)

    if not quick:
        variants = [("final, no prompt", {"prompt": ""}), ("final, Indonesian only", {"en_threshold": 1.0})]
        for name, opts in variants:
            log.info("[%s] %s", case.name, name)
            w = fresh_work(case, base, name.replace(", ", "_").replace(" ", "-"),
                           keep_steps=("post/regions.json", "post/diarization.json", "post/embeddings.npz"))
            r = run_post(w, opts, from_step="asr")
            res["systems"][name] = {**score(ref, r["lines"]), "seconds": r["seconds"]}
        log.info("[%s] final, denoised", case.name)
        with tempfile.TemporaryDirectory() as tmp:
            den = Path(tmp) / "audio.wav"
            denoise_file(case / "audio.wav", den)
            w = fresh_work(case, base, "final_denoised", audio=den)
        r = run_post(w, {})
        res["systems"]["final, denoised"] = {**score(ref, r["lines"]), "seconds": r["seconds"]}

    runs = []
    for i in range(live_runs):
        log.info("[%s] live replay %d/%d", case.name, i + 1, live_runs)
        t0 = time.perf_counter()
        lines = live_replay(case / "audio.wav")
        runs.append({**score(ref, lines), "seconds": round(time.perf_counter() - t0, 1)})
        (base / f"live_{i + 1}.json").write_text(json.dumps(lines, ensure_ascii=False))
    if runs:
        keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float)) and runs[0][k] is not None]
        res["systems"]["live"] = {k: float(np.mean([x[k] for x in runs if x[k] is not None])) for k in keys}
        res["systems"]["live"]["range_wer"] = [min(x["wer"] for x in runs), max(x["wer"] for x in runs)]
        res["systems"]["live"]["runs"] = len(runs)
    return res


def fmt(x, pct=True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:g}"


def report(results: list[dict], out: Path) -> str:
    md = [f"# Evaluation, {datetime.now():%Y-%m-%d %H:%M}", ""]
    if all(r["synthetic"] for r in results):
        md += ["**Every case here is synthetic** (macOS text-to-speech voices with simulated reverb and noise). "
               "These numbers check the machinery and compare settings; they do not predict a real lecture hall.", ""]
    for r in results:
        md += [f"## {r['case']}" + (f" (SNR {r['snr_db']:g} dB)" if r.get("snr_db") is not None else ""), ""]
        md += ["| System | WER | CER | English terms kept | DER | Speakers found / real | Lecturer under one label |",
               "|---|---|---|---|---|---|---|"]
        for name, s in r["systems"].items():
            wer_s = fmt(s["wer"]) + (f" ({fmt(s['range_wer'][0])}–{fmt(s['range_wer'][1])}, {s['runs']} runs)"
                                     if "range_wer" in s else "")
            md.append(f"| {name} | {wer_s} | {fmt(s['cer'])} | {fmt(s['terms'])} | {fmt(s['der'])} | "
                      f"{s['speakers']:.0f} / {s['speakers_ref']:.0f} | {fmt(s['lecturer_coverage'])} |")
        if "final" in r["systems"] and r["systems"]["final"].get("seconds"):
            f = r["systems"]["final"]
            md += ["", f"After-class pass: {f['seconds']:.0f} s for {r['duration_s']:.0f} s of audio, peak "
                       f"{f['peak_mb']} MB in its largest step."]
        md += ["", "Offline clustering threshold (cosine distance):", "",
               "| Threshold | DER | Speakers found / real | Lecturer coverage | Lecturer split into |", "|---|---|---|---|---|"]
        md += [f"| {x['threshold']:g} | {fmt(x['der'])} | {x['speakers']} / {x['speakers_ref']} | "
               f"{fmt(x['lecturer_coverage'])} | {x['lecturer_split']} |" for x in r["sweep_clustering"]]
        md += ["", "VAD threshold (speech found vs the reference turns, 100 ms frames):", "",
               "| Threshold | Missed speech | False alarm |", "|---|---|---|"]
        md += [f"| {x['threshold']:g} | {fmt(x['missed'])} | {fmt(x['false_alarm'])} |" for x in r["sweep_vad"]]
        md.append("")
    md += ["WER/CER: word/character error rate against the reference text (lower is better). DER: diarization "
           "error rate, 10 ms frames, 0.25 s collar, Hungarian speaker mapping. Lecturer under one label: the share "
           "of the lecturer's speech carrying the single label mapped to them.", ""]
    text = "\n".join(md)
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def make_reference(session: str, start: float | None, end: float | None) -> Path:
    from ..config import SESSIONS_DIR
    from ..review import load_doc, text_only
    folder = SESSIONS_DIR / session
    doc = load_doc(folder)
    if doc is None:
        raise SystemExit("this session has no transcript.json yet: open it in the app first")
    if not doc.get("history"):
        log.warning("nothing in this transcript was corrected: it's the machine's output, not a reference")
    lines = text_only(doc)
    ref = {"lines": lines, "range": [start or 0.0, end or 1e9] if start or end else None, "synthetic": False,
           "terms": [], "source_session": session, "made": datetime.now().isoformat(timespec="seconds")}
    dest = ROOT / "eval" / "cases" / session
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "reference.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    link = dest / "audio.wav"
    if not link.exists():
        os.symlink(folder / "audio.wav", link)
    return dest


def stability(sessions: list[str]) -> str:
    """Memory, drops and latency over a session, from its engine.log health lines."""
    import re
    from ..config import SESSIONS_DIR
    md = ["| Session | Length | Memory start → end (max) | Worker drops | Underruns | Reconnects | Latency (measured) |",
          "|---|---|---|---|---|---|---|"]
    for sid in sessions:
        text = (SESSIONS_DIR / sid / "engine.log").read_text(encoding="utf-8", errors="replace")
        mem = [int(m) for m in re.findall(r"mem (\d+) MB", text)]
        up = re.findall(r"health: up (\d+:\d+:\d+)", text)
        drops = re.findall(r"drops: worker (\d+) in-ovf \d+ underrun (\d+)", text)
        rec = re.findall(r"reconnects (\d+)", text)
        lat = [int(x) for x in re.findall(r"measured latency (\d+) ms", text)]
        md.append(f"| {sid} | {up[-1] if up else '?'} | {mem[0] if mem else '?'} → {mem[-1] if mem else '?'} "
                  f"({max(mem) if mem else '?'}) MB | {drops[-1][0] if drops else '?'} | {drops[-1][1] if drops else '?'} | "
                  f"{rec[-1] if rec else '?'} | {', '.join(map(str, lat)) or '–'} ms |")
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="focus_ear.eval", description="evaluate transcripts and speaker labels")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cases", nargs="+", type=Path)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--live-runs", type=int, default=3)
    r.add_argument("--quick", action="store_true", help="only the shipped pass and the sweeps")
    m = sub.add_parser("reference")
    m.add_argument("session")
    m.add_argument("--from", dest="start", type=float)
    m.add_argument("--to", dest="end", type=float)
    s = sub.add_parser("stability")
    s.add_argument("sessions", nargs="+")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("focus_ear").setLevel(logging.WARNING)
    log.setLevel(logging.INFO)
    if args.cmd == "reference":
        print(make_reference(args.session, args.start, args.end))
        return 0
    if args.cmd == "stability":
        print(stability(args.sessions))
        return 0
    out = args.out or ROOT / "eval" / "results" / datetime.now().strftime("%Y-%m-%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    results = [run_case(c.resolve(), out, args.live_runs, args.quick) for c in args.cases]
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(report(results, out))
    shutil.rmtree(out / "work", ignore_errors=True) if os.environ.get("FOCUS_EAR_EVAL_CLEAN") else None
    return 0


if __name__ == "__main__":
    sys.exit(main())
