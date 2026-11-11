"""Step 4: words + speaker turns -> transcript lines (final.json).

Each word takes the speaker of the turn its midpoint falls in (the nearest
turn if it falls in none). A word belongs to the chunk whose unpadded span
holds its midpoint, so the 0.15 s padding never duplicates a word at a
chunk edge. A new line starts on a change of speaker, after a pause of
``gap_s``, or once a line runs past ``max_line_s`` (at a sentence end if
there is one in the last 10 s).
"""
from __future__ import annotations

import bisect

from ..sessions import now_iso


def word_speaker(turns: list[dict], starts: list[float], t: float, fallback: str) -> str:
    i = bisect.bisect_right(starts, t) - 1
    if 0 <= i < len(turns) and turns[i]["start"] <= t <= turns[i]["end"]:
        return turns[i]["speaker"]
    best, dist = fallback, float("inf")
    for j in (i, i + 1):
        if 0 <= j < len(turns):
            d = min(abs(turns[j]["start"] - t), abs(turns[j]["end"] - t))
            if d < dist:
                best, dist = turns[j]["speaker"], d
    return best


def build_lines(asr: dict, turns: list[dict], gap_s: float = 1.0, max_line_s: float = 30.0) -> list[dict]:
    turns = sorted(turns, key=lambda t: t["start"])
    starts = [t["start"] for t in turns]
    words = []
    for c in asr["chunks"]:
        for seg in c["segments"]:
            for s, e, w, p in seg["words"]:
                mid = (s + e) / 2
                if not c["start"] - 1e-6 <= mid <= c["end"] + 1e-6:
                    continue
                words.append({"start": s, "end": e, "word": w, "p": p, "flagged": seg["flagged"],
                              "logprob": seg["avg_logprob"], "lang": c["language"],
                              "speaker": word_speaker(turns, starts, mid, c["speaker"])})
    words.sort(key=lambda w: w["start"])
    lines: list[dict] = []
    cur: list[dict] = []

    def close():
        if not cur:
            return
        text = "".join(w["word"] for w in cur).strip()
        if text:
            lp = [w["logprob"] for w in cur]
            lines.append({"id": f"L{len(lines) + 1}", "start": round(cur[0]["start"], 3), "end": round(cur[-1]["end"], 3),
                          "speaker": cur[0]["speaker"], "text": text,
                          "words": [[round(w["start"], 3), round(w["end"], 3), w["word"], w["p"]] for w in cur],
                          "flagged": any(w["flagged"] for w in cur), "lang": cur[0]["lang"],
                          "avg_logprob": round(sum(lp) / len(lp), 3)})
        cur.clear()

    for w in words:
        if cur:
            long_line = w["start"] - cur[0]["start"] > max_line_s
            sentence_end = cur[-1]["word"].rstrip().endswith((".", "?", "!"))
            if (w["speaker"] != cur[-1]["speaker"] or w["start"] - cur[-1]["end"] > gap_s
                    or (long_line and (sentence_end or w["start"] - cur[0]["start"] > max_line_s + 10))):
                close()
        cur.append(w)
    close()
    return lines


def run(asr: dict, diar: dict, regions: dict) -> dict:
    lines = build_lines(asr, diar["turns"])
    used = {ln["speaker"] for ln in lines}
    speakers = {sid: s for sid, s in diar["speakers"].items() if sid in used}
    return {"version": 1, "created": now_iso(),
            "models": {"asr": asr["model"], "speaker": "ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)",
                       "vad": "silero-vad v6.2.3"},
            "params": {"asr": {k: asr[k] for k in ("language", "en_threshold", "temperature", "prompt")},
                       "diarize": diar["params"], "vad_threshold": regions["threshold"]},
            "speakers": speakers, "lines": lines,
            "stats": {"duration_s": regions["duration_s"], "speech_s": regions["speech_s"],
                      "lines": len(lines), "flagged": sum(ln["flagged"] for ln in lines),
                      "speakers": len(speakers)}}
