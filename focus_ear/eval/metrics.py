"""Accuracy metrics: word and character error rate, technical-term recall, diarization error rate.

WER/CER are Levenshtein distances over normalised words/characters
(lowercase, punctuation removed). DER is frame-based at 10 ms: missed
speech + false alarm + speaker confusion, over the reference's speech time,
with hypothesis speakers mapped one-to-one to reference speakers by the
Hungarian algorithm (scipy's linear_sum_assignment) to maximise overlap.
A collar (default 0.25 s) around each reference boundary isn't scored, as
is customary: nobody agrees on a turn's exact edge.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np

_PUNCT = re.compile(r"[^\w\s'-]", re.UNICODE)


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    words = [w.strip("'-") for w in text.split()]
    return [w for w in words if w]


def edit_ops(ref: list, hyp: list) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of the cheapest alignment."""
    n, m = len(ref), len(hyp)
    # Rows of (cost, subs, dels, ins), kept to two rows.
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            c_s, c_d, c_i = prev[j - 1], prev[j], cur[j - 1]
            best = min((c_s[0], 0), (c_d[0], 1), (c_i[0], 2))
            if best[1] == 0:
                cur[j] = (c_s[0] + 1, c_s[1] + 1, c_s[2], c_s[3])
            elif best[1] == 1:
                cur[j] = (c_d[0] + 1, c_d[1], c_d[2] + 1, c_d[3])
            else:
                cur[j] = (c_i[0] + 1, c_i[1], c_i[2], c_i[3] + 1)
        prev = cur
    _, s, d, ins = prev[m]
    return s, d, ins


@dataclass
class ErrorRate:
    rate: float
    substitutions: int
    deletions: int
    insertions: int
    reference_len: int

    def to_dict(self) -> dict:
        return asdict(self)


def wer(reference: str, hypothesis: str) -> ErrorRate:
    ref, hyp = normalize(reference), normalize(hypothesis)
    s, d, i = edit_ops(ref, hyp)
    return ErrorRate((s + d + i) / max(len(ref), 1), s, d, i, len(ref))


def cer(reference: str, hypothesis: str) -> ErrorRate:
    ref, hyp = list(" ".join(normalize(reference))), list(" ".join(normalize(hypothesis)))
    s, d, i = edit_ops(ref, hyp)
    return ErrorRate((s + d + i) / max(len(ref), 1), s, d, i, len(ref))


def _occurrences(words: list[str], term: list[str]) -> int:
    n = len(term)
    return sum(1 for i in range(len(words) - n + 1) if words[i:i + n] == term) if n else 0


def term_recall(terms: list[str], reference: str, hypothesis: str) -> dict:
    """Of the technical terms that occur in the reference, how many occurrences the hypothesis kept."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    found = total = 0
    missed = []
    for term in terms:
        t = normalize(term)
        n_ref = _occurrences(ref, t)
        if not n_ref:
            continue
        n_hyp = min(_occurrences(hyp, t), n_ref)
        total += n_ref
        found += n_hyp
        if n_hyp < n_ref:
            missed.append(term)
    return {"recall": found / total if total else float("nan"), "found": found, "total": total, "missed": missed}


# --------------------------------------------------------------------------
# Diarization

FRAME_S = 0.01


def _frames(turns: list[dict], n: int, labels: dict[str, int]) -> np.ndarray:
    """Per-frame speaker index, -1 for nobody. Overlap: the later turn wins (no overlap scoring)."""
    out = np.full(n, -1, dtype=np.int32)
    for t in turns:
        a, b = int(round(t["start"] / FRAME_S)), int(round(t["end"] / FRAME_S))
        out[max(a, 0):min(b, n)] = labels.setdefault(str(t["speaker"]), len(labels))
    return out


def der(reference: list[dict], hypothesis: list[dict], collar: float = 0.25) -> dict:
    """Diarization error rate. Turns are dicts with start, end (seconds) and speaker (any label)."""
    from scipy.optimize import linear_sum_assignment

    end = max([t["end"] for t in reference] + [t["end"] for t in hypothesis] + [0.0])
    n = int(np.ceil(end / FRAME_S)) + 1
    ref_labels: dict[str, int] = {}
    hyp_labels: dict[str, int] = {}
    ref = _frames(reference, n, ref_labels)
    hyp = _frames(hypothesis, n, hyp_labels)
    scored = np.ones(n, dtype=bool)
    c = int(round(collar / FRAME_S))
    if c:
        for t in reference:
            for edge in (t["start"], t["end"]):
                e = int(round(edge / FRAME_S))
                scored[max(0, e - c):min(n, e + c)] = False
    ref, hyp = ref[scored], hyp[scored]
    speech = ref >= 0
    total = int(speech.sum())
    missed = int((speech & (hyp < 0)).sum())
    false_alarm = int((~speech & (hyp >= 0)).sum())
    both = speech & (hyp >= 0)
    overlap = np.zeros((len(ref_labels), max(len(hyp_labels), 1)), dtype=np.int64)
    np.add.at(overlap, (ref[both], hyp[both]), 1)
    rows, cols = linear_sum_assignment(-overlap)
    matched = int(overlap[rows, cols].sum())
    confusion = int(both.sum()) - matched
    ref_names = {v: k for k, v in ref_labels.items()}
    hyp_names = {v: k for k, v in hyp_labels.items()}
    mapping = {hyp_names[c_]: ref_names[r] for r, c_ in zip(rows, cols) if c_ in hyp_names and overlap[r, c_] > 0}
    per_speaker = {}
    for r, name in ref_names.items():
        ref_time = int((ref == r).sum())
        mapped = [c_ for c_, rr in zip(cols, rows) if rr == r]
        best = mapped[0] if mapped else None
        hyp_time = int((hyp == best).sum()) if best is not None else 0
        hit = int(overlap[r, best]) if best is not None else 0
        # How many hypothesis speakers hold at least 10% of this person's speech: >1 means "split".
        pieces = int(((overlap[r] / max(ref_time, 1)) >= 0.10).sum())
        per_speaker[name] = {
            "speech_s": round(ref_time * FRAME_S, 1),
            "coverage": hit / ref_time if ref_time else float("nan"),   # their speech under their label
            "purity": hit / hyp_time if hyp_time else float("nan"),     # their label's speech that is theirs
            "split_into": pieces,
            "label": hyp_names.get(best) if best is not None else None,
        }
    return {
        "der": (missed + false_alarm + confusion) / total if total else float("nan"),
        "missed": missed / total if total else float("nan"),
        "false_alarm": false_alarm / total if total else float("nan"),
        "confusion": confusion / total if total else float("nan"),
        "speech_s": round(total * FRAME_S, 1),
        "speakers_ref": len(ref_labels), "speakers_hyp": len(hyp_labels),
        "mapping": mapping, "per_speaker": per_speaker, "collar_s": collar,
    }


def lines_to_turns(lines: list[dict], key: str = "speaker", merge_gap: float = 0.5) -> list[dict]:
    """Transcript lines -> speaker turns (adjacent lines of one speaker less than ``merge_gap`` apart joined)."""
    turns: list[dict] = []
    for ln in sorted(lines, key=lambda x: x["start"]):
        who = ln.get(key)
        if who is None:
            who = "?"
        if turns and turns[-1]["speaker"] == who and ln["start"] - turns[-1]["end"] <= merge_gap:
            turns[-1]["end"] = max(turns[-1]["end"], ln["end"])
        else:
            turns.append({"start": ln["start"], "end": ln["end"], "speaker": who})
    return turns
