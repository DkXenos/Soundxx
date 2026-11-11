"""transcript.json: the transcript the app shows and you correct.

Built from final.json (the after-class pass) or, before that, from
live.jsonl. Neither of those is ever edited: they're what the machine
produced, the evaluation's input. Your corrections live only here, and
the corrected transcript is the reference the evaluation scores against.

    {"version": 1, "source": "final" | "live", "built": iso,
     "speakers": {"S1": {"label": "Dosen", "role": "lecturer", "profile": null, "remembered": false}},
     "lines": [{"id": "L1", "start": 3.2, "end": 9.8, "speaker": "S1", "text": "...",
                "flagged": false, "edited": false, "lang": "id"}],
     "history": [...]}      what undo() needs, newest last (at most 100)

Edits (apply()):
    rename_speaker {speaker, label}
    merge_speakers {source, target}     every line of source becomes target's; source disappears
    move_line      {line, speaker}      speaker "new" makes a new speaker
    edit_text      {line, text}
    mark_remembered {speaker, name}     after the app saved their voice
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

from .sessions import LIVE_STEM, read_jsonl, now_iso
from .settings import write_json_atomic
from .transcript import clock

DOC_FILE = "transcript.json"
MD_FILE = "transcript.md"
HISTORY_MAX = 100
UNKNOWN = "S0"


class EditError(ValueError):
    pass


# Building ----------------------------------------------------------------

def doc_from_final(final: dict) -> dict:
    speakers = {sid: {"label": s["label"], "role": s.get("role"), "profile": s.get("profile"), "remembered": False}
                for sid, s in final.get("speakers", {}).items()}
    lines = [{"id": ln["id"], "start": ln["start"], "end": ln["end"], "speaker": ln["speaker"], "text": ln["text"],
              "flagged": bool(ln.get("flagged")), "edited": False, "lang": ln.get("lang")}
             for ln in final.get("lines", [])]
    return {"version": 1, "source": "final", "built": now_iso(), "speakers": speakers, "lines": lines, "history": []}


def doc_from_live(records: list[dict]) -> dict:
    speakers: dict[str, dict] = {}
    lines = []
    for rec in records:
        if not (rec.get("text") or "").strip():
            continue
        sid = f"S{rec['speaker_id']}" if rec.get("speaker_id") is not None else UNKNOWN
        if sid not in speakers:
            label = rec.get("speaker") or ("Unknown" if sid == UNKNOWN else f"Speaker {sid[1:]}")
            speakers[sid] = {"label": label, "role": None, "profile": None, "remembered": False}
        garbled = bool(rec.get("garbled"))
        lines.append({"id": f"L{len(lines) + 1}", "start": rec["start"], "end": rec["end"], "speaker": sid,
                      "text": "(unclear)" if garbled else rec["text"].strip(), "flagged": garbled, "edited": False,
                      "lang": rec.get("language")})
    return {"version": 1, "source": "live", "built": now_iso(), "speakers": speakers, "lines": lines, "history": []}


def load_doc(folder: Path) -> dict | None:
    p = folder / DOC_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_or_build(folder: Path, session: dict | None = None) -> dict:
    """transcript.json, built first from final.json or the live transcript if it doesn't exist yet."""
    doc = load_doc(folder)
    if doc is not None:
        return doc
    final = folder / "final.json"
    if final.exists():
        doc = doc_from_final(json.loads(final.read_text(encoding="utf-8")))
    else:
        doc = doc_from_live(read_jsonl(folder / f"{LIVE_STEM}.jsonl"))
    save(folder, doc, session)
    return doc


def install_final(folder: Path, final: dict, session: dict | None = None) -> dict:
    """After processing: transcript.json from final.json. Corrections of an earlier final are kept aside."""
    old = load_doc(folder)
    if old is not None and old.get("history"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        write_json_atomic(folder / f"transcript.before-{stamp}.json", old)
    doc = doc_from_final(final)
    save(folder, doc, session)
    return doc


def save(folder: Path, doc: dict, session: dict | None = None) -> None:
    write_json_atomic(folder / DOC_FILE, doc, indent=None)
    md = render_markdown(doc, session)
    tmp = folder / f".{MD_FILE}.tmp"
    tmp.write_text(md, encoding="utf-8")
    tmp.replace(folder / MD_FILE)


def render_markdown(doc: dict, session: dict | None = None) -> str:
    s = session or {}
    title = s.get("title") or "Lecture"
    started = s.get("started", "")
    try:
        when = datetime.fromisoformat(started).strftime("%A %-d %B %Y, %H:%M")
    except (TypeError, ValueError):
        when = started or "?"
    speakers = ", ".join(sp["label"] for sid, sp in doc["speakers"].items()
                         if any(ln["speaker"] == sid for ln in doc["lines"]))
    head = [f"# {title}", "", f"- **Date:** {when}"]
    if s.get("course"):
        head.append(f"- **Course:** {s['course']}")
    head.append(f"- **Speakers:** {speakers or 'none'}")
    head.append(f"- **Transcript:** {'after-class pass' if doc['source'] == 'final' else 'live (not processed yet)'}"
                + (", corrected" if doc.get("history") else ""))
    head.append("")
    body = []
    prev = None
    for ln in doc["lines"]:
        who = doc["speakers"].get(ln["speaker"], {}).get("label", ln["speaker"])
        text = ln["text"] + (" *(unsure)*" if ln.get("flagged") and not ln.get("edited") else "")
        if who != prev:
            body.append(f"**[{clock(ln['start'])}] {who}:** {text}\n")
        else:
            body.append(f"[{clock(ln['start'])}] {text}\n")
        prev = who
    return "\n".join(head) + "\n" + "\n".join(body)


# Editing -----------------------------------------------------------------

def _line(doc: dict, line_id: str) -> dict:
    for ln in doc["lines"]:
        if ln["id"] == line_id:
            return ln
    raise EditError(f"no line {line_id}")


def _speaker(doc: dict, sid: str) -> dict:
    if sid not in doc["speakers"]:
        raise EditError(f"no speaker {sid}")
    return doc["speakers"][sid]


def _new_speaker_id(doc: dict) -> str:
    """Never an id used before in this document: a merged-away id still names a voiceprint in speakers.npz."""
    used = [int(k[1:]) for k in doc["speakers"] if k[1:].isdigit()]
    used += [int(ln["speaker"][1:]) for ln in doc["lines"] if ln["speaker"][1:].isdigit()]
    for h in doc.get("history", []):
        used += [int(k[1:]) for k in h["before"]["speakers"] if k[1:].isdigit()]
    n = max([doc.get("next_speaker", 0)] + [u + 1 for u in used] + [1])
    doc["next_speaker"] = n + 1
    return f"S{n}"


def apply(doc: dict, op: dict) -> dict:
    """Apply one edit in place (and return doc); records what undo() needs."""
    kind = op.get("op")
    before = {"speakers": copy.deepcopy(doc["speakers"]), "lines": {}}
    if kind == "rename_speaker":
        label = str(op.get("label", "")).strip()
        if not label:
            raise EditError("a name can't be empty")
        _speaker(doc, op["speaker"])["label"] = label
    elif kind == "merge_speakers":
        src, dst = op["source"], op["target"]
        _speaker(doc, src), _speaker(doc, dst)
        if src == dst:
            raise EditError("pick two different speakers")
        for ln in doc["lines"]:
            if ln["speaker"] == src:
                before["lines"][ln["id"]] = copy.deepcopy(ln)
                ln["speaker"] = dst
        del doc["speakers"][src]
    elif kind == "move_line":
        ln = _line(doc, op["line"])
        target = op["speaker"]
        if target == "new":
            target = _new_speaker_id(doc)
            n = sum(1 for s in doc["speakers"].values() if s.get("role") == "student") + 1
            doc["speakers"][target] = {"label": op.get("label") or f"Mahasiswa {n}", "role": "student",
                                       "profile": None, "remembered": False}
        else:
            _speaker(doc, target)
        before["lines"][ln["id"]] = copy.deepcopy(ln)
        ln["speaker"] = target
    elif kind == "edit_text":
        ln = _line(doc, op["line"])
        text = str(op.get("text", "")).strip()
        before["lines"][ln["id"]] = copy.deepcopy(ln)
        ln["text"] = text
        ln["edited"] = True
    elif kind == "mark_remembered":
        sp = _speaker(doc, op["speaker"])
        sp["remembered"] = True
        sp["profile"] = op.get("name") or sp["label"]
        sp["label"] = op.get("name") or sp["label"]
    else:
        raise EditError(f"unknown edit {kind!r}")
    doc.setdefault("history", []).append({"op": op, "at": now_iso(), "before": before})
    del doc["history"][:-HISTORY_MAX]
    return doc


def undo(doc: dict) -> dict | None:
    """Revert the last edit in place; returns the edit undone, or None if there was nothing to undo."""
    if not doc.get("history"):
        return None
    last = doc["history"].pop()
    before = last["before"]
    doc["speakers"] = before["speakers"]
    by_id = {ln["id"]: i for i, ln in enumerate(doc["lines"])}
    for line_id, old in before["lines"].items():
        if line_id in by_id:
            doc["lines"][by_id[line_id]] = old
    return last["op"]


def text_only(doc: dict) -> list[dict]:
    """What the notes and the evaluation need: id, times, speaker label, text (no audio, no voiceprints)."""
    return [{"id": ln["id"], "start": ln["start"], "end": ln["end"],
             "speaker": doc["speakers"].get(ln["speaker"], {}).get("label", ln["speaker"]), "text": ln["text"]}
            for ln in doc["lines"] if ln["text"]]
