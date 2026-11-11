"""Lecture notes from the transcript, by Claude Code in headless mode (claude -p).

Only the transcript TEXT is sent: line ids, times, speaker labels and
words. No audio and no voiceprints ever leave the Mac. The model answers
with strict JSON (summary, key points, chapters, terms); chapters point at a
transcript line id, and their timestamps are taken from the transcript,
never from the model, so a click on a chapter always lands on real audio.

Offline (no route to api.anthropic.com) or with the CLI missing, a session's
notes are "queued" and generated later: on one click, or when the app
starts with a connection.

    python -m focus_ear.notes SESSION [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import SESSIONS_DIR
from .review import load_or_build, text_only
from .sessions import SessionStore, now_iso
from .settings import write_json_atomic
from .transcript import clock

log = logging.getLogger("focus_ear.notes")

TIMEOUT_S = 600
API_HOST = "api.anthropic.com"

SYSTEM = """You write study notes from a university lecture transcript for a student who attended it.
The transcript comes from speech recognition: expect misheard words, especially English technical terms
in Indonesian speech ("away" for "array", "link list" for "linked list"). Use the obvious intended term.
Never invent content that isn't in the transcript.

Write in the language the lecture is mostly in (usually Indonesian), keeping English technical terms in English.

Answer with ONE JSON object and nothing else (no Markdown fences), exactly this shape:
{
  "summary": "3-6 sentences: what the lecture covered and why it matters",
  "key_points": ["the most important ideas, facts, definitions, formulas; 4-12 items"],
  "chapters": [{"title": "short topic title", "line_id": "L12", "summary": "1-2 sentences"}],
  "terms": [{"term": "technical term", "explanation": "one sentence, as the lecturer used it"}],
  "tasks": ["assignments, deadlines, exam hints the lecturer gave; [] if none"],
  "questions": [{"line_id": "L40", "question": "a question someone asked", "answer": "the answer given"}]
}
Chapters follow the order of the lecture; each line_id must be an id from the transcript, the line where that
topic starts. Use 3-12 chapters for a full lecture, fewer for a short one."""


class NotesError(RuntimeError):
    pass


class Offline(NotesError):
    pass


def online(timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((API_HOST, 443), timeout=timeout):
            return True
    except OSError:
        return False


def claude_path() -> str | None:
    found = shutil.which("claude")
    if found:
        return found
    for p in (Path.home() / ".local/bin/claude", Path("/opt/homebrew/bin/claude"), Path("/usr/local/bin/claude")):
        if p.exists():
            return str(p)
    return None


def build_prompt(lines: list[dict], session: dict | None) -> str:
    s = session or {}
    head = []
    if s.get("course"):
        head.append(f"Course: {s['course']}")
    if s.get("title"):
        head.append(f"Lecture: {s['title']}")
    body = "\n".join(f"[{ln['id']} {clock(ln['start'])}] {ln['speaker']}: {ln['text']}" for ln in lines)
    return ("\n".join(head) + "\n\n" if head else "") + "Transcript:\n" + body


def call_claude(prompt: str, model: str | None = None, timeout: float = TIMEOUT_S) -> tuple[str, dict]:
    exe = claude_path()
    if exe is None:
        raise Offline("the claude command isn't installed (https://claude.com/claude-code)")
    if not online():
        raise Offline(f"no connection to {API_HOST}")
    cmd = [exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence", "--strict-mcp-config",
           "--disable-slash-commands", "--system-prompt", SYSTEM]
    if model:
        cmd += ["--model", model]
    with tempfile.TemporaryDirectory(prefix="focus-ear-notes-") as cwd:  # no project files in reach
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired:
            raise NotesError(f"claude didn't answer within {timeout:.0f} s") from None
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[-500:]
        if re.search(r"ENOTFOUND|ECONNREFUSED|network|fetch failed|Unable to connect|offline", err, re.I):
            raise Offline(err)
        raise NotesError(f"claude failed (exit {proc.returncode}): {err}")
    try:
        envelope = json.loads(proc.stdout)
    except ValueError:
        raise NotesError(f"claude's answer wasn't JSON: {proc.stdout[:300]}") from None
    if envelope.get("is_error"):
        raise NotesError(f"claude reported an error: {str(envelope.get('result'))[:300]}")
    meta = {k: envelope.get(k) for k in ("duration_ms", "total_cost_usd", "session_id") if k in envelope}
    usage = envelope.get("modelUsage") or {}
    if usage:
        meta["model"] = next(iter(usage))
    return str(envelope.get("result", "")), meta


def parse_notes(text: str, lines: list[dict]) -> dict:
    """The model's JSON, validated against the transcript. Raises NotesError if it isn't usable."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < start:
        raise NotesError("no JSON object in the answer")
    try:
        data = json.loads(t[start:end + 1])
    except ValueError as exc:
        raise NotesError(f"the answer isn't valid JSON: {exc}") from None
    by_id = {ln["id"]: ln for ln in lines}

    def strs(x):
        return [str(i).strip() for i in x if str(i).strip()] if isinstance(x, list) else []

    chapters = []
    for c in data.get("chapters") or []:
        if not isinstance(c, dict) or c.get("line_id") not in by_id:
            continue  # a chapter must land on real audio
        ln = by_id[c["line_id"]]
        chapters.append({"title": str(c.get("title", "")).strip() or "(untitled)", "line_id": ln["id"],
                         "start": ln["start"], "summary": str(c.get("summary", "")).strip()})
    chapters.sort(key=lambda c: c["start"])
    questions = []
    for q in data.get("questions") or []:
        if isinstance(q, dict) and q.get("question"):
            ln = by_id.get(q.get("line_id"))
            questions.append({"question": str(q["question"]).strip(), "answer": str(q.get("answer", "")).strip(),
                              "line_id": ln["id"] if ln else None, "start": ln["start"] if ln else None})
    terms = [{"term": str(x.get("term", "")).strip(), "explanation": str(x.get("explanation", "")).strip()}
             for x in data.get("terms") or [] if isinstance(x, dict) and x.get("term")]
    summary = str(data.get("summary", "")).strip()
    if not summary and not chapters:
        raise NotesError("the answer has neither a summary nor chapters")
    return {"summary": summary, "key_points": strs(data.get("key_points")), "chapters": chapters, "terms": terms,
            "tasks": strs(data.get("tasks")), "questions": questions}


def render_markdown(notes: dict, session: dict | None) -> str:
    s = session or {}
    out = [f"# Notes: {s.get('title') or 'Lecture'}", ""]
    if s.get("course"):
        out += [f"*{s['course']}*", ""]
    out += ["## Summary", "", notes["summary"], ""]
    if notes["key_points"]:
        out += ["## Key points", ""] + [f"- {p}" for p in notes["key_points"]] + [""]
    if notes["chapters"]:
        out += ["## Chapters", ""]
        for c in notes["chapters"]:
            out.append(f"- **[{clock(c['start'])}] {c['title']}**" + (f": {c['summary']}" if c["summary"] else ""))
        out.append("")
    if notes["terms"]:
        out += ["## Terms", ""] + [f"- **{t['term']}**: {t['explanation']}" for t in notes["terms"]] + [""]
    if notes.get("tasks"):
        out += ["## Tasks and deadlines", ""] + [f"- {t}" for t in notes["tasks"]] + [""]
    if notes.get("questions"):
        out += ["## Questions asked", ""]
        for q in notes["questions"]:
            when = f"[{clock(q['start'])}] " if q.get("start") is not None else ""
            out.append(f"- {when}**{q['question']}** {q['answer']}")
        out.append("")
    out += ["---", f"*Generated {notes['generated'][:16].replace('T', ' ')} by {notes.get('model') or 'Claude'} from "
            f"the {'corrected ' if notes.get('corrected') else ''}transcript text (no audio was sent).*", ""]
    return "\n".join(out)


def generate(folder: Path, session: dict | None = None, model: str | None = None, call=None) -> dict:
    """Generate and save notes.json + notes.md for a session folder; returns the notes."""
    doc = load_or_build(folder, session)
    lines = text_only(doc)
    if not lines:
        raise NotesError("the transcript is empty")
    t0 = time.monotonic()
    call = call or call_claude  # looked up now, so tests can replace it
    text, meta = call(build_prompt(lines, session), model)
    notes = parse_notes(text, lines)
    notes.update(generated=now_iso(), model=meta.get("model") or model, source=doc["source"],
                 corrected=bool(doc.get("history")), seconds=round(time.monotonic() - t0, 1),
                 cost_usd=meta.get("total_cost_usd"))
    write_json_atomic(folder / "notes.json", notes)
    tmp = folder / ".notes.md.tmp"
    tmp.write_text(render_markdown(notes, session), encoding="utf-8")
    tmp.replace(folder / "notes.md")
    return notes


MIN_WORDS = 40  # below this there's nothing to take notes on (a test, a quiet room, a false start)


def generate_for_session(store: SessionStore, sid: str, model: str | None = None, force: bool = False) -> str:
    """Generate one session's notes, keeping notes_status up to date. Returns the new status."""
    session = store.load(sid)
    words = sum(len(ln["text"].split()) for ln in text_only(load_or_build(store.path(sid), session)))
    if words < MIN_WORDS and not force:
        store.update(sid, notes_status="skipped", notes_error=f"only {words} words were transcribed")
        return "skipped"
    store.update(sid, notes_status="generating", notes_error=None)
    try:
        generate(store.path(sid), session, model)
    except Offline as exc:
        store.update(sid, notes_status="queued", notes_error=f"waiting for a connection ({exc})")
        return "queued"
    except NotesError as exc:
        store.update(sid, notes_status="failed", notes_error=str(exc))
        return "failed"
    except Exception as exc:  # noqa: BLE001
        log.exception("notes failed")
        store.update(sid, notes_status="failed", notes_error=f"{type(exc).__name__}: {exc}")
        return "failed"
    store.update(sid, notes_status="ready", notes_error=None, notes_generated=now_iso())
    return "ready"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="focus_ear.notes", description="lecture notes for a session (claude -p)")
    p.add_argument("session")
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store = SessionStore(SESSIONS_DIR)
    status = generate_for_session(store, args.session, args.model)
    data = store.load(args.session)
    print(status, data.get("notes_error") or store.path(args.session) / "notes.md")
    return 0 if status == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
