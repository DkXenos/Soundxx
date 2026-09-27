"""Stage 5: transcribing the selected speaker, off the audio path.

  worker thread  UtteranceSegmenter (a FrameAnalyzer) cuts the raw 16 kHz
                 analysis stream into utterances and hands each one to the
                 Transcriber without waiting
  asr thread     Transcriber: bounded queue -> Whisper -> TranscriptWriter,
                 which appends and flushes every utterance to Markdown + JSONL
  UI             drains finished lines and reads counters

The audio callbacks never see any of this. The worker only appends frames
and, once per utterance, takes the queue's lock for an append. If Whisper
falls behind, the oldest waiting utterance is dropped and counted; the
worker never waits for it (except in --file mode, where nothing is live).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from .config import ANALYSIS_FRAME, ANALYSIS_RATE

if TYPE_CHECKING:
    from .asr import AsrBackend
    from .pipeline import BlockContext

log = logging.getLogger(__name__)

FRAME_S = ANALYSIS_FRAME / ANALYSIS_RATE
GARBLED_COMPRESSION_RATIO = 2.4  # Whisper's compression_ratio_threshold


def clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class Utterance:
    audio: np.ndarray           # 16 kHz mono, pre-roll included
    start: float                # speech onset, seconds since the session (or file) started
    end: float                  # end of the last speech frame
    speaker_id: int | None
    closed_at: float = field(default_factory=time.monotonic)


class UtteranceSegmenter:
    """Pipeline FrameAnalyzer: turns speech into utterances for the Transcriber.

    Opens an utterance at speech onset, starting with ~``preroll_ms`` of the
    audio before it, so the first phoneme isn't clipped. Closes it after
    ``gap_ms`` of silence, or at ``max_s`` (cutting at the latest pause in
    the last few seconds, so a word isn't split, if there is one).
    Utterances with less than ``min_ms`` of speech are discarded.

    Every speech frame is buffered, not only frames already attributed to
    the selected speaker: identification lags speech by about a second, and
    speaker_id goes back to unknown after each pause, so buffering only
    attributed frames would clip the start of every turn. An utterance
    belongs to whoever most of its identified frames were attributed to
    (or, if none were, the previous utterance's speaker). When a speaker is
    selected, other speakers' utterances are discarded when they close.

    When a different speaker is identified mid-utterance and stays
    identified for ``confirm_s``, the utterance is split at the latest pause
    shortly before (identification lags), or ``SPLIT_BACK_S`` before if
    there is none. A single mis-identified window doesn't split anything.
    """

    name = "utterance"
    SPLIT_BACK_S = 0.75     # ~half an embedding window: when a new voice typically wins
    SPLIT_SEARCH_S = 1.5    # look this far back for a pause to split at
    CAP_SEARCH_S = 5.0      # at the length cap, cut at a pause at most this far back
    TAIL_S = 0.2            # silence kept after the last speech frame

    def __init__(self, submit: Callable[[Utterance], None], selected: Callable[[], int | None] = lambda: None,
                 gap_ms: float = 800.0, max_s: float = 20.0, preroll_ms: float = 300.0,
                 min_ms: float = 400.0, confirm_s: float = 0.5):
        self.submit = submit
        self.selected = selected
        self._gap = max(1, math.ceil(gap_ms / 1000 / FRAME_S))
        self._max = max(2, int(max_s / FRAME_S))
        self._min_speech = math.ceil(min_ms / 1000 / FRAME_S - 1e-9)
        self._confirm = max(1, round(confirm_s / FRAME_S))
        self._tail = round(self.TAIL_S / FRAME_S)
        self._preroll: deque[np.ndarray] = deque(maxlen=math.ceil(preroll_ms / 1000 / FRAME_S))
        self._origin: float | None = None
        self._last_speaker: int | None = None
        self._clear()
        # Read by the UI; only ever increase.
        self.utterances = 0      # handed to the transcriber
        self.too_short = 0
        self.other_speaker = 0   # not the selected speaker

    def _clear(self) -> None:
        self._frames: list[np.ndarray] = []
        self._speech: list[bool] = []
        self._who: list[int | None] = []    # ctx.speaker_id on speech frames, else None
        self._t0 = 0.0                      # time of _frames[0]
        self._silence = 0                   # trailing non-speech frames
        self._current: int | None = None    # last speaker identified in this utterance
        self._candidate: int | None = None  # a different speaker, not yet confirmed
        self._candidate_at = 0

    @property
    def recording_s(self) -> float:
        """Length of the utterance being collected; 0 when idle."""
        return len(self._frames) * FRAME_S

    def analyze(self, frame: np.ndarray, ctx: BlockContext) -> None:
        if self._origin is None:
            self._origin = ctx.adc_time
        speech = ctx.is_speech
        if not self._frames:
            if not speech:
                self._preroll.append(frame)
                return
            self._frames = list(self._preroll)
            self._speech = [False] * len(self._frames)
            self._who = [None] * len(self._frames)
            self._t0 = ctx.adc_time - self._origin - len(self._frames) * FRAME_S
            self._preroll.clear()

        who = ctx.speaker_id if speech else None
        self._frames.append(frame)
        self._speech.append(speech)
        self._who.append(who)
        self._silence = 0 if speech else self._silence + 1

        if who is not None:
            if self._current is None or who == self._current:
                self._current, self._candidate = who, None
            elif who != self._candidate:
                self._candidate, self._candidate_at = who, len(self._frames) - 1
            elif len(self._frames) - self._candidate_at >= self._confirm:
                self._split_speaker_change()
                return

        if self._silence >= self._gap:
            self.flush()
        elif len(self._frames) >= self._max:
            cut = self._last_pause(len(self._frames), self.CAP_SEARCH_S)
            self._cut(cut if cut is not None else len(self._frames))

    def flush(self) -> None:
        """Close the open utterance, if any (also at shutdown and at the end of a file)."""
        if not self._frames:
            return
        tail = self._frames[max(0, len(self._frames) - self._preroll.maxlen):]
        self._emit(self._frames, self._speech, self._who, self._t0)
        self._clear()
        self._preroll.extend(tail)  # trailing silence is the next utterance's pre-roll

    close = flush  # Pipeline.stop() closes analyzers: don't lose the last words at quit

    def reset(self) -> None:
        self._clear()
        self._preroll.clear()

    def _last_pause(self, end: int, search_s: float) -> int | None:
        for i in range(end - 1, max(0, end - round(search_s / FRAME_S)) - 1, -1):
            if not self._speech[i]:
                return i
        return None

    def _split_speaker_change(self) -> None:
        at = self._candidate_at
        cut = self._last_pause(at, self.SPLIT_SEARCH_S)
        if cut is None:
            cut = at - round(self.SPLIT_BACK_S / FRAME_S)
        cut = max(1, cut)
        speaker = self._candidate
        # Frames between the cut and the moment identification caught up were
        # still tagged with the old speaker: they belong to nobody known.
        for i in range(cut, at):
            self._who[i] = None
        self._cut(cut)
        self._current, self._candidate = speaker, None

    def _cut(self, cut: int) -> None:
        """Emit frames[:cut] as an utterance; the rest carries on as the open one."""
        self._emit(self._frames[:cut], self._speech[:cut], self._who[:cut], self._t0)
        self._frames, self._speech, self._who = self._frames[cut:], self._speech[cut:], self._who[cut:]
        self._t0 += cut * FRAME_S
        self._current, self._candidate = next((w for w in reversed(self._who) if w is not None), None), None
        if not self._frames:
            self._clear()

    def _emit(self, frames: list[np.ndarray], speech: list[bool], who: list[int | None], t0: float) -> None:
        spoken = [i for i, s in enumerate(speech) if s]
        if len(spoken) < self._min_speech:
            if spoken:
                self.too_short += 1
            return
        known = Counter(w for w in who if w is not None)
        if known:
            self._last_speaker = known.most_common(1)[0][0]
        speaker = self._last_speaker
        selected = self.selected()
        if selected is not None and speaker != selected:
            self.other_speaker += 1
            return
        end = min(len(frames), spoken[-1] + 1 + self._tail)
        self.utterances += 1
        self.submit(Utterance(audio=np.concatenate(frames[:end]), start=t0 + spoken[0] * FRAME_S,
                              end=t0 + (spoken[-1] + 1) * FRAME_S, speaker_id=speaker))


# --------------------------------------------------------------------------

@dataclass
class TranscriptLine:
    start: float
    end: float
    speaker_id: int | None
    speaker: str | None
    text: str
    language: str
    lang_probs: dict[str, float] | None
    avg_logprob: float | None
    no_speech_prob: float | None
    compression_ratio: float | None
    audio_s: float          # length of the audio Whisper saw (pre-roll and tail included)
    queue_s: float          # waiting in the queue
    processing_s: float     # Whisper itself
    model: str
    # Whisper's own failure test (compression ratio > 2.4) still failed at its
    # highest temperature: a repetition loop, not words. Kept in the JSONL
    # for evaluation; shown as "(unclear)" in the Markdown and the UI.
    garbled: bool = False

    @property
    def display_text(self) -> str:
        return "(unclear)" if self.garbled else self.text


class AsrUnavailable(RuntimeError):
    pass


class Transcriber:
    """Owns the ASR thread, its bounded queue, and the transcript writer.

    ``drop_oldest=True`` (live): a full queue drops its oldest utterance and
    counts it, so submit() never waits. ``False`` (--file): submit() waits
    for room, so a file is transcribed completely.

    The backend loads on the ASR thread (MLX requires it); the constructor
    waits for that and raises AsrUnavailable if no backend loads.
    """

    def __init__(self, model: str, language: str | None = "id", initial_prompt: str | None = None,
                 queue_size: int = 4, drop_oldest: bool = True, detect_language: bool = False,
                 label_of: Callable[[int | None], str | None] = lambda speaker_id: None,
                 load: Callable[[str], AsrBackend] | None = None):
        if load is None:
            from .asr import load_backend as load
        self.model = model
        self.language = language            # None = detect per utterance
        self.prompt = initial_prompt or None
        self.detect_language = detect_language
        self.label_of = label_of
        self.writer: TranscriptWriter | None = None  # set before audio starts
        self.backend: AsrBackend | None = None

        self._size = max(1, queue_size)
        self._drop_oldest = drop_oldest
        self._queue: deque[Utterance] = deque()
        self._cv = threading.Condition()     # worker <-> asr thread; never taken by an audio callback
        self._busy = False
        self._closing = False
        self._lines: deque[TranscriptLine] = deque(maxlen=500)  # finished, not yet drained by the UI
        self._lines_lock = threading.Lock()
        # Read by the UI; only ever increase.
        self.done = 0
        self.dropped = 0
        self.failed = 0
        self.last_audio_s = 0.0
        self.last_processing_s = 0.0

        self._load = load
        self._load_error: str | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="asr", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._load_error is not None:
            raise AsrUnavailable(self._load_error)

    @property
    def pending(self) -> int:
        """Utterances waiting or being transcribed."""
        return len(self._queue) + self._busy

    @property
    def description(self) -> str:
        return self.backend.description if self.backend is not None else self.model

    def submit(self, utt: Utterance) -> None:
        with self._cv:
            if len(self._queue) >= self._size:
                if self._drop_oldest:
                    self._queue.popleft()
                    self.dropped += 1
                else:
                    while len(self._queue) >= self._size and self._thread.is_alive():
                        self._cv.wait(0.5)
            self._queue.append(utt)
            self._cv.notify_all()

    def drain_lines(self) -> list[TranscriptLine]:
        with self._lines_lock:
            lines = list(self._lines)
            self._lines.clear()
        return lines

    def close(self, timeout: float | None = None) -> bool:
        """Transcribe what's queued, close the files, and stop. False if it timed out."""
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def abandon(self, timeout: float = 3.0) -> int:
        """Stop without transcribing what's still queued; returns how many were given up.

        The utterance Whisper is on can't be interrupted, so it's waited for up
        to ``timeout``. The files are closed either way (TranscriptWriter is
        safe to close from here while the ASR thread might still write).
        """
        with self._cv:
            abandoned = len(self._queue)
            self._queue.clear()
            self.dropped += abandoned
            self._closing = True
            self._cv.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            abandoned += 1
        if self.writer is not None:
            self.writer.close()
        return abandoned

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            self.backend = self._load(self.model)
        except Exception as exc:  # noqa: BLE001 - reported by the constructor
            log.debug("ASR backend failed to load", exc_info=True)
            self._load_error = f"{type(exc).__name__}: {exc}"
            return
        finally:
            self._ready.set()
        log.info("transcription: %s, language %s", self.backend.description, self.language or "auto")
        try:
            while True:
                with self._cv:
                    while not self._queue and not self._closing:
                        self._cv.wait()
                    if not self._queue:
                        return
                    utt = self._queue.popleft()
                    self._busy = True
                    self._cv.notify_all()  # room for a waiting submit()
                try:
                    self._transcribe(utt)
                except Exception:  # noqa: BLE001 - keep going; the next utterance may work
                    log.exception("transcription failed")
                    self.failed += 1
                finally:
                    self._busy = False
        finally:
            if self.writer is not None:
                self.writer.close()

    def _transcribe(self, utt: Utterance) -> None:
        queue_s = time.monotonic() - utt.closed_at
        t = time.perf_counter()
        r = self.backend.transcribe(utt.audio, self.language, self.prompt, detect=self.detect_language)
        processing_s = time.perf_counter() - t
        audio_s = len(utt.audio) / ANALYSIS_RATE
        self.last_audio_s, self.last_processing_s = audio_s, processing_s
        line = TranscriptLine(
            start=utt.start, end=utt.end, speaker_id=utt.speaker_id, speaker=self.label_of(utt.speaker_id),
            text=r.text, language=r.language, lang_probs=r.lang_probs, avg_logprob=r.avg_logprob,
            no_speech_prob=r.no_speech_prob, compression_ratio=r.compression_ratio,
            audio_s=audio_s, queue_s=queue_s, processing_s=processing_s, model=self.description,
            garbled=(r.compression_ratio or 0.0) > GARBLED_COMPRESSION_RATIO)
        self.done += 1
        if self.writer is not None:
            self.writer.write(line)
        if line.text:
            with self._lines_lock:
                self._lines.append(line)
        if log.isEnabledFor(logging.DEBUG):
            probs = " ".join(f"{k} {v:.2f}" for k, v in (r.lang_probs or {}).items())
            log.debug("asr [%s-%s] %s: lang %s%s avg_logprob %s no_speech %s | %.1f s audio in %.2f s "
                      "(queued %.2f s) | %s",
                      clock(utt.start), clock(utt.end), line.speaker or "-", r.language,
                      f" ({probs})" if probs else "", _fmt(r.avg_logprob), _fmt(r.no_speech_prob),
                      audio_s, processing_s, queue_s,
                      ("GARBLED " if line.garbled else "") + (r.text[:200] or "(nothing)"))


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


# --------------------------------------------------------------------------

SPEAKERS_PLACEHOLDER = "- **Speakers:** (listed when the session ends)"


class TranscriptWriter:
    """~/.focus-ear/sessions/<YYYY-MM-DD_HHMM>.md and .jsonl, appended and flushed per utterance.

    A crash loses at most the utterance being transcribed. The Markdown
    heading's speaker list is filled in when the session closes cleanly.
    The JSONL has one record per transcribed utterance, including those
    Whisper found no words in (empty text), which the Markdown leaves out.
    """

    def __init__(self, directory: Path, started: datetime, source: str, model: str, language: str,
                 suffix: str = ""):
        directory.mkdir(parents=True, exist_ok=True)
        stem = started.strftime("%Y-%m-%d_%H%M") + suffix
        n = 1
        while True:
            name = stem if n == 1 else f"{stem}_{n}"
            self.md_path, self.jsonl_path = directory / f"{name}.md", directory / f"{name}.jsonl"
            if not self.md_path.exists() and not self.jsonl_path.exists():
                break
            n += 1
        self._md = open(self.md_path, "x", encoding="utf-8")
        self._jsonl = open(self.jsonl_path, "x", encoding="utf-8")
        self._lock = threading.Lock()  # the ASR thread writes; shutdown may close from the main thread
        self._speakers: list[str] = []
        self.lines = 0
        self._md.write(
            f"# Lecture transcript: {started:%Y-%m-%d %H:%M}\n\n"
            f"- **Date:** {started:%A %-d %B %Y, %H:%M}\n"
            f"{SPEAKERS_PLACEHOLDER}\n"
            f"- **Source:** {source}\n"
            f"- **Model:** {model}, language {language}\n\n")
        self._sync()

    def write(self, line: TranscriptLine) -> None:
        rec = {k: round(v, 3) if isinstance(v, float) else v for k, v in dataclasses.asdict(line).items()}
        with self._lock:
            if self._md.closed:
                return  # shut down while this utterance was being transcribed
            self._jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if line.text:
                who = line.speaker or "Speaker"
                if line.speaker and line.speaker not in self._speakers:
                    self._speakers.append(line.speaker)
                text = "*(unclear)*" if line.garbled else line.text
                self._md.write(f"**[{clock(line.start)}] {who}:** {text}\n\n")
                self.lines += 1
            self._sync()

    def _sync(self) -> None:
        for f in (self._md, self._jsonl):
            f.flush()
            os.fsync(f.fileno())

    def close(self) -> None:
        with self._lock:
            if self._md.closed:
                return
            self._md.close()
            self._jsonl.close()
        speakers = ", ".join(self._speakers) or "none transcribed"
        text = self.md_path.read_text(encoding="utf-8").replace(
            SPEAKERS_PLACEHOLDER, f"- **Speakers:** {speakers}", 1)
        tmp = self.md_path.with_suffix(".md.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.md_path)  # atomic: a crash here leaves the old file, never half of one
