"""Stage 3: online speaker clustering, and the speaker state shared with the UI.

OnlineClusterer is plain numpy and single-threaded. SpeakerTracker wraps it
with a lock so the worker (which feeds it embeddings) and the UI (which reads
snapshots and changes the selection) can share it. The lock is held only for
bookkeeping, never during inference, and no audio callback ever takes it.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

SILENT_DB = -100.0


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


@dataclass
class Speaker:
    id: int
    centroid: np.ndarray          # L2-normalised embedding
    first_seen: float
    last_seen: float
    count: int = 0                # embeddings matched this session
    weight: int = 0               # how many embeddings the centroid averages (for the running mean)
    name: str | None = None
    enrolled: bool = False        # has a saved profile: never pruned
    saved_centroid: np.ndarray | None = None
    level_db: float = SILENT_DB   # live level while they talk, decaying otherwise
    speaking_s: float = 0.0

    @property
    def label(self) -> str:
        return self.name or f"Speaker {self.id}"


@dataclass
class Assignment:
    speaker_id: int | None        # None: no room for a new speaker
    similarities: dict[int, float]  # cosine similarity to every speaker, before the update
    created: bool = False
    merged: list[tuple[int, int]] = field(default_factory=list)  # (absorbed id, surviving id)


class OnlineClusterer:
    """Assigns each embedding to a speaker as it arrives. No offline pass.

    An embedding joins the most similar centroid if the cosine similarity is
    at least ``threshold``, and that centroid moves towards it (a running
    mean for the first 1/alpha embeddings, then an exponential average with
    ``alpha``). Otherwise it founds a new speaker. Because single
    short-window embeddings are noisy, one voice sometimes founds a second cluster; once
    the two centroids converge to ``merge_threshold`` they're merged.
    """

    def __init__(self, threshold: float = 0.5, merge_threshold: float = 0.6, alpha: float = 0.1,
                 max_speakers: int = 8, timeout_s: float = 120.0, min_count: int = 3,
                 phantom_timeout_s: float = 30.0):
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self.alpha = alpha
        self.max_speakers = max_speakers
        self.timeout_s = timeout_s
        self.min_count = min_count
        # Unconfirmed clusters (seen fewer than min_count times) are usually a
        # cough or a mixed window; they expire sooner than real speakers.
        self.phantom_timeout_s = phantom_timeout_s
        self.speakers: dict[int, Speaker] = {}
        self._next_id = 1

    def confirmed(self, s: Speaker) -> bool:
        return s.enrolled or s.count >= self.min_count

    def add_enrolled(self, name: str, centroid: np.ndarray, now: float = 0.0) -> Speaker:
        c = _unit(centroid)
        s = Speaker(id=self._next_id, centroid=c, first_seen=now, last_seen=now, name=name,
                    enrolled=True, saved_centroid=c.copy(), weight=round(1 / self.alpha))
        self._next_id += 1
        self.speakers[s.id] = s
        return s

    def assign(self, embedding: np.ndarray, now: float, protected: frozenset[int] = frozenset()) -> Assignment:
        """``protected`` ids (the selected speaker) are never evicted and win merges."""
        e = _unit(embedding)
        sims = {sid: float(s.centroid @ e) for sid, s in self.speakers.items()}
        best = max(sims, key=sims.__getitem__, default=None)
        if best is not None and sims[best] >= self.threshold:
            s = self.speakers[best]
            a = max(self.alpha, 1.0 / (s.weight + 1))
            s.centroid = _unit((1 - a) * s.centroid + a * e)
            s.weight += 1
            s.count += 1
            s.last_seen = now
            created = False
        else:
            s = self._create(e, now, protected)
            if s is None:
                return Assignment(None, sims)
            created = True
        merged: list[tuple[int, int]] = []
        s = self._merge_neighbours(s, protected, merged)
        return Assignment(s.id, sims, created, merged)

    def prune(self, now: float, protected: frozenset[int] = frozenset()) -> list[int]:
        gone = []
        for s in list(self.speakers.values()):
            if s.enrolled or s.id in protected:
                continue
            limit = self.timeout_s if self.confirmed(s) else self.phantom_timeout_s
            if now - s.last_seen > limit:
                del self.speakers[s.id]
                gone.append(s.id)
        return gone

    def reset(self) -> None:
        """Forget everyone except enrolled speakers, who go back to their saved voiceprint."""
        for s in list(self.speakers.values()):
            if not s.enrolled:
                del self.speakers[s.id]
                continue
            s.centroid = s.saved_centroid.copy()
            s.weight = round(1 / self.alpha)
            s.count, s.speaking_s, s.level_db = 0, 0.0, SILENT_DB
        self._next_id = max(self.speakers, default=0) + 1

    def _create(self, e: np.ndarray, now: float, protected: frozenset[int]) -> Speaker | None:
        if len(self.speakers) >= self.max_speakers:
            victims = [s for s in self.speakers.values() if not s.enrolled and s.id not in protected]
            if not victims:
                return None
            # Unconfirmed clusters go first, then whoever was heard least recently.
            victim = min(victims, key=lambda s: (self.confirmed(s), s.last_seen))
            del self.speakers[victim.id]
        s = Speaker(id=self._next_id, centroid=e, first_seen=now, last_seen=now, count=1, weight=1)
        self._next_id += 1
        self.speakers[s.id] = s
        return s

    def _merge_neighbours(self, s: Speaker, protected: frozenset[int],
                          merged: list[tuple[int, int]]) -> Speaker:
        for other in list(self.speakers.values()):
            if other.id not in self.speakers or other.id == s.id:
                continue  # already absorbed earlier in this loop
            if other.enrolled and s.enrolled:
                continue  # two enrolled people stay distinct, however similar
            if float(other.centroid @ s.centroid) < self.merge_threshold:
                continue
            rank = lambda x: (x.id in protected, x.enrolled, self.confirmed(x), -x.id)  # noqa: E731
            keep, drop = (s, other) if rank(s) > rank(other) else (other, s)
            cap = round(1 / self.alpha)
            wk, wd = min(keep.weight, cap), min(drop.weight, cap)
            keep.centroid = _unit(wk * keep.centroid + wd * drop.centroid)
            keep.weight += drop.weight
            keep.count += drop.count
            keep.speaking_s += drop.speaking_s
            keep.first_seen = min(keep.first_seen, drop.first_seen)
            keep.last_seen = max(keep.last_seen, drop.last_seen)
            keep.level_db = max(keep.level_db, drop.level_db)
            del self.speakers[drop.id]
            merged.append((drop.id, keep.id))
            s = keep
        return s


# --------------------------------------------------------------------------
# Shared state for the worker and the UI
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeakerView:
    key: int            # the number key that selects them (1-9)
    id: int
    label: str
    level_db: float
    speaking_s: float
    talking: bool
    selected: bool
    enrolled: bool


@dataclass(frozen=True)
class TrackerSnapshot:
    speakers: tuple[SpeakerView, ...]   # confirmed speakers only, in key order
    selected_label: str | None          # None: passthrough
    unconfirmed: int                    # clusters not yet shown (seen < min_count times)
    last_similarities: tuple[tuple[str, float], ...]  # for the last embedding, best first
    last_assigned: str | None
    threshold: float
    embeddings: int


class SpeakerTracker:
    ACTIVE_S = 1.5              # a speaker counts as talking this long after their last match
    LEVEL_DECAY_DB_PER_S = 30.0

    def __init__(self, clusterer: OnlineClusterer, boost_gain: float = 1.0, attenuation_gain: float = 0.1):
        self._lock = threading.Lock()
        self.clusterer = clusterer
        self.boost_gain = boost_gain
        self.attenuation_gain = attenuation_gain
        self.selected_id: int | None = None   # None = passthrough
        self._user_chose = False              # manual choice beats auto-selection
        self._active_id: int | None = None
        self._active_time = 0.0
        self._speech_now = False
        self._last: Assignment | None = None
        self._embeddings = 0
        self._events: deque[str] = deque(maxlen=20)
        # Embedding quality, for comparing --denoise-scope settings: how close
        # each voiceprint was to the speaker it matched, and how often one
        # matched nobody and founded a new cluster. Only increase.
        self.match_sim_sum = 0.0
        self.matches = 0
        self.clusters_founded = 0

    # Worker side ---------------------------------------------------------
    def observe(self, embedding: np.ndarray, now: float) -> int | None:
        """Cluster one embedding. Returns the speaker's id if they're confirmed, else None."""
        with self._lock:
            c = self.clusterer
            protected = frozenset() if self.selected_id is None else frozenset({self.selected_id})
            a = c.assign(embedding, now, protected)
            c.prune(now, protected)
            self._last = a
            self._embeddings += 1
            if a.created:
                self.clusters_founded += 1
            elif a.similarities:
                self.match_sim_sum += max(a.similarities.values())
                self.matches += 1
            s = c.speakers.get(a.speaker_id)
            if s is None or not c.confirmed(s):
                return None
            self._active_id, self._active_time = s.id, now
            if (s.enrolled and self.selected_id is None and not self._user_chose
                    and s.count >= c.min_count):
                self.selected_id = s.id
                self._events.append(f"Recognised {s.label}: selected automatically")
            return s.id

    def update_level(self, rms_db: float, is_speech: bool, dt: float) -> None:
        with self._lock:
            self._speech_now = is_speech
            decay = self.LEVEL_DECAY_DB_PER_S * dt
            for s in self.clusterer.speakers.values():
                s.level_db = max(SILENT_DB, s.level_db - decay)
            s = self.clusterer.speakers.get(self._active_id)
            if is_speech and s is not None:
                s.level_db = max(s.level_db, rms_db)
                s.speaking_s += dt

    def gain_target(self, speaker_id: int | None) -> float:
        """Linear gain for audio attributed to ``speaker_id``.

        None means nobody identified yet (startup, or a new turn after a
        pause): full volume, so a lagging identification never silences the
        person you selected.
        """
        selected = self.selected_id
        if selected is None or speaker_id is None:
            return 1.0
        return self.boost_gain if speaker_id == selected else self.attenuation_gain

    # UI side -------------------------------------------------------------
    def snapshot(self, now: float | None = None) -> TrackerSnapshot:
        now = time.monotonic() if now is None else now
        with self._lock:
            c = self.clusterer
            shown = [s for s in c.speakers.values() if c.confirmed(s)]
            talking_id = (self._active_id if self._speech_now and now - self._active_time < self.ACTIVE_S
                          else None)
            views = tuple(
                SpeakerView(key=i, id=s.id, label=s.label, level_db=s.level_db, speaking_s=s.speaking_s,
                            talking=s.id == talking_id, selected=s.id == self.selected_id, enrolled=s.enrolled)
                for i, s in enumerate(shown, start=1))
            sims: tuple[tuple[str, float], ...] = ()
            assigned = None
            if self._last is not None:
                sims = tuple(sorted(((c.speakers[i].label if i in c.speakers else f"#{i} (gone)", v)
                                     for i, v in self._last.similarities.items()),
                                    key=lambda kv: -kv[1]))
                s = c.speakers.get(self._last.speaker_id)
                assigned = (s.label + ("" if c.confirmed(s) else " (unconfirmed)")) if s else None
            selected = c.speakers.get(self.selected_id)
            return TrackerSnapshot(
                speakers=views, selected_label=selected.label if selected else None,
                unconfirmed=len(c.speakers) - len(shown), last_similarities=sims,
                last_assigned=assigned, threshold=c.threshold, embeddings=self._embeddings)

    def label(self, speaker_id: int | None) -> str | None:
        """Current label of a speaker, also after they've been merged away or forgotten."""
        if speaker_id is None:
            return None
        with self._lock:
            s = self.clusterer.speakers.get(speaker_id)
            return s.label if s is not None else f"Speaker {speaker_id}"

    def drain_events(self) -> list[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def select(self, key: int) -> str:
        """Select by number key: 0 = passthrough, n = the n-th shown speaker."""
        with self._lock:
            self._user_chose = True
            if key == 0:
                self.selected_id = None
                return "Passthrough: every speaker at full volume"
            shown = [s for s in self.clusterer.speakers.values() if self.clusterer.confirmed(s)]
            if not 1 <= key <= len(shown):
                return f"No speaker {key}"
            self.selected_id = shown[key - 1].id
            return f"Focusing on {shown[key - 1].label}"

    def enroll_selected(self, name: str) -> list[tuple[str, np.ndarray]] | None:
        """Name the selected speaker and mark them enrolled.

        Returns every enrolled profile (to save), or None if nobody is selected.
        """
        name = name.strip()
        with self._lock:
            s = self.clusterer.speakers.get(self.selected_id)
            if s is None or not name:
                return None
            for other in self.clusterer.speakers.values():
                if other is not s and other.name == name:
                    other.name, other.enrolled = None, False  # the name moves to this voice
            s.name, s.enrolled = name, True
            s.saved_centroid = s.centroid.copy()
            s.weight = max(s.weight, round(1 / self.clusterer.alpha))
            return [(x.name, x.saved_centroid.copy()) for x in self.clusterer.speakers.values() if x.enrolled]

    def reset(self) -> None:
        with self._lock:
            self.clusterer.reset()
            if self.selected_id not in self.clusterer.speakers:
                self.selected_id = None
            self._active_id = None
            self._last = None
