"""Step 2: who spoke when, over the whole recording at once.

The live clusterer decides from 1.5 s of audio, online, and splits a
reverberant lecturer into several speakers in noise (STATUS.md). Offline,
every voiceprint of the lecture is available at once:

  1. ECAPA voiceprints of 1.5 s windows every 0.75 s inside speech regions
     (shorter regions get one voiceprint of their own).
  2. Agglomerative clustering (average linkage) on cosine distance, cut at
     ``threshold``; clusters holding less than ``min_cluster_s`` of speech
     are folded into the nearest big one (coughs, mixed windows).
  3. Each speech frame takes the label of the nearest window centre; runs
     shorter than ``min_turn_s`` are absorbed by their neighbours.
  4. Names: a cluster whose centroid matches a saved voice (cosine >=
     ``profile_threshold``) takes that name. If no saved voice names the
     biggest cluster, it's "Dosen" (in a lecture, the lecturer talks most);
     the rest are "Mahasiswa 1, 2, ..." in order of appearance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import ANALYSIS_RATE

log = logging.getLogger("focus_ear.post")

WINDOW_S = 1.5
HOP_S = 0.75
MIN_WINDOW_S = 0.5
LABEL_RES_S = 0.05
# Cosine distance to cut the tree at. On the synthetic lecture (focus_ear.eval.synth, 3 voices, 10 and
# 0 dB SNR): 0.7 split the lecturer into 7-14 clusters, 0.95 merged a student into them; 0.85 found
# exactly 3 at both SNRs (DER 0.1% and 0.3%). Re-check on real lectures with python -m focus_ear.eval.
DEFAULT_THRESHOLD = 0.85


def windows(regions: list[tuple[float, float]], window_s: float = WINDOW_S, hop_s: float = HOP_S
            ) -> list[tuple[float, float]]:
    out = []
    for a, b in regions:
        if b - a < MIN_WINDOW_S:
            continue
        if b - a <= window_s:
            out.append((a, b))
            continue
        t = a
        while t + window_s <= b + 1e-6:
            out.append((t, t + window_s))
            t += hop_s
        if out[-1][1] < b - hop_s / 2:
            out.append((b - window_s, b))
    return out


def embed_windows(audio: np.ndarray, wins: list[tuple[float, float]], device: str = "auto", progress=None,
                  batch: int = 32) -> np.ndarray:
    import torch

    from ..embeddings import EcapaEmbedder

    emb = EcapaEmbedder(device)
    out = np.zeros((len(wins), 192), np.float32)
    full = [i for i, (a, b) in enumerate(wins) if abs((b - a) - WINDOW_S) < 1e-3]
    short = [i for i in range(len(wins)) if i not in set(full)]
    n_full = int(round(WINDOW_S * ANALYSIS_RATE))
    done = 0
    with torch.inference_mode():
        for k in range(0, len(full), batch):
            idx = full[k:k + batch]
            x = np.stack([_cut(audio, *wins[i], n_full) for i in idx])
            e = emb.model.encode_batch(torch.from_numpy(x).to(emb.device)).reshape(len(idx), -1).cpu().numpy()
            out[idx] = e
            done += len(idx)
            if progress is not None:
                progress(done / len(wins), f"listening for voices: {done} of {len(wins)}")
        for i in short:
            out[i] = emb(_cut(audio, *wins[i]))
            done += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def _cut(audio: np.ndarray, a: float, b: float, n: int | None = None) -> np.ndarray:
    i, j = int(a * ANALYSIS_RATE), int(b * ANALYSIS_RATE)
    x = audio[i:j]
    if n is not None and len(x) != n:
        x = np.pad(x, (0, max(0, n - len(x))))[:n]
    return np.ascontiguousarray(x, dtype=np.float32)


def cluster(emb: np.ndarray, threshold: float, min_cluster_windows: int = 3) -> np.ndarray:
    """Cluster labels 0..k-1, biggest first; small clusters folded into the nearest big one."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(emb)
    if n == 0:
        return np.zeros(0, np.int32)
    if n == 1:
        return np.zeros(1, np.int32)
    z = linkage(emb.astype(np.float64), method="average", metric="cosine")
    labels = fcluster(z, t=threshold, criterion="distance") - 1
    counts = np.bincount(labels)
    big = [c for c in range(len(counts)) if counts[c] >= min_cluster_windows]
    if not big:
        big = [int(np.argmax(counts))]
    cents = np.stack([_unit(emb[labels == c].mean(axis=0)) for c in big])
    for c in range(len(counts)):
        if c in big:
            continue
        members = labels == c
        target = big[int(np.argmax(cents @ _unit(emb[members].mean(axis=0))))]
        labels[members] = target
    # Renumber: biggest cluster first.
    order = [c for c, _ in sorted(((c, int((labels == c).sum())) for c in set(labels.tolist())),
                                  key=lambda kv: -kv[1])]
    remap = {c: i for i, c in enumerate(order)}
    return np.array([remap[c] for c in labels], np.int32)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def turns_from_windows(regions: list[tuple[float, float]], wins: list[tuple[float, float]], labels: np.ndarray,
                       min_turn_s: float = 0.6) -> list[dict]:
    """Label every LABEL_RES_S step of speech by the nearest window centre, then smooth into turns."""
    if not len(wins):
        return []
    centres = np.array([(a + b) / 2 for a, b in wins])
    turns: list[dict] = []
    for a, b in regions:
        t = np.arange(a, b, LABEL_RES_S)
        if not len(t):
            continue
        idx = np.searchsorted(centres, t)
        lo = np.clip(idx - 1, 0, len(centres) - 1)
        hi = np.clip(idx, 0, len(centres) - 1)
        nearest = np.where(np.abs(centres[lo] - t) <= np.abs(centres[hi] - t), lo, hi)
        lab = labels[nearest]
        runs = []
        start = 0
        for i in range(1, len(lab) + 1):
            if i == len(lab) or lab[i] != lab[start]:
                runs.append([int(lab[start]), t[start], min(b, t[i - 1] + LABEL_RES_S)])
                start = i
        # Absorb short runs into the longer neighbour, repeatedly.
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for i, (lb, s, e) in enumerate(runs):
                if e - s >= min_turn_s:
                    continue
                left = runs[i - 1] if i > 0 else None
                right = runs[i + 1] if i + 1 < len(runs) else None
                into = max((r for r in (left, right) if r is not None), key=lambda r: r[2] - r[1])
                runs[i][0] = into[0]
                changed = True
                break
            # merge equal neighbours
            merged = [runs[0]]
            for r in runs[1:]:
                if r[0] == merged[-1][0]:
                    merged[-1][2] = r[2]
                else:
                    merged.append(r)
            runs = merged
        for lb, s, e in runs:
            if turns and turns[-1]["cluster"] == lb and s - turns[-1]["end"] < 1.0:
                turns[-1]["end"] = round(float(e), 3)
            else:
                turns.append({"start": round(float(s), 3), "end": round(float(e), 3), "cluster": lb})
    return turns


def name_clusters(centroids: np.ndarray, speech_s: list[float], first_s: list[float],
                  profiles: list[tuple[str, np.ndarray]], profile_threshold: float = 0.45) -> list[dict]:
    """Names per cluster index. Each saved voice names at most one cluster (its best match)."""
    k = len(centroids)
    names: list[str | None] = [None] * k
    sims: list[float | None] = [None] * k
    if profiles and k:
        P = np.stack([_unit(v) for _, v in profiles])
        S = centroids @ P.T  # clusters x profiles
        pairs = sorted(((S[c, p], c, p) for c in range(k) for p in range(len(profiles))), reverse=True)
        used_c, used_p = set(), set()
        for s, c, p in pairs:
            if s < profile_threshold or c in used_c or p in used_p:
                continue
            names[c], sims[c] = profiles[p][0], round(float(s), 3)
            used_c.add(c)
            used_p.add(p)
    biggest = int(np.argmax(speech_s)) if k else None
    out = []
    n_student = 0
    for c in sorted(range(k), key=lambda c: first_s[c]):
        if names[c] is not None:
            label, role = names[c], "profile"
        elif c == biggest:
            label, role = "Dosen", "lecturer"
        else:
            n_student += 1
            label, role = f"Mahasiswa {n_student}", "student"
        out.append({"cluster": c, "label": label, "role": role, "profile": names[c], "profile_similarity": sims[c],
                    "speech_s": round(speech_s[c], 1), "first_s": round(first_s[c], 1)})
    return sorted(out, key=lambda x: x["cluster"])


def run(folder: Path, audio: np.ndarray, regions: list, progress, threshold: float = DEFAULT_THRESHOLD,
        min_cluster_s: float = 2.0, device: str = "auto", profiles: list | None = None) -> dict:
    from ..profiles import load_profiles

    regions = [tuple(r) for r in regions]
    wins = windows(regions)
    progress(0.0, f"listening for voices: {len(wins)} windows", force=True)
    emb = embed_windows(audio, wins, device, progress)
    labels = cluster(emb, threshold, min_cluster_windows=max(1, int(round(min_cluster_s / HOP_S))))
    turns = turns_from_windows(regions, wins, labels)
    k = int(labels.max()) + 1 if len(labels) else 0
    centroids = np.stack([_unit(emb[labels == c].mean(axis=0)) for c in range(k)]) if k else np.zeros((0, 192))
    speech = [sum(t["end"] - t["start"] for t in turns if t["cluster"] == c) for c in range(k)]
    first = [min((t["start"] for t in turns if t["cluster"] == c), default=1e9) for c in range(k)]
    profiles = load_profiles() if profiles is None else profiles
    named = name_clusters(centroids, speech, first, profiles)
    speakers = {f"S{x['cluster'] + 1}": {k_: v for k_, v in x.items() if k_ != "cluster"} for x in named}
    for t in turns:
        t["speaker"] = f"S{t.pop('cluster') + 1}"
    # The session's voice clusters, for "remember this voice" in the review screen.
    ids = [f"S{c + 1}" for c in range(k)]
    tmp = folder / ".speakers.npz.tmp.npz"
    np.savez(tmp, ids=np.array(ids), labels=np.array([speakers[i]["label"] for i in ids]),
             centroids=centroids.astype(np.float32), source=np.array("offline"))
    tmp.replace(folder / "speakers.npz")
    return {"turns": turns, "speakers": speakers, "windows": len(wins),
            "params": {"threshold": threshold, "min_cluster_s": min_cluster_s, "window_s": WINDOW_S, "hop_s": HOP_S},
            # Kept for the eval sweeps: re-clustering at other thresholds without re-embedding.
            "_embeddings_file": "post/embeddings.npz"} | _save_embeddings(folder, wins, emb)


def _save_embeddings(folder: Path, wins, emb) -> dict:
    tmp = folder / "post" / ".embeddings.tmp.npz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp, windows=np.array(wins, np.float32), embeddings=emb.astype(np.float16))
    tmp.replace(folder / "post" / "embeddings.npz")
    return {}
