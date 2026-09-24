"""Online speaker clustering (phase 2, not implemented yet).

Plan: keep one centroid per speaker. An embedding joins the most similar
centroid if cosine similarity >= threshold (running-average update, then
re-normalise); otherwise it founds a new speaker. At most max_speakers
(one per number key); speakers unseen for prune_after_s are dropped, least
recently seen first when over the cap. Named, enrolled speakers are never
pruned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Speaker:
    id: int
    centroid: np.ndarray
    count: int = 0
    last_seen: float = 0.0
    name: str | None = None


class OnlineClusterer:
    def __init__(self, threshold: float = 0.65, max_speakers: int = 9, prune_after_s: float = 300.0):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.prune_after_s = prune_after_s

    def assign(self, embedding: np.ndarray, now: float) -> Speaker:
        raise NotImplementedError("clustering arrives in phase 2")

    def reset(self) -> None:
        raise NotImplementedError("clustering arrives in phase 2")
