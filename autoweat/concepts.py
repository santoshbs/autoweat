"""
concepts.py — Concept label canonicalization and word-list cache.

Purpose:
  Ensure that when the LLM proposes the "same" concept across different
  tests (e.g. "women", "women and feminine terms", "feminine terms"),
  we recognize them as the same concept and reuse the same word list.

Two jobs:

  1. CANONICALIZATION. Turn an LLM-proposed label into a stable key
     by lowercasing, dropping stopwords, sorting tokens. Two labels
     that normalize to overlapping token sets (Jaccard >= 0.6) are
     treated as the same concept.

  2. WORD-LIST CACHE. A JSON file on disk mapping canonical keys to
     the frozen word list for that concept. On first use, the LLM's
     list is stored. On subsequent uses, the cached list is the
     source of truth; the LLM's new proposal must overlap it at
     Jaccard >= 0.70 to be accepted.

Failure modes and how we handle them:
  - LLM proposes drifted list for a known concept → proposer retries
    with explicit feedback showing the cached list.
  - LLM labels two distinct concepts identically (e.g. uses "power"
    for both "political power" and "physical power") → canonical key
    collision. Unavoidable without stronger intent signals; flagged
    in logs for human review.
  - Cache file corrupted or missing → treated as empty cache; new
    entries created as tests run.

The cache file is plain JSON and human-editable. If the first
cached list for a concept is bad, fix it by editing the file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Canonicalization
# ──────────────────────────────────────────────────────────────────────

# Stopwords to drop when canonicalizing labels. These are words that
# add no information about what the concept IS — they just pad out the
# LLM's natural-language phrasing.
_STOPWORDS: frozenset[str] = frozenset({
    "and", "or", "the", "a", "an", "of", "in", "on", "for", "to",
    "with", "from", "by", "at", "as",
    "terms", "words", "vocabulary", "language",
    "related", "associated",
})

_TOKEN_RE = re.compile(r"[a-z]+")


def canonical_tokens(label: str) -> list[str]:
    """Tokens of a label, lowercased, stopwords dropped, sorted, deduped."""
    if not label:
        return []
    lower = label.lower()
    tokens = _TOKEN_RE.findall(lower)
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    # Dedup preserving no particular order (we sort)
    seen: set[str] = set()
    out: list[str] = []
    for t in filtered:
        if t not in seen:
            seen.add(t)
            out.append(t)
    out.sort()
    return out


def canonical_key(label: str) -> str:
    """Stable canonical key for a label, used as the cache key.

    Empty labels → empty string (never a valid cache key).
    """
    return "|".join(canonical_tokens(label))


# ──────────────────────────────────────────────────────────────────────
# Set similarity
# ──────────────────────────────────────────────────────────────────────

def jaccard(a: list[str] | set[str], b: list[str] | set[str]) -> float:
    """Jaccard similarity between two collections (treated as sets).

    Returns 0.0 if either is empty (including if both are empty).
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def concept_match_score(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Similarity score for concept labels, built on Jaccard but
    boosted when one token set is a strict subset of the other.

    Rationale: "women" and "women and feminine terms" are clearly the
    same concept, but their raw Jaccard is only 0.5 because the longer
    label has an extra token. When the shorter label's tokens are all
    contained in the longer label, we treat that as a match at score =
    1.0 (perfect subset). This correctly collapses across stylistic
    variation in how the LLM labels the same concept.

    Two token sets that share no tokens always return 0.0.
    """
    sa, sb = set(tokens_a), set(tokens_b)
    if not sa or not sb:
        return 0.0
    if sa.issubset(sb) or sb.issubset(sa):
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def labels_match(label_a: str, label_b: str, threshold: float = 0.6) -> bool:
    """Two labels refer to the same concept iff their canonical token
    sets score at or above threshold via `concept_match_score`."""
    ta = canonical_tokens(label_a)
    tb = canonical_tokens(label_b)
    return concept_match_score(ta, tb) >= threshold


# ──────────────────────────────────────────────────────────────────────
# Word-list cache
# ──────────────────────────────────────────────────────────────────────

class ConceptCache:
    """On-disk JSON cache of canonical-key → {label, words} for every
    concept that has ever been accepted into the feed.

    The cache is small and rewritten entirely on each save — no need
    for locking for single-process use.

    Cache file format (JSON):
      {
        "feminine|women": {
          "display_label": "women and feminine terms",
          "words": ["woman", "women", "female", ...],
          "first_seen": "2026-04-17T10:00:00+00:00"
        },
        ...
      }
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                self._data = obj
            else:
                self._data = {}
        except Exception:
            # Corrupted cache — start fresh, don't crash the daemon
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # --- Lookup --------------------------------------------------

    def lookup(self, label: str, fuzzy_threshold: float = 0.6
               ) -> tuple[str, dict] | None:
        """Find the cache entry that matches this label.

        First tries an exact canonical-key hit. If none, falls back
        to a fuzzy search over all cached keys using Jaccard on
        canonical token sets.

        Returns (canonical_key, entry_dict) or None.
        """
        key = canonical_key(label)
        if not key:
            return None
        if key in self._data:
            return (key, self._data[key])
        # Fuzzy fallback using the subset-aware score
        tokens = canonical_tokens(label)
        best_key: str | None = None
        best_sim: float = 0.0
        for k in self._data:
            k_tokens = k.split("|")
            sim = concept_match_score(tokens, k_tokens)
            if sim > best_sim:
                best_sim = sim
                best_key = k
        if best_key is not None and best_sim >= fuzzy_threshold:
            return (best_key, self._data[best_key])
        return None

    # --- Insert / update ----------------------------------------

    def insert(self, label: str, words: list[str], timestamp: str) -> str:
        """Store a new concept. Returns the canonical key used.

        If the canonical key already exists, this OVERWRITES it. Use
        `lookup` first to decide whether to overwrite.
        """
        key = canonical_key(label)
        if not key:
            raise ValueError(f"label canonicalizes to empty key: {label!r}")
        self._data[key] = {
            "display_label": label,
            "words": list(words),
            "first_seen": timestamp,
        }
        return key

    def has(self, label: str) -> bool:
        return self.lookup(label) is not None

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def size(self) -> int:
        return len(self._data)
