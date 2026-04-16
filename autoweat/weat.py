"""
weat.py — Locked WEAT computer. WEAT-only (SC-WEAT removed 2026-04-16).

AUTOWEAT_WEAT_VERSION = "2026-04-16-weat-only"

Why only WEAT:

SC-WEAT was removed after empirical diagnosis showed that its absolute
signs are contaminated by attribute-pool corpus-frequency asymmetry in
GloVe-style embeddings. Specifically: when the A and B attribute pools
differ in how common their words are in ordinary web text, almost any
target word will appear to "lean toward" the denser pool in SC-WEAT,
regardless of actual semantic association.

WEAT (relative) is immune to this: any constant bias in cos(w, A∪B)
cancels in the mean(s over X) − mean(s over Y) difference.

DEFINITIONS (Caliskan, Bryson & Narayanan 2017):

  Association of a single word w with attribute pools A and B:

    s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b)

  WEAT effect size (relative test, eq. 3):

    es(X, Y, A, B) = ( mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B) )
                     / std_dev_{w in X ∪ Y} s(w, A, B)

  WEAT permutation p (two-sided):

    p = Pr[ |S(X', Y', A, B)| >= |S(X, Y, A, B)| ]
        over equal-size partitions (X', Y') of X ∪ Y, where
        S(X, Y, A, B) = sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)

EQUAL-SIZE REQUIREMENT:

  Caliskan's procedure requires |X| = |Y| (for WEAT) AND |A| = |B|.
  This module enforces both via random downsampling of the larger set,
  seeded by the rng for reproducibility.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np


AUTOWEAT_WEAT_VERSION = "2026-04-16-weat-only"


# ---------- data types ----------

@dataclass
class WEATResult:
    effect_size: float          # Cohen's d, Caliskan eq. 3
    p_value: float              # two-sided permutation p
    test_statistic: float       # S(X, Y, A, B)

    n_X: int = 0
    n_Y: int = 0
    n_A: int = 0
    n_B: int = 0
    n_permutations: int = 0
    exact: bool = False
    X_used: list[str] = field(default_factory=list)
    Y_used: list[str] = field(default_factory=list)
    A_used: list[str] = field(default_factory=list)
    B_used: list[str] = field(default_factory=list)
    X_dropped: list[str] = field(default_factory=list)
    Y_dropped: list[str] = field(default_factory=list)
    A_dropped: list[str] = field(default_factory=list)
    B_dropped: list[str] = field(default_factory=list)
    per_word_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- vocab filtering & balancing ----------

def filter_in_vocab(words: Sequence[str], vocab: set[str]) -> tuple[list[str], list[str]]:
    """Split `words` into (kept, dropped) based on vocab membership."""
    kept, dropped, seen = [], [], set()
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        if w in vocab:
            kept.append(w)
        elif w.lower() in vocab:
            kept.append(w.lower())
        else:
            dropped.append(w)
    return kept, dropped


def _balance_pair(
    P: list[str], Q: list[str], rng: random.Random
) -> tuple[list[str], list[str]]:
    """Force |P| == |Q| by random downsampling of the larger set."""
    n = min(len(P), len(Q))
    P2 = P[:] if len(P) == n else rng.sample(P, n)
    Q2 = Q[:] if len(Q) == n else rng.sample(Q, n)
    return P2, Q2


# Backwards-compatible alias
def balance_targets(X, Y, rng):
    return _balance_pair(X, Y, rng)


# ---------- math ----------

def _cosine_matrix(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix between U (n, d) and V (m, d)."""
    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return U_norm @ V_norm.T


def compute_weat(
    X_words: Sequence[str],
    Y_words: Sequence[str],
    A_words: Sequence[str],
    B_words: Sequence[str],
    embed_fn,
    vocab: set[str],
    rng: random.Random,
    max_permutations: int = 100_000,
    exact_threshold: int = 12,
    min_pool_size: int = 20,
) -> WEATResult:
    """Run WEAT (relative test only). Returns effect size + two-sided p."""
    # 1. vocab filter
    X_kept, X_dropped = filter_in_vocab(X_words, vocab)
    Y_kept, Y_dropped = filter_in_vocab(Y_words, vocab)
    A_kept, A_dropped = filter_in_vocab(A_words, vocab)
    B_kept, B_dropped = filter_in_vocab(B_words, vocab)

    if (
        len(X_kept) < min_pool_size
        or len(Y_kept) < min_pool_size
        or len(A_kept) < min_pool_size
        or len(B_kept) < min_pool_size
    ):
        raise ValueError(
            f"Insufficient in-vocab words (need >= {min_pool_size} per pool): "
            f"|X|={len(X_kept)} |Y|={len(Y_kept)} "
            f"|A|={len(A_kept)} |B|={len(B_kept)}"
        )

    # 2. balance both pairs
    X_bal, Y_bal = _balance_pair(X_kept, Y_kept, rng)
    A_bal, B_bal = _balance_pair(A_kept, B_kept, rng)

    # 3. embed everything
    X_vec = np.vstack([embed_fn(w) for w in X_bal])
    Y_vec = np.vstack([embed_fn(w) for w in Y_bal])
    A_vec = np.vstack([embed_fn(w) for w in A_bal])
    B_vec = np.vstack([embed_fn(w) for w in B_bal])

    # 4. s(w, A, B) for every target word
    XY_vec = np.vstack([X_vec, Y_vec])
    XY_words = X_bal + Y_bal
    cos_XY_A = _cosine_matrix(XY_vec, A_vec).mean(axis=1)
    cos_XY_B = _cosine_matrix(XY_vec, B_vec).mean(axis=1)
    s_all = cos_XY_A - cos_XY_B

    n = len(X_bal)
    s_X = s_all[:n]
    s_Y = s_all[n:]

    # WEAT effect size
    weat_numerator = s_X.mean() - s_Y.mean()
    weat_denominator = s_all.std(ddof=0)
    weat_d = float(weat_numerator / weat_denominator) if weat_denominator > 0 else 0.0

    # WEAT test statistic
    weat_test_stat = float(s_X.sum() - s_Y.sum())
    weat_abs_test_stat = abs(weat_test_stat)

    # 5. WEAT permutation (two-sided)
    idx_all = np.arange(2 * n)
    total_sum = s_all.sum()

    if n <= exact_threshold:
        weat_count = 0
        total = 0
        for combo in itertools.combinations(idx_all, n):
            s_sum = s_all[list(combo)].sum()
            S_i = 2 * s_sum - total_sum
            if abs(S_i) >= weat_abs_test_stat:
                weat_count += 1
            total += 1
        weat_p = weat_count / total
        weat_exact = True
        n_perms_weat = total
    else:
        weat_count = 0
        for _ in range(max_permutations):
            perm = rng.sample(range(2 * n), n)
            s_sum = s_all[perm].sum()
            S_i = 2 * s_sum - total_sum
            if abs(S_i) >= weat_abs_test_stat:
                weat_count += 1
        weat_p = weat_count / max_permutations
        weat_exact = False
        n_perms_weat = max_permutations

    per_word_s = {w: float(v) for w, v in zip(XY_words, s_all)}

    return WEATResult(
        effect_size=weat_d,
        p_value=weat_p,
        test_statistic=weat_test_stat,
        n_X=len(X_bal),
        n_Y=len(Y_bal),
        n_A=len(A_bal),
        n_B=len(B_bal),
        n_permutations=n_perms_weat,
        exact=weat_exact,
        X_used=X_bal,
        Y_used=Y_bal,
        A_used=A_bal,
        B_used=B_bal,
        X_dropped=X_dropped,
        Y_dropped=Y_dropped,
        A_dropped=A_dropped,
        B_dropped=B_dropped,
        per_word_s=per_word_s,
    )
