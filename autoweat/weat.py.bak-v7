"""
weat.py — Locked WEAT computer with correct Caliskan SC-WEAT decomposition.

AUTOWEAT_WEAT_VERSION = "2026-04-11-sc-weat-correct"

This module is intentionally boring. The LLM never writes any of this code;
it only proposes word lists. All math happens here, under version control.

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

  SC-WEAT effect size for a SINGLE target word w (Caliskan eq.):

    es(w, A, B) = ( mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b) )
                  / std_dev_{x in A ∪ B} cos(w, x)

  Note that the numerator is exactly s(w, A, B). The denominator is the
  std-dev of cosines from w to every individual word in A∪B (NOT a std
  over the per-word s scores, which is the WEAT denominator).

  SC-WEAT permutation p (two-sided, for a single target word w):

    p_w = Pr[ |s(w, A', B')| >= |s(w, A, B)| ]
          over equal-size partitions (A', B') of A ∪ B.

  IMPORTANT: SC-WEAT permutes the ATTRIBUTE pools (A, B), not the
  target pools (X, Y). The target word w is held fixed. This is the
  opposite of what an earlier version of this code did.

AGGREGATION FOR DECOMPOSITION:

  For AutoWEAT we want to know whether the X side or the Y side is
  driving the WEAT contrast. We compute SC-WEAT individually for every
  target word (|X| + |Y| values), then:

    sc_weat_X_mean = mean_{x in X} es(x, A, B)
    sc_weat_Y_mean = mean_{y in Y} es(y, A, B)

  These two numbers tell you how strongly the AVERAGE X word and the
  AVERAGE Y word lean toward A vs B in this corpus. The difference
  between them mirrors the WEAT effect size's direction.

  For each side we also produce a permutation p value, computed by
  shuffling A∪B partitions and counting how often the random mean SC-
  WEAT (across the SAME target words) is at least as extreme as the
  observed mean SC-WEAT.

EQUAL-SIZE REQUIREMENT:

  Caliskan's procedure requires |X| = |Y| (for WEAT) AND |A| = |B|
  (for SC-WEAT). This module enforces both via random downsampling
  of the larger set, seeded by the rng for reproducibility.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np


AUTOWEAT_WEAT_VERSION = "2026-04-11-sc-weat-correct"


# ---------- data types ----------

@dataclass
class WEATResult:
    # Relative WEAT (the headline)
    effect_size: float          # Cohen's d, Caliskan eq. 3
    p_value: float              # two-sided permutation p
    test_statistic: float       # S(X, Y, A, B)

    # SC-WEAT decomposition by TARGET SIDE (Caliskan formula, aggregated)
    sc_weat_x_mean: float       # mean SC-WEAT effect size across X target words
    sc_p_x: float               # two-sided permutation p for the X-side mean
    sc_weat_y_mean: float       # mean SC-WEAT effect size across Y target words
    sc_p_y: float               # two-sided permutation p for the Y-side mean

    # Per-word SC-WEAT effect sizes for transparency / future analyses
    per_word_sc_weat: dict[str, float] = field(default_factory=dict)

    n_X: int = 0
    n_Y: int = 0
    n_A: int = 0
    n_B: int = 0
    n_permutations: int = 0          # for the relative WEAT permutation
    n_permutations_sc: int = 0       # for the SC-WEAT permutation (over A∪B)
    exact: bool = False
    exact_sc: bool = False
    X_used: list[str] = field(default_factory=list)
    Y_used: list[str] = field(default_factory=list)
    A_used: list[str] = field(default_factory=list)
    B_used: list[str] = field(default_factory=list)
    X_dropped: list[str] = field(default_factory=list)
    Y_dropped: list[str] = field(default_factory=list)
    A_dropped: list[str] = field(default_factory=list)
    B_dropped: list[str] = field(default_factory=list)
    per_word_s: dict[str, float] = field(default_factory=dict)  # s(w, A, B) for w in X∪Y

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
    min_pool_size: int = 12,
) -> WEATResult:
    """Run a full WEAT test plus correct Caliskan SC-WEAT decomposition.

    The relative WEAT and SC-WEAT have DIFFERENT permutation procedures:
      - WEAT permutes the targets (X ∪ Y), holding A and B fixed
      - SC-WEAT permutes the attributes (A ∪ B), holding the target word fixed

    Both are reported as two-sided p-values.
    """
    # 1. vocab filter
    X_kept, X_dropped = filter_in_vocab(X_words, vocab)
    Y_kept, Y_dropped = filter_in_vocab(Y_words, vocab)
    A_kept, A_dropped = filter_in_vocab(A_words, vocab)
    B_kept, B_dropped = filter_in_vocab(B_words, vocab)

    # Require at least min_pool_size vocab-present words in each of X, Y,
    # A, B. This is the per-pool floor for statistical power under the
    # permutation test; at n=12 the exact permutation null has C(24,12) =
    # 2,704,156 partitions, enough to resolve p-values at ~.00005 precision
    # and narrow enough to give medium effect sizes a chance at .05.
    # If the LLM proposal falls below this floor after vocab filtering,
    # the test is rejected and the daemon moves on.
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

    # 2. balance both pairs (Caliskan requires |X|=|Y| AND |A|=|B|)
    X_bal, Y_bal = _balance_pair(X_kept, Y_kept, rng)
    A_bal, B_bal = _balance_pair(A_kept, B_kept, rng)

    # 3. embed everything
    X_vec = np.vstack([embed_fn(w) for w in X_bal])
    Y_vec = np.vstack([embed_fn(w) for w in Y_bal])
    A_vec = np.vstack([embed_fn(w) for w in A_bal])
    B_vec = np.vstack([embed_fn(w) for w in B_bal])

    # 4. WEAT setup: s(w, A, B) for every target word
    XY_vec = np.vstack([X_vec, Y_vec])
    XY_words = X_bal + Y_bal

    cos_XY_A = _cosine_matrix(XY_vec, A_vec).mean(axis=1)
    cos_XY_B = _cosine_matrix(XY_vec, B_vec).mean(axis=1)
    s_all = cos_XY_A - cos_XY_B                # shape (2n,)

    n = len(X_bal)
    s_X = s_all[:n]
    s_Y = s_all[n:]

    # WEAT effect size — Caliskan eq. 3
    weat_numerator = s_X.mean() - s_Y.mean()
    weat_denominator = s_all.std(ddof=0)
    weat_d = float(weat_numerator / weat_denominator) if weat_denominator > 0 else 0.0

    # WEAT test statistic S
    weat_test_stat = float(s_X.sum() - s_Y.sum())
    weat_abs_test_stat = abs(weat_test_stat)

    # 5. WEAT permutation (over X ∪ Y, two-sided)
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

    # 6. SC-WEAT (per-target-word, Caliskan formula, permutation over A ∪ B)
    #
    # For each target word w, compute:
    #   numerator   = mean cos(w, A) - mean cos(w, B)
    #   denominator = std-dev cos(w, A ∪ B)
    #   es(w, A, B) = numerator / denominator
    #
    # The permutation test holds w fixed and shuffles A ∪ B into random
    # equal-size partitions A' and B'. For each partition we recompute
    # the numerator (the denominator is invariant since A ∪ B is fixed).
    #
    # The "side" p-value is computed by aggregating across X (or Y) target
    # words: under each random A/B partition, compute the mean SC-WEAT
    # numerator across X words, divide by the same denominator (per word),
    # and check whether that mean effect size is at least as extreme as
    # the observed mean. Same for Y.

    AB_vec = np.vstack([A_vec, B_vec])         # shape (2m, d) where m=|A|=|B|
    m = len(A_bal)
    AB_words = A_bal + B_bal

    # Cosines from every target word to every word in A ∪ B
    cos_XY_AB = _cosine_matrix(XY_vec, AB_vec)  # shape (2n, 2m)

    # Per-word SC-WEAT denominator: std of cos(w, x) over x in A ∪ B
    sc_denom_per_target = cos_XY_AB.std(axis=1, ddof=0)  # shape (2n,)
    sc_denom_per_target = np.where(sc_denom_per_target > 0, sc_denom_per_target, 1e-12)

    # Per-word SC-WEAT numerator under the OBSERVED partition (A = first m, B = last m)
    sc_num_obs = cos_XY_AB[:, :m].mean(axis=1) - cos_XY_AB[:, m:].mean(axis=1)
    sc_es_obs = sc_num_obs / sc_denom_per_target            # shape (2n,)

    sc_es_X = sc_es_obs[:n]
    sc_es_Y = sc_es_obs[n:]

    sc_weat_x_mean = float(sc_es_X.mean())
    sc_weat_y_mean = float(sc_es_Y.mean())

    # Build per-word SC-WEAT dict
    per_word_sc_weat = {w: float(v) for w, v in zip(XY_words, sc_es_obs)}

    # 7. SC-WEAT permutation (over A ∪ B, two-sided, aggregated by side)
    #
    # For each random partition A' / B' of A ∪ B into two equal halves,
    # compute per-word SC-WEAT numerator (denominator stays fixed), divide,
    # then take the mean across X targets and the mean across Y targets.
    # Count how often each absolute mean is at least as extreme as observed.
    abs_sc_x_obs = abs(sc_weat_x_mean)
    abs_sc_y_obs = abs(sc_weat_y_mean)

    ab_idx = np.arange(2 * m)
    if m <= exact_threshold:
        sc_x_count = 0
        sc_y_count = 0
        total_sc = 0
        for combo in itertools.combinations(ab_idx, m):
            combo_arr = np.array(combo)
            mask = np.ones(2 * m, dtype=bool)
            mask[combo_arr] = False
            # numerator under this partition: mean over A' minus mean over B'
            num_perm = (
                cos_XY_AB[:, combo_arr].mean(axis=1)
                - cos_XY_AB[:, mask].mean(axis=1)
            )                                                  # shape (2n,)
            es_perm = num_perm / sc_denom_per_target           # shape (2n,)
            x_mean_perm = es_perm[:n].mean()
            y_mean_perm = es_perm[n:].mean()
            if abs(x_mean_perm) >= abs_sc_x_obs:
                sc_x_count += 1
            if abs(y_mean_perm) >= abs_sc_y_obs:
                sc_y_count += 1
            total_sc += 1
        sc_p_x = sc_x_count / total_sc
        sc_p_y = sc_y_count / total_sc
        sc_exact = True
        n_perms_sc = total_sc
    else:
        sc_x_count = 0
        sc_y_count = 0
        for _ in range(max_permutations):
            perm = rng.sample(range(2 * m), m)
            perm_arr = np.array(perm)
            mask = np.ones(2 * m, dtype=bool)
            mask[perm_arr] = False
            num_perm = (
                cos_XY_AB[:, perm_arr].mean(axis=1)
                - cos_XY_AB[:, mask].mean(axis=1)
            )
            es_perm = num_perm / sc_denom_per_target
            x_mean_perm = es_perm[:n].mean()
            y_mean_perm = es_perm[n:].mean()
            if abs(x_mean_perm) >= abs_sc_x_obs:
                sc_x_count += 1
            if abs(y_mean_perm) >= abs_sc_y_obs:
                sc_y_count += 1
        sc_p_x = sc_x_count / max_permutations
        sc_p_y = sc_y_count / max_permutations
        sc_exact = False
        n_perms_sc = max_permutations

    per_word_s = {w: float(v) for w, v in zip(XY_words, s_all)}

    return WEATResult(
        effect_size=weat_d,
        p_value=weat_p,
        test_statistic=weat_test_stat,
        sc_weat_x_mean=sc_weat_x_mean,
        sc_p_x=sc_p_x,
        sc_weat_y_mean=sc_weat_y_mean,
        sc_p_y=sc_p_y,
        per_word_sc_weat=per_word_sc_weat,
        n_X=len(X_bal),
        n_Y=len(Y_bal),
        n_A=len(A_bal),
        n_B=len(B_bal),
        n_permutations=n_perms_weat,
        n_permutations_sc=n_perms_sc,
        exact=weat_exact,
        exact_sc=sc_exact,
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
