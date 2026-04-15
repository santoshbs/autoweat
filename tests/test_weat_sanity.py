"""Sanity test for the WEAT computer using a tiny hand-built embedding space."""
import random
import sys
sys.path.insert(0, "/home/claude/autoweat")

import numpy as np
from autoweat.weat import compute_weat


def main():
    # Build a fake 3-D embedding space.
    # X words live near [1, 0, 0], Y near [-1, 0, 0],
    # A near [1, 0, 0],  B near [-1, 0, 0].
    # So X should associate strongly with A, Y with B → big positive d.
    rng = np.random.default_rng(42)
    vectors = {}
    for w in ["x1", "x2", "x3", "x4", "x5", "a1", "a2", "a3", "a4", "a5"]:
        vectors[w] = np.array([1.0, 0, 0]) + 0.05 * rng.standard_normal(3)
    for w in ["y1", "y2", "y3", "y4", "y5", "b1", "b2", "b3", "b4", "b5"]:
        vectors[w] = np.array([-1.0, 0, 0]) + 0.05 * rng.standard_normal(3)
    # An OOV word to test filtering
    vectors_vocab = set(vectors.keys())

    def embed(w):
        return vectors[w]

    res = compute_weat(
        X_words=["x1", "x2", "x3", "x4", "x5", "OOV_WORD"],
        Y_words=["y1", "y2", "y3", "y4", "y5"],
        A_words=["a1", "a2", "a3", "a4", "a5"],
        B_words=["b1", "b2", "b3", "b4", "b5"],
        embed_fn=embed,
        vocab=vectors_vocab,
        rng=random.Random(0),
        exact_threshold=10,
    )
    print(f"effect_size = {res.effect_size:.4f}  (expect ~ +1.8)")
    print(f"p_value     = {res.p_value:.6f}      (expect near 0)")
    print(f"n_X={res.n_X} n_Y={res.n_Y} n_A={res.n_A} n_B={res.n_B}")
    print(f"exact       = {res.exact}  perms = {res.n_permutations}")
    print(f"X dropped   = {res.X_dropped}  (expect ['OOV_WORD'])")

    assert res.effect_size > 1.5, "effect size should be strongly positive"
    assert res.p_value < 0.01, "p-value should be tiny"
    assert res.X_dropped == ["OOV_WORD"]
    assert res.exact is True

    # Now the null case: X, Y, and A all drawn from same distribution; B elsewhere.
    # Use a FRESH rng so noise is iid.
    rng2 = np.random.default_rng(123)
    vectors2 = {}
    for w in ["x1", "x2", "x3", "x4", "x5", "y1", "y2", "y3", "y4", "y5",
              "a1", "a2", "a3", "a4", "a5"]:
        vectors2[w] = np.array([1.0, 0, 0]) + 0.05 * rng2.standard_normal(3)
    for w in ["b1", "b2", "b3", "b4", "b5"]:
        vectors2[w] = np.array([-1.0, 0, 0]) + 0.05 * rng2.standard_normal(3)

    def embed2(w):
        return vectors2[w]

    res2 = compute_weat(
        X_words=["x1", "x2", "x3", "x4", "x5"],
        Y_words=["y1", "y2", "y3", "y4", "y5"],
        A_words=["a1", "a2", "a3", "a4", "a5"],
        B_words=["b1", "b2", "b3", "b4", "b5"],
        embed_fn=embed2,
        vocab=set(vectors2.keys()),
        rng=random.Random(0),
    )
    print()
    print(f"NULL effect = {res2.effect_size:.4f}  (expect near 0)")
    print(f"NULL p      = {res2.p_value:.4f}      (expect ~0.5)")
    assert abs(res2.effect_size) < 1.0, "should be small, actual noise can be modest at n=5"
    assert res2.p_value > 0.05, "null should not reject"
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
