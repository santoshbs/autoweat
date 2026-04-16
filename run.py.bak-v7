"""
run.py — AutoWEAT v13 runtime loop.

Pipeline per iteration:
  1. Read persona, config, current feed.
  2. Phase 1: LLM proposes X, Y, A, B labels and word lists.
  3. Locked Python computes WEAT D/p and SC-WEAT X/Y means and p-values.
  4. Phase 2 (conditional): IF WEAT is significant AND SC-WEAT directions
     are consistent with WEAT, LLM writes one interpretive paragraph.
     Otherwise no paragraph is produced.
  5. Feed entry written to docs/feed.json with schema `inductive.v7`.
  6. Optional git commit+push.

Feed entry schema (inductive.v7):
  id, timestamp, schema_version="inductive.v7",
  domain, contrast_label,
  labels: {X, Y, A, B},
  words:  {X_used, Y_used, A_used, B_used, X_dropped, Y_dropped, A_dropped, B_dropped},
  effect_size, p_value,
  sc_weat_x_mean, sc_p_x,
  sc_weat_y_mean, sc_p_y,
  n: {X, Y, A, B},
  n_permutations, exact,
  interpretive_paragraph: str (empty if consistency condition not met),
  interpretive_fired: bool,
  config: {embedding_backend, proposer_model, thinking_style, thinking_effort,
           history_size_seen, sampling},
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from autoweat.proposer import (
    OllamaProposer,
    Proposal,
    Interpretation,
    interpretive_fires,
)
from autoweat.weat import compute_weat, WEATResult
from autoweat.embeddings import load_backend


AUTOWEAT_SCHEMA_VERSION = "inductive.v7"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feed(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            return []
    return data if isinstance(data, list) else []


def write_feed(path: str, feed: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feed, f, ensure_ascii=False, indent=2)


def make_test_id(prop: Proposal, cfg: dict) -> str:
    """Deterministic id from labels + word lists + model."""
    h = hashlib.sha1()
    h.update(prop.X_label.encode("utf-8"))
    h.update(b"|")
    h.update(prop.Y_label.encode("utf-8"))
    h.update(b"|")
    h.update(prop.A_label.encode("utf-8"))
    h.update(b"|")
    h.update(prop.B_label.encode("utf-8"))
    h.update(b"|")
    h.update(",".join(sorted(prop.X_words)).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(sorted(prop.Y_words)).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(sorted(prop.A_words)).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(sorted(prop.B_words)).encode("utf-8"))
    h.update(b"|")
    h.update(cfg.get("proposer", {}).get("model", "").encode("utf-8"))
    return h.hexdigest()[:12]


def git_push(repo_root: str, message: str) -> None:
    try:
        subprocess.run(
            ["git", "add", "docs/feed.json"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "push"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        print(f"  pushed to git: {message}")
    except subprocess.CalledProcessError as e:
        # nothing to commit is fine; other failures just print
        msg = (e.stderr.decode() if e.stderr else "") + (
            e.stdout.decode() if e.stdout else ""
        )
        if "nothing to commit" in msg.lower():
            return
        print(f"  git push failed (non-fatal): {msg.strip()[:200]}")


def run_one(
    proposer: OllamaProposer,
    backend: Any,
    rng: Any,
    cfg: dict,
    feed: list[dict],
    recent_domains: list[str],
) -> dict | None:
    """Run one iteration: propose → compute → (maybe) interpret → store."""
    history_slice = feed[-15:]

    print("→ Phase 1: requesting proposal…")
    prop = proposer.propose(history=history_slice, recent_domains=recent_domains)
    print(f"  domain:   {prop.domain}")
    print(f"  contrast: {prop.contrast_label}")
    print(
        f"  X = {prop.X_label} ({len(prop.X_words)} words)\n"
        f"  Y = {prop.Y_label} ({len(prop.Y_words)} words)\n"
        f"  A = {prop.A_label} ({len(prop.A_words)} words)\n"
        f"  B = {prop.B_label} ({len(prop.B_words)} words)"
    )

    print("→ Computing WEAT + SC-WEAT…")
    weat_cfg = cfg.get("weat", {}) or {}
    try:
        result: WEATResult = compute_weat(
            X_words=prop.X_words,
            Y_words=prop.Y_words,
            A_words=prop.A_words,
            B_words=prop.B_words,
            embed_fn=backend.embed,
            vocab=backend.vocab,
            rng=rng,
            max_permutations=weat_cfg.get("max_permutations", 100_000),
            exact_threshold=weat_cfg.get("exact_threshold", 10),
        )
    except ValueError as e:
        print(f"  rejected (insufficient vocab): {e}")
        return None
    print(
        f"  WEAT:       D = {result.effect_size:+.3f}, "
        f"p = {result.p_value:.3f}"
    )
    print(
        f"  SC-WEAT(X): {result.sc_weat_x_mean:+.3f} "
        f"(p = {result.sc_p_x:.3f})"
    )
    print(
        f"  SC-WEAT(Y): {result.sc_weat_y_mean:+.3f} "
        f"(p = {result.sc_p_y:.3f})"
    )

    # Determine id and skip duplicates
    test_id = make_test_id(prop, cfg)
    if any(e.get("id") == test_id for e in feed):
        print(f"  skipped: duplicate id {test_id}")
        return None

    # Phase 2: conditional interpretation
    fires = interpretive_fires(
        result.effect_size,
        result.p_value,
        result.sc_weat_x_mean,
        result.sc_weat_y_mean,
    )
    if fires:
        print("→ Phase 2: requesting interpretive paragraph (fires)…")
        result_dict_for_interp = {
            "effect_size": result.effect_size,
            "p_value": result.p_value,
            "sc_weat_x_mean": result.sc_weat_x_mean,
            "sc_p_x": result.sc_p_x,
            "sc_weat_y_mean": result.sc_weat_y_mean,
            "sc_p_y": result.sc_p_y,
        }
        try:
            interp = proposer.interpret(prop, result_dict_for_interp)
            preview = interp.paragraph
            if len(preview) > 200:
                preview = preview[:197] + "..."
            print(f"  paragraph: {preview}")
        except Exception as e:
            print(f"  interpretation failed (test still recorded): {e}")
            interp = Interpretation(paragraph="", fired=False, raw="")
    else:
        print("→ Phase 2: skipped (consistency condition not met)")
        interp = Interpretation(paragraph="", fired=False, raw="")

    entry = {
        "id": test_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": AUTOWEAT_SCHEMA_VERSION,
        "domain": prop.domain,
        "contrast_label": prop.contrast_label,
        "labels": {
            "X": prop.X_label,
            "Y": prop.Y_label,
            "A": prop.A_label,
            "B": prop.B_label,
        },
        "effect_size": result.effect_size,
        "p_value": result.p_value,
        "sc_weat_x_mean": result.sc_weat_x_mean,
        "sc_p_x": result.sc_p_x,
        "sc_weat_y_mean": result.sc_weat_y_mean,
        "sc_p_y": result.sc_p_y,
        "n": {
            "X": result.n_X,
            "Y": result.n_Y,
            "A": result.n_A,
            "B": result.n_B,
        },
        "n_permutations": result.n_permutations,
        "exact": result.exact,
        "words": {
            "X_used": result.X_used,
            "Y_used": result.Y_used,
            "A_used": result.A_used,
            "B_used": result.B_used,
            "X_dropped": result.X_dropped,
            "Y_dropped": result.Y_dropped,
            "A_dropped": result.A_dropped,
            "B_dropped": result.B_dropped,
        },
        "interpretive_paragraph": interp.paragraph,
        "interpretive_fired": interp.fired,
        "config": {
            "embedding_backend": backend.name,
            "proposer_model": cfg["proposer"]["model"],
            "thinking_style": cfg["proposer"].get("thinking_style", "none"),
            "thinking_effort": cfg["proposer"].get("thinking_effort"),
            "history_size_seen": len(history_slice),
            "sampling": cfg["proposer"].get("sampling", {}),
        },
    }
    feed.append(entry)
    recent_domains.append(prop.domain)
    print(f"  ✓ accepted as {test_id}")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--once", action="store_true", help="run one iteration and exit")
    ap.add_argument("--no-push", action="store_true", help="do not git push")
    ap.add_argument("--rounds", type=int, default=0, help="number of rounds (0 = until killed)")
    ap.add_argument("--sleep", type=int, default=20, help="seconds between rounds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo_root = os.path.dirname(os.path.abspath(args.config))
    feed_path = os.path.join(repo_root, cfg.get("feed_path", "docs/feed.json"))
    persona_path = os.path.join(repo_root, cfg.get("persona_path", "persona.md"))

    print(f"autoweat v13 | config={args.config}")
    print(f"  feed:    {feed_path}")
    print(f"  persona: {persona_path}")

    backend = load_backend(cfg["embedding"])
    print(f"  backend: {backend.name}")

    # Seeded RNG for reproducible permutations and balancing.
    import random as _random
    rng = _random.Random(cfg.get("seed", 20260413))

    proposer = OllamaProposer(
        model=cfg["proposer"]["model"],
        persona_path=persona_path,
        host=cfg["proposer"].get("host", "http://localhost:11434"),
        sampling=cfg["proposer"].get("sampling", {}),
        thinking_style=cfg["proposer"].get("thinking_style", "none"),
        thinking_effort=cfg["proposer"].get("thinking_effort"),
        num_ctx=cfg["proposer"].get("num_ctx"),
    )

    feed = load_feed(feed_path)
    print(f"  feed entries: {len(feed)}")
    recent_domains: list[str] = [
        e.get("domain", "") for e in feed[-12:] if e.get("domain")
    ]

    round_num = 0
    while True:
        round_num += 1
        print(f"\n─── round {round_num} ───")
        try:
            entry = run_one(proposer, backend, rng, cfg, feed, recent_domains)
            if entry is not None:
                write_feed(feed_path, feed)
                if not args.no_push:
                    git_push(
                        repo_root,
                        f"autoweat: {entry['id']} {entry['domain']} "
                        f"d={entry['effect_size']:+.2f}",
                    )
        except Exception as e:
            print(f"  round failed: {e}")

        if args.once:
            break
        if args.rounds and round_num >= args.rounds:
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
