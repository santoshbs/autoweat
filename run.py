"""
run.py — AutoWEAT v15 runtime loop.

Pipeline per iteration:

  1. Load feed + concept cache.
  2. Pick mode (alternating: even feed length → well-studied, odd → novel).
  3. Derive cooled domains (used >= 3 times in last 12 feed entries),
     with a 12-entry warm-up period during which no cooling is applied.
  4. Build prompt inputs: history signatures + cached-concept summary.
  5. Ask the LLM for a proposal. Validate against six rules:
       V1. Domain is in taxonomy.
       V2. Full 4-tuple signature not already in feed.
       V3. For each of X, Y, A, B: if the concept is cached, the
           proposed word list must Jaccard-overlap the cached list >= 0.70.
           If the LLM proposed a label that canonicalizes to the same
           key as a cached concept but used a different display label,
           that's also a violation (rule R1).
       V4. Proposal has >= 20 words per list and the lists are alphabetic-only.
           (Hard-checked here so run.py does not have to trust the LLM.)
       V5. Novel mode: `prediction` block is present, valid, and
           confidence is NOT "high" (high confidence → contrast is not
           novel enough).
       V6. Domain cooling: if the chosen domain is cooled AND the
           warmup period has passed, issue a gentle warning (not a
           rejection — cooling is a nudge, not a hard rule).
     Retry up to 5 times, passing structured feedback to the LLM on
     each retry so it can correct specific problems.
  6. Compute WEAT on the validated proposal.
  7. Update concept cache: any new concept not previously seen gets
     added with the current proposal's word list.
  8. (Optional) Phase 2 interpretation if p < .05.
  9. Write feed entry (schema inductive.v9), save cache, git push.

Feed entry schema (inductive.v9):
  id, timestamp, schema_version="inductive.v9",
  domain, contrast_label,
  labels: {X, Y, A, B},
  canonical_keys: {X, Y, A, B},  # concept cache keys, for filtering
  words: {X_used, Y_used, A_used, B_used, X_dropped, ...},
  effect_size, p_value,
  n, n_permutations, exact,
  prediction: {expected_direction, confidence, rationale} | null,
  interpretive_paragraph, interpretive_fired,
  config: {
    embedding_backend, proposer_model, proposer_mode,
    thinking_style, thinking_effort, history_size_seen, sampling
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import yaml

from autoweat.proposer import (
    OllamaProposer,
    Proposal,
    Prediction,
    Interpretation,
    interpretive_fires,
)
from autoweat.weat import compute_weat, WEATResult
from autoweat.embeddings import load_backend
from autoweat.domains import normalize_domain, VALID_DOMAINS
from autoweat.concepts import (
    ConceptCache,
    canonical_key,
    canonical_tokens,
    jaccard,
)


AUTOWEAT_SCHEMA_VERSION = "inductive.v9"

# ── Thresholds (matching the design decisions) ─────────────────────
COOLING_THRESHOLD = 3      # domain uses in last 12 entries to cool it
COOLING_WINDOW = 12
COOLING_WARMUP = 12        # no cooling until feed has this many entries
WORD_LIST_REUSE_THRESHOLD = 0.70   # Jaccard for reused-concept word lists
CONCEPT_MATCH_THRESHOLD = 0.60     # Jaccard for concept label matching
MAX_PROPOSAL_RETRIES = 5
MIN_POOL_SIZE_FLOOR = 20


# ─── I/O ───────────────────────────────────────────────────────────

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


# ─── Dedup signature ───────────────────────────────────────────────
#
# The dedup key for a test is the canonical 4-tuple. We use the canonical
# KEYS (not the raw labels) so that "women and feminine terms" and
# "feminine terms" collide when they should.
#
# Order-sensitivity: (X, Y, A, B) and (Y, X, B, A) are the same test with
# sign flipped. We canonicalize by sorting within the target pair and
# within the attribute pair — the resulting signature ignores ordering.

def dedup_signature(x_key: str, y_key: str, a_key: str, b_key: str) -> str:
    target_pair = "~".join(sorted([x_key, y_key]))
    attr_pair = "~".join(sorted([a_key, b_key]))
    return target_pair + "//" + attr_pair


def make_test_id(prop: Proposal, cfg: dict) -> str:
    """Deterministic id from labels + word lists + model (unchanged from v14)."""
    h = hashlib.sha1()
    for field_val in (
        prop.X_label, prop.Y_label, prop.A_label, prop.B_label,
        ",".join(sorted(prop.X_words)),
        ",".join(sorted(prop.Y_words)),
        ",".join(sorted(prop.A_words)),
        ",".join(sorted(prop.B_words)),
        cfg.get("proposer", {}).get("model", ""),
    ):
        h.update(field_val.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


# ─── Cooled-domain logic ───────────────────────────────────────────

def compute_cooled_domains(feed: list[dict]) -> list[str]:
    """Return the list of domains currently cooled.

    A domain is cooled when:
      - the feed has >= COOLING_WARMUP entries total (warmup complete)
      - AND the domain appears >= COOLING_THRESHOLD times in the last
        COOLING_WINDOW feed entries.
    """
    if len(feed) < COOLING_WARMUP:
        return []
    window = feed[-COOLING_WINDOW:]
    counts = Counter()
    for entry in window:
        d = entry.get("domain", "")
        if d:
            counts[d] += 1
    return [d for d, c in counts.items() if c >= COOLING_THRESHOLD]


# ─── History + cache summaries for the prompt ──────────────────────

def format_history_summary(feed: list[dict], n: int = 15) -> str:
    """Format the last n feed entries as a compact signature list
    for the user-message of the next proposal."""
    if not feed:
        return ""
    recent = feed[-n:]
    lines: list[str] = []
    for h in recent:
        labels = h.get("labels", {}) or {}
        x = labels.get("X", "?")
        y = labels.get("Y", "?")
        a = labels.get("A", "?")
        b = labels.get("B", "?")
        d = h.get("effect_size", 0.0) or 0.0
        p = h.get("p_value", 1.0) or 1.0
        dom = h.get("domain", "?")
        lines.append(
            f"  · [{dom}] {x} vs {y} × {a} vs {b}  (d={d:+.2f}, p={p:.3f})"
        )
    return "\n".join(lines)


def format_cached_concepts(cache: ConceptCache, max_show: int = 40) -> str:
    """Summarize the concept cache for inclusion in the user message.

    Shows each cached concept's display label and first 6 words so the
    LLM can recognize reuse and match the cached form. If the cache is
    too large to show in full, we trim to max_show entries (most recent).
    """
    if cache.size() == 0:
        return ""
    # Get all entries with first_seen timestamp
    entries: list[tuple[str, dict]] = list(
        cache._data.items()  # Intentional internal access; single-module use.
    )
    # Sort by first_seen descending (most recently introduced first)
    entries.sort(
        key=lambda kv: kv[1].get("first_seen", ""),
        reverse=True,
    )
    entries = entries[:max_show]
    lines: list[str] = []
    for key, obj in entries:
        label = obj.get("display_label", key)
        words = obj.get("words", [])
        preview = ", ".join(words[:6])
        if len(words) > 6:
            preview += f", … ({len(words)} total)"
        lines.append(f"  · {label}  →  [{preview}]")
    return "\n".join(lines)


# ─── Validation ────────────────────────────────────────────────────

def validate_proposal(
    prop: Proposal,
    feed: list[dict],
    cache: ConceptCache,
    mode: str,
) -> list[str]:
    """Run all validation checks on a proposal. Return a list of human-
    readable failure messages. Empty list means validation passed.

    The messages are phrased so they can be concatenated and fed back to
    the LLM verbatim on retry.
    """
    problems: list[str] = []

    # ── V1: domain is in taxonomy ───────────────────────────────
    canonical_domain = normalize_domain(prop.domain)
    if canonical_domain is None:
        problems.append(
            f"Your domain '{prop.domain}' is not in the taxonomy. "
            f"You must pick one canonical domain name from the "
            f"taxonomy list exactly as written."
        )
    else:
        prop.domain = canonical_domain  # Normalize in place

    # ── V4: word-list hygiene ───────────────────────────────────
    for side, words in [
        ("X", prop.X_words), ("Y", prop.Y_words),
        ("A", prop.A_words), ("B", prop.B_words),
    ]:
        if len(words) < MIN_POOL_SIZE_FLOOR:
            problems.append(
                f"Your {side} word list has only {len(words)} words after "
                f"cleaning. Minimum is {MIN_POOL_SIZE_FLOOR}. All words must "
                f"be single common English words (lowercase, alphabetic, no "
                f"phrases, no hyphens)."
            )

    # ── V3: concept-reuse rules (R1 and R2) ─────────────────────
    # For each proposed concept, check whether the canonical key matches
    # a cached concept. If so, verify (a) the label is the same display
    # form and (b) the word list overlaps >= 70%.
    concept_keys: dict[str, str] = {}
    for side, label, words in [
        ("X", prop.X_label, prop.X_words),
        ("Y", prop.Y_label, prop.Y_words),
        ("A", prop.A_label, prop.A_words),
        ("B", prop.B_label, prop.B_words),
    ]:
        hit = cache.lookup(label, fuzzy_threshold=CONCEPT_MATCH_THRESHOLD)
        if hit is None:
            # Not a reused concept — fine
            concept_keys[side] = canonical_key(label)
            continue
        cached_key, cached_obj = hit
        concept_keys[side] = cached_key

        # Rule R1: same concept, same display label
        cached_label = cached_obj.get("display_label", "")
        if cached_label and cached_label != label:
            problems.append(
                f"Your {side}_label '{label}' refers to the same concept "
                f"as the cached concept '{cached_label}'. When reusing a "
                f"concept, you must use the exact same label: "
                f"'{cached_label}'."
            )

        # Rule R2: word list overlap >= 70%
        cached_words = cached_obj.get("words", [])
        overlap = jaccard(words, cached_words)
        if overlap < WORD_LIST_REUSE_THRESHOLD:
            problems.append(
                f"Your {side} word list for '{label}' only overlaps "
                f"{overlap*100:.0f}% with the cached list (need >= "
                f"{int(WORD_LIST_REUSE_THRESHOLD*100)}%). The cached word "
                f"list for this concept is: {cached_words}. Either match "
                f"that list more closely, or pick a different concept "
                f"with a distinguishing label."
            )

    # ── V2: exact 4-tuple signature must not already exist ──────
    sig = dedup_signature(
        concept_keys["X"], concept_keys["Y"],
        concept_keys["A"], concept_keys["B"],
    )
    for entry in feed:
        entry_keys = entry.get("canonical_keys", {}) or {}
        if not entry_keys:
            continue
        entry_sig = dedup_signature(
            entry_keys.get("X", ""), entry_keys.get("Y", ""),
            entry_keys.get("A", ""), entry_keys.get("B", ""),
        )
        if entry_sig == sig:
            labels = entry.get("labels", {})
            problems.append(
                f"The 4-tuple ({prop.X_label} vs {prop.Y_label} × "
                f"{prop.A_label} vs {prop.B_label}) has already been run "
                f"as test {entry.get('id','?')} "
                f"({labels.get('X','?')} vs {labels.get('Y','?')} × "
                f"{labels.get('A','?')} vs {labels.get('B','?')}). "
                f"Propose something different."
            )
            break

    # ── V5: novel mode prediction ──────────────────────────────
    if mode == "novel":
        if prop.prediction is None or not prop.prediction.is_valid():
            problems.append(
                "You are in novel mode. You must include a `prediction` "
                "block with valid `expected_direction` (positive|negative|"
                "uncertain), `confidence` (low|medium|high), and a "
                "`rationale` (one sentence)."
            )
        elif prop.prediction.confidence == "high":
            problems.append(
                "Your prediction confidence is 'high'. In novel mode, the "
                "contrast must be one whose result you cannot confidently "
                "predict. If you can predict it with high confidence, it "
                "is not novel — redesign the contrast so the answer is "
                "genuinely open, or pick an entirely different contrast."
            )

    return problems


# ─── git ───────────────────────────────────────────────────────────

def git_push(repo_root: str, message: str) -> None:
    try:
        subprocess.run(
            ["git", "add", "docs/feed.json", "docs/concept_cache.json"],
            cwd=repo_root, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_root, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "push"],
            cwd=repo_root, check=True, capture_output=True,
        )
        print(f"  pushed to git: {message}")
    except subprocess.CalledProcessError as e:
        msg = (e.stderr.decode() if e.stderr else "") + (
            e.stdout.decode() if e.stdout else ""
        )
        if "nothing to commit" in msg.lower():
            return
        print(f"  git push failed (non-fatal): {msg.strip()[:200]}")


# ─── One iteration ─────────────────────────────────────────────────

def run_one(
    proposer: OllamaProposer,
    backend: Any,
    rng: Any,
    cfg: dict,
    feed: list[dict],
    cache: ConceptCache,
) -> dict | None:
    """Propose → validate → compute → record. Returns feed entry or None."""
    # Mode alternation
    mode = "novel" if (len(feed) % 2 == 1) else "well-studied"

    # Cooled domains
    cooled = compute_cooled_domains(feed)

    # Prompt inputs
    history_summary = format_history_summary(feed, n=15)
    cached_summary = format_cached_concepts(cache, max_show=40)

    print(f"→ Phase 1: requesting proposal… (mode: {mode}"
          + (f", cooled: {', '.join(cooled)}" if cooled else "")
          + ")")

    # Retry loop with feedback
    prop: Proposal | None = None
    feedback = ""
    problems: list[str] = []
    for attempt in range(1, MAX_PROPOSAL_RETRIES + 1):
        try:
            candidate = proposer.propose_raw(
                mode=mode,
                history_summary=history_summary,
                cooled_domains=cooled,
                cached_concept_summary=cached_summary,
                feedback=feedback,
            )
        except Exception as e:
            print(f"  attempt {attempt}: proposal parse failed — {e}")
            feedback = (
                "Your previous output was not valid JSON or was missing "
                "required fields. Re-read the schema and emit exactly the "
                "fields requested. No markdown fences, no commentary."
            )
            continue

        problems = validate_proposal(candidate, feed, cache, mode)
        if not problems:
            prop = candidate
            print(f"  attempt {attempt}: accepted")
            break

        # Problems — prepare feedback and retry
        print(f"  attempt {attempt}: {len(problems)} validation problem(s)")
        for p in problems:
            print(f"    · {p[:120]}" + ("…" if len(p) > 120 else ""))
        feedback = (
            "Your previous proposal had these problems — fix them all and "
            "resubmit:\n\n"
            + "\n".join(f"  {i+1}. {msg}" for i, msg in enumerate(problems))
        )

    if prop is None:
        print(f"  ✗ giving up after {MAX_PROPOSAL_RETRIES} attempts")
        return None

    # Print the accepted proposal
    print(f"  domain:   {prop.domain}")
    print(f"  contrast: {prop.contrast_label}")
    print(
        f"  X = {prop.X_label} ({len(prop.X_words)} words)\n"
        f"  Y = {prop.Y_label} ({len(prop.Y_words)} words)\n"
        f"  A = {prop.A_label} ({len(prop.A_words)} words)\n"
        f"  B = {prop.B_label} ({len(prop.B_words)} words)"
    )
    if prop.prediction is not None:
        print(
            f"  prediction: {prop.prediction.expected_direction} "
            f"({prop.prediction.confidence}) — "
            f"{prop.prediction.rationale}"
        )

    # Compute WEAT
    print("→ Computing WEAT…")
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
            exact_threshold=weat_cfg.get("exact_threshold", 12),
            min_pool_size=weat_cfg.get("min_pool_size", MIN_POOL_SIZE_FLOOR),
        )
    except ValueError as e:
        print(f"  rejected (insufficient in-vocab words): {e}")
        return None
    print(
        f"  WEAT:  d = {result.effect_size:+.3f}, "
        f"p = {result.p_value:.3f}  "
        f"(n={result.n_X}, exact={result.exact}, "
        f"perms={result.n_permutations})"
    )

    # Assemble canonical keys for this test (what we'll store in the entry)
    canonical_keys = {
        "X": canonical_key(prop.X_label),
        "Y": canonical_key(prop.Y_label),
        "A": canonical_key(prop.A_label),
        "B": canonical_key(prop.B_label),
    }

    # Update the concept cache. For each of the four concepts, if it's
    # not already in the cache (by canonical key OR fuzzy match), add it.
    # If it IS already there, leave the cached list alone — validation
    # has ensured the new list matches at >= 70% anyway.
    now = datetime.now(timezone.utc).isoformat()
    for side, label, words in [
        ("X", prop.X_label, prop.X_words),
        ("Y", prop.Y_label, prop.Y_words),
        ("A", prop.A_label, prop.A_words),
        ("B", prop.B_label, prop.B_words),
    ]:
        if cache.lookup(label, fuzzy_threshold=CONCEPT_MATCH_THRESHOLD) is None:
            cache.insert(label, words, now)
    cache.save()

    # Phase 2: interpretation if significant
    fires = interpretive_fires(result.effect_size, result.p_value)
    if fires:
        print("→ Phase 2: requesting interpretive paragraph (p < .05)…")
        try:
            interp = proposer.interpret(prop, {
                "effect_size": result.effect_size,
                "p_value": result.p_value,
            })
            preview = interp.paragraph
            if len(preview) > 200:
                preview = preview[:197] + "..."
            print(f"  paragraph: {preview}")
        except Exception as e:
            print(f"  interpretation failed (test still recorded): {e}")
            interp = Interpretation(paragraph="", fired=False, raw="")
    else:
        print("→ Phase 2: skipped (p >= .05)")
        interp = Interpretation(paragraph="", fired=False, raw="")

    # Feed entry
    test_id = make_test_id(prop, cfg)
    entry: dict[str, Any] = {
        "id": test_id,
        "timestamp": now,
        "schema_version": AUTOWEAT_SCHEMA_VERSION,
        "domain": prop.domain,
        "contrast_label": prop.contrast_label,
        "labels": {
            "X": prop.X_label, "Y": prop.Y_label,
            "A": prop.A_label, "B": prop.B_label,
        },
        "canonical_keys": canonical_keys,
        "effect_size": result.effect_size,
        "p_value": result.p_value,
        "n": {
            "X": result.n_X, "Y": result.n_Y,
            "A": result.n_A, "B": result.n_B,
        },
        "n_permutations": result.n_permutations,
        "exact": result.exact,
        "words": {
            "X_used": result.X_used, "Y_used": result.Y_used,
            "A_used": result.A_used, "B_used": result.B_used,
            "X_dropped": result.X_dropped, "Y_dropped": result.Y_dropped,
            "A_dropped": result.A_dropped, "B_dropped": result.B_dropped,
        },
        "prediction": (
            prop.prediction.to_dict() if prop.prediction is not None else None
        ),
        "interpretive_paragraph": interp.paragraph,
        "interpretive_fired": interp.fired,
        "config": {
            "embedding_backend": backend.name,
            "proposer_model": cfg["proposer"]["model"],
            "proposer_mode": mode,
            "thinking_style": cfg["proposer"].get("thinking_style", "none"),
            "thinking_effort": cfg["proposer"].get("thinking_effort"),
            "history_size_seen": len(feed[-15:]),
            "cooled_domains_at_proposal": cooled,
            "sampling": cfg["proposer"].get("sampling", {}),
        },
    }
    feed.append(entry)
    print(f"  ✓ accepted as {test_id}")
    return entry


# ─── main ──────────────────────────────────────────────────────────

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
    cache_path = os.path.join(repo_root, cfg.get("cache_path", "docs/concept_cache.json"))

    print(f"autoweat v15 (domains + prediction + dedup) | config={args.config}")
    print(f"  feed:    {feed_path}")
    print(f"  persona: {persona_path}")
    print(f"  cache:   {cache_path}")

    backend = load_backend(cfg["embedding"])
    print(f"  backend: {backend.name}")

    import random as _random
    rng = _random.Random(cfg.get("seed", 20260417))

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
    cache = ConceptCache(cache_path)
    print(f"  feed entries: {len(feed)}  |  cached concepts: {cache.size()}")

    round_num = 0
    while True:
        round_num += 1
        print(f"\n─── round {round_num} ───")
        try:
            entry = run_one(proposer, backend, rng, cfg, feed, cache)
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
            import traceback
            traceback.print_exc()

        if args.once:
            break
        if args.rounds and round_num >= args.rounds:
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
