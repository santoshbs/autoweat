"""
proposer.py — AutoWEAT v15 proposer.

Changes from v14 → v15:
  - Injects the domain taxonomy (from autoweat.domains) into the system
    prompt, and requires the LLM to pick a primary domain.
  - Adds a `prediction` field to proposals (required in novel mode,
    optional in well-studied mode).
  - Adds `cooled_domains` to the propose() signature so the runtime
    can tell the LLM which domains to avoid.
  - Adds `cached_concepts` (brief summaries from the concept cache)
    so the LLM can see which concepts have frozen word lists.
  - Retry-on-failure loop now passes structured feedback back to the
    LLM so it can correct specific violations (not just re-roll).

Proposal JSON schema (v15):
  {
    "domain": "<canonical domain name>",
    "contrast_label": "<short description>",
    "X_label": "...", "Y_label": "...",
    "A_label": "...", "B_label": "...",
    "X_words": [...], "Y_words": [...],
    "A_words": [...], "B_words": [...],
    "prediction": {                    # required in novel mode
      "expected_direction": "positive" | "negative" | "uncertain",
      "confidence": "low" | "medium" | "high",
      "rationale": "<one sentence>"
    }
  }

Phase 2 (interpretation) is unchanged from v14.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from autoweat.domains import (
    DOMAINS,
    VALID_DOMAINS,
    format_taxonomy_for_prompt,
    normalize_domain,
)


AUTOWEAT_PROPOSER_VERSION = "2026-04-17-v15-domains-prediction"


# ─── Phase 1: proposal schema text (injected into system prompt) ───

PROPOSAL_SCHEMA = """Output exactly this JSON structure. Do not add fields or
comments. Do not wrap in markdown fences.

{
  "domain": "<canonical domain name from the taxonomy, e.g. 'sociology'>",
  "contrast_label": "<short natural-English description of what you are probing>",
  "X_label": "<short label for target set X>",
  "Y_label": "<short label for target set Y>",
  "A_label": "<short label for attribute set A>",
  "B_label": "<short label for attribute set B>",
  "X_words": [<20-25 common English words representing X>],
  "Y_words": [<20-25 common English words representing Y>],
  "A_words": [<20-25 common English words representing A>],
  "B_words": [<20-25 common English words representing B>],
  "prediction": {
    "expected_direction": "<one of: positive, negative, uncertain>",
    "confidence": "<one of: low, medium, high>",
    "rationale": "<one sentence, no citations>"
  }
}

In novel mode the `prediction` block is REQUIRED. In well-studied mode
it is optional — you may omit the whole `prediction` key if you wish,
but if you include it you must fill it fully.

Labels must be natural English phrases. Words must be common lowercase
single English words: no phrases, no hyphens, no proper nouns, no rare
jargon. All four word lists should be roughly 20–25 words each."""


# ─── Phase 2: interpretation prompt (unchanged from v14) ───────────

INTERPRETATION_PROMPT = """You are being asked to write ONE short paragraph (4-7 sentences)
interpreting a WEAT result. The WEAT is statistically significant
(p < .05), meaning the relative association between the two target
sets and the two attribute sets is unlikely to be noise.

Write the paragraph in three movements:

(1) What the result shows in plain English. State the finding as an
    observation about how the corpus uses this vocabulary. Name the
    four labels (X, Y, A, B) naturally; do not invent new categories.
    The direction of the effect: if WEAT d is positive, X-words lean
    toward A relative to Y-words; if negative, X-words lean toward B
    relative to Y-words.

(2) A plausible explanation for why this pattern might exist in a
    corpus of English-language web text. Not a proof, just a
    thoughtful possibility.

(3) Why a social scientist studying the relevant domain might find
    this interesting or important. Connect it to real questions
    about how language encodes social structure.

DO NOT name any specific authors, papers, theories, or frameworks.
DO NOT hedge excessively. DO NOT apologize for the numbers. Write
as a researcher sharing a finding.

Output ONLY the paragraph itself. No JSON, no preamble, no headings,
no markdown. Just the paragraph text."""


# ─── dataclasses ───────────────────────────────────────────────────


@dataclass
class Prediction:
    expected_direction: str = ""   # "positive" | "negative" | "uncertain"
    confidence: str = ""           # "low" | "medium" | "high"
    rationale: str = ""

    def is_valid(self) -> bool:
        return (
            self.expected_direction in ("positive", "negative", "uncertain")
            and self.confidence in ("low", "medium", "high")
            and bool(self.rationale.strip())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_direction": self.expected_direction,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass
class Proposal:
    domain: str
    contrast_label: str
    X_label: str
    Y_label: str
    A_label: str
    B_label: str
    X_words: list[str]
    Y_words: list[str]
    A_words: list[str]
    B_words: list[str]
    prediction: Prediction | None = None
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "domain": self.domain,
            "contrast_label": self.contrast_label,
            "X_label": self.X_label, "Y_label": self.Y_label,
            "A_label": self.A_label, "B_label": self.B_label,
            "X_words": self.X_words, "Y_words": self.Y_words,
            "A_words": self.A_words, "B_words": self.B_words,
        }
        if self.prediction is not None:
            d["prediction"] = self.prediction.to_dict()
        return d


@dataclass
class Interpretation:
    paragraph: str = ""
    fired: bool = False
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"paragraph": self.paragraph, "fired": self.fired}


# ─── Significance gate ─────────────────────────────────────────────


def interpretive_fires(effect_size: float, p_value: float) -> bool:
    """Fires iff WEAT p < .05 and the effect is non-zero."""
    if p_value is None or p_value >= 0.05:
        return False
    if effect_size is None or effect_size == 0:
        return False
    return True


# ─── Proposer class ────────────────────────────────────────────────


class OllamaProposer:
    """LLM proposer for AutoWEAT v15."""

    GPT_OSS_EFFORTS = ("low", "medium", "high")

    def __init__(
        self,
        model: str,
        persona_path: str,
        host: str = "http://localhost:11434",
        sampling: dict | None = None,
        thinking_style: str = "none",
        thinking_effort: str | None = None,
        num_ctx: int | None = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        with open(persona_path, "r", encoding="utf-8") as f:
            self.persona = f.read()
        self.sampling = dict(sampling or {})
        self.thinking_style = thinking_style
        self.thinking_effort = thinking_effort
        self.num_ctx = num_ctx

    # ─── Shared request helper ──────────────────────────────────

    def _chat(self, system: str, user: str) -> str:
        """Use /api/generate with `system` field."""
        options: dict[str, Any] = {}
        options.update(self.sampling)
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        sys_prompt = system
        if self.thinking_style == "gpt-oss" and self.thinking_effort:
            if self.thinking_effort in self.GPT_OSS_EFFORTS:
                sys_prompt = f"Reasoning: {self.thinking_effort}\n\n" + system
        payload: dict[str, Any] = {
            "model": self.model,
            "system": sys_prompt,
            "prompt": user,
            "stream": False,
            "options": options,
        }
        if self.thinking_style == "qwen3":
            payload["think"] = (
                self.thinking_effort != "off"
                and self.thinking_effort is not None
            )

        r = requests.post(
            f"{self.host}/api/generate", json=payload, timeout=900,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    # ─── Phase 1: propose ───────────────────────────────────────

    def propose_raw(
        self,
        mode: str,
        history_summary: str,
        cooled_domains: list[str],
        cached_concept_summary: str,
        feedback: str = "",
    ) -> Proposal:
        """Single proposal attempt. Raises on parse failure.

        Validation is handled by the caller (run.py). This method just
        returns whatever the LLM said, parsed into a Proposal. If it
        fails validation, the caller passes `feedback` back into a
        subsequent call so the LLM can correct specific problems.
        """
        system = self._propose_system_prompt(mode, cooled_domains)
        user = self._propose_user_message(
            mode=mode,
            history_summary=history_summary,
            cached_concept_summary=cached_concept_summary,
            feedback=feedback,
        )
        raw = self._chat(system, user)
        data = self._extract_json(raw)

        # Prediction sub-object
        pred_obj = data.get("prediction")
        prediction: Prediction | None = None
        if isinstance(pred_obj, dict):
            prediction = Prediction(
                expected_direction=str(pred_obj.get("expected_direction", "")).strip().lower(),
                confidence=str(pred_obj.get("confidence", "")).strip().lower(),
                rationale=str(pred_obj.get("rationale", "")).strip(),
            )

        return Proposal(
            domain=str(data.get("domain", "")).strip().lower(),
            contrast_label=str(data.get("contrast_label", "")).strip(),
            X_label=str(data.get("X_label", "")).strip(),
            Y_label=str(data.get("Y_label", "")).strip(),
            A_label=str(data.get("A_label", "")).strip(),
            B_label=str(data.get("B_label", "")).strip(),
            X_words=self._clean_words(data.get("X_words", [])),
            Y_words=self._clean_words(data.get("Y_words", [])),
            A_words=self._clean_words(data.get("A_words", [])),
            B_words=self._clean_words(data.get("B_words", [])),
            prediction=prediction,
            raw=raw,
        )

    def _propose_system_prompt(self, mode: str, cooled_domains: list[str]) -> str:
        taxonomy = format_taxonomy_for_prompt()

        if mode == "novel":
            mode_directive = (
                "MODE FOR THIS ITERATION: NOVEL. Propose a contrast that is "
                "interesting, potentially unexpected or counter-intuitive, "
                "AND substantively important. If you can predict the result "
                "with high confidence before it runs, the contrast is not "
                "novel — pick a different one. You MUST fill in the "
                "`prediction` block."
            )
        else:
            mode_directive = (
                "MODE FOR THIS ITERATION: WELL-STUDIED. Propose a classic, "
                "substantively important baseline contrast from the chosen "
                "domain. Classic does not mean 'only gender and race' — "
                "every domain has classic contrasts. The `prediction` "
                "block is optional in this mode."
            )

        cooled_note = ""
        if cooled_domains:
            cooled_note = (
                "\n\nDOMAINS CURRENTLY COOLED (used too often recently — "
                "avoid these unless you have a strong reason):\n  "
                + ", ".join(cooled_domains)
            )

        return (
            self.persona
            + "\n\n═══ DOMAIN TAXONOMY ═══\n"
            + "Pick ONE domain below as the primary domain. Use the exact "
              "canonical name (the string before the colon).\n\n"
            + taxonomy
            + cooled_note
            + "\n\n═══ THIS ITERATION ═══\n"
            + mode_directive
            + "\n\n═══ OUTPUT ═══\n"
            + PROPOSAL_SCHEMA
        )

    def _propose_user_message(
        self,
        mode: str,
        history_summary: str,
        cached_concept_summary: str,
        feedback: str = "",
    ) -> str:
        parts: list[str] = []
        parts.append(
            f"You are in {mode.upper()} mode for this iteration. "
            f"Propose one WEAT test."
        )
        parts.append("")
        parts.append("─── Recently completed tests (do not duplicate the full 4-tuple) ───")
        parts.append(history_summary or "(no prior tests in this run)")
        parts.append("")
        parts.append("─── Cached concepts (if you reuse any, match the cached label and word list) ───")
        parts.append(cached_concept_summary or "(cache is empty — any concept you pick will be a new one)")

        if feedback:
            parts.append("")
            parts.append("─── Feedback on your previous attempt (fix these issues) ───")
            parts.append(feedback)

        parts.append("")
        parts.append("Output the JSON object now. No prose, no preamble, no markdown.")
        return "\n".join(parts)

    # ─── Phase 2: interpret (unchanged from v14) ───────────────

    def interpret(self, proposal: Proposal, result_dict: dict) -> Interpretation:
        d = float(result_dict.get("effect_size", 0.0))
        p_two = float(result_dict.get("p_value", 1.0))

        if not interpretive_fires(d, p_two):
            return Interpretation(paragraph="", fired=False, raw="")

        system = INTERPRETATION_PROMPT
        user = self._interpret_user_message(proposal, result_dict)

        raw = self._chat(system, user)
        paragraph = self._clean_paragraph(raw)
        return Interpretation(paragraph=paragraph, fired=True, raw=raw)

    def _interpret_user_message(
        self, proposal: Proposal, result_dict: dict
    ) -> str:
        d = float(result_dict.get("effect_size", 0.0))
        p = float(result_dict.get("p_value", 1.0))

        return (
            f"Test that was run:\n"
            f"  X = {proposal.X_label}\n"
            f"  Y = {proposal.Y_label}\n"
            f"  A = {proposal.A_label}\n"
            f"  B = {proposal.B_label}\n"
            f"\n"
            f"Result:\n"
            f"  WEAT:  d = {d:+.3f},  p = {p:.3f}\n"
            f"\n"
            f"The WEAT is statistically significant. Write the interpretive "
            f"paragraph now. Output the paragraph text only."
        )

    # ─── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_json(raw: str) -> dict:
        s = raw.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            raise ValueError("no JSON object found in response")
        return json.loads(m.group(0))

    @staticmethod
    def _clean_words(words: Any) -> list[str]:
        if not isinstance(words, list):
            return []
        out: list[str] = []
        for w in words:
            if not isinstance(w, str):
                continue
            ww = w.strip().lower()
            if not ww:
                continue
            if " " in ww or "-" in ww or "_" in ww:
                continue
            if not ww.isalpha():
                continue
            out.append(ww)
        seen = set()
        deduped = []
        for w in out:
            if w not in seen:
                seen.add(w)
                deduped.append(w)
        return deduped

    @staticmethod
    def _clean_paragraph(raw: str) -> str:
        s = raw.strip()
        s = re.sub(r"^```(?:\w+)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    for k in ("paragraph", "interpretation", "text"):
                        if isinstance(obj.get(k), str):
                            s = obj[k]
                            break
            except Exception:
                pass
        return s.strip()
