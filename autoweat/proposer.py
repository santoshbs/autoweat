"""
proposer.py — AutoWEAT v13: minimal Charlesworth-faithful proposer.

Two phases per iteration:

  Phase 1 (propose):   LLM picks X, Y, A, B labels and word lists.
                       No headline format, no verdict prediction,
                       no prior. Just a test to run.

  Phase 2 (interpret): Runs ONLY when the relative WEAT is significant
                       (p < .05) AND the SC-WEAT directions are
                       consistent with the WEAT direction (opposite
                       signs, SC-WEAT(X) sign matches WEAT sign).
                       Under that condition, the LLM writes one short
                       paragraph: what the result shows, a possible
                       explanation, why it might matter. No format
                       rules, no verdict patterns, no forbidden
                       language lists. Just the paragraph.

                       When the condition is NOT met, Phase 2 is
                       skipped and the test is stored with no
                       interpretive paragraph.

Output schema: inductive.v7 (see run.py).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

AUTOWEAT_PROPOSER_VERSION = "2026-04-13-v13-charlesworth-minimal"


# ─── Phase 1: proposal schema ──────────────────────────────────────

PROPOSAL_SCHEMA = """Output this exact JSON structure:

{
  "domain": "<one-or-two-word domain tag, e.g. 'sociology', 'gender', 'occupation'>",
  "contrast_label": "<short natural-English description of what you're probing, e.g. 'gender with agency versus communion'>",
  "X_label": "<short label for target set X, e.g. 'women and feminine terms'>",
  "Y_label": "<short label for target set Y, e.g. 'men and masculine terms'>",
  "A_label": "<short label for attribute set A, e.g. 'communion and care terms'>",
  "B_label": "<short label for attribute set B, e.g. 'agency and power terms'>",
  "X_words": [<12-16 common English words representing X>],
  "Y_words": [<12-16 common English words representing Y>],
  "A_words": [<12-16 common English words representing A>],
  "B_words": [<12-16 common English words representing B>]
}

Labels must be natural English phrases (not identifiers). Words must
be common lowercase English, single words, no phrases, no hyphens,
no proper nouns. All four word lists should be roughly the same size
(12-16 words each).
"""


# ─── Phase 2: interpretation prompt (short, no schema) ─────────────

INTERPRETATION_PROMPT = """You are being asked to write ONE short paragraph (4-7 sentences)
interpreting a WEAT result. The relative WEAT is significant AND the
single-category decomposition is directionally consistent with the
relative effect.

Write the paragraph in three movements:

(1) What the result shows in plain English. State the finding as an
    observation about how the corpus uses this vocabulary. Name the
    four labels (X, Y, A, B) naturally; do not invent new categories.

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
no markdown. Just the paragraph text.
"""


# ─── dataclasses ───────────────────────────────────────────────────


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
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("raw", None)
        return d


@dataclass
class Interpretation:
    """The interpretive paragraph, or empty string if not fired."""

    paragraph: str = ""
    fired: bool = False
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"paragraph": self.paragraph, "fired": self.fired}


# ─── Consistency condition ─────────────────────────────────────────


def interpretive_fires(
    effect_size: float,
    p_value: float,
    sc_weat_x_mean: float,
    sc_weat_y_mean: float,
) -> bool:
    """Charlesworth-style consistency check.

    Fires IFF:
    (1) WEAT p < .05
    (2) SC-WEAT(X) and SC-WEAT(Y) have opposite signs
    (3) sign(SC-WEAT(X)) == sign(WEAT D)

    SC-WEAT p-values are NOT checked (per Charlesworth).
    SC-WEAT magnitudes are NOT checked (per Charlesworth).
    """
    if p_value is None or p_value >= 0.05:
        return False
    if effect_size is None or effect_size == 0:
        return False
    if sc_weat_x_mean is None or sc_weat_y_mean is None:
        return False
    # Condition 2: opposite signs
    if sc_weat_x_mean * sc_weat_y_mean >= 0:
        return False
    # Condition 3: SC-WEAT(X) sign matches WEAT sign
    if (sc_weat_x_mean > 0) != (effect_size > 0):
        return False
    return True


# ─── Proposer class ────────────────────────────────────────────────


class OllamaProposer:
    """Minimal LLM proposer for AutoWEAT v13."""

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
        """Use /api/generate (not /api/chat) — Qwen3.5 hangs on /api/chat
        when a system role message is included. /api/generate with the
        `system` field is the format that works for both Qwen3 and gpt-oss.
        """
        options: dict[str, Any] = {}
        options.update(self.sampling)
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        # gpt-oss family reads "Reasoning: <effort>" from the very first
        # line of the system prompt; for that family, prepend it here.
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
        # Qwen3 family uses Ollama's top-level `think` field.
        if self.thinking_style == "qwen3":
            payload["think"] = (self.thinking_effort != "off"
                                and self.thinking_effort is not None)

        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=900)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    # ─── Phase 1: propose ───────────────────────────────────────

    def propose(
        self,
        history: list[dict] | None = None,
        recent_domains: list[str] | None = None,
        max_retries: int = 3,
    ) -> Proposal:
        """Phase 1: ask the LLM for one WEAT test proposal."""
        history = history or []
        recent_domains = recent_domains or []

        system = self._propose_system_prompt(recent_domains)
        user = self._propose_user_message(history)

        last_err = None
        for _ in range(max_retries):
            raw = self._chat(system, user)
            try:
                data = self._extract_json(raw)
                return Proposal(
                    domain=str(data.get("domain", "")).strip(),
                    contrast_label=str(data.get("contrast_label", "")).strip(),
                    X_label=str(data.get("X_label", "")).strip(),
                    Y_label=str(data.get("Y_label", "")).strip(),
                    A_label=str(data.get("A_label", "")).strip(),
                    B_label=str(data.get("B_label", "")).strip(),
                    X_words=self._clean_words(data.get("X_words", [])),
                    Y_words=self._clean_words(data.get("Y_words", [])),
                    A_words=self._clean_words(data.get("A_words", [])),
                    B_words=self._clean_words(data.get("B_words", [])),
                    raw=raw,
                )
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"Phase 1 (propose) returned unparseable JSON after "
            f"{max_retries} attempts. Last error: {last_err}"
        )

    def _propose_system_prompt(self, recent_domains: list[str]) -> str:
        recent_note = ""
        if recent_domains:
            uniq = list(dict.fromkeys(recent_domains[-8:]))
            recent_note = (
                "\n\nYou have recently proposed tests in these domains: "
                + ", ".join(uniq)
                + ". Do not repeat a contrast from this list. Aim for a "
                "different substantive question."
            )
        return (
            self.persona
            + recent_note
            + "\n\nYou are AutoWEAT's proposer (Phase 1). Your sole output "
            "is a JSON object describing one WEAT test. No prose, no "
            "preamble, no markdown, no explanation. Just the JSON."
            "\n\n" + PROPOSAL_SCHEMA
        )

    def _propose_user_message(self, history: list[dict]) -> str:
        parts = [
            "Previous WEAT tests you have already proposed in this run "
            "(do not repeat or trivially relabel):",
        ]
        if not history:
            parts.append("(none yet)")
        else:
            for h in history[-15:]:
                labels = h.get("labels", {}) or {}
                x = labels.get("X", "?")
                y = labels.get("Y", "?")
                a = labels.get("A", "?")
                b = labels.get("B", "?")
                d = h.get("effect_size", 0.0) or 0.0
                p = h.get("p_value", 1.0) or 1.0
                parts.append(
                    f"  · {x} vs {y} × {a} vs {b}  "
                    f"(d={d:+.2f}, p={p:.3f})"
                )
        parts.append("\nPropose the next WEAT test now. Output JSON only.")
        return "\n".join(parts)

    # ─── Phase 2: interpret (conditional) ───────────────────────

    def interpret(self, proposal: Proposal, result_dict: dict) -> Interpretation:
        """Phase 2: IF the consistency condition holds, ask the LLM for one
        interpretive paragraph. Otherwise return an empty Interpretation.
        """
        d = float(result_dict.get("effect_size", 0.0))
        p_two = float(result_dict.get("p_value", 1.0))
        sc_x_mean = float(result_dict.get("sc_weat_x_mean", 0.0))
        sc_y_mean = float(result_dict.get("sc_weat_y_mean", 0.0))

        if not interpretive_fires(d, p_two, sc_x_mean, sc_y_mean):
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
        sc_x = float(result_dict.get("sc_weat_x_mean", 0.0))
        sc_p_x = float(result_dict.get("sc_p_x", 1.0))
        sc_y = float(result_dict.get("sc_weat_y_mean", 0.0))
        sc_p_y = float(result_dict.get("sc_p_y", 1.0))

        return (
            f"Test that was run:\n"
            f"  X = {proposal.X_label}\n"
            f"  Y = {proposal.Y_label}\n"
            f"  A = {proposal.A_label}\n"
            f"  B = {proposal.B_label}\n"
            f"\n"
            f"Results:\n"
            f"  WEAT:       D = {d:+.3f}, p = {p:.3f}\n"
            f"  SC-WEAT(X): {sc_x:+.3f} (p = {sc_p_x:.3f})  "
            f"[X = {proposal.X_label}]\n"
            f"  SC-WEAT(Y): {sc_y:+.3f} (p = {sc_p_y:.3f})  "
            f"[Y = {proposal.Y_label}]\n"
            f"\n"
            f"The WEAT is significant and the SC-WEAT decomposition is "
            f"directionally consistent (opposite signs, matching the "
            f"WEAT direction). Write the interpretive paragraph now. "
            f"Output the paragraph text only."
        )

    # ─── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Pull the first JSON object out of an LLM response."""
        # Strip common wrappers
        s = raw.strip()
        # Remove fences if present
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        # Try direct parse first
        try:
            return json.loads(s)
        except Exception:
            pass
        # Find the first { ... } block
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
            # reject phrases, hyphens, numbers
            if " " in ww or "-" in ww or "_" in ww:
                continue
            if not ww.isalpha():
                continue
            out.append(ww)
        # dedupe preserving order
        seen = set()
        deduped = []
        for w in out:
            if w not in seen:
                seen.add(w)
                deduped.append(w)
        return deduped

    @staticmethod
    def _clean_paragraph(raw: str) -> str:
        """Strip fences, leading/trailing whitespace, and any JSON."""
        s = raw.strip()
        s = re.sub(r"^```(?:\w+)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        # If the model accidentally wrapped it in JSON, try to extract
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
