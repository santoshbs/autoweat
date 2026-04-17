"""
domains.py — Social-science domain taxonomy for AutoWEAT (v16+).

Design in v16:
  The taxonomy is a FLAT LIST of canonical domain names. No concept
  lists under each domain. The domain field is a filing tag used for
  rotation and filtering; it does not constrain what targets or
  attributes the LLM can propose within that domain.

  Earlier versions had suggestive concept lists under each domain.
  Those lists tended to become sampling distributions — the LLM would
  grab a listed concept rather than think about what the domain
  actually covers. Stripping them returns the choice of substance to
  the LLM.

Edit this file to add or remove domains. All consumers (persona,
proposer, run.py validation) read from `VALID_DOMAINS` here.
"""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────
# The taxonomy. Canonical domain names, lowercase. Order matters only
# for display in the prompt.
# ──────────────────────────────────────────────────────────────────────

DOMAINS: tuple[str, ...] = (
    "management",
    "strategy",
    "economics",
    "organizational theory",
    "organizational behavior",
    "io psychology",
    "psychology",
    "cognition",
    "social psychology",
    "sociology",
    "anthropology",
    "linguistics",
    "arts",
    "humanities",
)

VALID_DOMAINS: frozenset[str] = frozenset(DOMAINS)


# Common abbreviations and alternate spellings the LLM might emit.
_ALIASES: dict[str, str] = {
    "ob": "organizational behavior",
    "org behavior": "organizational behavior",
    "org. behavior": "organizational behavior",
    "organisational behaviour": "organizational behavior",
    "organisational behavior": "organizational behavior",
    "organizational behaviour": "organizational behavior",
    "ot": "organizational theory",
    "org theory": "organizational theory",
    "organisational theory": "organizational theory",
    "i/o psychology": "io psychology",
    "i-o psychology": "io psychology",
    "io psych": "io psychology",
    "industrial organizational psychology": "io psychology",
    "industrial-organizational psychology": "io psychology",
    "soc psych": "social psychology",
    "social psych": "social psychology",
    "mgmt": "management",
    "econ": "economics",
    "ling": "linguistics",
    "psych": "psychology",
    "soc": "sociology",
    "anthro": "anthropology",
}


def normalize_domain(raw: str) -> str | None:
    """Normalize a raw LLM-proposed domain string to a canonical name.

    Returns the canonical domain name if the input resolves (directly or
    via alias) to a known domain. Returns None otherwise.
    """
    if not raw:
        return None
    s = raw.strip().lower()
    # Strip common decorative suffixes the LLM sometimes adds
    for suffix in (" studies", " research", " theory", " science"):
        if s.endswith(suffix):
            candidate = s[: -len(suffix)].strip()
            if candidate in VALID_DOMAINS:
                return candidate
    if s in VALID_DOMAINS:
        return s
    if s in _ALIASES:
        return _ALIASES[s]
    return None


def format_taxonomy_for_prompt() -> str:
    """Render the taxonomy as a compact list to inject into the system
    prompt. Just the names — no concept hints, no per-domain guidance.
    """
    return "\n".join(f"  {name}" for name in DOMAINS)
