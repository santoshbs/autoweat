"""
domains.py — Social-science domain taxonomy for AutoWEAT (v15+).

Single source of truth for:
  - which domains are valid
  - the main concepts (suggestive, not constraining) inside each domain
  - normalization of LLM-proposed domain labels to canonical form

Edit this file to add/remove domains. The persona and proposer read
from it, so changes propagate without edits elsewhere.

Design notes:
  - Concepts under each domain are SUGGESTIVE. The LLM uses them as
    seeds when brainstorming a contrast; it is not restricted to them.
  - Cross-domain contrasts are allowed. The proposer tags the PRIMARY
    domain — the one whose literature would most naturally discuss the
    contrast. Ties are broken by picking whichever domain has been used
    less recently.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# The taxonomy. Keys are canonical domain names (lowercase).
# Values are suggestive concept lists — each one is a PHRASE the LLM
# can grab as the seed of a contrast.
# ──────────────────────────────────────────────────────────────────────

DOMAINS: dict[str, list[str]] = {
    "management": [
        "strategy",
        "leadership",
        "decision making",
        "planning",
        "control",
        "coordination",
        "delegation",
        "performance management",
    ],
    "strategy": [
        "competitive advantage",
        "positioning",
        "resources and capabilities",
        "stakeholders",
        "growth and scaling",
        "diversification",
        "innovation",
        "industry structure",
    ],
    "economics": [
        "markets and exchange",
        "incentives",
        "scarcity",
        "welfare and distribution",
        "rationality",
        "information asymmetry",
        "public goods",
        "labor",
    ],
    "organizational theory": [
        "structure",
        "bureaucracy",
        "institutions and legitimacy",
        "environment",
        "adaptation and change",
        "networks",
        "ecology",
        "isomorphism",
    ],
    "organizational behavior": [
        "personality",
        "motivation",
        "power and influence",
        "teams",
        "conflict",
        "culture",
        "identity",
        "justice and fairness",
        "trust",
    ],
    "io psychology": [
        "selection and hiring",
        "performance appraisal",
        "training and development",
        "well-being",
        "leadership",
        "person-environment fit",
        "work engagement",
        "burnout",
    ],
    "psychology": [
        "cognition",
        "emotion",
        "learning",
        "development",
        "memory",
        "attention",
        "personality",
        "motivation",
    ],
    "cognition": [
        "reasoning",
        "attention",
        "memory",
        "perception",
        "judgment and decision",
        "heuristics and biases",
        "language processing",
        "problem solving",
    ],
    "social psychology": [
        "attitudes",
        "stereotypes and prejudice",
        "norms and conformity",
        "attribution",
        "persuasion",
        "groups",
        "self and identity",
        "intergroup relations",
    ],
    "sociology": [
        "class and stratification",
        "race and ethnicity",
        "gender",
        "family",
        "religion",
        "institutions",
        "deviance",
        "urbanism",
        "mobility",
    ],
    "anthropology": [
        "ritual",
        "kinship",
        "exchange and reciprocity",
        "ethnicity and boundary",
        "modernity and tradition",
        "material culture",
        "symbolic meaning",
    ],
    "linguistics": [
        "register and formality",
        "politeness",
        "metaphor",
        "pragmatics",
        "discourse",
        "style",
        "code switching",
        "address terms",
    ],
    "arts": [
        "genre",
        "medium",
        "style and movement",
        "canon",
        "reception and audience",
        "aesthetic value",
        "patronage",
    ],
    "humanities": [
        "history and memory",
        "ethics and virtue",
        "ideology",
        "narrative",
        "interpretation",
        "tradition and modernity",
        "authority of texts",
    ],
}

# All valid canonical names (for validation on proposal parse)
VALID_DOMAINS: frozenset[str] = frozenset(DOMAINS.keys())

# Common abbreviations and alternate spellings the LLM might emit.
# Mapped to the canonical form in DOMAINS.
_ALIASES: dict[str, str] = {
    "ob": "organizational behavior",
    "org behavior": "organizational behavior",
    "org. behavior": "organizational behavior",
    "organisational behaviour": "organizational behavior",  # UK spelling
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
    """Normalize a raw LLM-proposed domain string to a canonical key.

    Returns the canonical domain name if the input resolves to one of
    the known domains (directly or via alias). Returns None otherwise.
    """
    if not raw:
        return None
    s = raw.strip().lower()
    # Strip common suffixes
    for suffix in (" studies", " research", " theory", " science"):
        if s.endswith(suffix):
            candidate = s[: -len(suffix)].strip()
            if candidate in VALID_DOMAINS:
                return candidate
    # Direct hit
    if s in VALID_DOMAINS:
        return s
    # Alias hit
    if s in _ALIASES:
        return _ALIASES[s]
    # No match
    return None


def format_taxonomy_for_prompt() -> str:
    """Render the taxonomy as a readable block to inject into the system
    prompt. One line per domain, concepts comma-separated."""
    lines: list[str] = []
    for name, concepts in DOMAINS.items():
        lines.append(f"  {name}: {', '.join(concepts)}")
    return "\n".join(lines)
