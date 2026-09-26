"""
Blocking stopword configuration for Business Entity Resolution.

Stopwords are tokens that are too generic or too common to be useful
as blocking tokens. They should be EXCLUDED when deriving blocking
tokens (keys C, D, E) but NOT destroyed from the stored normalized name.

The stopword list is built from:
1. Legal suffix tokens canonicalized by business_vocabulary.py
2. High-frequency generic business terms observed in the dataset
3. Common honorifics and organizational prefixes

This list is intentionally conservative — only tokens that appear
across many unrelated entities and would create oversized blocks.
"""

from __future__ import annotations

from typing import FrozenSet, Set

# =============================================================================
# 1. Legal/Corporate suffix tokens
# =============================================================================
# These are the canonical forms produced by apply_legal_suffix() in
# business_vocabulary.py, plus their common un-canonicalized variants.
# When these appear as tokens in a normalized business name, they
# carry no discriminating value for blocking.
LEGAL_SUFFIX_TOKENS: FrozenSet[str] = frozenset({
    # Canonical forms
    "pvt", "ltd", "llc", "llp", "inc", "corp", "co", "plc",
    # Expanded forms that may survive normalization
    "private", "limited", "company", "corporation", "incorporated",
    "liability", "partnership", "public",
    # Indic transliterated forms that may survive if transliteration
    # didn't fully canonicalize them
    "limiteda", "limidhedh", "limirrad", "limatida",
    "praiveta", "praivet", "praibheta", "praibhet",
    "bhiraivedh", "praivarr", "elaelapi", "elelbhi", "elelpi",
})

# =============================================================================
# 2. Generic business/industry terms
# =============================================================================
# High-frequency terms that appear across many unrelated businesses.
# Derived from dataset frequency analysis:
#   - Source1 top repeated: "primary care group", "pediatric group", etc.
#   - These terms describe industry/type, not unique identity.
# Also includes transliterated equivalents from BUSINESS_TERMS in
# business_vocabulary.py.
GENERIC_BUSINESS_TOKENS: FrozenSet[str] = frozenset({
    # Organizational structure words
    "group", "associates", "association", "society", "trust",
    "foundation", "federation", "council", "committee", "board",
    "agency", "bureau", "institute", "institution",
    # Industry descriptors — these create huge blocks
    "services", "solutions", "enterprises", "industries", "technologies",
    "technology", "tech", "trading", "traders", "marketing",
    "properties", "logistics", "energy", "global", "international",
    "foods", "food", "impex", "engineering", "constructions",
    "construction", "products", "developers", "estate", "media",
    "agro", "projects", "ventures", "exports", "producer", "power",
    "systems", "builders", "investments", "investment", "business",
    "consulting", "consultants", "consultancy", "finance",
    "care", "infra", "infrastructure", "software", "management",
    "infotech", "healthcare", "hospitality", "hotel", "digital",
    "hitech", "creative", "unique", "universal", "supreme",
    "premier", "motors", "pharmacy", "furniture", "provision",
    # Common generic qualifiers
    "new", "old", "big", "sri", "shri", "shree", "sree",
    "the", "and", "of", "for", "in", "at", "to", "on",
    "&",
})

# =============================================================================
# 3. Honorifics and organizational prefixes
# =============================================================================
HONORIFIC_TOKENS: FrozenSet[str] = frozenset({
    "mr", "mrs", "ms", "m/s", "m-s", "messrs", "dr", "prof", "sir", "smt",
    "shri", "sri", "shree", "sree", "kumari",
    "late", "son", "sons", "brothers", "bros",
})

# =============================================================================
# Combined stopword set
# =============================================================================
BLOCKING_STOPWORDS: FrozenSet[str] = (
    LEGAL_SUFFIX_TOKENS | GENERIC_BUSINESS_TOKENS | HONORIFIC_TOKENS
)


def is_stopword(token: str) -> bool:
    """Check if a token is a blocking stopword."""
    return token.lower().strip() in BLOCKING_STOPWORDS
