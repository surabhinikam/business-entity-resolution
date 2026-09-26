"""
Meaningful name token extraction for blocking.

Extracts tokens from normalized business names after filtering out
blocking stopwords (legal suffixes, generic business terms, honorifics).

IMPORTANT: This does NOT modify the stored normalized name — it only
provides a filtered view for deriving blocking keys.
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.candidate_generation.stopwords import BLOCKING_STOPWORDS


# Tokenize on whitespace and non-alphanumeric (preserving &, -, /)
_TOKEN_RE = re.compile(r"[^\w&/-]+", re.UNICODE)


def extract_meaningful_tokens(
    normalized_name: Optional[str],
    min_token_length: int = 2,
) -> List[str]:
    """
    Extract meaningful (non-stopword) tokens from a normalized business name.

    Steps:
    1. Tokenize the normalized name.
    2. Filter out tokens that are in BLOCKING_STOPWORDS.
    3. Filter out tokens shorter than min_token_length (default 2).
    4. Return the remaining tokens in order.

    If all tokens are filtered out, returns an empty list.
    The caller must handle this case gracefully (e.g., skip the blocking key).

    Parameters:
        normalized_name: The normalized business name string.
        min_token_length: Minimum length for a token to be kept (default 2).

    Returns:
        List of meaningful tokens in their original order.
    """
    if not normalized_name:
        return []

    # Split into tokens
    raw_tokens = [t for t in _TOKEN_RE.split(normalized_name) if t]

    # Filter stopwords and short tokens
    meaningful = []
    for token in raw_tokens:
        t_lower = token.lower().strip()
        if not t_lower:
            continue
        if len(t_lower) < min_token_length:
            continue
        if t_lower in BLOCKING_STOPWORDS:
            continue
        meaningful.append(t_lower)

    return meaningful
