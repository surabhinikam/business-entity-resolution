"""
Address number extraction for blocking.

Extracts the first usable street/plot/house number from a normalized
business address. Used by blocking keys D and E.

Key distinction for Key E fallback:
- "address is null" vs "no address number was successfully extracted"
  are different conditions. An address like "Suite B, Oak Plaza" is
  non-null but has no extractable address number.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Regex to extract address numbers.
# Matches patterns like: 123, 570/13, 12-a, 12A, #402, 3829
# Must start with a digit to be a usable address number.
# Excludes pure zip codes (5+ digit sequences at end of address) by
# requiring the number to appear early or in a recognizable pattern.
_ADDRESS_NUMBER_RE = re.compile(
    r"""
    (?:^|\s|[#])                    # Start of string, whitespace, or hash
    (\d+                            # Leading digits (required)
      (?:[/-][a-zA-Z0-9]+)?        # Optional slash or dash suffix (e.g., 570/13, 12-A)
    )
    (?:\s|$|[,;])                   # Followed by whitespace, end, or punctuation
    """,
    re.VERBOSE,
)

# Patterns that are NOT useful address numbers (zip codes, phone-like, etc.)
# We keep the first match that isn't a zip code
_ZIP_CODE_RE = re.compile(r"^\d{5,6}$")  # 5-6 digit sequences are likely zip codes


def extract_address_number(
    normalized_address: Optional[str],
) -> Tuple[Optional[str], bool]:
    """
    Extract the first usable address number from a normalized address.

    Returns:
        Tuple of (address_number, has_number):
        - address_number: The extracted number string, or None if not found.
        - has_number: True if a usable number was extracted.

    Examples:
        "108 norle street college twp pa" -> ("108", True)
        "570/13 mg road bangalore ka" -> ("570/13", True)
        "suite b oak plaza" -> (None, False)
        None -> (None, False)
        "" -> (None, False)
    """
    if not normalized_address:
        return None, False

    matches = _ADDRESS_NUMBER_RE.findall(normalized_address)
    for match in matches:
        match = match.strip()
        if not match:
            continue
        # Skip pure zip codes (5-6 digit sequences)
        if _ZIP_CODE_RE.match(match):
            continue
        return match, True

    return None, False


def has_address_number(normalized_address: Optional[str]) -> bool:
    """
    Check if a normalized address contains an extractable address number.
    Convenience wrapper around extract_address_number.
    """
    _, has_num = extract_address_number(normalized_address)
    return has_num
