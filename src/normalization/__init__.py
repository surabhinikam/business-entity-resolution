"""
Normalization package for Business Entity Resolution.
"""

from src.normalization.text_normalizer import normalize_text, strip_diacritics
from src.normalization.business_normalizer import (
    normalize_business_name,
    tokenize_business_name,
    canonicalize_legal_suffix,
)
from src.normalization.address_normalizer import (
    normalize_address,
    tokenize_address,
)
from src.normalization.country_normalizer import (
    normalize_country,
)

from src.normalization.business_vocabulary import (
    normalize_business_vocabulary,
    apply_legal_suffix,
    LEGAL_SUFFIX_RULES,
    BUSINESS_TERMS,
)

__all__ = [
    "normalize_text",
    "strip_diacritics",
    "normalize_business_name",
    "tokenize_business_name",
    "canonicalize_legal_suffix",
    "normalize_business_vocabulary",
    "apply_legal_suffix",
    "LEGAL_SUFFIX_RULES",
    "BUSINESS_TERMS",
    "normalize_address",
    "tokenize_address",
    "normalize_country",
]
