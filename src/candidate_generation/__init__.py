"""
Candidate Generation / Blocking module for Business Entity Resolution.

Generates candidate pairs using a union of independent blocking keys,
all prefixed with country. Supports configurable block-size capping
and comprehensive evaluation against ground truth.
"""

from src.candidate_generation.stopwords import BLOCKING_STOPWORDS, is_stopword
from src.candidate_generation.name_tokens import extract_meaningful_tokens
from src.candidate_generation.address_parser import (
    extract_address_number,
    has_address_number,
)
from src.candidate_generation.block_keys import (
    key_a_exact_norm_name,
    key_b_exact_translit_name,
    key_c_first_two_meaningful_tokens,
    key_d_first_token_addr_num,
    key_e_first_token_fallback,
    key_f_exact_norm_address,
    generate_all_keys,
    BLOCKING_KEY_NAMES,
)
from src.candidate_generation.block_index import BlockIndex
from src.candidate_generation.generate_candidates import generate_candidate_pairs
from src.candidate_generation.evaluate_blocking import evaluate_blocking

__all__ = [
    "BLOCKING_STOPWORDS",
    "is_stopword",
    "extract_meaningful_tokens",
    "extract_address_number",
    "has_address_number",
    "key_a_exact_norm_name",
    "key_b_exact_translit_name",
    "key_c_first_two_meaningful_tokens",
    "key_d_first_token_addr_num",
    "key_e_first_token_fallback",
    "key_f_exact_norm_address",
    "generate_all_keys",
    "BLOCKING_KEY_NAMES",
    "BlockIndex",
    "generate_candidate_pairs",
    "evaluate_blocking",
]
