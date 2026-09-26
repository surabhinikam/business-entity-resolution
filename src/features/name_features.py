"""
Pairwise business-name feature extraction module.

Computes pure, deterministic similarity and compatibility features between
two precomputed NameRepresentation objects (and optional transliterated names)
without re-tokenizing or re-extracting character n-grams.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import polars as pl

from src.features.record_representation import NameRepresentation
from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    NAME_FEATURE_NAMES,
    NAME_FEATURE_SCHEMA,
)
from src.analysis.text_similarity import (
    jaccard_similarity,
    overlap_coefficient,
    dice_coefficient,
    character_ngrams,
)
from src.normalization.business_vocabulary import LEGAL_SUFFIX_RULES

# Canonical corporate/legal suffixes derived directly from normalization layer,
# preserving first-seen and compound-first ordering (e.g. 'pvt ltd' before 'ltd')
CANONICAL_LEGAL_SUFFIXES: Tuple[str, ...] = tuple(
    dict.fromkeys(canonical for _, canonical in LEGAL_SUFFIX_RULES)
)

_SENTINEL_EMPTY_STRINGS = frozenset({"", "none", "nan", "null"})


def extract_legal_suffix(clean_name: str) -> Optional[str]:
    """
    Extract canonical legal suffix from the end of a normalized business name.

    Requires the suffix to be at a token boundary (preceded by a space or
    spanning the entire name). Returns None if no suffix is detected.
    """
    if not clean_name:
        return None

    for suffix in CANONICAL_LEGAL_SUFFIXES:
        if clean_name.endswith(suffix):
            suffix_len = len(suffix)
            if len(clean_name) == suffix_len or clean_name[-(suffix_len + 1)] == " ":
                return suffix
    return None


def compute_name_features(
    rep_a: NameRepresentation,
    rep_b: NameRepresentation,
    translit_a: Optional[str] = None,
    translit_b: Optional[str] = None,
) -> Dict[str, float | int]:
    """
    Compute pairwise business-name features between two NameRepresentation objects.

    Parameters:
        rep_a: Precomputed NameRepresentation for first entity (e.g. Source 1).
        rep_b: Precomputed NameRepresentation for second entity (e.g. Candidate Source 2/3).
        translit_a: Optional transliterated string for entity A.
        translit_b: Optional transliterated string for entity B.

    Returns:
        Dictionary mapping feature names to numeric values (int or float).
        Missing or empty values safely produce 0 or 0.0 (no artificial matches).
    """
    # 1. Exact normalized name match
    name_exact_norm = int(
        bool(rep_a.clean_name) and rep_a.clean_name == rep_b.clean_name
    )

    # 2. Exact transliterated name match (None == None must NOT count as a match)
    if translit_a is not None and translit_b is not None:
        t_a = str(translit_a).strip().lower()
        t_b = str(translit_b).strip().lower()
        if (
            t_a
            and t_b
            and t_a not in _SENTINEL_EMPTY_STRINGS
            and t_b not in _SENTINEL_EMPTY_STRINGS
        ):
            name_exact_translit = int(t_a == t_b)
        else:
            name_exact_translit = 0
    else:
        name_exact_translit = 0

    # 3-5. Token set similarities (Empty sets must produce 0.0, not 1.0)
    if rep_a.token_set and rep_b.token_set:
        name_token_jaccard = float(jaccard_similarity(rep_a.token_set, rep_b.token_set))
        name_token_overlap = float(overlap_coefficient(rep_a.token_set, rep_b.token_set))
        name_token_dice = float(dice_coefficient(rep_a.token_set, rep_b.token_set))
    else:
        name_token_jaccard = 0.0
        name_token_overlap = 0.0
        name_token_dice = 0.0

    # 6. Token count difference
    name_token_count_diff = int(abs(rep_a.token_count - rep_b.token_count))

    # 7. Character 3-gram Jaccard similarity (Empty sets must produce 0.0, not 1.0)
    if rep_a.char_3grams and rep_b.char_3grams:
        name_char_3gram_jaccard = float(
            jaccard_similarity(rep_a.char_3grams, rep_b.char_3grams)
        )
    else:
        name_char_3gram_jaccard = 0.0

    # 8. Character length difference
    name_char_len_diff = int(abs(rep_a.char_length - rep_b.char_length))

    # 9. Character length ratio (0.0 if either name is empty)
    if rep_a.char_length > 0 and rep_b.char_length > 0:
        name_char_len_ratio = float(
            min(rep_a.char_length, rep_b.char_length)
            / max(rep_a.char_length, rep_b.char_length)
        )
    else:
        name_char_len_ratio = 0.0

    # 10. First token exact match (0 if either token count is 0)
    if (
        rep_a.token_count > 0
        and rep_b.token_count > 0
        and rep_a.clean_name
        and rep_b.clean_name
    ):
        first_a = rep_a.clean_name.split()[0]
        first_b = rep_b.clean_name.split()[0]
        name_first_token_exact = int(first_a == first_b)
    else:
        name_first_token_exact = 0

    # 11. Acronym match (0 if either acronym is empty)
    if rep_a.acronym and rep_b.acronym:
        name_acronym_match = int(rep_a.acronym == rep_b.acronym)
    else:
        name_acronym_match = 0

    # 12. Legal suffix match (0 if either has no detectable suffix)
    suffix_a = extract_legal_suffix(rep_a.clean_name)
    suffix_b = extract_legal_suffix(rep_b.clean_name)
    if suffix_a is not None and suffix_b is not None:
        name_legal_suffix_match = int(suffix_a == suffix_b)
    else:
        name_legal_suffix_match = 0

    return {
        "name_exact_norm": name_exact_norm,
        "name_exact_translit": name_exact_translit,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_token_dice": name_token_dice,
        "name_token_count_diff": name_token_count_diff,
        "name_char_3gram_jaccard": name_char_3gram_jaccard,
        "name_char_len_diff": name_char_len_diff,
        "name_char_len_ratio": name_char_len_ratio,
        "name_first_token_exact": name_first_token_exact,
        "name_acronym_match": name_acronym_match,
        "name_legal_suffix_match": name_legal_suffix_match,
    }


def _name_representations_to_lookup_df(
    reps: Union[Dict[str, Any], pl.DataFrame],
    translits: Optional[Union[Dict[str, Optional[str]], pl.DataFrame]] = None,
) -> pl.DataFrame:
    """Converts a dictionary or DataFrame of name representations into a standardized lookup DataFrame."""
    if isinstance(reps, pl.DataFrame):
        return reps

    # Build transliteration lookup if provided
    t_lookup: Dict[str, Optional[str]] = {}
    if translits is not None:
        if isinstance(translits, dict):
            t_lookup = translits
        elif isinstance(translits, pl.DataFrame) and "entity_id" in translits.columns and "translit" in translits.columns:
            t_lookup = dict(zip(translits["entity_id"].to_list(), translits["translit"].to_list()))

    rows: List[Dict[str, Any]] = []
    for eid, r in reps.items():
        if isinstance(r, dict):
            clean_name = (r.get("clean_name") or r.get("business_name_normalized") or "").strip().lower()
            toks = r.get("token_set") or r.get("business_name_tokens") or clean_name.split()
            if isinstance(toks, (set, frozenset)):
                toks = list(toks)
            grams = r.get("char_3grams")
            if grams is None:
                grams = character_ngrams(clean_name, n=3) if clean_name else []
            elif isinstance(grams, (set, frozenset)):
                grams = list(grams)
            char_len = int(r.get("char_length", len(clean_name)))
            tok_count = int(r.get("token_count", len(toks)))
            acronym = r.get("acronym") or ("".join(t[0] for t in toks if t) if toks else None)
            first_tok = clean_name.split()[0] if clean_name else None
            legal = extract_legal_suffix(clean_name)
            raw_t = t_lookup.get(eid, r.get("translit", r.get("business_name_transliterated")))
        else:
            clean_name = getattr(r, "clean_name", "") or ""
            tok_set = getattr(r, "token_set", None)
            toks = list(tok_set) if tok_set is not None else []
            grams = list(getattr(r, "char_3grams", []))
            char_len = int(getattr(r, "char_length", len(clean_name)))
            tok_count = int(getattr(r, "token_count", len(toks)))
            acronym = getattr(r, "acronym", None) or None
            first_tok = clean_name.split()[0] if clean_name else None
            legal = extract_legal_suffix(clean_name)
            raw_t = t_lookup.get(eid, getattr(r, "translit", None))

        # Clean transliterated string
        if raw_t is not None:
            t_clean = str(raw_t).strip().lower()
            if not t_clean or t_clean in _SENTINEL_EMPTY_STRINGS:
                t_clean = None
        else:
            t_clean = None

        rows.append({
            "entity_id": str(eid),
            "clean_name": clean_name,
            "translit": t_clean,
            "token_set": [str(t) for t in toks],
            "char_3grams": [str(g) for g in grams],
            "char_len": char_len,
            "token_count": tok_count,
            "first_token": first_tok,
            "acronym": acronym,
            "legal_suffix": legal,
        })

    lookup_schema = {
        "entity_id": pl.Utf8,
        "clean_name": pl.Utf8,
        "translit": pl.Utf8,
        "token_set": pl.List(pl.Utf8),
        "char_3grams": pl.List(pl.Utf8),
        "char_len": pl.Int64,
        "token_count": pl.Int64,
        "first_token": pl.Utf8,
        "acronym": pl.Utf8,
        "legal_suffix": pl.Utf8,
    }

    if not rows:
        return pl.DataFrame({
            "entity_id": pl.Series([], dtype=pl.Utf8),
            "clean_name": pl.Series([], dtype=pl.Utf8),
            "translit": pl.Series([], dtype=pl.Utf8),
            "token_set": pl.Series([], dtype=pl.List(pl.Utf8)),
            "char_3grams": pl.Series([], dtype=pl.List(pl.Utf8)),
            "char_len": pl.Series([], dtype=pl.Int64),
            "token_count": pl.Series([], dtype=pl.Int64),
            "first_token": pl.Series([], dtype=pl.Utf8),
            "acronym": pl.Series([], dtype=pl.Utf8),
            "legal_suffix": pl.Series([], dtype=pl.Utf8),
        }, schema=lookup_schema)

    return pl.DataFrame(rows, schema=lookup_schema)


def extract_name_features_batch(
    candidate_pairs_df: pl.DataFrame,
    s1_representations: Union[Dict[str, Any], pl.DataFrame],
    cand_representations: Union[Dict[str, Any], pl.DataFrame],
    s1_translit: Optional[Union[Dict[str, Optional[str]], pl.DataFrame]] = None,
    cand_translit: Optional[Union[Dict[str, Optional[str]], pl.DataFrame]] = None,
) -> pl.DataFrame:
    """
    Vectorized batch calculation of all 12 pairwise business name features.

    Avoids row-by-row Python iteration over candidate pairs by joining lookup
    tables and evaluating SIMD-accelerated Polars expressions.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS.
        s1_representations: Mapping of S1 entity_id -> NameRepresentation (or lookup DataFrame).
        cand_representations: Mapping of Candidate entity_id -> NameRepresentation (or lookup DataFrame).
        s1_translit: Optional mapping or DataFrame of S1 entity_id -> transliterated string.
        cand_translit: Optional mapping or DataFrame of Cand entity_id -> transliterated string.

    Returns:
        Polars DataFrame containing PAIR_ID_COLUMNS + NAME_FEATURE_NAMES.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(f"Required identity column '{col}' missing from candidate_pairs_df")

    if candidate_pairs_df.height == 0:
        schema = {col: pl.Utf8 for col in PAIR_ID_COLUMNS}
        schema.update(NAME_FEATURE_SCHEMA)
        return pl.DataFrame(schema=schema)

    s1_lookup = _name_representations_to_lookup_df(s1_representations, s1_translit)
    cand_lookup = _name_representations_to_lookup_df(cand_representations, cand_translit)

    joined = (
        candidate_pairs_df.select(PAIR_ID_COLUMNS)
        .join(s1_lookup, left_on="source1_entity_id", right_on="entity_id", how="left")
        .join(cand_lookup, left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_cand")
        .with_columns([
            pl.col("token_set").fill_null([]),
            pl.col("token_set_cand").fill_null([]),
            pl.col("char_3grams").fill_null([]),
            pl.col("char_3grams_cand").fill_null([]),
        ])
    )

    token_inter = pl.col("token_set").list.set_intersection(pl.col("token_set_cand")).list.len()
    token_union = pl.col("token_set").list.set_union(pl.col("token_set_cand")).list.len()
    token_len_a = pl.col("token_set").list.len()
    token_len_b = pl.col("token_set_cand").list.len()
    token_min = pl.min_horizontal(token_len_a, token_len_b)

    char_inter = pl.col("char_3grams").list.set_intersection(pl.col("char_3grams_cand")).list.len()
    char_union = pl.col("char_3grams").list.set_union(pl.col("char_3grams_cand")).list.len()

    char_len_a = pl.col("char_len").fill_null(0)
    char_len_b = pl.col("char_len_cand").fill_null(0)
    char_min = pl.min_horizontal(char_len_a, char_len_b)
    char_max = pl.max_horizontal(char_len_a, char_len_b)

    exprs = [
        pl.when(
            pl.col("clean_name").is_not_null()
            & (pl.col("clean_name") != "")
            & (pl.col("clean_name") == pl.col("clean_name_cand"))
        )
        .then(1).otherwise(0).cast(pl.Int8).alias("name_exact_norm"),

        pl.when(
            pl.col("translit").is_not_null()
            & pl.col("translit_cand").is_not_null()
            & (pl.col("translit") != "")
            & (pl.col("translit_cand") != "")
            & (pl.col("translit") == pl.col("translit_cand"))
        )
        .then(1).otherwise(0).cast(pl.Int8).alias("name_exact_translit"),

        pl.when((token_union == 0) | token_union.is_null()).then(0.0)
        .otherwise(token_inter / token_union).cast(pl.Float32).alias("name_token_jaccard"),

        pl.when((token_min == 0) | token_min.is_null()).then(0.0)
        .otherwise(token_inter / token_min).cast(pl.Float32).alias("name_token_overlap"),

        pl.when((token_len_a + token_len_b == 0) | (token_len_a + token_len_b).is_null()).then(0.0)
        .otherwise((2.0 * token_inter) / (token_len_a + token_len_b)).cast(pl.Float32).alias("name_token_dice"),

        (pl.col("token_count").fill_null(0) - pl.col("token_count_cand").fill_null(0)).abs().cast(pl.Int16).alias("name_token_count_diff"),

        pl.when((char_union == 0) | char_union.is_null()).then(0.0)
        .otherwise(char_inter / char_union).cast(pl.Float32).alias("name_char_3gram_jaccard"),

        (char_len_a - char_len_b).abs().cast(pl.Int16).alias("name_char_len_diff"),

        pl.when((char_max == 0) | char_max.is_null()).then(0.0)
        .otherwise(char_min / char_max).cast(pl.Float32).alias("name_char_len_ratio"),

        pl.when(
            pl.col("first_token").is_not_null()
            & (pl.col("first_token") != "")
            & (pl.col("first_token") == pl.col("first_token_cand"))
        )
        .then(1).otherwise(0).cast(pl.Int8).alias("name_first_token_exact"),

        pl.when(
            pl.col("acronym").is_not_null()
            & (pl.col("acronym") != "")
            & (pl.col("acronym") == pl.col("acronym_cand"))
        )
        .then(1).otherwise(0).cast(pl.Int8).alias("name_acronym_match"),

        pl.when(
            pl.col("legal_suffix").is_not_null()
            & (pl.col("legal_suffix") != "")
            & (pl.col("legal_suffix") == pl.col("legal_suffix_cand"))
        )
        .then(1).otherwise(0).cast(pl.Int8).alias("name_legal_suffix_match"),
    ]

    return joined.select(PAIR_ID_COLUMNS + exprs)
