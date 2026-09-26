"""
Candidate pair generation pipeline.

Loads processed parquet data, builds the block index,
generates candidate pairs, and returns results with provenance.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

from src.analysis.data_loader import load_processed_parquet
from src.candidate_generation.block_index import BlockIndex

logger = logging.getLogger(__name__)

# Columns needed from parquet for blocking
BLOCKING_COLUMNS = [
    "entity_id",
    "source",
    "business_name_normalized",
    "business_name_transliterated",
    "business_address_normalized",
    "country_normalized",
]


def _df_to_records(df: pl.DataFrame) -> List[Dict[str, Any]]:
    """Convert a Polars DataFrame to list of dicts, handling nulls."""
    return df.to_dicts()


def generate_candidate_pairs(
    s1_dir: str = "data/processed/train/source1",
    s2_dir: str = "data/processed/train/source2",
    s3_dir: str = "data/processed/train/source3",
    active_keys: Optional[List[str]] = None,
    max_block_size: Optional[int] = None,
    cap_blocks: bool = False,
) -> Tuple[BlockIndex, Set[Tuple[str, str]], Dict[Tuple[str, str], Set[str]]]:
    """
    Full pipeline: load data, build index, generate candidate pairs.

    Parameters:
        s1_dir: Path to source1 processed parquet directory.
        s2_dir: Path to source2 processed parquet directory.
        s3_dir: Path to source3 processed parquet directory.
        active_keys: List of key labels to use (default: all A-F).
        max_block_size: Block size cap for detection/reporting.
        cap_blocks: Whether to apply block size capping to pair generation.

    Returns:
        Tuple of (BlockIndex, candidate_pairs, pair_provenance).
    """
    t0 = time.time()

    # Build block index
    index = BlockIndex(max_block_size=max_block_size)

    # Load and index source1
    logger.info("Loading Source 1...")
    s1_lf = load_processed_parquet(s1_dir, source="source1", columns=BLOCKING_COLUMNS)
    s1_df = s1_lf.collect()
    logger.info(f"  Source 1: {len(s1_df):,} records loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    s1_records = _df_to_records(s1_df)
    for rec in s1_records:
        index.add_s1_record(rec["entity_id"], rec, active_keys=active_keys)
    logger.info(f"  Source 1 indexed in {time.time()-t1:.1f}s")
    del s1_df, s1_records

    # Load and index source2
    logger.info("Loading Source 2...")
    t2 = time.time()
    s2_lf = load_processed_parquet(s2_dir, source="source2", columns=BLOCKING_COLUMNS)
    s2_df = s2_lf.collect()
    logger.info(f"  Source 2: {len(s2_df):,} records loaded in {time.time()-t2:.1f}s")

    t2b = time.time()
    s2_records = _df_to_records(s2_df)
    for rec in s2_records:
        index.add_candidate_record(rec["entity_id"], rec, active_keys=active_keys)
    logger.info(f"  Source 2 indexed in {time.time()-t2b:.1f}s")
    del s2_df, s2_records

    # Load and index source3
    logger.info("Loading Source 3...")
    t3 = time.time()
    s3_lf = load_processed_parquet(s3_dir, source="source3", columns=BLOCKING_COLUMNS)
    s3_df = s3_lf.collect()
    logger.info(f"  Source 3: {len(s3_df):,} records loaded in {time.time()-t3:.1f}s")

    t3b = time.time()
    s3_records = _df_to_records(s3_df)
    for rec in s3_records:
        index.add_candidate_record(rec["entity_id"], rec, active_keys=active_keys)
    logger.info(f"  Source 3 indexed in {time.time()-t3b:.1f}s")
    del s3_df, s3_records

    # Generate pairs
    logger.info("Generating candidate pairs...")
    t4 = time.time()
    pairs, provenance = index.generate_pairs(cap_blocks=cap_blocks)
    logger.info(
        f"  Generated {len(pairs):,} candidate pairs in {time.time()-t4:.1f}s "
        f"(total time: {time.time()-t0:.1f}s)"
    )

    return index, pairs, provenance
