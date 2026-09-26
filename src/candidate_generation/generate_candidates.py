"""
Candidate pair generation pipeline.

Loads processed parquet data, builds the block index,
generates candidate pairs, and returns results with provenance.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
    limit_source1: Optional[int] = None,
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
        limit_source1: Limit number of source1 entities.

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

    if limit_source1 is not None and limit_source1 > 0:
        unique_s1 = sorted(s1_df["entity_id"].unique().to_list())[:limit_source1]
        s1_df = s1_df.filter(pl.col("entity_id").is_in(unique_s1))
        logger.info(f"  Applied limit_source1={limit_source1}: kept {len(s1_df):,} records")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate candidate pairs.")
    parser.add_argument("--s1-dir", default="data/processed/train/source1", help="Path to Source 1 processed directory.")
    parser.add_argument("--s2-dir", default="data/processed/train/source2", help="Path to Source 2 processed directory.")
    parser.add_argument("--s3-dir", default="data/processed/train/source3", help="Path to Source 3 processed directory.")
    parser.add_argument("--output", default=None, help="Output path for candidate pairs (Parquet).")
    parser.add_argument("--max-block-size", type=int, default=5000, help="Max block size.")
    parser.add_argument("--cap-blocks", action="store_true", help="Apply capping.")
    parser.add_argument("--limit-source1", type=int, default=None, help="Limit number of source1 entities for smoke testing.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    _, pairs, provenance = generate_candidate_pairs(
        s1_dir=args.s1_dir,
        s2_dir=args.s2_dir,
        s3_dir=args.s3_dir,
        max_block_size=args.max_block_size,
        cap_blocks=args.cap_blocks,
        limit_source1=args.limit_source1,
    )

    logger.info(f"Total candidate pairs generated: {len(pairs):,}")

    if args.output:
        keys_list = ["A", "C", "D", "E", "F"]
        prov_cols: Dict[str, List[int]] = {f"matched_key_{k}": [] for k in keys_list}
        for p in pairs:
            matched = provenance.get(p, set())
            for k in keys_list:
                prov_cols[f"matched_key_{k}"].append(1 if k in matched else 0)

        df = pl.DataFrame({
            "source1_entity_id": [p[0] for p in pairs],
            "candidate_entity_id": [p[1] for p in pairs],
            **{k: pl.Series(v, dtype=pl.Int8) for k, v in prov_cols.items()},
        }).with_columns(
            pl.when(pl.col("candidate_entity_id").str.to_uppercase().str.starts_with("S2-"))
            .then(pl.lit("source2"))
            .when(pl.col("candidate_entity_id").str.to_uppercase().str.starts_with("S3-"))
            .then(pl.lit("source3"))
            .otherwise(pl.lit("unknown"))
            .alias("candidate_source")
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df.write_parquet(args.output)
        logger.info(f"Saved {df.height:,} candidate pairs with provenance to {args.output}")
