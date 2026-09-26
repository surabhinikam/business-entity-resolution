"""
Data loading and partition discovery utilities for raw and processed datasets.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List, Optional, Tuple, Dict, Any, Set
import logging

import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def discover_parquet_files(
    directory: str,
    source: Optional[str] = None,
    warn_on_misplaced: bool = True,
) -> List[str]:
    """
    Discover parquet files in directory, handling partitions and filtering.

    If source is specified (e.g. 'source1', 'source2', 'source3'), filters to
    files specifically belonging to that source (e.g. matching 'train_{source}_part_*.parquet').
    Detects and warns about misplaced files (e.g. 79 stray source1 files in source2 dir).
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_files = sorted(glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True))

    if not source:
        dir_name = os.path.basename(os.path.normpath(directory))
        if dir_name in ("source1", "source2", "source3"):
            source = dir_name

    if source:
        matching_files = []
        misplaced_files = []
        expected_stem = f"_{source}_"
        for f in all_files:
            base = os.path.basename(f)
            if expected_stem in base:
                matching_files.append(f)
            else:
                misplaced_files.append(f)

        if misplaced_files and warn_on_misplaced:
            logger.warning(
                f"[Partition Discovery] Detected {len(misplaced_files)} misplaced files in "
                f"{directory} not matching source '{source}'. "
                f"Example: {os.path.basename(misplaced_files[0])}. These are filtered out."
            )
        return matching_files

    return all_files


def load_raw_tsv(
    file_path: str,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Load raw TSV dataset using Polars.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pl.read_csv(
        file_path,
        separator="\t",
        columns=columns,
        n_rows=limit,
        truncate_ragged_lines=True,
        null_values=["", "NULL", "null", "None", "NaN"],
    )
    return df


def load_processed_parquet(
    directory: str,
    source: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pl.LazyFrame:
    """
    Load processed parquet dataset lazily using Polars.
    Discovers all valid partition files and returns a LazyFrame.
    """
    files = discover_parquet_files(directory, source=source)
    if not files:
        raise FileNotFoundError(f"No valid parquet files found in {directory} for source {source}")

    lf = pl.scan_parquet(files)
    if columns:
        lf = lf.select([c for c in columns if c in lf.collect_schema().names()])
    return lf


def load_ground_truth(
    file_path: str = "data/raw/train/train_ground_truth.tsv",
) -> pl.DataFrame:
    """
    Load ground truth linkage TSV.
    Columns: source1_entity_id, matched_entity_ids
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    df = pl.read_csv(
        file_path,
        separator="\t",
        truncate_ragged_lines=True,
    )
    return df


def explode_ground_truth(gt_df: pl.DataFrame) -> pl.DataFrame:
    """
    Explode comma-separated matched_entity_ids into individual rows.
    Returns DataFrame with columns:
      - source1_entity_id: str
      - matched_entity_id: str
      - target_source: 'source2' or 'source3'
    Filters out singletons (empty matches).
    """
    valid_matches = gt_df.filter(
        pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids").str.strip_chars() != "")
    )

    exploded = (
        valid_matches.with_columns(
            pl.col("matched_entity_ids").str.split(",")
        )
        .explode("matched_entity_ids", empty_as_null=True)
        .with_columns(
            pl.col("matched_entity_ids").str.strip_chars().alias("matched_entity_id")
        )
        .filter(pl.col("matched_entity_id") != "")
        .with_columns(
            pl.when(pl.col("matched_entity_id").str.starts_with("S2-"))
            .then(pl.lit("source2"))
            .when(pl.col("matched_entity_id").str.starts_with("S3-"))
            .then(pl.lit("source3"))
            .otherwise(pl.lit("unknown"))
            .alias("target_source")
        )
        .select(["source1_entity_id", "matched_entity_id", "target_source"])
    )
    return exploded
