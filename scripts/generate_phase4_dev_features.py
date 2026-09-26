#!/usr/bin/env python3
"""Generate a bounded Phase 4 development feature sample."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.data_loader import discover_parquet_files
from src.candidate_generation.block_index import BlockIndex
from src.features.feature_pipeline import FeaturePipeline
from src.features.feature_schema import (
    FULL_FEATURE_SCHEMA,
    FULL_PIPELINE_COLUMNS,
    PAIR_ID_COLUMNS,
)

ACTIVE_KEYS = ["A", "C", "D", "E", "F"]
MAX_BLOCK_SIZE = 5000
DEFAULT_SAMPLE_SIZE = 500_000
DEFAULT_OUTPUT = "data/processed/train/dev_features/phase4_dev_corrected.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase4_dev_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persist a deterministic bounded Phase 4 feature sample."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of candidate pairs to persist (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output Parquet path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--allow-smaller",
        action="store_true",
        default=True,
        help="Write all available candidates when fewer than --sample-size exist (default: True).",
    )
    parser.add_argument(
        "--s1-files",
        type=int,
        default=3,
        help="Number of source1 partition files to load (default: 3).",
    )
    parser.add_argument(
        "--s2-files",
        type=int,
        default=6,
        help="Number of genuine source2 partition files to load (default: 6).",
    )
    parser.add_argument(
        "--s3-files",
        type=int,
        default=6,
        help="Number of genuine source3 partition files to load (default: 6).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    return parser.parse_args()


def load_bounded_partitions(
    num_s1: int = 3,
    num_s2: int = 6,
    num_s3: int = 6,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    s1_files = discover_parquet_files("data/processed/train/source1", source="source1")[:num_s1]
    s2_files = discover_parquet_files("data/processed/train/source2", source="source2")[:num_s2]
    s3_files = discover_parquet_files("data/processed/train/source3", source="source3")[:num_s3]

    if not s1_files or not s2_files or not s3_files:
        raise FileNotFoundError(
            "Expected bounded processed partitions were not found for source1, source2, and source3."
        )

    # Strict partition verification
    for f in s1_files:
        base = os.path.basename(f)
        if "source1" not in base:
            raise ValueError(f"Invalid non-source1 file in source1 partitions: {f}")
    for f in s2_files:
        base = os.path.basename(f)
        if "source2" not in base or base.startswith("train_source1"):
            raise ValueError(f"Invalid non-source2 file in source2 partitions: {f}")
    for f in s3_files:
        base = os.path.basename(f)
        if "source3" not in base:
            raise ValueError(f"Invalid non-source3 file in source3 partitions: {f}")

    logger.info("Discovered genuine partition files:")
    logger.info("  S1 files: %s", [os.path.basename(f) for f in s1_files])
    logger.info("  S2 files: %s", [os.path.basename(f) for f in s2_files])
    logger.info("  S3 files: %s", [os.path.basename(f) for f in s3_files])

    return (
        pl.concat([pl.read_parquet(path) for path in s1_files]),
        pl.concat([pl.read_parquet(path) for path in s2_files]),
        pl.concat([pl.read_parquet(path) for path in s3_files]),
    )


def build_candidate_sample(
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    sample_size: int,
    allow_smaller: bool,
) -> tuple[pl.DataFrame, dict[tuple[str, str], set[str]]]:
    index = BlockIndex(max_block_size=MAX_BLOCK_SIZE)

    for row in s1_df.iter_rows(named=True):
        entity_id = row["entity_id"]
        if not entity_id.startswith("S1-"):
            raise ValueError(f"Invalid entity_id format for source1 record: {entity_id}")
        index.add_s1_record(entity_id, row, active_keys=ACTIVE_KEYS)

    candidate_source_by_id: dict[str, str] = {}
    for row in s2_df.iter_rows(named=True):
        entity_id = row["entity_id"]
        if not entity_id.startswith("S2-"):
            raise ValueError(f"Invalid entity_id format for source2 record: {entity_id}")
        candidate_source_by_id[entity_id] = "source2"
        index.add_candidate_record(entity_id, row, active_keys=ACTIVE_KEYS)

    for row in s3_df.iter_rows(named=True):
        entity_id = row["entity_id"]
        if not entity_id.startswith("S3-"):
            raise ValueError(f"Invalid entity_id format for source3 record: {entity_id}")
        candidate_source_by_id[entity_id] = "source3"
        index.add_candidate_record(entity_id, row, active_keys=ACTIVE_KEYS)

    pairs, provenance = index.generate_pairs(cap_blocks=True)
    if len(pairs) < sample_size:
        if not allow_smaller:
            raise RuntimeError(
                f"Only {len(pairs):,} candidate pairs are available after frozen blocking; "
                f"cannot produce the requested {sample_size:,} rows."
            )
        logger.warning(
            "Fewer candidates than requested after frozen blocking: "
            "requested=%s, actual=%s. Writing all available candidates because "
            "--allow-smaller was provided.",
            f"{sample_size:,}",
            f"{len(pairs):,}",
        )

    selected_pairs = sorted(pairs)[:sample_size]
    candidate_pairs_df = pl.DataFrame(
        {
            "source1_entity_id": [pair[0] for pair in selected_pairs],
            "candidate_entity_id": [pair[1] for pair in selected_pairs],
            "candidate_source": [candidate_source_by_id[pair[1]] for pair in selected_pairs],
        }
    )
    selected_provenance = {pair: provenance[pair] for pair in selected_pairs}
    return candidate_pairs_df, selected_provenance


def validate_features(features_df: pl.DataFrame, expected_rows: int) -> None:
    if features_df.height != expected_rows:
        raise ValueError(
            f"Feature row count mismatch: expected {expected_rows:,}, got {features_df.height:,}."
        )

    if features_df.columns != FULL_PIPELINE_COLUMNS:
        raise ValueError(
            "Feature columns do not match FULL_PIPELINE_COLUMNS. "
            f"Expected {FULL_PIPELINE_COLUMNS}, got {features_df.columns}."
        )

    if features_df.width != 32:
        raise ValueError(f"Expected 32 columns, got {features_df.width}.")

    if features_df.schema != FULL_FEATURE_SCHEMA:
        mismatches = {
            column: (features_df.schema.get(column), FULL_FEATURE_SCHEMA.get(column))
            for column in FULL_PIPELINE_COLUMNS
            if features_df.schema.get(column) != FULL_FEATURE_SCHEMA.get(column)
        }
        raise ValueError(f"Feature schema mismatch: {mismatches}")

    for column in PAIR_ID_COLUMNS:
        if features_df[column].null_count() != 0:
            raise ValueError(f"Identity column '{column}' contains null values.")
        if features_df.filter(pl.col(column).str.strip_chars() == "").height != 0:
            raise ValueError(f"Identity column '{column}' contains empty values.")

    if features_df.select(PAIR_ID_COLUMNS).n_unique() != expected_rows:
        raise ValueError("Duplicate candidate pair IDs detected in the feature output.")


def write_features(features_df: pl.DataFrame, output_path: str, overwrite: bool = False) -> None:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {path}. Pass --overwrite to overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.write_parquet(path)


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer.")

    started = time.perf_counter()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output_path}. Pass --overwrite to overwrite."
        )

    logger.info("Loading bounded processed partitions...")
    s1_df, s2_df, s3_df = load_bounded_partitions(
        num_s1=args.s1_files,
        num_s2=args.s2_files,
        num_s3=args.s3_files,
    )
    logger.info(
        "Loaded partitions: S1=%s, S2=%s, S3=%s records",
        f"{s1_df.height:,}",
        f"{s2_df.height:,}",
        f"{s3_df.height:,}",
    )

    logger.info(
        "Generating candidates with active keys %s and MAX_BLOCK_SIZE=%s...",
        ACTIVE_KEYS,
        MAX_BLOCK_SIZE,
    )
    candidate_pairs_df, pair_provenance = build_candidate_sample(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        sample_size=args.sample_size,
        allow_smaller=args.allow_smaller,
    )

    logger.info("Generating %s Phase 3 features...", f"{args.sample_size:,}")
    features_df = FeaturePipeline().generate_features_from_records(
        candidate_pairs_df=candidate_pairs_df,
        s1_records=s1_df,
        cand_records=pl.concat([s2_df, s3_df]),
        pair_provenance=pair_provenance,
    )
    validate_features(features_df, candidate_pairs_df.height)
    write_features(features_df, args.output, overwrite=args.overwrite)

    source_counts = (
        features_df.group_by("candidate_source")
        .len()
        .sort("candidate_source")
        .iter_rows()
    )
    elapsed = time.perf_counter() - started

    print("Phase 4 development feature sample written successfully.")
    print(f"Rows written: {features_df.height:,}")
    print(f"Columns: {features_df.width}")
    print(f"Output path: {args.output}")
    print(
        "Candidate source counts: "
        + ", ".join(f"{source}={count:,}" for source, count in source_counts)
    )
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
