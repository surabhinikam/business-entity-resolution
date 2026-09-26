#!/usr/bin/env python3
"""Generate labeled bounded Phase 4 development dataset using SupervisedDatasetBuilder."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.training_dataset import SupervisedDatasetBuilder
from src.features.feature_schema import PAIR_ID_COLUMNS, LABEL_COLUMN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_phase4_dev_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a bounded labeled Phase 4 dev dataset.")
    parser.add_argument(
        "--features",
        default="data/processed/train/dev_features/phase4_dev_500k.parquet",
        help="Input candidate features Parquet path.",
    )
    parser.add_argument(
        "--ground-truth",
        default="data/raw/train/train_ground_truth.tsv",
        help="Input ground-truth TSV path.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/train/dev_dataset",
        help="Output directory for train.parquet, validation.parquet, and manifest.json.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of Source1 entities for validation split (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    return parser.parse_args()


def validate_split(df: pl.DataFrame, name: str) -> None:
    if df.height == 0:
        logger.warning(f"Split {name} is empty.")
        return

    for col in PAIR_ID_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Split {name} missing identity column {col}")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Split {name} missing label column {LABEL_COLUMN}")

    labels = df[LABEL_COLUMN].unique().to_list()
    invalid_labels = [l for l in labels if l not in (0, 1)]
    if invalid_labels:
        raise ValueError(f"Split {name} contains invalid labels: {invalid_labels}")

    dupes = df.height - df.select(PAIR_ID_COLUMNS).n_unique()
    if dupes > 0:
        raise ValueError(f"Split {name} contains {dupes} duplicate candidate pairs.")

    logger.info(f"Split {name} passed validation.")


def write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"Manifest written to {path}")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "validation.parquet"
    manifest_path = out_dir / "manifest.json"

    logger.info(f"Loading candidate features from {args.features}...")
    candidate_features = pl.read_parquet(args.features)

    logger.info(f"Initializing SupervisedDatasetBuilder with GT {args.ground_truth}...")
    builder = SupervisedDatasetBuilder(ground_truth=args.ground_truth, label_col=LABEL_COLUMN)

    logger.info(f"Creating strictly disjoint train/validation splits (val_fraction={args.val_fraction}, seed={args.seed})...")
    split = builder.create_splits(
        candidate_pairs_df=candidate_features,
        val_fraction=args.val_fraction,
        stratify_by_positive=True,
        seed=args.seed,
    )

    train_df = split.train
    val_df = split.validation

    logger.info(f"Validating {train_df.height} training rows...")
    validate_split(train_df, "Train")
    logger.info(f"Validating {val_df.height} validation rows...")
    validate_split(val_df, "Validation")

    # Assert schemas match
    if train_df.schema != val_df.schema:
        raise ValueError("Train and validation schemas do not match.")

    # Assert disjoint
    train_s1 = set(train_df["source1_entity_id"].unique().to_list())
    val_s1 = set(val_df["source1_entity_id"].unique().to_list())
    overlap = train_s1 & val_s1
    if overlap:
        raise ValueError(f"Leakage detected! {len(overlap)} S1 entities in both splits.")

    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)

    feature_cols = [c for c in train_df.columns if c not in PAIR_ID_COLUMNS and c != LABEL_COLUMN]

    # Candidate source coverage
    t_src_counts = train_df.group_by("candidate_source").len().to_dicts()
    v_src_counts = val_df.group_by("candidate_source").len().to_dicts()
    src_coverage = {
        "train": {d["candidate_source"]: d["len"] for d in t_src_counts},
        "validation": {d["candidate_source"]: d["len"] for d in v_src_counts}
    }

    manifest = {
        "feature_artifact_path": args.features,
        "ground_truth_path": args.ground_truth,
        "train_output_path": str(train_path).replace("\\", "/"),
        "validation_output_path": str(val_path).replace("\\", "/"),
        "feature_columns": feature_cols,
        "label_column": LABEL_COLUMN,
        "identity_columns": PAIR_ID_COLUMNS,
        "candidate_source_coverage": src_coverage,
        "source1_level_split_policy": "strict_disjoint_stratified",
        "split_parameters": {
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "metrics": {
            "train": {
                "row_count": split.train_stats.total_pairs,
                "positive_count": split.train_stats.positive_pairs,
                "negative_count": split.train_stats.negative_pairs,
                "unique_source1_entities": split.train_stats.unique_s1_entities,
            },
            "validation": {
                "row_count": split.validation_stats.total_pairs,
                "positive_count": split.validation_stats.positive_pairs,
                "negative_count": split.validation_stats.negative_pairs,
                "unique_source1_entities": split.validation_stats.unique_s1_entities,
            }
        }
    }

    write_manifest(manifest, manifest_path)

    elapsed = time.perf_counter() - started
    logger.info(f"Done in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
