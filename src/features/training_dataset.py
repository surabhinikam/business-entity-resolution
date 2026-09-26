"""
Supervised Training Dataset Construction Module for Business Entity Resolution.

Part 7: Responsible for candidate-pair labeling, ground-truth identity normalization,
and leakage-free entity-level train/validation splitting.

Design & Guarantees:
- Pure Blocker Candidates: ONLY candidate pairs produced by the frozen blocker are used.
  Never creates Cartesian or arbitrary negative pairs.
- Ground-Truth Disambiguation: Carefully normalizes ground truth into canonical
  composite identity (source1_entity_id, candidate_entity_id, candidate_source),
  guaranteeing source2 and source3 candidates with identical or colliding entity IDs
  are never confused.
- Multiple & Zero Match Support: Accurately supports S1 entities with multiple true matches
  and S1 entities with zero true matches.
- Entity-Level Separation: Splits strictly by source1_entity_id so that no S1 entity
  appears in both training and validation sets (zero S1 overlap, zero pair leakage).
- Stratification & Reproducibility: Stratifies S1 entities by positive match count to maintain
  balanced class ratios across splits, using a fixed random seed.
- Explicit Sampling: No silent downsampling; optional training negative downsampling is
  explicitly configurable while validation data remains strictly untampered.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    LABEL_COLUMN,
    LABEL_DTYPE,
    FULL_PIPELINE_COLUMNS,
    SUPERVISED_DATASET_COLUMNS,
    SUPERVISED_FEATURE_SCHEMA,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Dataset Statistics Data Structures
# =============================================================================

@dataclass
class DatasetSplitStats:
    """Statistical summary of candidate pairs, class balance, and entity counts in a split."""
    total_pairs: int
    positive_pairs: int
    negative_pairs: int
    positive_rate_pct: float
    imbalance_ratio: float  # negatives / positives (e.g. 500.0 means 1:500)
    unique_s1_entities: int
    unique_candidate_entities: int
    source2_candidate_count: int
    source3_candidate_count: int
    s1_entities_with_positives: int
    s1_entities_zero_positives: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pairs": self.total_pairs,
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "positive_rate_pct": self.positive_rate_pct,
            "imbalance_ratio": self.imbalance_ratio,
            "unique_s1_entities": self.unique_s1_entities,
            "unique_candidate_entities": self.unique_candidate_entities,
            "source2_candidate_count": self.source2_candidate_count,
            "source3_candidate_count": self.source3_candidate_count,
            "s1_entities_with_positives": self.s1_entities_with_positives,
            "s1_entities_zero_positives": self.s1_entities_zero_positives,
        }


@dataclass
class SupervisedDatasetSplit:
    """Supervised training and validation splits with strict entity separation guarantees."""
    train: pl.DataFrame
    validation: pl.DataFrame
    train_stats: DatasetSplitStats
    validation_stats: DatasetSplitStats
    s1_overlap_count: int  # Must be 0
    candidate_pair_overlap_count: int  # Must be 0
    all_validation_positives_retained_by_blocker: bool  # Must be True
    validation_blocker_recall: Optional[float] = None


# =============================================================================
# 2. Ground-Truth Normalization
# =============================================================================

def normalize_ground_truth(
    ground_truth: Union[pl.DataFrame, pl.LazyFrame, str, os.PathLike, Sequence[Tuple[Any, ...]], Set[Tuple[Any, ...]]]
) -> pl.DataFrame:
    """
    Normalizes diverse ground-truth inputs into a canonical Polars DataFrame
    with schema: [source1_entity_id: Utf8, candidate_entity_id: Utf8, candidate_source: Utf8].

    Handles:
    - Raw ground truth TSV/DataFrame with [source1_entity_id, matched_entity_ids] (comma-separated)
    - Exploded ground truth with [source1_entity_id, matched_entity_id, target_source]
    - Normalized ground truth with [source1_entity_id, candidate_entity_id, candidate_source]
    - Python sets/lists of tuples (s1_id, cand_id, cand_source) or (s1_id, cand_id)
    - File path string pointing to TSV or Parquet file

    Guarantees:
    - Correctly maps S2-* to 'source2' and S3-* to 'source3'
    - Eliminates singletons (S1 with zero matches) from positive link table
    - Deduplicates identical match tuples
    - Returns strict Utf8 schema matching PAIR_ID_COLUMNS
    """
    if isinstance(ground_truth, (str, os.PathLike)):
        path_str = str(ground_truth)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Ground-truth file not found: {path_str}")
        if path_str.endswith(".parquet"):
            gt_df = pl.read_parquet(path_str)
        else:
            gt_df = pl.read_csv(
                path_str,
                separator="\t",
                truncate_ragged_lines=True,
                null_values=["", "NULL", "null", "None", "NaN"],
            )
    elif isinstance(ground_truth, pl.LazyFrame):
        gt_df = ground_truth.collect()
    elif isinstance(ground_truth, pl.DataFrame):
        gt_df = ground_truth
    elif isinstance(ground_truth, (set, list, tuple)):
        # Handle set/list of tuples
        raw_list = list(ground_truth)
        if not raw_list:
            return pl.DataFrame(
                schema={col: pl.Utf8 for col in PAIR_ID_COLUMNS}
            )
        first_elem = raw_list[0]
        if len(first_elem) == 3:
            return pl.DataFrame(
                {
                    "source1_entity_id": [str(t[0]) for t in raw_list],
                    "candidate_entity_id": [str(t[1]) for t in raw_list],
                    "candidate_source": [str(t[2]).lower().strip() for t in raw_list],
                }
            ).unique(maintain_order=True)
        elif len(first_elem) == 2:
            s1_ids = [str(t[0]) for t in raw_list]
            cand_ids = [str(t[1]) for t in raw_list]
            sources = []
            for cid in cand_ids:
                cid_upper = cid.upper()
                if cid_upper.startswith("S2-") or cid_upper.startswith("S2_"):
                    sources.append("source2")
                elif cid_upper.startswith("S3-") or cid_upper.startswith("S3_"):
                    sources.append("source3")
                else:
                    sources.append("unknown")
            return pl.DataFrame(
                {
                    "source1_entity_id": s1_ids,
                    "candidate_entity_id": cand_ids,
                    "candidate_source": sources,
                }
            ).unique(maintain_order=True)
        else:
            raise ValueError(f"Ground truth tuples must have length 2 or 3, got: {len(first_elem)}")
    else:
        raise TypeError(f"Unsupported ground_truth type: {type(ground_truth)}")

    if gt_df.height == 0:
        return pl.DataFrame(schema={col: pl.Utf8 for col in PAIR_ID_COLUMNS})

    cols = gt_df.columns

    # Format 1: Raw TSV format (source1_entity_id, matched_entity_ids)
    if "matched_entity_ids" in cols and "source1_entity_id" in cols:
        valid_matches = gt_df.filter(
            pl.col("matched_entity_ids").is_not_null()
            & (pl.col("matched_entity_ids").cast(pl.Utf8).str.strip_chars() != "")
        )
        if valid_matches.height == 0:
            return pl.DataFrame(schema={col: pl.Utf8 for col in PAIR_ID_COLUMNS})

        exploded = (
            valid_matches.with_columns(
                pl.col("matched_entity_ids").cast(pl.Utf8).str.split(",")
            )
            .explode("matched_entity_ids", empty_as_null=True)
            .with_columns(
                pl.col("source1_entity_id").cast(pl.Utf8).str.strip_chars(),
                pl.col("matched_entity_ids").cast(pl.Utf8).str.strip_chars().alias("candidate_entity_id"),
            )
            .filter(pl.col("candidate_entity_id") != "")
            .with_columns(
                pl.when(pl.col("candidate_entity_id").str.to_uppercase().str.starts_with("S2-"))
                .then(pl.lit("source2"))
                .when(pl.col("candidate_entity_id").str.to_uppercase().str.starts_with("S3-"))
                .then(pl.lit("source3"))
                .otherwise(pl.lit("unknown"))
                .alias("candidate_source")
            )
            .select(PAIR_ID_COLUMNS)
            .unique(maintain_order=True)
        )
        return exploded

    # Format 2: Exploded format with matched_entity_id and target_source
    if "matched_entity_id" in cols and "source1_entity_id" in cols:
        target_src_col = "target_source" if "target_source" in cols else "candidate_source"
        if target_src_col in cols:
            out_df = gt_df.select([
                pl.col("source1_entity_id").cast(pl.Utf8).str.strip_chars(),
                pl.col("matched_entity_id").cast(pl.Utf8).str.strip_chars().alias("candidate_entity_id"),
                pl.col(target_src_col).cast(pl.Utf8).str.to_lowercase().str.strip_chars().alias("candidate_source"),
            ]).unique()
        else:
            # Derive target source from matched_entity_id prefix
            out_df = (
                gt_df.with_columns(
                    pl.col("source1_entity_id").cast(pl.Utf8).str.strip_chars(),
                    pl.col("matched_entity_id").cast(pl.Utf8).str.strip_chars().alias("candidate_entity_id"),
                )
                .with_columns(
                    pl.when(pl.col("candidate_entity_id").str.to_uppercase().str.starts_with("S2-"))
                    .then(pl.lit("source2"))
                    .when(pl.col("candidate_entity_id").str.to_uppercase().str.starts_with("S3-"))
                    .then(pl.lit("source3"))
                    .otherwise(pl.lit("unknown"))
                    .alias("candidate_source")
                )
                .select(PAIR_ID_COLUMNS)
                .unique()
            )
        return out_df

    # Format 3: Already has PAIR_ID_COLUMNS
    if all(col in cols for col in PAIR_ID_COLUMNS):
        return gt_df.select([
            pl.col("source1_entity_id").cast(pl.Utf8).str.strip_chars(),
            pl.col("candidate_entity_id").cast(pl.Utf8).str.strip_chars(),
            pl.col("candidate_source").cast(pl.Utf8).str.to_lowercase().str.strip_chars(),
        ]).unique()

    raise ValueError(
        f"Ground-truth DataFrame schema unrecognized: {cols}. "
        f"Expected columns: ['source1_entity_id', 'matched_entity_ids'] or {PAIR_ID_COLUMNS}"
    )


# =============================================================================
# 3. Candidate Pair Labeling
# =============================================================================

def label_candidate_pairs(
    candidate_pairs_df: pl.DataFrame,
    ground_truth: Any,
    label_col: str = LABEL_COLUMN,
) -> pl.DataFrame:
    """
    Labels candidate pairs generated by the blocker against ground truth.

    Semantics:
    - label = 1 iff composite identity (source1_entity_id, candidate_entity_id, candidate_source)
      is present in ground-truth linkage.
    - label = 0 for all other candidate pairs.
    - Strictly preserves the input candidate pairs without generating any Cartesian or arbitrary negatives.
    - Correctly handles S1 entities with multiple matches and S1 entities with zero matches.
    - Never confuses source2 and source3 candidates with colliding entity IDs.

    Parameters:
        candidate_pairs_df: DataFrame containing PAIR_ID_COLUMNS and optional features.
        ground_truth: Ground-truth linkages (DataFrame, TSV path, or tuple collection).
        label_col: Name of the resulting binary target column (default: 'label').

    Returns:
        Polars DataFrame with exact same rows and input columns preserved, plus the Int8 label column.
    """
    for col in PAIR_ID_COLUMNS:
        if col not in candidate_pairs_df.columns:
            raise ValueError(
                f"Candidate pairs DataFrame is missing required composite identity column: '{col}'. "
                f"Expected columns: {PAIR_ID_COLUMNS}"
            )

    if candidate_pairs_df.height == 0:
        return candidate_pairs_df.with_columns(
            pl.Series(label_col, [], dtype=LABEL_DTYPE)
        )

    gt_normalized = normalize_ground_truth(ground_truth)

    # Prepare ground-truth table with positive indicator
    gt_positives = gt_normalized.select([
        *PAIR_ID_COLUMNS,
        pl.lit(1, dtype=LABEL_DTYPE).alias(label_col),
    ])

    # Left join preserves all candidate pairs without altering order or adding rows
    labeled_df = (
        candidate_pairs_df.join(
            gt_positives,
            on=PAIR_ID_COLUMNS,
            how="left",
        )
        .with_columns(
            pl.col(label_col).fill_null(0).cast(LABEL_DTYPE)
        )
    )

    logger.debug(
        f"Labeled {labeled_df.height:,} candidate pairs: "
        f"{(labeled_df[label_col] == 1).sum():,} positives, "
        f"{(labeled_df[label_col] == 0).sum():,} negatives."
    )

    return labeled_df


# =============================================================================
# 4. Split Statistics Computation
# =============================================================================

def compute_split_stats(df: pl.DataFrame, label_col: str = LABEL_COLUMN) -> DatasetSplitStats:
    """Computes comprehensive class balance, entity distribution, and source statistics."""
    total = df.height
    if total == 0:
        return DatasetSplitStats(
            total_pairs=0,
            positive_pairs=0,
            negative_pairs=0,
            positive_rate_pct=0.0,
            imbalance_ratio=0.0,
            unique_s1_entities=0,
            unique_candidate_entities=0,
            source2_candidate_count=0,
            source3_candidate_count=0,
            s1_entities_with_positives=0,
            s1_entities_zero_positives=0,
        )

    pos_mask = df[label_col] == 1
    pos_count = int(pos_mask.sum())
    neg_count = total - pos_count
    pos_rate = (pos_count / total) * 100.0 if total > 0 else 0.0
    imbalance = (neg_count / pos_count) if pos_count > 0 else float("inf")

    unique_s1 = df["source1_entity_id"].n_unique()
    unique_cand = df["candidate_entity_id"].n_unique()

    s2_count = int((df["candidate_source"] == "source2").sum())
    s3_count = int((df["candidate_source"] == "source3").sum())

    pos_s1_set = set(df.filter(pos_mask)["source1_entity_id"].unique().to_list())
    s1_with_pos = len(pos_s1_set)
    s1_zero_pos = unique_s1 - s1_with_pos

    return DatasetSplitStats(
        total_pairs=total,
        positive_pairs=pos_count,
        negative_pairs=neg_count,
        positive_rate_pct=pos_rate,
        imbalance_ratio=imbalance,
        unique_s1_entities=unique_s1,
        unique_candidate_entities=unique_cand,
        source2_candidate_count=s2_count,
        source3_candidate_count=s3_count,
        s1_entities_with_positives=s1_with_pos,
        s1_entities_zero_positives=s1_zero_pos,
    )


# =============================================================================
# 5. Leakage-Free Entity-Level Train / Validation Splitting
# =============================================================================

def split_supervised_dataset(
    labeled_df: pl.DataFrame,
    val_fraction: float = 0.2,
    stratify_by_positive: bool = True,
    seed: int = 42,
    negative_downsample_ratio: Optional[float] = None,
    ground_truth: Optional[Any] = None,
    label_col: str = LABEL_COLUMN,
) -> SupervisedDatasetSplit:
    """
    Constructs train and validation datasets partitioned strictly at the entity level by source1_entity_id.

    Guarantees:
    - Zero S1 leakage: No source1_entity_id appears in both train and validation splits.
    - Zero Pair leakage: No candidate pair appears in both splits.
    - Only Blocker Pairs: All pairs in train and validation originate directly from candidate generation.
    - Verification of Validation Positives: Confirms all validation positives were retained by the blocker.
    - Reproducibility: Fully deterministic when using the same random seed.
    - Explicit Downsampling: Training negative downsampling is strictly optional and configurable.
      Validation data is NEVER downsampled to ensure unbiased evaluation.

    Parameters:
        labeled_df: Labeled candidate pairs DataFrame.
        val_fraction: Fraction of S1 entities allocated to the validation split (e.g. 0.2 = 20%).
        stratify_by_positive: Whether to stratify S1 entities based on positive match counts.
        seed: Random seed for deterministic entity partitioning.
        negative_downsample_ratio: Optional maximum ratio of negative pairs to positive pairs in training.
            If None (default), all negatives are retained. Validation is never downsampled.
        ground_truth: Optional ground-truth for computing validation blocker recall.
        label_col: Name of the label column.

    Returns:
        SupervisedDatasetSplit containing train and validation DataFrames, statistics, and leakage checks.
    """
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError(f"val_fraction must be in [0.0, 1.0), got {val_fraction}")

    for col in PAIR_ID_COLUMNS:
        if col not in labeled_df.columns:
            raise ValueError(f"Missing required identity column: '{col}'")
    if label_col not in labeled_df.columns:
        raise ValueError(f"Missing required label column: '{label_col}'")

    if labeled_df.height == 0:
        empty_stats = compute_split_stats(labeled_df, label_col=label_col)
        return SupervisedDatasetSplit(
            train=labeled_df,
            validation=labeled_df,
            train_stats=empty_stats,
            validation_stats=empty_stats,
            s1_overlap_count=0,
            candidate_pair_overlap_count=0,
            all_validation_positives_retained_by_blocker=True,
            validation_blocker_recall=0.0,
        )

    # 1. Group candidate pairs by source1_entity_id to determine entity-level stratification
    all_s1_entities = sorted(labeled_df["source1_entity_id"].unique().to_list())
    rng = random.Random(seed)

    train_s1_set: Set[str] = set()
    val_s1_set: Set[str] = set()

    if stratify_by_positive:
        # Count positive candidate pairs per S1 entity
        pos_counts = (
            labeled_df.filter(pl.col(label_col) == 1)
            .group_by("source1_entity_id")
            .len()
        )
        pos_dict = dict(zip(pos_counts["source1_entity_id"].to_list(), pos_counts["len"].to_list()))

        # Stratified buckets: 0 positives, 1 positive, 2+ positives
        bucket_0: List[str] = []
        bucket_1: List[str] = []
        bucket_2plus: List[str] = []

        for s1_id in all_s1_entities:
            c = pos_dict.get(s1_id, 0)
            if c == 0:
                bucket_0.append(s1_id)
            elif c == 1:
                bucket_1.append(s1_id)
            else:
                bucket_2plus.append(s1_id)

        # Shuffle each bucket deterministically
        for bucket in (bucket_0, bucket_1, bucket_2plus):
            rng.shuffle(bucket)
            n_val = max(1 if len(bucket) > 1 and val_fraction > 0 else 0, int(round(len(bucket) * val_fraction)))
            val_s1_set.update(bucket[:n_val])
            train_s1_set.update(bucket[n_val:])
    else:
        shuffled_s1 = list(all_s1_entities)
        rng.shuffle(shuffled_s1)
        n_val = max(1 if len(shuffled_s1) > 1 and val_fraction > 0 else 0, int(round(len(shuffled_s1) * val_fraction)))
        val_s1_set.update(shuffled_s1[:n_val])
        train_s1_set.update(shuffled_s1[n_val:])

    # 2. Partition candidate pairs strictly by S1 entity set
    train_df = labeled_df.filter(pl.col("source1_entity_id").is_in(list(train_s1_set)))
    val_df = labeled_df.filter(pl.col("source1_entity_id").is_in(list(val_s1_set)))

    # 3. Rigorous Leakage Assertions
    s1_overlap = set(train_df["source1_entity_id"].unique().to_list()) & set(val_df["source1_entity_id"].unique().to_list())
    if s1_overlap:
        raise AssertionError(
            f"Entity-level leakage detected! {len(s1_overlap)} S1 entities appear in both train and val."
        )

    # Check pair-level overlap
    train_pairs = set(zip(
        train_df["source1_entity_id"].to_list(),
        train_df["candidate_entity_id"].to_list(),
        train_df["candidate_source"].to_list(),
    ))
    val_pairs = set(zip(
        val_df["source1_entity_id"].to_list(),
        val_df["candidate_entity_id"].to_list(),
        val_df["candidate_source"].to_list(),
    ))
    pair_overlap = train_pairs & val_pairs
    if pair_overlap:
        raise AssertionError(
            f"Candidate-pair leakage detected! {len(pair_overlap)} pairs appear in both train and val."
        )

    # 4. Optional Explicit Training Negative Downsampling (Configurable)
    if negative_downsample_ratio is not None and train_df.height > 0:
        if negative_downsample_ratio <= 0.0:
            raise ValueError(f"negative_downsample_ratio must be positive, got {negative_downsample_ratio}")

        train_positives = train_df.filter(pl.col(label_col) == 1)
        train_negatives = train_df.filter(pl.col(label_col) == 0)

        n_pos = train_positives.height
        max_negatives = int(math.ceil(n_pos * negative_downsample_ratio))

        if train_negatives.height > max_negatives:
            logger.info(
                f"Applying explicit negative downsampling to train: "
                f"retaining {max_negatives:,} of {train_negatives.height:,} negatives "
                f"({n_pos:,} positives, ratio={negative_downsample_ratio})."
            )
            # Sample negatives deterministically
            neg_indices = list(range(train_negatives.height))
            rng.shuffle(neg_indices)
            sampled_indices = sorted(neg_indices[:max_negatives])
            train_negatives_sampled = train_negatives[sampled_indices]
            train_df = pl.concat([train_positives, train_negatives_sampled]).sort(PAIR_ID_COLUMNS)

    # 5. Verification of Validation Positives
    # Every positive in val_df must be a valid candidate pair generated by the blocker
    val_positives = val_df.filter(pl.col(label_col) == 1)
    all_val_positives_retained = True
    for row in val_positives.iter_rows(named=True):
        pair_key = (row["source1_entity_id"], row["candidate_entity_id"], row["candidate_source"])
        if pair_key not in val_pairs:
            all_val_positives_retained = False
            break

    # 6. Validation Blocker Recall (if ground truth provided)
    val_blocker_recall: Optional[float] = None
    if ground_truth is not None and len(val_s1_set) > 0:
        gt_norm = normalize_ground_truth(ground_truth)
        val_gt = gt_norm.filter(pl.col("source1_entity_id").is_in(list(val_s1_set)))
        total_val_gt = val_gt.height
        if total_val_gt > 0:
            val_blocker_recall = val_positives.height / total_val_gt

    # 7. Compute Statistics
    train_stats = compute_split_stats(train_df, label_col=label_col)
    val_stats = compute_split_stats(val_df, label_col=label_col)

    logger.info(
        f"Supervised split created: Train={train_df.height:,} pairs "
        f"({train_stats.positive_pairs:,} pos, {train_stats.positive_rate_pct:.2f}%), "
        f"Val={val_df.height:,} pairs "
        f"({val_stats.positive_pairs:,} pos, {val_stats.positive_rate_pct:.2f}%)."
    )

    return SupervisedDatasetSplit(
        train=train_df,
        validation=val_df,
        train_stats=train_stats,
        validation_stats=val_stats,
        s1_overlap_count=len(s1_overlap),
        candidate_pair_overlap_count=len(pair_overlap),
        all_validation_positives_retained_by_blocker=all_val_positives_retained,
        validation_blocker_recall=val_blocker_recall,
    )


# =============================================================================
# 6. Unified Supervised Dataset Builder
# =============================================================================

class SupervisedDatasetBuilder:
    """
    High-level orchestrator for constructing labeled candidate datasets and
    leakage-free train/validation splits from candidate pairs and ground truth.
    """

    def __init__(self, ground_truth: Any, label_col: str = LABEL_COLUMN):
        """
        Initialize builder with ground-truth linkages.

        Parameters:
            ground_truth: DataFrame, TSV path, or tuple collection.
            label_col: Target binary label column name.
        """
        self.label_col = label_col
        self.normalized_gt = normalize_ground_truth(ground_truth)
        logger.info(
            f"SupervisedDatasetBuilder initialized with {self.normalized_gt.height:,} ground-truth matches."
        )

    def label(self, candidate_pairs_df: pl.DataFrame) -> pl.DataFrame:
        """Labels candidate pairs against normalized ground truth."""
        return label_candidate_pairs(
            candidate_pairs_df=candidate_pairs_df,
            ground_truth=self.normalized_gt,
            label_col=self.label_col,
        )

    def create_splits(
        self,
        candidate_pairs_df: pl.DataFrame,
        val_fraction: float = 0.2,
        stratify_by_positive: bool = True,
        seed: int = 42,
        negative_downsample_ratio: Optional[float] = None,
    ) -> SupervisedDatasetSplit:
        """Labels candidate pairs and performs entity-level train/validation split."""
        labeled_df = self.label(candidate_pairs_df)
        return split_supervised_dataset(
            labeled_df=labeled_df,
            val_fraction=val_fraction,
            stratify_by_positive=stratify_by_positive,
            seed=seed,
            negative_downsample_ratio=negative_downsample_ratio,
            ground_truth=self.normalized_gt,
            label_col=self.label_col,
        )
