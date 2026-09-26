#!/usr/bin/env python3
"""
Baseline Supervised Matching Model Runner (Part 8).

Trains and evaluates a LightGBM classifier on blocker-generated candidate pairs
using the 29 engineered features and entity-level train/val splitting.

Outputs:
- Training and validation dataset sizes and positive/negative counts
- Validation metrics at default threshold 0.5 (F0.5, Precision, Recall, F1, Confusion Matrix, PR-AUC, ROC-AUC)
- Threshold sweep table for probability analysis across [0.05 ... 0.95]
- Feature importance ranking (gain and relative %)
- Comparison between natural imbalance baseline and downsampled training
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.data_loader import load_ground_truth
from src.candidate_generation.block_index import BlockIndex
from src.features.feature_pipeline import FeaturePipeline
from src.features.feature_schema import ALL_FEATURE_NAMES, PAIR_ID_COLUMNS
from src.features.training_dataset import (
    SupervisedDatasetBuilder,
    split_supervised_dataset,
)
from src.models.baseline_model import BaselineMatchingModel
from src.models.metrics import evaluate_matching_probabilities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("baseline_runner")

CANDIDATE_PAIR_SAMPLE_SIZE = 50_000
RANDOM_SEED = 42
MAX_BLOCK_SIZE = 5000
ACTIVE_KEYS = ["A", "C", "D", "E", "F"]


def main():
    logger.info("=" * 80)
    logger.info("PHASE 3 PART 8: BASELINE SUPERVISED MATCHING MODEL (LIGHTGBM)")
    logger.info("=" * 80)

    # 1. Load Data Partitions
    logger.info("Step 1: Loading raw processed parquet partitions...")
    s1_files = sorted(glob.glob("data/processed/train/source1/*.parquet"))[:3]
    s2_files = sorted(glob.glob("data/processed/train/source2/*.parquet"))[:4]
    s3_files = sorted(glob.glob("data/processed/train/source3/*.parquet"))[:4]

    s1_df = pl.concat([pl.read_parquet(f) for f in s1_files])
    s2_df = pl.concat([pl.read_parquet(f) for f in s2_files])
    s3_df = pl.concat([pl.read_parquet(f) for f in s3_files])

    logger.info(
        f"Loaded partitions: S1={len(s1_df):,} records, S2={len(s2_df):,} records, S3={len(s3_df):,} records"
    )

    # 2. Generate Candidate Pairs with Frozen Blocker
    logger.info(f"Step 2: Building BlockIndex with {ACTIVE_KEYS} and MAX_BLOCK_SIZE={MAX_BLOCK_SIZE}...")
    idx = BlockIndex(max_block_size=MAX_BLOCK_SIZE)

    for row in s1_df.iter_rows(named=True):
        idx.add_s1_record(row["entity_id"], row, active_keys=ACTIVE_KEYS)

    cand_source_map = {}
    for row in s2_df.iter_rows(named=True):
        eid = row["entity_id"]
        cand_source_map[eid] = "source2"
        idx.add_candidate_record(eid, row, active_keys=ACTIVE_KEYS)

    for row in s3_df.iter_rows(named=True):
        eid = row["entity_id"]
        cand_source_map[eid] = "source3"
        idx.add_candidate_record(eid, row, active_keys=ACTIVE_KEYS)

    pairs_set, prov_dict = idx.generate_pairs(cap_blocks=True)
    logger.info(f"Total deduplicated candidate pairs generated: {len(pairs_set):,}")

    # Deterministic candidate sample
    sorted_pairs = sorted(list(pairs_set))
    sample_pairs = sorted_pairs[:CANDIDATE_PAIR_SAMPLE_SIZE]
    logger.info(f"Using reproducible candidate subset: {len(sample_pairs):,} pairs")

    candidate_pairs_df = pl.DataFrame({
        "source1_entity_id": [p[0] for p in sample_pairs],
        "candidate_entity_id": [p[1] for p in sample_pairs],
        "candidate_source": [cand_source_map[p[1]] for p in sample_pairs],
    })
    pair_provenance_sub = {p: prov_dict[p] for p in sample_pairs}

    # 3. Precompute Representations and Extract Features
    logger.info("Step 3: Precomputing record representations & generating 29 features...")
    pipeline = FeaturePipeline()
    all_cand_df = pl.concat([s2_df, s3_df])

    t_feat_start = time.perf_counter()
    features_df = pipeline.generate_features_from_records(
        candidate_pairs_df=candidate_pairs_df,
        s1_records=s1_df,
        cand_records=all_cand_df,
        pair_provenance=pair_provenance_sub,
    )
    t_feat_elapsed = time.perf_counter() - t_feat_start
    logger.info(f"Feature extraction completed in {t_feat_elapsed:.2f}s ({features_df.width} columns, {features_df.height} rows).")

    # 4. Supervised Dataset Construction & Entity-Level Split
    logger.info("Step 4: Labeling candidate pairs & entity-level splitting (80/20)...")
    gt_path = "data/raw/train/train_ground_truth.tsv"
    builder = SupervisedDatasetBuilder(ground_truth=gt_path)

    # Baseline 1: Natural imbalance (no training downsampling)
    split_natural = builder.create_splits(
        candidate_pairs_df=features_df,
        val_fraction=0.20,
        stratify_by_positive=True,
        seed=RANDOM_SEED,
        negative_downsample_ratio=None,
    )

    logger.info(
        f"Train set: {split_natural.train.height:,} pairs "
        f"({split_natural.train_stats.positive_pairs} pos, {split_natural.train_stats.negative_pairs} neg, "
        f"rate={split_natural.train_stats.positive_rate_pct:.3f}%, imbalance=1:{split_natural.train_stats.imbalance_ratio:.1f})"
    )
    logger.info(
        f"Validation set: {split_natural.validation.height:,} pairs "
        f"({split_natural.validation_stats.positive_pairs} pos, {split_natural.validation_stats.negative_pairs} neg, "
        f"rate={split_natural.validation_stats.positive_rate_pct:.3f}%, imbalance=1:{split_natural.validation_stats.imbalance_ratio:.1f})"
    )
    logger.info(f"Entity leakage check: S1 overlap = {split_natural.s1_overlap_count}, pair overlap = {split_natural.candidate_pair_overlap_count}")

    # 5. Train LightGBM Baseline Model (Natural Imbalance)
    logger.info("Step 5: Training LightGBM baseline model (natural imbalance)...")
    model = BaselineMatchingModel(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
        importance_type="gain",
    )

    t_train_start = time.perf_counter()
    model.fit(split_natural.train)
    t_train_elapsed = time.perf_counter() - t_train_start
    logger.info(f"Model training completed in {t_train_elapsed:.2f}s.")

    # 6. Evaluate Model on Validation Set
    logger.info("Step 6: Evaluating model on natural validation set across thresholds...")
    threshold_sweep = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    report = model.evaluate(split_natural.validation, thresholds=threshold_sweep)

    # 7. Feature Importance
    fi_df = model.get_feature_importances()

    # 8. Compare with Configurable Negative Downsampling (e.g. 10:1 ratio)
    logger.info("Step 7: Evaluating training negative downsampling (10:1 ratio)...")
    split_downsampled = builder.create_splits(
        candidate_pairs_df=features_df,
        val_fraction=0.20,
        stratify_by_positive=True,
        seed=RANDOM_SEED,
        negative_downsample_ratio=10.0,
    )
    model_downsampled = BaselineMatchingModel(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
        importance_type="gain",
    )
    model_downsampled.fit(split_downsampled.train)
    report_downsampled = model_downsampled.evaluate(split_natural.validation, thresholds=threshold_sweep)

    # Print Summary Tables
    print("\n" + "=" * 80)
    print("BASELINE SUPERVISED MATCHING MODEL REPORT")
    print("=" * 80)

    print("\n### 1. Dataset & Split Summary")
    print(f"- Total Candidate Subset: {len(sample_pairs):,} pairs")
    print(f"- Training Pairs (Natural): {split_natural.train.height:,} ({split_natural.train_stats.positive_pairs} Pos, {split_natural.train_stats.negative_pairs:,} Neg, Imbalance 1:{split_natural.train_stats.imbalance_ratio:.1f})")
    print(f"- Validation Pairs (Natural): {split_natural.validation.height:,} ({split_natural.validation_stats.positive_pairs} Pos, {split_natural.validation_stats.negative_pairs:,} Neg, Imbalance 1:{split_natural.validation_stats.imbalance_ratio:.1f})")
    print(f"- S1 Entity Overlap: {split_natural.s1_overlap_count} (Zero Leakage)")
    print(f"- Candidate Pair Overlap: {split_natural.candidate_pair_overlap_count} (Zero Leakage)")
    print(f"- All Validation Positives in Blocker: {split_natural.all_validation_positives_retained_by_blocker}")

    print("\n### 2. Validation Metrics at Default Threshold (0.50)")
    d05 = report.default_metrics_at_05
    print(f"- **Primary Metric (F0.5)**: **{d05.f05:.4f}**")
    print(f"- **Precision**: {d05.precision:.4f}")
    print(f"- **Recall**: {d05.recall:.4f}")
    print(f"- **F1 Score**: {d05.f1:.4f}")
    print(f"- **PR-AUC (Average Precision)**: **{report.pr_auc:.4f}**")
    print(f"- **ROC-AUC**: **{report.roc_auc:.4f}**")
    print(f"- **Confusion Matrix**: TP={d05.true_positives}, FP={d05.false_positives}, FN={d05.false_negatives}, TN={d05.true_negatives}")

    print("\n### 3. Probability & Threshold Analysis (Natural Imbalance Baseline)")
    print(f"| Threshold | Pred Pos | True Pos | False Pos | Precision | Recall | F1 Score | **F0.5 Score** |")
    print(f"|:---------:|:--------:|:--------:|:---------:|:---------:|:------:|:--------:|:--------------:|")
    for te in report.threshold_evaluations:
        print(
            f"| {te.threshold:.2f} | {te.predicted_positives:4d} | {te.true_positives:3d} | "
            f"{te.false_positives:4d} | {te.precision:8.4f} | {te.recall:6.4f} | "
            f"{te.f1:8.4f} | **{te.f05:8.4f}** |"
        )

    print("\n### 4. Comparison: Natural Imbalance vs 10:1 Downsampled Training")
    print(f"| Model Setting | Train Size | PR-AUC | Best F0.5 (and Thresh) | F0.5 at 0.50 | Prec at 0.50 | Rec at 0.50 |")
    print(f"|:-------------|:----------:|:------:|:---------------------:|:------------:|:------------:|:-----------:|")
    best_nat = max(report.threshold_evaluations, key=lambda x: x.f05)
    best_down = max(report_downsampled.threshold_evaluations, key=lambda x: x.f05)
    d05_down = report_downsampled.default_metrics_at_05
    print(
        f"| Natural Imbalance (1:{split_natural.train_stats.imbalance_ratio:.0f}) | "
        f"{split_natural.train.height:,} | {report.pr_auc:.4f} | "
        f"{best_nat.f05:.4f} (th={best_nat.threshold:.2f}) | {d05.f05:.4f} | "
        f"{d05.precision:.4f} | {d05.recall:.4f} |"
    )
    print(
        f"| Downsampled (1:10) | {split_downsampled.train.height:,} | {report_downsampled.pr_auc:.4f} | "
        f"{best_down.f05:.4f} (th={best_down.threshold:.2f}) | {d05_down.f05:.4f} | "
        f"{d05_down.precision:.4f} | {d05_down.recall:.4f} |"
    )

    print("\n### 5. Top 15 Feature Importances (Gain)")
    print(f"| Rank | Feature Name | Importance (Gain) | Relative Importance (%) |")
    print(f"|:----:|:-------------|:-----------------:|:-----------------------:|")
    for i, row in enumerate(fi_df.head(15).iter_rows(named=True), 1):
        print(f"| {i:2d} | `{row['feature_name']}` | {row['importance']:12.2f} | {row['relative_pct']:6.2f}% |")

    logger.info("Baseline supervised matching model runner completed successfully.")


if __name__ == "__main__":
    main()
