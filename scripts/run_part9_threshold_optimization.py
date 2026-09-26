#!/usr/bin/env python3
"""
Phase 3 Part 9: Threshold Optimization and Match Post-Processing Runner.

Workflow:
1. Candidate pair generation with frozen blocker (A + C + D + E + F, cap=5000).
2. Unified 29-feature extraction via FeaturePipeline.
3. Supervised dataset construction with entity-level split (80/20).
4. Leakage-safe threshold optimization for F0.5 using 5-fold OOF cross-validation
   STRICTLY on the training set (never touching validation labels).
5. Train final LightGBM model on full training set.
6. Predict match probabilities on untouched natural validation set.
7. Evaluate selected threshold vs default 0.50 baseline on validation set.
8. Execute post-processing preserving zero-match and multi-match entities.
9. Generate detailed diagnostics:
   - Match count per S1 entity
   - Zero-match and multi-match counts
   - Probability distributions of accepted / rejected pairs
   - Error analysis: FP and FN inspection
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
from src.features.training_dataset import SupervisedDatasetBuilder
from src.models.baseline_model import BaselineMatchingModel
from src.models.metrics import (
    evaluate_matching_probabilities,
    evaluate_predictions_at_threshold,
)
from src.models.post_processing import (
    optimize_threshold_oof,
    MatchPostProcessor,
    compute_prediction_diagnostics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("part9_runner")

CANDIDATE_PAIR_SAMPLE_SIZE = 50_000
RANDOM_SEED = 42
MAX_BLOCK_SIZE = 5000
ACTIVE_KEYS = ["A", "C", "D", "E", "F"]


def main():
    logger.info("=" * 80)
    logger.info("PHASE 3 PART 9: THRESHOLD OPTIMIZATION & MATCH POST-PROCESSING")
    logger.info("=" * 80)

    # 1. Load Parquet Partitions
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

    # 2. Frozen Blocker Candidate Generation
    logger.info(f"Step 2: Building BlockIndex with keys {ACTIVE_KEYS} and cap={MAX_BLOCK_SIZE}...")
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
    sorted_pairs = sorted(list(pairs_set))
    sample_pairs = sorted_pairs[:CANDIDATE_PAIR_SAMPLE_SIZE]
    logger.info(f"Sampled {len(sample_pairs):,} candidate pairs for Part 9 benchmark.")

    candidate_pairs_df = pl.DataFrame({
        "source1_entity_id": [p[0] for p in sample_pairs],
        "candidate_entity_id": [p[1] for p in sample_pairs],
        "candidate_source": [cand_source_map[p[1]] for p in sample_pairs],
    })
    pair_provenance_sub = {p: prov_dict[p] for p in sample_pairs}

    # 3. Extract Unified 29 Features
    logger.info("Step 3: Extracting 29 features using FeaturePipeline...")
    pipeline = FeaturePipeline()
    all_cand_df = pl.concat([s2_df, s3_df])

    t_feat_start = time.perf_counter()
    features_df = pipeline.generate_features_from_records(
        candidate_pairs_df=candidate_pairs_df,
        s1_records=s1_df,
        cand_records=all_cand_df,
        pair_provenance=pair_provenance_sub,
    )
    logger.info(f"Feature extraction completed in {time.perf_counter() - t_feat_start:.2f}s.")

    # 4. Supervised Dataset Construction & Entity-Level Split
    logger.info("Step 4: Supervised dataset labeling and entity-level 80/20 split...")
    gt_path = "data/raw/train/train_ground_truth.tsv"
    builder = SupervisedDatasetBuilder(ground_truth=gt_path)

    split = builder.create_splits(
        candidate_pairs_df=features_df,
        val_fraction=0.20,
        stratify_by_positive=True,
        seed=RANDOM_SEED,
        negative_downsample_ratio=None,
    )

    logger.info(
        f"Train split: {split.train.height:,} pairs ({split.train_stats.positive_pairs} pos, "
        f"{split.train_stats.negative_pairs:,} neg, 1:{split.train_stats.imbalance_ratio:.1f} imbalance)"
    )
    logger.info(
        f"Validation split: {split.validation.height:,} pairs ({split.validation_stats.positive_pairs} pos, "
        f"{split.validation_stats.negative_pairs:,} neg, 1:{split.validation_stats.imbalance_ratio:.1f} imbalance)"
    )

    # 5. Leakage-Safe Out-Of-Fold Threshold Optimization on TRAINING Data ONLY
    logger.info("Step 5: Performing 5-fold OOF threshold optimization on TRAINING data only...")
    t_opt_start = time.perf_counter()
    opt_result = optimize_threshold_oof(
        train_df=split.train,
        n_splits=5,
        seed=RANDOM_SEED,
        threshold_candidates=[round(t, 2) for t in np.linspace(0.05, 0.95, 37)],
        beta=0.5,
        model_params={
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": RANDOM_SEED,
        },
    )
    logger.info(
        f"Threshold optimization completed in {time.perf_counter() - t_opt_start:.2f}s. "
        f"Optimal Training OOF Threshold: {opt_result.best_threshold:.2f} (OOF F0.5={opt_result.best_f05:.4f})"
    )

    # 6. Train Final Model on Full Training Set & Predict on Validation
    logger.info("Step 6: Fitting final model on full training set and evaluating on validation...")
    model = BaselineMatchingModel(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
    )
    model.fit(split.train)

    val_probs = model.predict_proba(split.validation)
    val_y_true = split.validation["label"].to_numpy().astype(int)

    # 7. Evaluate on Validation Set at Selected Threshold vs Baseline 0.50
    eval_opt = evaluate_predictions_at_threshold(
        y_true=val_y_true,
        y_prob=val_probs,
        threshold=opt_result.best_threshold,
    )
    eval_base = evaluate_predictions_at_threshold(
        y_true=val_y_true,
        y_prob=val_probs,
        threshold=0.50,
    )
    full_report = model.evaluate(split.validation)

    # 8. Post-Processing & Diagnostics
    logger.info("Step 7: Executing post-processing and computing diagnostics...")
    processor = MatchPostProcessor(threshold=opt_result.best_threshold)

    val_with_probs = split.validation.with_columns(
        pl.Series("probability", val_probs, dtype=pl.Float64)
    )
    accepted_pairs = processor.filter_candidate_pairs(val_with_probs)

    all_val_s1 = sorted(split.validation["source1_entity_id"].unique().to_list())
    submission_df = processor.format_submission_linkages(accepted_pairs, all_val_s1)

    diagnostics = compute_prediction_diagnostics(
        val_df=split.validation,
        probs=val_probs,
        threshold=opt_result.best_threshold,
        label_col="label",
        top_errors_to_inspect=10,
    )

    # Print Full Structured Report
    print("\n" + "=" * 80)
    print("PHASE 3 PART 9: THRESHOLD OPTIMIZATION & POST-PROCESSING REPORT")
    print("=" * 80)

    print("\n### 1. Threshold Selection Methodology")
    print("- **Optimization Dataset**: Strictly TRAINING set (zero validation exposure).")
    print("- **Validation Technique**: 5-Fold Entity-Disjoint Out-Of-Fold (OOF) Cross-Validation on `source1_entity_id`.")
    print("- **Target Objective**: Maximize $F_{0.5}$ score (weighting precision $2\\times$ over recall).")
    print(f"- **Selected Threshold**: **{opt_result.best_threshold:.2f}** (OOF $F_{{0.5}} = {opt_result.best_f05:.4f}$, OOF Prec = {opt_result.best_precision:.4f}, OOF Rec = {opt_result.best_recall:.4f})")
    print(f"- **OOF PR-AUC**: **{opt_result.oof_pr_auc:.4f}** | **OOF ROC-AUC**: **{opt_result.oof_roc_auc:.4f}**")

    print("\n### 2. Validation Metrics: Selected Threshold vs. Baseline (0.50)")
    print(f"| Metric | Baseline Threshold (0.50) | Selected Threshold ({opt_result.best_threshold:.2f}) | Delta |")
    print(f"|:-------|:--------------------------:|:-----------------------------------:|:-----:|")
    print(f"| **Primary Metric ($F_{{0.5}}$)** | **{eval_base.f05:.4f}** | **{eval_opt.f05:.4f}** | **{eval_opt.f05 - eval_base.f05:+.4f}** |")
    print(f"| **Precision** | {eval_base.precision:.4f} | {eval_opt.precision:.4f} | {eval_opt.precision - eval_base.precision:+.4f} |")
    print(f"| **Recall** | {eval_base.recall:.4f} | {eval_opt.recall:.4f} | {eval_opt.recall - eval_base.recall:+.4f} |")
    print(f"| **$F_1$ Score** | {eval_base.f1:.4f} | {eval_opt.f1:.4f} | {eval_opt.f1 - eval_base.f1:+.4f} |")
    print(f"| **Predicted Positives** | {eval_base.predicted_positives} | {eval_opt.predicted_positives} | {eval_opt.predicted_positives - eval_base.predicted_positives:+d} |")
    print(f"| **True Positives (TP)** | {eval_base.true_positives} | {eval_opt.true_positives} | {eval_opt.true_positives - eval_base.true_positives:+d} |")
    print(f"| **False Positives (FP)** | {eval_base.false_positives} | {eval_opt.false_positives} | {eval_opt.false_positives - eval_base.false_positives:+d} |")
    print(f"| **False Negatives (FN)** | {eval_base.false_negatives} | {eval_opt.false_negatives} | {eval_opt.false_negatives - eval_base.false_negatives:+d} |")
    print(f"| **True Negatives (TN)** | {eval_base.true_negatives} | {eval_opt.true_negatives} | {eval_opt.true_negatives - eval_base.true_negatives:+d} |")
    print(f"| **PR-AUC (Untouched Val)** | {full_report.pr_auc:.4f} | {full_report.pr_auc:.4f} | 0.0000 |")

    print("\n### 3. Match Count & Entity Distribution Diagnostics (Validation Set)")
    print(f"- **Total S1 Entities in Validation**: {diagnostics['total_s1_entities']:,}")
    print(f"- **Zero-Match S1 Entities**: {diagnostics['zero_match_s1_count']:,} ({diagnostics['zero_match_s1_pct']:.2f}%)")
    print(f"- **Single-Match S1 Entities**: {diagnostics['single_match_s1_count']:,} ({diagnostics['single_match_s1_pct']:.2f}%)")
    print(f"- **Multi-Match S1 Entities**: {diagnostics['multi_match_s1_count']:,} ({diagnostics['multi_match_s1_pct']:.2f}%)")
    print(f"- **Max Matches for Single S1**: {diagnostics['max_matches_single_s1']}")
    print(f"- **Source 2 Accepted Matches**: {diagnostics['source2_accepted']}")
    print(f"- **Source 3 Accepted Matches**: {diagnostics['source3_accepted']}")

    print("\n### 4. Probability Distribution Diagnostics")
    ap = diagnostics["accepted_prob_stats"]
    rp = diagnostics["rejected_prob_stats"]
    print(f"| Prediction Pool | Min Prob | Median Prob | Mean Prob | Max Prob | Std Dev |")
    print(f"|:----------------|:--------:|:-----------:|:---------:|:--------:|:-------:|")
    print(f"| **Accepted Pairs ($P \\ge {opt_result.best_threshold:.2f}$)** | {ap['min']:.4f} | {ap['median']:.4f} | {ap['mean']:.4f} | {ap['max']:.4f} | {ap['std']:.4f} |")
    print(f"| **Rejected Pairs ($P < {opt_result.best_threshold:.2f}$)** | {rp['min']:.4f} | {rp['median']:.4f} | {rp['mean']:.4f} | {rp['max']:.4f} | {rp['std']:.4f} |")

    print("\n### 5. Error Analysis: False Positives on Validation")
    if diagnostics["false_positives"]:
        print(f"Found {len(diagnostics['false_positives'])} False Positive pair(s):")
        for fp in diagnostics["false_positives"]:
            print(f"- Pair: (`{fp['source1_entity_id']}`, `{fp['candidate_entity_id']}`, `{fp['candidate_source']}`) | Prob: {fp['probability']:.4f} | True Label: {fp['label']}")
            if "name_char_3gram_jaccard" in fp:
                print(f"  Name 3-gram: {fp['name_char_3gram_jaccard']:.3f} | Addr 3-gram: {fp['address_char_3gram_similarity']:.3f} | Shared Num: {fp['shared_address_number_count']} | Keys: {fp['matched_key_count']}")
    else:
        print("Zero False Positives on validation set.")

    print("\n### 6. Error Analysis: False Negatives on Validation")
    if diagnostics["false_negatives"]:
        print(f"Found {len(diagnostics['false_negatives'])} False Negative pair(s):")
        for fn in diagnostics["false_negatives"]:
            print(f"- Pair: (`{fn['source1_entity_id']}`, `{fn['candidate_entity_id']}`, `{fn['candidate_source']}`) | Prob: {fn['probability']:.4f} | True Label: {fn['label']}")
            if "name_char_3gram_jaccard" in fn:
                print(f"  Name 3-gram: {fn['name_char_3gram_jaccard']:.3f} | Addr 3-gram: {fn['address_char_3gram_similarity']:.3f} | Shared Num: {fn['shared_address_number_count']} | Keys: {fn['matched_key_count']}")
    else:
        print("Zero False Negatives on validation set.")

    logger.info("Part 9 runner executed successfully.")


if __name__ == "__main__":
    main()
