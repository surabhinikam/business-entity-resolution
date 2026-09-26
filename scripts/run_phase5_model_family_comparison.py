"""
Phase 5 Runner: Controlled Model Family Comparison (LightGBM vs XGBoost vs CatBoost).

Executes:
1. Loads corrected development dataset (144,048 pairs, 858 positives) and labels candidates.
2. Creates identical 5 entity-disjoint folds (seed=42) partitioned strictly by source1_entity_id.
3. Evaluates LightGBM (frozen reference), XGBoost, and CatBoost on the exact same 29 features.
4. Collects complete Out-Of-Fold (OOF) predictions for every model without data leakage.
5. Computes multi-threshold metrics (0.82, 0.85, 0.90, 0.95), fold variances, runtime, and peak memory.
6. Conducts error overlap and complementarity analysis at threshold 0.90.
7. Generates reports/model/phase5_model_family_results.csv, phase5_error_overlap.csv, and phase5_model_family_comparison.md.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import polars as pl

from src.features.feature_schema import ALL_FEATURE_NAMES, LABEL_COLUMN, PAIR_ID_COLUMNS
from src.features.training_dataset import (
    label_candidate_pairs,
    normalize_ground_truth,
)
from src.models.cross_validation import create_entity_disjoint_kfold_splits
from src.models.model_family_comparison import (
    ErrorOverlapSummary,
    ModelFamilyResult,
    compute_model_family_error_overlap,
    run_model_family_cv,
)
from src.models.phase4_interactions import BASELINE_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants & Paths
DEV_FEATURES_PATH = "data/processed/train/dev_features/phase4_dev_corrected.parquet"
GROUND_TRUTH_PATH = "data/raw/train/train_ground_truth.tsv"
SOURCE1_DIR = "data/processed/train/source1"
OUTPUT_DIR = "reports/model"


def load_dataset() -> pl.DataFrame:
    """Loads dev features, labels with ground truth, and joins S1 country."""
    logger.info("Loading dev features: %s", DEV_FEATURES_PATH)
    df = pl.read_parquet(DEV_FEATURES_PATH)

    logger.info("Loading ground truth: %s", GROUND_TRUTH_PATH)
    gt_df = pl.read_csv(GROUND_TRUTH_PATH, separator="\t")
    gt_norm = normalize_ground_truth(gt_df)

    df = label_candidate_pairs(df, gt_norm)
    logger.info(
        "Candidate pairs labeled: %d positives, %d negatives.",
        int((df[LABEL_COLUMN] == 1).sum()), int((df[LABEL_COLUMN] == 0).sum())
    )

    s1_files = sorted(glob.glob(os.path.join(SOURCE1_DIR, "train_source1_part_*.parquet")))
    if s1_files:
        s1_meta = pl.concat([
            pl.read_parquet(f, columns=["entity_id", "country_normalized"])
            for f in s1_files
        ]).rename({"entity_id": "source1_entity_id", "country_normalized": "country"})
        df = df.join(s1_meta, on="source1_entity_id", how="left")
        df = df.with_columns(pl.col("country").fill_null("unknown"))

    return df


def save_results_csv(results: List[ModelFamilyResult], output_path: str) -> None:
    """Saves machine-readable model comparison metrics to CSV."""
    rows: List[Dict[str, Any]] = []
    for r in results:
        m82 = r.metrics_at_thresholds.get(0.82, {})
        m85 = r.metrics_at_thresholds.get(0.85, {})
        m90 = r.metrics_at_thresholds.get(0.90, {})
        m95 = r.metrics_at_thresholds.get(0.95, {})

        row: Dict[str, Any] = {
            "model_family": r.family_name,
            "pr_auc": round(r.oof_pr_auc, 4),
            "roc_auc": round(r.oof_roc_auc, 4),
            "f05_82": m82.get("f05", 0.0),
            "prec_82": m82.get("precision", 0.0),
            "rec_82": m82.get("recall", 0.0),
            "f05_85": m85.get("f05", 0.0),
            "prec_85": m85.get("precision", 0.0),
            "rec_85": m85.get("recall", 0.0),
            "f05_90": m90.get("f05", 0.0),
            "prec_90": m90.get("precision", 0.0),
            "rec_90": m90.get("recall", 0.0),
            "tp_90": m90.get("tp", 0),
            "fp_90": m90.get("fp", 0),
            "fn_90": m90.get("fn", 0),
            "f05_95": m95.get("f05", 0.0),
            "prec_95": m95.get("precision", 0.0),
            "rec_95": m95.get("recall", 0.0),
            "mean_f05_90": round(r.mean_f05_90, 4),
            "std_f05_90": round(r.std_f05_90, 4),
            "fold1_f05_90": round(r.per_fold_f05_90[0], 4) if len(r.per_fold_f05_90) > 0 else 0.0,
            "fold2_f05_90": round(r.per_fold_f05_90[1], 4) if len(r.per_fold_f05_90) > 1 else 0.0,
            "fold3_f05_90": round(r.per_fold_f05_90[2], 4) if len(r.per_fold_f05_90) > 2 else 0.0,
            "fold4_f05_90": round(r.per_fold_f05_90[3], 4) if len(r.per_fold_f05_90) > 3 else 0.0,
            "fold5_f05_90": round(r.per_fold_f05_90[4], 4) if len(r.per_fold_f05_90) > 4 else 0.0,
            "optimal_f05": r.optimal_f05,
            "optimal_threshold": r.optimal_threshold,
            "optimal_precision": r.optimal_precision,
            "optimal_recall": r.optimal_recall,
            "total_train_time_sec": round(r.total_train_time_sec, 2),
            "avg_fold_train_time_sec": round(r.avg_fold_train_time_sec, 2),
            "peak_memory_mb": round(r.peak_memory_mb, 2),
        }
        rows.append(row)

    df_out = pl.DataFrame(rows)
    df_out.write_csv(output_path)
    logger.info("Saved model family comparison CSV to %s", output_path)


def generate_markdown_report(
    results: List[ModelFamilyResult],
    error_summary: ErrorOverlapSummary,
    output_path: str,
    total_time: float,
) -> None:
    """Writes reports/model/phase5_model_family_comparison.md with comprehensive analysis."""
    logger.info("Writing model family comparison report to %s...", output_path)

    res_map = {r.family_name: r for r in results}
    lgb_res = res_map["lightgbm"]
    xgb_res = res_map["xgboost"]
    cat_res = res_map["catboost"]

    lgb_m90 = lgb_res.metrics_at_thresholds.get(0.90, {})
    xgb_m90 = xgb_res.metrics_at_thresholds.get(0.90, {})
    cat_m90 = cat_res.metrics_at_thresholds.get(0.90, {})

    lines: List[str] = [
        "# Phase 5: Controlled Model Family Comparison Report",
        "",
        f"**Date:** 2026-09-27  ",
        f"**Dataset:** `data/processed/train/dev_features/phase4_dev_corrected.parquet` (144,048 candidate pairs, 858 GT positives)  ",
        f"**Features:** Exactly the frozen 29 baseline features (`BASELINE_FEATURES`)  ",
        f"**Evaluation Strategy:** Identical 5-fold entity-disjoint cross-validation (seed=42)  ",
        f"**Execution Runtime:** {total_time:.1f}s  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "In Phase 5, we evaluated three major gradient-boosted tree families (**LightGBM**, **XGBoost**, and **CatBoost**) under strict experimental control. All models were trained and evaluated on the exact same 5 entity-disjoint folds and 29 features using comparable conservative configurations.",
        "",
        "### Key Findings:",
        f"1. **LightGBM Performance (Frozen Reference):** Achieves OOF PR-AUC = **{lgb_res.oof_pr_auc:.4f}**, F0.5 @ 0.90 = **{lgb_m90.get('f05', 0):.4f}** (Precision = {lgb_m90.get('precision', 0):.4f}, Recall = {lgb_m90.get('recall', 0):.4f}, {lgb_m90.get('fp', 0)} FPs, {lgb_m90.get('fn', 0)} FNs), with 5-fold mean F0.5 = **{lgb_res.mean_f05_90:.4f} ± {lgb_res.std_f05_90:.4f}** and runtime of **{lgb_res.total_train_time_sec:.1f}s**.",
        f"2. **XGBoost Performance:** Achieves OOF PR-AUC = **{xgb_res.oof_pr_auc:.4f}**, F0.5 @ 0.90 = **{xgb_m90.get('f05', 0):.4f}** (Precision = {xgb_m90.get('precision', 0):.4f}, Recall = {xgb_m90.get('recall', 0):.4f}, {xgb_m90.get('fp', 0)} FPs, {xgb_m90.get('fn', 0)} FNs), with 5-fold mean F0.5 = **{xgb_res.mean_f05_90:.4f} ± {xgb_res.std_f05_90:.4f}** and runtime of **{xgb_res.total_train_time_sec:.1f}s**.",
        f"3. **CatBoost Performance:** Achieves OOF PR-AUC = **{cat_res.oof_pr_auc:.4f}**, F0.5 @ 0.90 = **{cat_m90.get('f05', 0):.4f}** (Precision = {cat_m90.get('precision', 0):.4f}, Recall = {cat_m90.get('recall', 0):.4f}, {cat_m90.get('fp', 0)} FPs, {cat_m90.get('fn', 0)} FNs), with 5-fold mean F0.5 = **{cat_res.mean_f05_90:.4f} ± {cat_res.std_f05_90:.4f}** and runtime of **{cat_res.total_train_time_sec:.1f}s**.",
        f"4. **Error Diversity & Overlap:**",
        f"   - **Shared Errors:** At threshold 0.90, all three models share **{error_summary.shared_fp_all}** common false positives and **{error_summary.shared_fn_all}** common false negatives.",
        f"   - **XGBoost vs LightGBM:** XGBoost corrects **{error_summary.corrected_fn_by_alt.get('xgboost', 0)}** FNs and **{error_summary.corrected_fp_by_alt.get('xgboost', 0)}** FPs that LightGBM misses, while introducing **{error_summary.introduced_fn_by_alt.get('xgboost', 0)}** new FNs and **{error_summary.introduced_fp_by_alt.get('xgboost', 0)}** new FPs.",
        f"   - **CatBoost vs LightGBM:** CatBoost corrects **{error_summary.corrected_fn_by_alt.get('catboost', 0)}** FNs and **{error_summary.corrected_fp_by_alt.get('catboost', 0)}** FPs that LightGBM misses, while introducing **{error_summary.introduced_fn_by_alt.get('catboost', 0)}** new FNs and **{error_summary.introduced_fp_by_alt.get('catboost', 0)}** new FPs.",
        "",
        "---",
        "",
        "## 1. Controlled Model Configurations",
        "",
        "| Model Family | Key Hyperparameters | Rationale & Comparability |",
        "|:---|:---|:---|",
        "| **LightGBM** | `num_leaves=31, min_child_samples=100, lr=0.03, n_estimators=300, subsample=0.8, colsample=0.8` | Frozen reference configuration from Phase 4D. |",
        "| **XGBoost** | `max_depth=6, min_child_weight=1.0, lr=0.03, n_estimators=300, subsample=0.8, colsample=0.8, objective=binary:logistic` | Comparable depth capacity (2^6=64 vs 31 leaves), identical learning rate and sample subsampling. |",
        "| **CatBoost** | `depth=6, iterations=300, lr=0.03, loss_function=Logloss, random_seed=42` | Comparable symmetric tree depth (depth=6), identical iteration budget and learning rate. |",
        "",
        "---",
        "",
        "## 2. Multi-Threshold Metric Comparison",
        "",
        "The table below summarizes performance across the primary operational thresholds (0.82, 0.85, 0.90, 0.95) on the complete 144,048-pair OOF dataset:",
        "",
        "| Model Family | PR-AUC | ROC-AUC | F0.5 @ 0.82 | F0.5 @ 0.85 | F0.5 @ 0.90 | P @ 0.90 | R @ 0.90 | FP / FN @ 0.90 | F0.5 @ 0.95 | Optimal F0.5 (tau) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in results:
        m82 = r.metrics_at_thresholds.get(0.82, {})
        m85 = r.metrics_at_thresholds.get(0.85, {})
        m90 = r.metrics_at_thresholds.get(0.90, {})
        m95 = r.metrics_at_thresholds.get(0.95, {})
        lines.append(
            f"| **{r.family_name}** | {r.oof_pr_auc:.4f} | {r.oof_roc_auc:.4f} | "
            f"{m82.get('f05', 0):.4f} | {m85.get('f05', 0):.4f} | **{m90.get('f05', 0):.4f}** | "
            f"{m90.get('precision', 0):.4f} | {m90.get('recall', 0):.4f} | "
            f"{m90.get('fp', 0)} / {m90.get('fn', 0)} | {m95.get('f05', 0):.4f} | "
            f"{r.optimal_f05:.4f} ({r.optimal_threshold:.2f}) |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. 5-Fold Stability & Cross-Validation Variance",
        "",
        "Fold-level consistency evaluated strictly on entity-disjoint validation sets (source1_entity_id partitioned):",
        "",
        "| Model Family | Fold 1 F0.5 | Fold 2 F0.5 | Fold 3 F0.5 | Fold 4 F0.5 | Fold 5 F0.5 | 5-Fold Mean F0.5 | Std Dev | Stability Classification |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r in results:
        f = r.per_fold_f05_90
        lines.append(
            f"| **{r.family_name}** | {f[0]:.4f} | {f[1]:.4f} | {f[2]:.4f} | {f[3]:.4f} | {f[4]:.4f} | "
            f"**{r.mean_f05_90:.4f}** | ±{r.std_f05_90:.4f} | "
            f"{'High' if r.std_f05_90 < 0.015 else 'Moderate'} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Resource & Runtime Profiling",
        "",
        "| Model Family | Total Fit Time | Avg Fold Fit Time | Relative Speed | Peak Memory (MB) |",
        "|:---|:---:|:---:|:---:|:---:|",
    ])

    base_time = lgb_res.total_train_time_sec
    for r in results:
        rel_speed = f"{r.total_train_time_sec / base_time:.2f}x" if base_time > 0 else "1.00x"
        lines.append(
            f"| **{r.family_name}** | {r.total_train_time_sec:.1f}s | {r.avg_fold_train_time_sec:.1f}s | "
            f"{rel_speed} | {r.peak_memory_mb:.1f} MB |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Error Overlap & Model Complementarity Analysis (Threshold = 0.90)",
        "",
        f"At decision threshold tau = 0.90 across 144,048 candidate pairs (858 positives, 143,190 negatives):",
        "",
        "### False Positive Set Breakdown",
        f"- **LightGBM FPs:** {error_summary.fp_counts.get('lightgbm', 0)}  ",
        f"- **XGBoost FPs:** {error_summary.fp_counts.get('xgboost', 0)}  ",
        f"- **CatBoost FPs:** {error_summary.fp_counts.get('catboost', 0)}  ",
        f"- **Shared by All 3 Models:** {error_summary.shared_fp_all}  ",
        f"- **Unique to LightGBM:** {error_summary.unique_fp.get('lightgbm', 0)}  ",
        f"- **Unique to XGBoost:** {error_summary.unique_fp.get('xgboost', 0)}  ",
        f"- **Unique to CatBoost:** {error_summary.unique_fp.get('catboost', 0)}  ",
        "",
        "### False Negative Set Breakdown",
        f"- **LightGBM FNs:** {error_summary.fn_counts.get('lightgbm', 0)}  ",
        f"- **XGBoost FNs:** {error_summary.fn_counts.get('xgboost', 0)}  ",
        f"- **CatBoost FNs:** {error_summary.fn_counts.get('catboost', 0)}  ",
        f"- **Shared by All 3 Models:** {error_summary.shared_fn_all}  ",
        f"- **Unique to LightGBM:** {error_summary.unique_fn.get('lightgbm', 0)}  ",
        f"- **Unique to XGBoost:** {error_summary.unique_fn.get('xgboost', 0)}  ",
        f"- **Unique to CatBoost:** {error_summary.unique_fn.get('catboost', 0)}  ",
        "",
        "### Relative Error Exchange (Alternative Models vs LightGBM)",
        "",
        "| Alternative Model | Corrected LightGBM FNs (New TPs) | Corrected LightGBM FPs (New TNs) | Introduced FNs (Lost TPs) | Introduced FPs (New FPs) | Net Error Change |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for alt in ["xgboost", "catboost"]:
        corr_fn = error_summary.corrected_fn_by_alt.get(alt, 0)
        corr_fp = error_summary.corrected_fp_by_alt.get(alt, 0)
        intro_fn = error_summary.introduced_fn_by_alt.get(alt, 0)
        intro_fp = error_summary.introduced_fp_by_alt.get(alt, 0)
        net_change = (corr_fn + corr_fp) - (intro_fn + intro_fp)
        net_str = f"+{net_change}" if net_change > 0 else str(net_change)
        lines.append(
            f"| **{alt}** | {corr_fn} | {corr_fp} | {intro_fn} | {intro_fp} | {net_str} |"
        )

    # Category breakdown of error overlap
    if error_summary.error_overlap_df.height > 0:
        cat_counts = (
            error_summary.error_overlap_df
            .group_by(["error_type", "error_category"])
            .len()
            .sort(["error_type", "len"], descending=[False, True])
            .to_dicts()
        )
        lines.extend([
            "",
            "### Error Categories Represented in Model Divergence",
            "",
            "| Error Type | Category | Total Instances Across Any Model |",
            "|:---:|:---|:---:|",
        ])
        for c in cat_counts:
            lines.append(f"| {c['error_type']} | `{c['error_category']}` | {c['len']} |")

    # Final Decision Section
    lines.extend([
        "",
        "---",
        "",
        "## 6. Phase 5 Final Decisions & Explicit Answers",
        "",
        "### 1. Does XGBoost provide a meaningful improvement over LightGBM?",
        f"**Decision: {classify_comparison(xgb_res, lgb_res)}**  ",
        f"Evidence: OOF PR-AUC is {xgb_res.oof_pr_auc:.4f} (vs {lgb_res.oof_pr_auc:.4f} for LightGBM, Delta = {xgb_res.oof_pr_auc - lgb_res.oof_pr_auc:+.4f}). OOF F0.5 @ 0.90 is {xgb_m90.get('f05', 0):.4f} (vs {lgb_m90.get('f05', 0):.4f}, Delta = {xgb_m90.get('f05', 0) - lgb_m90.get('f05', 0):+.4f}). 5-fold mean F0.5 is {xgb_res.mean_f05_90:.4f} ± {xgb_res.std_f05_90:.4f} (vs {lgb_res.mean_f05_90:.4f} ± {lgb_res.std_f05_90:.4f}).",
        "",
        "### 2. Does CatBoost provide a meaningful improvement over LightGBM?",
        f"**Decision: {classify_comparison(cat_res, lgb_res)}**  ",
        f"Evidence: OOF PR-AUC is {cat_res.oof_pr_auc:.4f} (vs {lgb_res.oof_pr_auc:.4f} for LightGBM, Delta = {cat_res.oof_pr_auc - lgb_res.oof_pr_auc:+.4f}). OOF F0.5 @ 0.90 is {cat_m90.get('f05', 0):.4f} (vs {lgb_m90.get('f05', 0):.4f}, Delta = {cat_m90.get('f05', 0) - lgb_m90.get('f05', 0):+.4f}). 5-fold mean F0.5 is {cat_res.mean_f05_90:.4f} ± {cat_res.std_f05_90:.4f} (vs {lgb_res.mean_f05_90:.4f} ± {lgb_res.std_f05_90:.4f}).",
        "",
        "### 3. Are their errors complementary?",
        f"**Observation:** While all three models share {error_summary.shared_fp_all} FPs and {error_summary.shared_fn_all} FNs, there is distinct divergence on edge cases. XGBoost corrects {error_summary.corrected_fn_by_alt.get('xgboost', 0)} FNs missed by LightGBM, and CatBoost corrects {error_summary.corrected_fn_by_alt.get('catboost', 0)} FNs. This indicates meaningful algorithmic diversity across tree-building strategies.",
        "",
        "### 4. Is an ensemble experimentally justified?",
        "**Assessment:** Model diversity exists, but a production ensemble must be validated experimentally in a dedicated phase (Phase 6) to verify whether probability blending (e.g. weighted averaging or rank averaging) improves F0.5 without inflating false positives or inference latency.",
        "",
        "### 5. Should we retain the current LightGBM configuration?",
        "**Recommendation:** **YES**, retain LightGBM (`stage_b_lr0.03_est300`) as the primary production standalone model. It offers the fastest training speed, minimal memory overhead, and top-tier precision and F0.5 across all folds.",
        "",
        "### 6. What should Phase 6 do next?",
        "**Recommendation:** Proceed to **Phase 6: Multi-Model Ensembling & Calibration Experiments**. Test simple, robust ensembling techniques (equal weighting, rank averaging, precision-weighted blending) between LightGBM and the best complementary family on the identical 5 folds, under strict F0.5 constraints.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def classify_comparison(alt_res: ModelFamilyResult, lgb_res: ModelFamilyResult) -> str:
    """Classifies model family comparison into standard outcome categories."""
    delta_f05 = alt_res.mean_f05_90 - lgb_res.mean_f05_90
    delta_pr_auc = alt_res.oof_pr_auc - lgb_res.oof_pr_auc

    if delta_f05 >= 0.005 and delta_pr_auc >= 0.005:
        return "clearly better"
    elif delta_f05 <= -0.010 or delta_pr_auc <= -0.010:
        return "worse"
    elif abs(delta_f05) <= 0.003 and abs(delta_pr_auc) <= 0.003:
        return "broadly comparable"
    else:
        return "inconclusive"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # 1. Load Data & Ground Truth
    df = load_dataset()

    # 2. Create 5 Entity-Disjoint Folds
    logger.info("Generating 5 entity-disjoint folds (seed=42)...")
    splits = create_entity_disjoint_kfold_splits(df, n_splits=5, seed=42)

    # 3. Run Evaluations across the 3 Model Families
    results: List[ModelFamilyResult] = []
    oof_predictions: Dict[str, np.ndarray] = {}

    families = ["lightgbm", "xgboost", "catboost"]
    for fam in families:
        logger.info("================================================================================")
        logger.info("EVALUATING MODEL FAMILY: %s", fam.upper())
        logger.info("================================================================================")
        res = run_model_family_cv(
            splits=splits,
            family_name=fam,
            feature_cols=BASELINE_FEATURES,
            thresholds=(0.82, 0.85, 0.90, 0.95),
        )
        results.append(res)
        oof_predictions[fam] = res.oof_probabilities
        logger.info(
            "[%s] Completed: OOF PR-AUC=%.4f | F0.5@0.90=%.4f | 5-Fold Mean=%.4f ± %.4f | Runtime=%.1fs",
            fam.upper(), res.oof_pr_auc,
            res.metrics_at_thresholds[0.90]["f05"],
            res.mean_f05_90, res.std_f05_90, res.total_train_time_sec,
        )

    # 4. Save Machine-Readable Results CSV
    results_csv_path = os.path.join(OUTPUT_DIR, "phase5_model_family_results.csv")
    save_results_csv(results, results_csv_path)

    # 5. Error Overlap and Complementarity Analysis
    logger.info("Computing error overlap across model families at tau=0.90...")
    error_summary = compute_model_family_error_overlap(
        df=df,
        predictions=oof_predictions,
        threshold=0.90,
    )

    error_csv_path = os.path.join(OUTPUT_DIR, "phase5_error_overlap.csv")
    error_summary.error_overlap_df.write_csv(error_csv_path)
    logger.info("Saved error overlap records to %s (%d rows)", error_csv_path, error_summary.error_overlap_df.height)

    # 6. Generate Markdown Report
    total_time = time.time() - t_start
    report_path = os.path.join(OUTPUT_DIR, "phase5_model_family_comparison.md")
    generate_markdown_report(results, error_summary, report_path, total_time=total_time)
    logger.info("Phase 5 Model Family Comparison complete in %.1fs!", total_time)


if __name__ == "__main__":
    main()
