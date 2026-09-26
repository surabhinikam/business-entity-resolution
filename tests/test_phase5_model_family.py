"""
Unit tests for Phase 5: Model Family Comparison Framework.

Tests:
1. build_model_for_family: Validates model instantiation for LightGBM, XGBoost, and CatBoost.
2. compute_model_family_error_overlap: Validates set logic for shared, unique, corrected, and introduced errors.
3. run_model_family_cv: End-to-end smoke test on small synthetic fold dataset.
"""

from __future__ import annotations

import catboost as cb
import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from src.features.training_dataset import compute_split_stats
from src.models.cross_validation import EntityDisjointKFoldSplit
from src.models.model_family_comparison import (
    build_model_for_family,
    compute_model_family_error_overlap,
    run_model_family_cv,
)


def test_build_model_for_family():
    # LightGBM
    lgbm = build_model_for_family("lightgbm")
    assert isinstance(lgbm, lgb.LGBMClassifier)
    assert lgbm.num_leaves == 31
    assert lgbm.min_child_samples == 100
    assert lgbm.n_estimators == 300

    # XGBoost
    xgb_model = build_model_for_family("xgboost")
    assert isinstance(xgb_model, xgb.XGBClassifier)
    assert xgb_model.n_estimators == 300
    assert xgb_model.max_depth == 6

    # CatBoost
    cb_model = build_model_for_family("catboost")
    assert isinstance(cb_model, cb.CatBoostClassifier)
    assert cb_model.get_params()["depth"] == 6
    assert cb_model.get_params()["iterations"] == 300

    # Custom params override
    custom_xgb = build_model_for_family("xgboost", custom_params={"n_estimators": 10})
    assert custom_xgb.n_estimators == 10

    # Invalid family
    with pytest.raises(ValueError, match="Unsupported model family"):
        build_model_for_family("random_forest")


def test_compute_model_family_error_overlap():
    # Mock candidate dataset with 6 rows:
    # Row 0: True pos (1) -> LGB: 0.95 (TP), XGB: 0.95 (TP), CAT: 0.95 (TP) [All correct]
    # Row 1: True pos (1) -> LGB: 0.40 (FN), XGB: 0.92 (TP), CAT: 0.40 (FN) [XGB corrects LGB FN]
    # Row 2: True pos (1) -> LGB: 0.92 (TP), XGB: 0.30 (FN), CAT: 0.92 (TP) [XGB introduces FN]
    # Row 3: True neg (0) -> LGB: 0.92 (FP), XGB: 0.92 (FP), CAT: 0.92 (FP) [Shared FP]
    # Row 4: True neg (0) -> LGB: 0.95 (FP), XGB: 0.10 (TN), CAT: 0.95 (FP) [XGB corrects LGB FP]
    # Row 5: True neg (0) -> LGB: 0.10 (TN), XGB: 0.95 (FP), CAT: 0.10 (TN) [XGB introduces FP]

    df = pl.DataFrame({
        "source1_entity_id": [f"E{i}" for i in range(6)],
        "candidate_entity_id": [f"C{i}" for i in range(6)],
        "candidate_source": ["source2"] * 6,
        "label": [1, 1, 1, 0, 0, 0],
        "name_char_3gram_jaccard": [0.8] * 6,
        "address_char_3gram_similarity": [0.8] * 6,
        "address_missing_candidate": [0] * 6,
        "address_missing_s1": [0] * 6,
        "shared_address_number_count": [0] * 6,
        "name_exact_translit": [0] * 6,
        "name_char_len_ratio": [1.0] * 6,
        "name_acronym_match": [0] * 6,
        "name_first_token_exact": [0] * 6,
    })

    preds = {
        "lightgbm": np.array([0.95, 0.40, 0.92, 0.92, 0.95, 0.10]),
        "xgboost":  np.array([0.95, 0.92, 0.30, 0.92, 0.10, 0.95]),
        "catboost": np.array([0.95, 0.40, 0.92, 0.92, 0.95, 0.10]),
    }

    summary = compute_model_family_error_overlap(df, predictions=preds, threshold=0.90)

    assert summary.threshold == 0.90
    assert summary.total_positives == 3
    assert summary.total_negatives == 3

    # FP counts at 0.90:
    # LGB: Row 3, Row 4 -> 2 FPs
    # XGB: Row 3, Row 5 -> 2 FPs
    # CAT: Row 3, Row 4 -> 2 FPs
    assert summary.fp_counts["lightgbm"] == 2
    assert summary.fp_counts["xgboost"] == 2
    assert summary.fp_counts["catboost"] == 2

    # FN counts at 0.90:
    # LGB: Row 1 -> 1 FN
    # XGB: Row 2 -> 1 FN
    # CAT: Row 1 -> 1 FN
    assert summary.fn_counts["lightgbm"] == 1
    assert summary.fn_counts["xgboost"] == 1
    assert summary.fn_counts["catboost"] == 1

    # Shared FP by all: Row 3 (1)
    assert summary.shared_fp_all == 1

    # Unique FP: Row 5 for XGB (1), 0 for LGB, 0 for CAT
    assert summary.unique_fp["xgboost"] == 1
    assert summary.unique_fp["lightgbm"] == 0

    # Corrections by XGBoost vs LightGBM:
    # Row 1: LGB was FN, XGB is TP -> corrected_fn = 1
    # Row 4: LGB was FP, XGB is TN -> corrected_fp = 1
    assert summary.corrected_fn_by_alt["xgboost"] == 1
    assert summary.corrected_fp_by_alt["xgboost"] == 1

    # Errors introduced by XGBoost vs LightGBM:
    # Row 2: LGB was TP, XGB is FN -> introduced_fn = 1
    # Row 5: LGB was TN, XGB is FP -> introduced_fp = 1
    assert summary.introduced_fn_by_alt["xgboost"] == 1
    assert summary.introduced_fp_by_alt["xgboost"] == 1

    # Error overlap DataFrame structure
    assert isinstance(summary.error_overlap_df, pl.DataFrame)
    assert summary.error_overlap_df.height == 5  # Rows 1, 2, 3, 4, 5 are errors for at least one model
    assert "error_type" in summary.error_overlap_df.columns
    assert "error_category" in summary.error_overlap_df.columns


def test_run_model_family_cv_smoke():
    # Create small synthetic dataset with 2 folds
    np.random.seed(42)
    n = 60
    f_cols = ["f1", "f2"]

    data = pl.DataFrame({
        "_row_idx": list(range(n)),
        "source1_entity_id": [f"S1_{i // 3}" for i in range(n)],
        "candidate_entity_id": [f"C_{i}" for i in range(n)],
        "candidate_source": ["source2"] * (n // 2) + ["source3"] * (n // 2),
        "f1": np.random.randn(n).astype(np.float32),
        "f2": np.random.randn(n).astype(np.float32),
        "label": [1 if i % 6 == 0 else 0 for i in range(n)],
    })

    # Split into 2 entity-disjoint folds
    train_fold0 = data.filter(pl.col("source1_entity_id").is_in([f"S1_{i}" for i in range(10)]))
    val_fold0 = data.filter(pl.col("source1_entity_id").is_in([f"S1_{i}" for i in range(10, 20)]))

    train_fold1 = val_fold0
    val_fold1 = train_fold0

    splits = [
        EntityDisjointKFoldSplit(
            fold_idx=0,
            train=train_fold0,
            validation=val_fold0,
            train_stats=compute_split_stats(train_fold0),
            validation_stats=compute_split_stats(val_fold0),
            s1_overlap_count=0,
            candidate_pair_overlap_count=0,
        ),
        EntityDisjointKFoldSplit(
            fold_idx=1,
            train=train_fold1,
            validation=val_fold1,
            train_stats=compute_split_stats(train_fold1),
            validation_stats=compute_split_stats(val_fold1),
            s1_overlap_count=0,
            candidate_pair_overlap_count=0,
        ),
    ]

    # Smoke test on LightGBM with small tree
    res = run_model_family_cv(
        splits=splits,
        family_name="lightgbm",
        feature_cols=f_cols,
        custom_params={"n_estimators": 5, "min_child_samples": 2, "num_leaves": 4},
        thresholds=[0.50, 0.90],
    )

    assert res.family_name == "lightgbm"
    assert len(res.fold_metrics) == 2
    assert len(res.oof_probabilities) == n
    assert 0.0 <= res.oof_pr_auc <= 1.0
    assert 0.50 in res.metrics_at_thresholds
    assert 0.90 in res.metrics_at_thresholds
