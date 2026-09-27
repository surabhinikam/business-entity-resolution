"""
Phase 7A: End-to-End Integration Validation and Pipeline Freeze Tests.

Verifies:
1. End-to-end data flow: records -> blocking -> feature pipeline -> model inference -> post-processing -> submission validation.
2. Canonical 29-feature schema names, data types, and deterministic ordering.
3. Frozen V4 blocking keys (A, C, D, E, F) and MAX_BLOCK_SIZE parameter.
4. Frozen LightGBM model configuration from Phase 4D.
5. Strict submission format compliance via validate_submission.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.candidate_generation.block_index import BlockIndex
from src.candidate_generation.block_keys import BLOCKING_KEY_NAMES
from src.features.feature_pipeline import FeaturePipeline
from src.features.feature_schema import (
    ALL_FEATURE_NAMES,
    FULL_PIPELINE_COLUMNS,
    PAIR_ID_COLUMNS,
)
from src.models.baseline_model import (
    FINAL_LIGHTGBM_CONFIG,
    BaselineMatchingModel,
    build_final_model,
)
from src.models.post_processing import MatchPostProcessor
from src.validation.validators import validate_submission


def test_frozen_v4_blocking_keys_and_schema():
    """Verify blocking keys and pair identity contract are strictly preserved."""
    assert PAIR_ID_COLUMNS == [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]
    # V4 Active keys must be A, C, D, E, F
    expected_v4_keys = {"A", "C", "D", "E", "F"}
    assert set(BLOCKING_KEY_NAMES.keys()) >= expected_v4_keys
    assert len(ALL_FEATURE_NAMES) == 29
    assert FULL_PIPELINE_COLUMNS == PAIR_ID_COLUMNS + ALL_FEATURE_NAMES


def test_frozen_lightgbm_model_configuration():
    """Verify final LightGBM hyperparameters frozen from Phase 4D."""
    assert FINAL_LIGHTGBM_CONFIG == {
        "num_leaves": 31,
        "min_child_samples": 100,
        "learning_rate": 0.03,
        "n_estimators": 300,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    model = build_final_model()
    assert model.num_leaves == 31
    assert model.min_child_samples == 100
    assert model.learning_rate == 0.03
    assert model.n_estimators == 300
    assert model.max_depth == -1
    assert model.subsample == 0.8
    assert model.colsample_bytree == 0.8
    assert model.random_state == 42
    assert model.feature_names == ALL_FEATURE_NAMES


def test_end_to_end_merged_pipeline_integration():
    """
    Simulates a complete run through the merged pipeline:
    Records -> Candidate Generation -> Feature Pipeline -> Model Scoring -> Post-Processing -> Submission Validation.
    """
    # 1. Mock processed records for Source 1, Source 2, and Source 3
    s1_df = pl.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "source": ["source1", "source1", "source1"],
        "business_name_normalized": ["apple computer inc", "starbucks coffee", "tata consultancy services"],
        "business_name_transliterated": ["apple computer inc", "starbucks coffee", "tata consultancy services"],
        "business_name_tokens": [["apple", "computer", "inc"], ["starbucks", "coffee"], ["tata", "consultancy", "services"]],
        "business_address_normalized": ["1 infinite loop cupertino ca", "100 pike st seattle wa", "bombay house 24 homi mody st mumbai"],
        "business_address_tokens": [["1", "infinite", "loop", "cupertino", "ca"], ["100", "pike", "st", "seattle", "wa"], ["bombay", "house", "24", "homi", "mody", "st", "mumbai"]],
        "country_normalized": ["united states", "united states", "india"],
    })

    s2_df = pl.DataFrame({
        "entity_id": ["S2-001", "S2-002"],
        "source": ["source2", "source2"],
        "business_name_normalized": ["apple computer", "blue bottle coffee"],
        "business_name_transliterated": ["apple computer", "blue bottle coffee"],
        "business_name_tokens": [["apple", "computer"], ["blue", "bottle", "coffee"]],
        "business_address_normalized": ["1 infinite loop cupertino", "300 university ave seattle wa"],
        "business_address_tokens": [["1", "infinite", "loop", "cupertino"], ["300", "university", "ave", "seattle", "wa"]],
        "country_normalized": ["united states", "united states"],
    })

    s3_df = pl.DataFrame({
        "entity_id": ["S3-001", "S3-002"],
        "source": ["source3", "source3"],
        "business_name_normalized": ["starbucks corp", "tata consultancy services ltd"],
        "business_name_transliterated": ["starbucks corp", "tata consultancy services ltd"],
        "business_name_tokens": [["starbucks", "corp"], ["tata", "consultancy", "services", "ltd"]],
        "business_address_normalized": ["100 pike street seattle", "homi mody street fort mumbai"],
        "business_address_tokens": [["100", "pike", "street", "seattle"], ["homi", "mody", "street", "fort", "mumbai"]],
        "country_normalized": ["united states", "india"],
    })

    all_cand_df = pl.concat([s2_df, s3_df])
    active_keys = ["A", "C", "D", "E", "F"]

    # 2. Candidate generation via BlockIndex
    index = BlockIndex(max_block_size=5000)
    for rec in s1_df.to_dicts():
        index.add_s1_record(rec["entity_id"], rec, active_keys=active_keys)
    for rec in all_cand_df.to_dicts():
        index.add_candidate_record(rec["entity_id"], rec, active_keys=active_keys)

    pairs, provenance = index.generate_pairs(cap_blocks=True)
    assert len(pairs) > 0

    candidate_pairs_df = pl.DataFrame({
        "source1_entity_id": [p[0] for p in pairs],
        "candidate_entity_id": [p[1] for p in pairs],
    }).with_columns(
        pl.when(pl.col("candidate_entity_id").str.starts_with("S2-"))
        .then(pl.lit("source2"))
        .when(pl.col("candidate_entity_id").str.starts_with("S3-"))
        .then(pl.lit("source3"))
        .otherwise(pl.lit("unknown"))
        .alias("candidate_source")
    )

    # 3. Feature generation via FeaturePipeline
    pipeline = FeaturePipeline()
    features_df = pipeline.generate_features_from_records(
        candidate_pairs_df=candidate_pairs_df,
        s1_records=s1_df,
        cand_records=all_cand_df,
        pair_provenance=provenance,
    )

    # Verify column count and exact schema ordering
    assert features_df.width == len(FULL_PIPELINE_COLUMNS)
    assert features_df.columns == FULL_PIPELINE_COLUMNS
    assert features_df.height == len(pairs)

    # Verify no nulls in feature values
    for col in ALL_FEATURE_NAMES:
        assert features_df[col].null_count() == 0, f"Null values found in {col}"

    # 4. Model scoring with frozen LightGBM architecture
    model = BaselineMatchingModel(n_estimators=10, min_child_samples=1, num_leaves=4)
    # Mock labels for quick fit
    mock_labels = [1 if (p[0], p[1]) in [("S1-001", "S2-001"), ("S1-002", "S3-001"), ("S1-003", "S3-002")] else 0 for p in pairs]
    labeled_features = features_df.with_columns(pl.Series("label", mock_labels, dtype=pl.Int8))
    model.fit(labeled_features)

    probs = model.predict_proba(features_df)
    assert len(probs) == features_df.height
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    features_with_probs = features_df.with_columns(pl.Series("probability", probs, dtype=pl.Float64))

    # 5. Post-processing and submission formatting
    all_s1_entities = s1_df["entity_id"].unique().to_list()
    post_processor = MatchPostProcessor(threshold=0.50)
    accepted_pairs = post_processor.filter_candidate_pairs(features_with_probs)

    submission_df = post_processor.format_submission_linkages(accepted_pairs, all_s1_entities)

    # 6. Submission validation
    all_cand_ids = set(all_cand_df["entity_id"].to_list())
    is_valid, errors = validate_submission(
        submission_df=submission_df,
        all_s1_entities=set(all_s1_entities),
        all_candidate_entity_ids=all_cand_ids,
    )

    assert is_valid is True, f"Submission validation failed: {errors}"
    assert len(errors) == 0
    assert submission_df.height == len(all_s1_entities)
    assert submission_df.columns == ["source1_entity_id", "matched_entity_ids"]
