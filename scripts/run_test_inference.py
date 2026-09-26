#!/usr/bin/env python3
"""
Test inference orchestrator for Business Entity Resolution.

Orchestrates the full test pipeline:
    processed test sources
        -> candidate generation  (frozen A/C/D/E/F keys, MAX_BLOCK_SIZE=5000)
        -> feature generation    (FeaturePipeline, 29 features)
        -> model inference       (BaselineMatchingModel.load_model)
        -> post-processing       (MatchPostProcessor)
        -> submission formatting (format_submission_linkages)
        -> submission validation (validate_submission)

Requires:
    - Processed test parquet directories (data/processed/test/source{1,2,3})
    - A trained LightGBM model artifact (produced by run_baseline_model.py)

IMPORTANT — Test data must be preprocessed first via scripts/process_dataset.py:
    python scripts/process_dataset.py --input data/raw/test/test_source1.tsv \\
        --source source1 --output-dir data/processed/test/source1
    (repeat for source2 and source3)

Usage:
    python scripts/run_test_inference.py \\
        --input-root data/processed/test \\
        --model models/lgbm_baseline.txt \\
        --submission submission.csv \\
        --threshold 0.5 \\
        [--limit-source1 N]  # bounded smoke-test only; do NOT use for final submission
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.candidate_generation.generate_candidates import generate_candidate_pairs
from src.features.feature_pipeline import FeaturePipeline
# NOTE: validate_submission is imported inside main() to avoid a circular-import
# cycle that fires when validators.py is the first module to load preprocessing.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Frozen blocking parameters — must not be changed
ACTIVE_KEYS = ["A", "C", "D", "E", "F"]
MAX_BLOCK_SIZE = 5000


def _infer_candidate_source(entity_id: str) -> str:
    """Derive candidate_source from entity_id namespace prefix."""
    uid = entity_id.upper()
    if uid.startswith("S2-"):
        return "source2"
    elif uid.startswith("S3-"):
        return "source3"
    return "unknown"


def _load_parquet_dir(directory: str) -> pl.DataFrame:
    """Load all Parquet part files in a directory."""
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in: {directory}")
    return pl.concat([pl.read_parquet(f) for f in files])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test inference orchestrator for Business Entity Resolution."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/processed/test",
        help="Root directory of processed test data (must contain source1/, source2/, source3/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained LightGBM model artifact (e.g. models/lgbm_baseline.txt).",
    )
    parser.add_argument(
        "--submission",
        type=str,
        default="submission.csv",
        help="Output path for submission CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for accepting a match (default: 0.5).",
    )
    parser.add_argument(
        "--limit-source1",
        type=int,
        default=None,
        help=(
            "Deterministic cap on the number of Source1 entities processed. "
            "For bounded smoke-testing ONLY — do NOT use for final submission."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 0. Pre-flight checks
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.model):
        logger.error(
            f"Trained model artifact not found: '{args.model}'\n"
            "Train a model first using scripts/run_baseline_model.py and save it via "
            "model.save_model('<path>')."
        )
        sys.exit(1)

    # Deferred imports — lightgbm and sklearn are only available when the full
    # model-training environment is active. Importing here (after the existence
    # check) gives a clear error message at the correct point.
    try:
        from src.models.baseline_model import BaselineMatchingModel
        from src.models.post_processing import MatchPostProcessor
    except ImportError as e:
        logger.error(
            f"Cannot import model components: {e}\n"
            "Ensure lightgbm and scikit-learn are installed in your environment."
        )
        sys.exit(1)

    # validate_submission is deferred here (not at module level) to avoid a
    # circular import: validators→preprocessing.schema→preprocessing.__init__→
    # ingestion→validators (partially initialized). By this point FeaturePipeline
    # has already loaded preprocessing, so the cycle does not fire.
    from src.validation.validators import validate_submission  # noqa: E402

    s1_dir = os.path.join(args.input_root, "source1")
    s2_dir = os.path.join(args.input_root, "source2")
    s3_dir = os.path.join(args.input_root, "source3")

    for d in [s1_dir, s2_dir, s3_dir]:
        if not os.path.isdir(d):
            logger.error(
                f"Processed test directory not found: '{d}'\n"
                "Run scripts/process_dataset.py on each raw test TSV first."
            )
            sys.exit(1)

    if args.limit_source1:
        logger.warning(
            f"--limit-source1={args.limit_source1}: BOUNDED DEVELOPMENT RUN — "
            "output is NOT a valid competition submission."
        )

    # ------------------------------------------------------------------ #
    # 1. Load model
    # ------------------------------------------------------------------ #
    logger.info(f"Step 1: Loading model from '{args.model}'...")
    model = BaselineMatchingModel()
    model.load_model(args.model)
    logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------ #
    # 2. Candidate generation (frozen A/C/D/E/F, MAX_BLOCK_SIZE=5000)
    # ------------------------------------------------------------------ #
    logger.info("Step 2: Generating candidate pairs...")
    _, pairs, provenance = generate_candidate_pairs(
        s1_dir=s1_dir,
        s2_dir=s2_dir,
        s3_dir=s3_dir,
        active_keys=ACTIVE_KEYS,
        max_block_size=MAX_BLOCK_SIZE,
        cap_blocks=True,
        limit_source1=args.limit_source1,
    )
    logger.info(f"  {len(pairs):,} candidate pairs generated.")

    # ------------------------------------------------------------------ #
    # 3. Load processed records for feature generation
    # ------------------------------------------------------------------ #
    logger.info("Step 3: Loading processed records for feature extraction...")
    s1_df = _load_parquet_dir(s1_dir)
    if args.limit_source1:
        unique_s1 = sorted(s1_df["entity_id"].unique().to_list())[: args.limit_source1]
        s1_df = s1_df.filter(pl.col("entity_id").is_in(unique_s1))
    s2_df = _load_parquet_dir(s2_dir)
    s3_df = _load_parquet_dir(s3_dir)
    all_cand_df = pl.concat([s2_df, s3_df])
    logger.info(
        f"  S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,} records loaded."
    )

    all_s1_entities = s1_df["entity_id"].unique().to_list()

    if not pairs:
        logger.warning(
            "No candidate pairs generated — writing zero-match submission for all S1 entities."
        )
        post_processor = MatchPostProcessor(threshold=args.threshold)
        empty_accepted = pl.DataFrame(
            {"source1_entity_id": [], "candidate_entity_id": [], "probability": []},
            schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8, "probability": pl.Float64},
        )
        submission_df = post_processor.format_submission_linkages(empty_accepted, all_s1_entities)
        os.makedirs(os.path.dirname(os.path.abspath(args.submission)) or ".", exist_ok=True)
        submission_df.write_csv(args.submission)
        logger.info(f"Empty submission written to '{args.submission}'.")
        sys.exit(0)

    # Build candidate pairs DataFrame with candidate_source
    candidate_pairs_df = pl.DataFrame(
        {
            "source1_entity_id": [p[0] for p in pairs],
            "candidate_entity_id": [p[1] for p in pairs],
        }
    ).with_columns(
        pl.col("candidate_entity_id")
        .map_elements(_infer_candidate_source, return_dtype=pl.Utf8)
        .alias("candidate_source")
    )

    # ------------------------------------------------------------------ #
    # 4. Feature generation
    # ------------------------------------------------------------------ #
    logger.info("Step 4: Extracting features via FeaturePipeline...")
    pipeline = FeaturePipeline()
    features_df = pipeline.generate_features_from_records(
        candidate_pairs_df=candidate_pairs_df,
        s1_records=s1_df,
        cand_records=all_cand_df,
        pair_provenance=provenance,
    )
    logger.info(
        f"  Feature extraction complete: {features_df.height:,} rows, {features_df.width} columns."
    )

    # ------------------------------------------------------------------ #
    # 5. Model inference
    # ------------------------------------------------------------------ #
    logger.info(f"Step 5: Running model inference (threshold={args.threshold})...")
    probs = model.predict_proba(features_df)
    features_with_probs = features_df.with_columns(
        pl.Series("probability", probs, dtype=pl.Float64)
    )

    # ------------------------------------------------------------------ #
    # 6. Post-processing
    # ------------------------------------------------------------------ #
    logger.info("Step 6: Applying match post-processing...")
    post_processor = MatchPostProcessor(threshold=args.threshold)
    accepted_pairs = post_processor.filter_candidate_pairs(features_with_probs)
    logger.info(f"  Accepted pairs above threshold: {accepted_pairs.height:,}")

    submission_df = post_processor.format_submission_linkages(
        accepted_pairs, all_s1_entities
    )
    logger.info(f"  Submission rows: {submission_df.height:,}")

    # ------------------------------------------------------------------ #
    # 7. Write submission
    # ------------------------------------------------------------------ #
    submission_dir = os.path.dirname(os.path.abspath(args.submission))
    if submission_dir:
        os.makedirs(submission_dir, exist_ok=True)
    submission_df.write_csv(args.submission)
    logger.info(f"Submission written to '{args.submission}'.")

    # ------------------------------------------------------------------ #
    # 8. Validate submission
    # ------------------------------------------------------------------ #
    logger.info("Step 8: Validating submission...")
    all_cand_ids = set(all_cand_df["entity_id"].to_list())
    is_valid, errors = validate_submission(
        submission_df=submission_df,
        all_s1_entities=set(all_s1_entities),
        all_candidate_entity_ids=all_cand_ids,
    )
    if not is_valid:
        logger.error("Submission validation FAILED:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)

    logger.info("Submission validation PASSED — all checks satisfied.")
    logger.info("=" * 70)
    logger.info("TEST INFERENCE COMPLETE")
    logger.info(f"  Submission: {args.submission}")
    logger.info(f"  S1 entities: {len(all_s1_entities):,}")
    logger.info(f"  Accepted matches: {accepted_pairs.height:,}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
