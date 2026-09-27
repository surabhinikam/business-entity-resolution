"""
Phase 7C: Train and Save Frozen Final LightGBM Model.

Trains the frozen LightGBM model configuration from Phase 4D on the
official corrected development benchmark (data/processed/train/dev_features/phase4_dev_corrected.parquet),
labeled with ground truth, using the exact 29 baseline features.

Saves the trained model booster to models/final_lightgbm_model.txt.
"""

from __future__ import annotations

import logging
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import polars as pl

from src.features.feature_schema import LABEL_COLUMN
from src.features.training_dataset import (
    label_candidate_pairs,
    normalize_ground_truth,
)
from src.models.baseline_model import build_final_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEV_FEATURES_PATH = "data/processed/train/dev_features/phase4_dev_corrected.parquet"
GROUND_TRUTH_PATH = "data/raw/train/train_ground_truth.tsv"
MODEL_OUTPUT_PATH = "models/final_lightgbm_model.txt"


def main():
    logger.info("=" * 70)
    logger.info("TRAINING FROZEN FINAL LIGHTGBM MODEL FOR TEST INFERENCE")
    logger.info("=" * 70)

    # 1. Load corrected development feature matrix
    logger.info("Loading feature dataset: %s", DEV_FEATURES_PATH)
    t0 = time.time()
    df = pl.read_parquet(DEV_FEATURES_PATH)
    logger.info("Loaded %d rows × %d columns in %.2fs", df.height, df.width, time.time() - t0)

    # 2. Load ground truth and label candidate pairs
    logger.info("Loading ground truth: %s", GROUND_TRUTH_PATH)
    gt_df = pl.read_csv(GROUND_TRUTH_PATH, separator="\t")
    gt_norm = normalize_ground_truth(gt_df)
    labeled_df = label_candidate_pairs(df, gt_norm)

    n_pos = int((labeled_df[LABEL_COLUMN] == 1).sum())
    n_neg = int((labeled_df[LABEL_COLUMN] == 0).sum())
    logger.info("Labeled pairs: %d positives (%.2f%%), %d negatives (%.2f%%)", n_pos, 100.0 * n_pos / labeled_df.height, n_neg, 100.0 * n_neg / labeled_df.height)

    # 3. Instantiate frozen model
    logger.info("Instantiating frozen LightGBM model with Phase 4D hyperparameters...")
    model = build_final_model()

    # 4. Train model
    logger.info("Fitting model on %d labeled candidate pairs...", labeled_df.height)
    t_fit = time.time()
    model.fit(labeled_df)
    fit_duration = time.time() - t_fit
    logger.info("Model training completed in %.2fs", fit_duration)

    # 5. Save model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model.save_model(MODEL_OUTPUT_PATH)
    logger.info("Saved final model booster to %s", MODEL_OUTPUT_PATH)

    # 6. Verify reload and self-consistency
    logger.info("Verifying model reloading...")
    reloaded_model = build_final_model()
    reloaded_model.load_model(MODEL_OUTPUT_PATH)
    probs_orig = model.predict_proba(df.head(100))
    probs_reload = reloaded_model.predict_proba(df.head(100))
    assert (probs_orig == probs_reload).all(), "Reloaded model output mismatch!"
    logger.info("Model verification PASSED: Outputs are 100%% identical.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
