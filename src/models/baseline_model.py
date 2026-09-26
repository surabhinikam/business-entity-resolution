"""
Baseline GBDT / LightGBM Supervised Matching Model for Business Entity Resolution.

Part 8:
- Consumes precomputed 29 engineered features from FeaturePipeline / SupervisedDatasetBuilder.
- Excludes all 3 composite identity columns (source1_entity_id, candidate_entity_id, candidate_source).
- Trains on blocker-generated candidate pairs using an entity-level split.
- Outputs continuous match probabilities in [0.0, 1.0].
- Evaluates using F0.5 as primary metric, alongside PR-AUC, Precision, Recall, and F1.
- Analyzes probability distributions across thresholds without premature threshold locking.
- Computes feature importances (gain and split).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import lightgbm as lgb
import numpy as np
import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    LABEL_COLUMN,
    ALL_FEATURE_NAMES,
)
from src.models.metrics import (
    SupervisedEvaluationReport,
    ThresholdEvaluation,
    evaluate_matching_probabilities,
    evaluate_predictions_at_threshold,
)

logger = logging.getLogger(__name__)

# Excluded non-feature metadata columns
NON_FEATURE_COLUMNS: Set[str] = set(PAIR_ID_COLUMNS) | {LABEL_COLUMN}


class BaselineMatchingModel:
    """
    Supervised LightGBM classifier for pairwise business entity matching.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        importance_type: str = "gain",
        feature_names: Optional[List[str]] = None,
        extra_lgb_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LightGBM baseline model.

        Parameters:
            n_estimators: Number of boosting iterations.
            learning_rate: Boosting learning rate.
            num_leaves: Maximum tree leaves per base learner.
            max_depth: Maximum tree depth (-1 for unlimited).
            min_child_samples: Minimum data points per leaf.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns when constructing each tree.
            random_state: Fixed random seed for deterministic training.
            class_weight: Optional class weights ('balanced' or dict).
            importance_type: Metric for feature importance ('gain' or 'split').
            feature_names: Explicit list of feature columns (defaults to ALL_FEATURE_NAMES).
            extra_lgb_params: Additional LightGBM parameters.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.class_weight = class_weight
        self.importance_type = importance_type
        self.feature_names = feature_names or list(ALL_FEATURE_NAMES)
        self.extra_lgb_params = extra_lgb_params or {}

        # Validate feature list excludes identity and target columns
        self._validate_feature_names()

        self.classifier: Optional[lgb.LGBMClassifier] = None
        self.is_fitted: bool = False

    def _validate_feature_names(self) -> None:
        """Ensures NO identity or label column is in feature names."""
        for col in NON_FEATURE_COLUMNS:
            if col in self.feature_names:
                raise ValueError(
                    f"Non-feature column '{col}' cannot be included in model features. "
                    f"Forbidden columns: {NON_FEATURE_COLUMNS}"
                )

    def prepare_features(
        self,
        df: pl.DataFrame,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extracts feature matrix X and target y from Polars DataFrame.

        Guarantees:
        - Exactly matches self.feature_names in deterministic column order.
        - Strictly excludes all composite identity columns.
        - Converts nulls / NaNs safely to 0.0.
        """
        # Verify all feature columns exist in DataFrame
        missing_feats = [f for f in self.feature_names if f not in df.columns]
        if missing_feats:
            raise ValueError(f"Input DataFrame is missing required feature columns: {missing_feats}")

        # Select exactly the 29 features in order
        feature_df = df.select(self.feature_names)

        # Cast to float32 numpy array
        X = feature_df.to_numpy().astype(np.float32)
        # Replace any residual NaNs with 0.0
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        y: Optional[np.ndarray] = None
        if LABEL_COLUMN in df.columns:
            y = df[LABEL_COLUMN].to_numpy().astype(int)

        return X, y

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: Optional[pl.DataFrame] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> BaselineMatchingModel:
        """
        Trains the LightGBM matching model on the training DataFrame.

        Parameters:
            train_df: Labeled training DataFrame with features and 'label'.
            val_df: Optional labeled validation DataFrame for early stopping.
            early_stopping_rounds: Optional early stopping patience.

        Returns:
            Self (fitted model).
        """
        if LABEL_COLUMN not in train_df.columns:
            raise ValueError(f"Training DataFrame is missing target column '{LABEL_COLUMN}'")

        X_train, y_train = self.prepare_features(train_df)
        assert y_train is not None

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        logger.info(
            f"Training LightGBM baseline on {len(y_train):,} pairs "
            f"({n_pos:,} positives, {n_neg:,} negatives, {len(self.feature_names)} features)..."
        )

        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "importance_type": self.importance_type,
            "n_jobs": -1,
            "verbose": -1,
            **self.extra_lgb_params,
        }
        if self.class_weight is not None:
            params["class_weight"] = self.class_weight

        self.classifier = lgb.LGBMClassifier(**params)

        fit_kwargs: Dict[str, Any] = {}
        if val_df is not None and LABEL_COLUMN in val_df.columns and early_stopping_rounds is not None:
            X_val, y_val = self.prepare_features(val_df)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            ]

        self.classifier.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True

        logger.info("LightGBM baseline model training completed.")
        return self

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """
        Generates continuous match probabilities for candidate pairs.

        Returns:
            1D numpy array of probabilities P(match=1) in [0.0, 1.0].
        """
        if not self.is_fitted or self.classifier is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict_proba().")

        X, _ = self.prepare_features(df)
        probs = self.classifier.predict_proba(X)
        # Return probability of positive class (label=1)
        return probs[:, 1].astype(float)

    def predict(self, df: pl.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Generates binary match predictions using a specified threshold.

        Returns:
            1D numpy array of 0s and 1s.
        """
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)

    def evaluate(
        self,
        val_df: pl.DataFrame,
        thresholds: Optional[Sequence[float]] = None,
    ) -> SupervisedEvaluationReport:
        """
        Evaluates the model on validation data, reporting F0.5, PR-AUC, ROC-AUC,
        precision, recall, and a threshold sweep.
        """
        if LABEL_COLUMN not in val_df.columns:
            raise ValueError(f"Validation DataFrame is missing target column '{LABEL_COLUMN}'")

        probs = self.predict_proba(val_df)
        y_true = val_df[LABEL_COLUMN].to_numpy().astype(int)

        return evaluate_matching_probabilities(
            y_true=y_true,
            y_prob=probs,
            thresholds=thresholds,
        )

    def get_feature_importances(self, importance_type: Optional[str] = None) -> pl.DataFrame:
        """
        Returns feature importances as a Polars DataFrame sorted descending by importance.
        """
        if not self.is_fitted or self.classifier is None:
            raise RuntimeError("Model has not been fitted yet.")

        imp_type = importance_type or self.importance_type
        importances = self.classifier.booster_.feature_importance(importance_type=imp_type)
        total_imp = float(np.sum(importances))

        rel_pct = (importances / total_imp * 100.0) if total_imp > 0 else np.zeros_like(importances)

        df = pl.DataFrame({
            "feature_name": self.feature_names,
            "importance": [float(v) for v in importances],
            "relative_pct": [round(float(v), 2) for v in rel_pct],
        }).sort("importance", descending=True)

        return df

    def save_model(self, file_path: str) -> None:
        """Saves LightGBM model booster to file."""
        if not self.is_fitted or self.classifier is None:
            raise RuntimeError("Cannot save an unfitted model.")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        self.classifier.booster_.save_model(file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> BaselineMatchingModel:
        """Loads LightGBM model booster from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        booster = lgb.Booster(model_file=file_path)
        self.classifier = lgb.LGBMClassifier()
        self.classifier._Booster = booster
        self.classifier.fitted_ = True
        self.is_fitted = True
        logger.info(f"Model loaded from {file_path}")
        return self
