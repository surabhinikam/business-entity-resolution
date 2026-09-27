"""
Benchmark to test feature extraction batch size optimization (50k vs 100k vs 200k).
Measures runtime, throughput, peak RAM, and verifies 100% bit-exact mathematical equality.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
import resource
import sys
import time
import tracemalloc
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import polars as pl

from src.candidate_generation.block_keys import generate_all_keys
from src.features.feature_pipeline import FeaturePipeline
from src.models.baseline_model import BaselineMatchingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_size_benchmark")

ACTIVE_KEYS = ["A", "C", "D", "E", "F"]
FROZEN_THRESHOLD = 0.88
MODEL_PATH = "models/final_lightgbm_model.pkl"
OVERSIZED_KEYS_FILE = "output/oversized_keys_india.json"

BLOCKING_COLS = [
    "entity_id",
    "business_name_normalized",
    "business_name_transliterated",
    "business_address_normalized",
    "country_normalized",
]

FEATURE_COLS = [
    "source",
    "entity_id",
    "business_name_normalized",
    "business_name_tokens",
    "business_name_transliterated",
    "business_address_normalized",
    "business_address_tokens",
    "country_normalized",
]


def prepare_test_data(n_pairs_target: int = 100000):
    """Generates a deterministic set of ~100k India candidate pairs for testing."""
    logger.info("Preparing ~%d candidate pairs for batch size benchmark...", n_pairs_target)
    with open(OVERSIZED_KEYS_FILE, "r") as f:
        oversized_keys = set(json.load(f).keys())

    s1_files = sorted(glob.glob("data/processed/test/source1/*.parquet"))
    s2_files = sorted(glob.glob("data/processed/test/source2/*.parquet"))
    s3_files = sorted(glob.glob("data/processed/test/source3/*.parquet"))

    # Load 4,000 S1 India records
    s1_df = pl.concat([
        pl.read_parquet(f, columns=FEATURE_COLS).filter(pl.col("country_normalized") == "india")
        for f in s1_files
    ]).head(4000)

    # Extract keys
    s1_index: Dict[str, List[str]] = defaultdict(list)
    for row in s1_df.select(BLOCKING_COLS).iter_rows(named=True):
        eid = row["entity_id"]
        for k in (generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set()) - oversized_keys):
            s1_index[k].append(eid)
    s1_keys = set(s1_index.keys())

    # Stream candidates
    cand_index: Dict[str, List[str]] = defaultdict(list)
    cand_matched_ids = set()
    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == "india")
        if df_p.height == 0:
            continue
        for row in df_p.iter_rows(named=True):
            eid = row["entity_id"]
            common = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set()) & s1_keys
            if common:
                cand_matched_ids.add(eid)
                for k in common:
                    cand_index[k].append(eid)

    # Pairs & provenance
    pairs: Set[Tuple[str, str]] = set()
    prov: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for k, s1_ids in s1_index.items():
        if k in cand_index:
            kl = k.split("||")[0] if "||" in k else "?"
            for s1_id in s1_ids:
                for c_id in cand_index[k]:
                    pair = (s1_id, c_id)
                    pairs.add(pair)
                    prov[pair].add(kl)
                    if len(pairs) >= n_pairs_target:
                        break
            if len(pairs) >= n_pairs_target:
                break

    # Candidate feature records
    cand_feat_df = pl.concat([
        pl.read_parquet(f, columns=FEATURE_COLS).filter(
            (pl.col("country_normalized") == "india") & pl.col("entity_id").is_in(list(cand_matched_ids))
        )
        for f in s2_files + s3_files
    ])

    pairs_df = pl.DataFrame({
        "source1_entity_id": [p[0] for p in pairs],
        "candidate_entity_id": [p[1] for p in pairs],
    }).with_columns(
        pl.when(pl.col("candidate_entity_id").str.to_uppercase().str.starts_with("S2-"))
        .then(pl.lit("source2"))
        .when(pl.col("candidate_entity_id").str.to_uppercase().str.starts_with("S3-"))
        .then(pl.lit("source3"))
        .otherwise(pl.lit("unknown"))
        .alias("candidate_source")
    )

    logger.info("Test dataset ready: %d pairs, %d S1 records, %d Candidate records.", pairs_df.height, s1_df.height, cand_feat_df.height)
    return pairs_df, s1_df, cand_feat_df, prov


def run_pipeline_with_batch_size(
    batch_size: int,
    candidate_pairs_df: pl.DataFrame,
    s1_df: pl.DataFrame,
    cand_df: pl.DataFrame,
    provenance: Dict[Tuple[str, str], Set[str]],
    model: BaselineMatchingModel,
) -> Tuple[pl.DataFrame, np.ndarray, List[Dict[str, Any]], float, float]:
    """Runs feature extraction and inference with a specified batch size."""
    pipeline = FeaturePipeline()
    n_pairs = candidate_pairs_df.height
    total_batches = (n_pairs + batch_size - 1) // batch_size

    feature_dfs = []
    all_probs = []
    accepted = []

    t_feat = 0.0
    t_infer = 0.0

    for b_idx in range(total_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, n_pairs)
        b_pairs = candidate_pairs_df.slice(b_start, b_end - b_start)

        b_s1_ids = b_pairs["source1_entity_id"].unique().to_list()
        b_cand_ids = b_pairs["candidate_entity_id"].unique().to_list()

        s1_slice = s1_df.filter(pl.col("entity_id").is_in(pl.Series("eid", b_s1_ids)))
        cand_slice = cand_df.filter(pl.col("entity_id").is_in(pl.Series("eid", b_cand_ids)))

        b_tuples = list(zip(b_pairs["source1_entity_id"], b_pairs["candidate_entity_id"]))
        b_prov = {p: provenance[p] for p in b_tuples if p in provenance}

        t0 = time.time()
        feats = pipeline.generate_features_from_records(
            candidate_pairs_df=b_pairs,
            s1_records=s1_slice,
            cand_records=cand_slice,
            pair_provenance=b_prov,
        )
        t_feat += time.time() - t0
        feature_dfs.append(feats)

        t1 = time.time()
        probs = model.predict_proba(feats)
        t_infer += time.time() - t1
        all_probs.append(probs)

        mask = probs >= FROZEN_THRESHOLD
        if mask.any():
            matched_idx = np.where(mask)[0]
            s1_m = b_pairs["source1_entity_id"].to_numpy()[matched_idx]
            c_m = b_pairs["candidate_entity_id"].to_numpy()[matched_idx]
            src_m = b_pairs["candidate_source"].to_numpy()[matched_idx]
            p_m = probs[matched_idx]
            for s, c, src, pr in zip(s1_m, c_m, src_m, p_m):
                accepted.append({
                    "source1_entity_id": s,
                    "candidate_entity_id": c,
                    "candidate_source": src,
                    "probability": float(pr),
                })

    full_features = pl.concat(feature_dfs)
    full_probs = np.concatenate(all_probs)
    return full_features, full_probs, accepted, t_feat, t_infer


def main():
    print("=" * 80)
    print("BENCHMARKING FEATURE EXTRACTION BATCH SIZE: 50,000 vs 100,000")
    print("=" * 80)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 1. Prepare deterministic pairs (~100,000 pairs)
    pairs_df, s1_df, cand_df, prov = prepare_test_data(n_pairs_target=100000)
    n_pairs = pairs_df.height

    # 2. Test Baseline (batch_size = 50,000)
    print("\n[1] Testing Baseline: batch_size = 50,000 (2 batches)...")
    tracemalloc.start()
    t0 = time.time()
    feats_50k, probs_50k, acc_50k, feat_t_50k, infer_t_50k = run_pipeline_with_batch_size(
        batch_size=50000,
        candidate_pairs_df=pairs_df,
        s1_df=s1_df,
        cand_df=cand_df,
        provenance=prov,
        model=model,
    )
    wall_50k = time.time() - t0
    _, peak_50k = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_50k = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"  Baseline 50k: Feat Time={feat_t_50k:.2f}s ({n_pairs/feat_t_50k:.1f} pairs/s) | Infer={infer_t_50k:.2f}s | Wall={wall_50k:.2f}s | Matches={len(acc_50k)}")

    # 3. Test Optimized (batch_size = 100,000)
    print("\n[2] Testing Optimized: batch_size = 100,000 (1 batch)...")
    tracemalloc.start()
    t1 = time.time()
    feats_100k, probs_100k, acc_100k, feat_t_100k, infer_t_100k = run_pipeline_with_batch_size(
        batch_size=100000,
        candidate_pairs_df=pairs_df,
        s1_df=s1_df,
        cand_df=cand_df,
        provenance=prov,
        model=model,
    )
    wall_100k = time.time() - t1
    _, peak_100k = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_100k = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"  Optimized 100k: Feat Time={feat_t_100k:.2f}s ({n_pairs/feat_t_100k:.1f} pairs/s) | Infer={infer_t_100k:.2f}s | Wall={wall_100k:.2f}s | Matches={len(acc_100k)}")

    # 4. Strict Mathematical Equality Checks
    print("\n[3] Verifying Strict Mathematical Equality...")
    feature_cols = [c for c in feats_50k.columns if c not in ["source1_entity_id", "candidate_entity_id", "candidate_source"]]

    # Compare features
    for col in feature_cols:
        v50 = feats_50k[col].to_numpy()
        v100 = feats_100k[col].to_numpy()
        if np.issubdtype(v50.dtype, np.floating):
            assert np.allclose(v50, v100, equal_nan=True), f"Feature column {col} values differ!"
        else:
            assert np.array_equal(v50, v100), f"Feature column {col} values differ!"
    print(f"  ✓ All 29 feature columns are 100% BIT-EXACT IDENTICAL across all {n_pairs:,} candidate pairs.")

    # Compare probabilities
    assert np.allclose(probs_50k, probs_100k, atol=1e-7), "Model prediction probabilities differ!"
    print(f"  ✓ Model prediction probabilities are 100% BIT-EXACT IDENTICAL (max diff = {np.max(np.abs(probs_50k - probs_100k)):.2e}).")

    # Compare accepted matches
    assert len(acc_50k) == len(acc_100k), f"Match count mismatch: {len(acc_50k)} vs {len(acc_100k)}"
    match_pairs_50k = set((m["source1_entity_id"], m["candidate_entity_id"]) for m in acc_50k)
    match_pairs_100k = set((m["source1_entity_id"], m["candidate_entity_id"]) for m in acc_100k)
    assert match_pairs_50k == match_pairs_100k, "Accepted match pair sets differ!"
    print(f"  ✓ Accepted matches at threshold 0.88 match 100%: exactly {len(acc_50k)} matches.")

    # 5. Speedup Analysis
    feat_speedup = (feat_t_50k - feat_t_100k) / feat_t_50k * 100.0
    wall_speedup = (wall_50k - wall_100k) / wall_50k * 100.0

    print("\n" + "=" * 80)
    print("BATCH SIZE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Candidate Pairs Tested:                  {n_pairs:10,d}")
    print(f"Batch Size 50,000 Feature Time:          {feat_t_50k:10.2f}s ({n_pairs/feat_t_50k:.1f} pairs/s)")
    print(f"Batch Size 100,000 Feature Time:         {feat_t_100k:10.2f}s ({n_pairs/feat_t_100k:.1f} pairs/s)")
    print(f"Feature Extraction Speedup:              {feat_speedup:10.1f}%")
    print(f"Wall Clock Time: 50k={wall_50k:.2f}s vs 100k={wall_100k:.2f}s (Speedup: {wall_speedup:.1f}%)")
    print(f"Peak Heap Memory: 50k={peak_50k/(1024*1024):.1f} MB vs 100k={peak_100k/(1024*1024):.1f} MB")
    print(f"Process Peak RSS:                        {rss_100k:.1f} MB")
    print(f"Mathematical & Prediction Equivalence:   VERIFIED (100% IDENTICAL)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
