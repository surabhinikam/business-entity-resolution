"""
50,000 S1 India Records Benchmark.

Evaluates memory-bounded blocking, 29-feature extraction, LightGBM scoring (threshold=0.88),
throughput, candidate pair counts, and peak RSS memory on a controlled 50k India sample.
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
logger = logging.getLogger("benchmark_50k")

ACTIVE_KEYS = ["A", "C", "D", "E", "F"]
MAX_BLOCK_SIZE = 5000
FROZEN_THRESHOLD = 0.88
MODEL_PATH = "models/final_lightgbm_model.pkl"
OVERSIZED_KEYS_FILE = "output/oversized_keys_india.json"
BENCHMARK_S1_COUNT = 50000
BATCH_SIZE = 50000

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


def main():
    logger.info("=" * 80)
    logger.info("BENCHMARK: 50,000 INDIA SOURCE 1 RECORDS")
    logger.info("=" * 80)

    tracemalloc.start()
    t_start = time.time()

    # 1. Load frozen model
    logger.info("Loading frozen LightGBM model from %s...", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2. Load precomputed oversized keys
    if not os.path.exists(OVERSIZED_KEYS_FILE):
        raise FileNotFoundError(f"Missing {OVERSIZED_KEYS_FILE}. Run prepare_india_oversized_keys.py first.")
    with open(OVERSIZED_KEYS_FILE, "r", encoding="utf-8") as f:
        oversized_data = json.load(f)
    oversized_keys = set(oversized_data.keys())
    logger.info("Loaded %d precomputed oversized keys.", len(oversized_keys))

    # 3. Load first 50,000 S1 India records
    s1_files = sorted(glob.glob("data/processed/test/source1/*.parquet"))
    logger.info("Loading first %d S1 India records...", BENCHMARK_S1_COUNT)
    s1_dfs = []
    loaded_s1 = 0
    for f in s1_files:
        df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(pl.col("country_normalized") == "india")
        if df_p.height > 0:
            remaining = BENCHMARK_S1_COUNT - loaded_s1
            s1_dfs.append(df_p.head(remaining))
            loaded_s1 += min(remaining, df_p.height)
            if loaded_s1 >= BENCHMARK_S1_COUNT:
                break
    s1_50k_df = pl.concat(s1_dfs)
    logger.info("Loaded %d S1 records (%.1f MB).", s1_50k_df.height, s1_50k_df.estimated_size() / (1024 * 1024))

    # 4. Extract keys for S1 chunk
    t_keys0 = time.time()
    s1_chunk_index: Dict[str, List[str]] = defaultdict(list)
    for row in s1_50k_df.select(BLOCKING_COLS).iter_rows(named=True):
        eid = row["entity_id"]
        keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
        valid_keys = keys - oversized_keys
        for k in valid_keys:
            s1_chunk_index[k].append(eid)
    s1_chunk_keys = set(s1_chunk_index.keys())
    logger.info("Extracted %d unique valid S1 keys for chunk in %.1fs.", len(s1_chunk_keys), time.time() - t_keys0)

    # 5. Stream candidate records and index ONLY for s1_chunk_keys
    logger.info("Streaming Candidate (S2+S3) records for chunk keys...")
    t_cand0 = time.time()
    s2_files = sorted(glob.glob("data/processed/test/source2/*.parquet"))
    s3_files = sorted(glob.glob("data/processed/test/source3/*.parquet"))

    cand_chunk_index: Dict[str, List[str]] = defaultdict(list)
    cand_matched_ids: Set[str] = set()
    total_cand_scanned = 0

    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == "india")
        if df_p.height == 0:
            continue
        total_cand_scanned += df_p.height
        for row in df_p.iter_rows(named=True):
            eid = row["entity_id"]
            keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
            common = keys & s1_chunk_keys
            if common:
                cand_matched_ids.add(eid)
                for k in common:
                    cand_chunk_index[k].append(eid)

    t_cand_time = time.time() - t_cand0
    logger.info(
        "Scanned %d candidates in %.1fs. Matched %d unique candidate entities across %d active keys.",
        total_cand_scanned, t_cand_time, len(cand_matched_ids), len(cand_chunk_index),
    )

    # 6. Generate candidate pairs
    t_pairs0 = time.time()
    pairs: Set[Tuple[str, str]] = set()
    provenance: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for k, s1_ids in s1_chunk_index.items():
        if k in cand_chunk_index:
            key_label = k.split("||")[0] if "||" in k else "?"
            c_ids = cand_chunk_index[k]
            for s1_id in s1_ids:
                for c_id in c_ids:
                    pair = (s1_id, c_id)
                    pairs.add(pair)
                    provenance[pair].add(key_label)

    n_pairs = len(pairs)
    t_pairs_time = time.time() - t_pairs0
    logger.info(
        "Generated %d candidate pairs in %.2fs (%.1f pairs/S1).",
        n_pairs, t_pairs_time, n_pairs / max(1, s1_50k_df.height),
    )

    # Free chunk indexes
    del s1_chunk_index, cand_chunk_index
    import gc
    gc.collect()

    # 7. Load candidate feature records ONLY for cand_matched_ids
    logger.info("Loading feature records for %d matched candidates...", len(cand_matched_ids))
    cand_feature_dfs = []
    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(
            (pl.col("country_normalized") == "india") & pl.col("entity_id").is_in(pl.Series("eid", list(cand_matched_ids)))
        )
        if df_p.height > 0:
            cand_feature_dfs.append(df_p)
    cand_matched_features_df = pl.concat(cand_feature_dfs)
    logger.info(
        "Loaded matched candidate feature DataFrame: %d rows (%.1f MB).",
        cand_matched_features_df.height, cand_matched_features_df.estimated_size() / (1024 * 1024),
    )

    # 8. Build Candidate Pairs DataFrame
    candidate_pairs_df = pl.DataFrame({
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
    del pairs
    gc.collect()

    # 9. Extract Features and Run Model Inference in Batches
    logger.info("Running 29-feature extraction & LightGBM scoring in batches of %d...", BATCH_SIZE)
    pipeline = FeaturePipeline()
    total_batches = (n_pairs + BATCH_SIZE - 1) // BATCH_SIZE
    total_feat_time = 0.0
    total_infer_time = 0.0
    accepted_matches = []

    for b_idx in range(total_batches):
        b_start = b_idx * BATCH_SIZE
        b_end = min(b_start + BATCH_SIZE, n_pairs)
        b_pairs = candidate_pairs_df.slice(b_start, b_end - b_start)

        b_s1_ids = b_pairs["source1_entity_id"].unique().to_list()
        b_cand_ids = b_pairs["candidate_entity_id"].unique().to_list()

        s1_slice = s1_50k_df.filter(pl.col("entity_id").is_in(pl.Series("eid", b_s1_ids)))
        cand_slice = cand_matched_features_df.filter(pl.col("entity_id").is_in(pl.Series("eid", b_cand_ids)))

        b_tuples = list(zip(b_pairs["source1_entity_id"], b_pairs["candidate_entity_id"]))
        b_prov = {p: provenance[p] for p in b_tuples if p in provenance}

        # Features
        t_f0 = time.time()
        features = pipeline.generate_features_from_records(
            candidate_pairs_df=b_pairs,
            s1_records=s1_slice,
            cand_records=cand_slice,
            pair_provenance=b_prov,
        )
        total_feat_time += time.time() - t_f0

        # Model Inference
        t_i0 = time.time()
        probs = model.predict_proba(features)
        total_infer_time += time.time() - t_i0

        # Thresholding
        mask = probs >= FROZEN_THRESHOLD
        if mask.any():
            matched_idx = np.where(mask)[0]
            s1_m = b_pairs["source1_entity_id"].to_numpy()[matched_idx]
            c_m = b_pairs["candidate_entity_id"].to_numpy()[matched_idx]
            src_m = b_pairs["candidate_source"].to_numpy()[matched_idx]
            p_m = probs[matched_idx]
            for s, c, src, pr in zip(s1_m, c_m, src_m, p_m):
                accepted_matches.append({
                    "source1_entity_id": s,
                    "candidate_entity_id": c,
                    "candidate_source": src,
                    "probability": float(pr),
                })

        del features, b_pairs, s1_slice, cand_slice, b_prov, probs, mask

    del candidate_pairs_df, provenance, cand_matched_features_df, s1_50k_df
    gc.collect()

    t_total = time.time() - t_start
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    s2_cnt = sum(1 for m in accepted_matches if m["candidate_source"] == "source2")
    s3_cnt = sum(1 for m in accepted_matches if m["candidate_source"] == "source3")

    print("\n" + "=" * 80)
    print("50,000 INDIA S1 BENCHMARK RESULTS")
    print("=" * 80)
    print(f"S1 Records Evaluated:                    {BENCHMARK_S1_COUNT:10,d}")
    print(f"Total Candidate Pairs Generated:         {n_pairs:10,d} ({n_pairs/BENCHMARK_S1_COUNT:.1f} pairs/S1)")
    print(f"Total Matches Predicted (P >= {FROZEN_THRESHOLD}):      {len(accepted_matches):10,d}")
    print(f"  - Source 2 Matches:                    {s2_cnt:10,d} ({100.0*s2_cnt/max(1, len(accepted_matches)):.1f}%)")
    print(f"  - Source 3 Matches:                    {s3_cnt:10,d} ({100.0*s3_cnt/max(1, len(accepted_matches)):.1f}%)")
    print("-" * 80)
    print("Throughput & Runtime:")
    print(f"  - Candidate Matching Time:             {t_cand_time + t_pairs_time:10.1f}s")
    print(f"  - Feature Extraction Time:             {total_feat_time:10.1f}s ({n_pairs / max(1e-3, total_feat_time):.1f} pairs/s)")
    print(f"  - Model Inference Time:                {total_infer_time:10.1f}s ({n_pairs / max(1e-3, total_infer_time):.1f} pairs/s)")
    print(f"  - Total Elapsed Wall Time:             {t_total:10.1f}s ({t_total/60:.2f} min)")
    print("-" * 80)
    print("Memory Utilization:")
    print(f"  - Tracemalloc Peak:                    {peak_traced / (1024*1024):10.1f} MB")
    print(f"  - Process Max RSS:                     {rss_mb:10.1f} MB (WSL2 limit: ~11,000 MB)")
    print(f"  - Safety Headroom:                     {11000 - rss_mb:10.1f} MB")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
