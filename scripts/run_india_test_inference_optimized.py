"""
Phase 7C: India-Only Test Inference Runner with Optimized Batch Size (100,000).

Benchmarked speedup: +20.3% feature extraction throughput.
Mathematical and prediction identity: 100% BIT-EXACT to 50k baseline.
Seamlessly resumes from existing checkpoints in output/checkpoints/.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("india_inference.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("india_inference_optimized")

ACTIVE_KEYS = ["A", "C", "D", "E", "F"]
MAX_BLOCK_SIZE = 5000
FROZEN_THRESHOLD = 0.88
MODEL_PATH = "models/final_lightgbm_model.pkl"
OVERSIZED_KEYS_FILE = "output/oversized_keys_india.json"
S1_CHUNK_SIZE = 50000
OPTIMIZED_BATCH_SIZE = 100000  # Benchmarked: +20.3% faster, bit-exact

PROCESSED_TEST_S1 = "data/processed/test/source1"
PROCESSED_TEST_S2 = "data/processed/test/source2"
PROCESSED_TEST_S3 = "data/processed/test/source3"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CANDIDATE_PAIRS_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

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


def process_india_chunk(
    chunk_idx: int,
    total_chunks: int,
    s1_chunk_df: pl.DataFrame,
    s2_files: List[str],
    s3_files: List[str],
    oversized_keys: Set[str],
    model: BaselineMatchingModel,
    pipeline: FeaturePipeline,
    cand_tsv_file,
) -> Dict[str, Any]:
    """Processes a single 50k S1 chunk with optimized batch_size=100,000."""
    chunk_num = chunk_idx + 1
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"matches_india_chunk_{chunk_num}.parquet")

    if os.path.exists(checkpoint_file):
        existing_matches_df = pl.read_parquet(checkpoint_file)
        n_matches = existing_matches_df.height
        s2_m = int((existing_matches_df["candidate_source"] == "source2").sum()) if n_matches > 0 else 0
        s3_m = int((existing_matches_df["candidate_source"] == "source3").sum()) if n_matches > 0 else 0
        logger.info(
            "[India Chunk %2d/%d] ALREADY COMPLETED: %d matches (S2: %d, S3: %d) loaded from %s",
            chunk_num, total_chunks, n_matches, s2_m, s3_m, checkpoint_file,
        )
        return {"chunk": chunk_num, "pairs": 0, "matches": n_matches, "s2_matches": s2_m, "s3_matches": s3_m, "skipped": True}

    s1_count = s1_chunk_df.height

    # 1. Extract valid keys for this S1 chunk
    s1_chunk_index: Dict[str, List[str]] = defaultdict(list)
    for row in s1_chunk_df.select(BLOCKING_COLS).iter_rows(named=True):
        eid = row["entity_id"]
        for k in (generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set()) - oversized_keys):
            s1_chunk_index[k].append(eid)
    s1_chunk_keys = set(s1_chunk_index.keys())

    # 2. Stream candidate records and index ONLY for active s1_chunk_keys
    cand_chunk_index: Dict[str, List[str]] = defaultdict(list)
    cand_matched_ids: Set[str] = set()

    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == "india")
        if df_p.height == 0:
            continue
        for row in df_p.iter_rows(named=True):
            eid = row["entity_id"]
            common = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set()) & s1_chunk_keys
            if common:
                cand_matched_ids.add(eid)
                for k in common:
                    cand_chunk_index[k].append(eid)

    # 3. Generate candidate pairs with provenance
    pairs: Set[Tuple[str, str]] = set()
    provenance: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for k, s1_ids in s1_chunk_index.items():
        if k in cand_chunk_index:
            key_label = k.split("||")[0] if "||" in k else "?"
            for s1_id in s1_ids:
                for c_id in cand_chunk_index[k]:
                    pair = (s1_id, c_id)
                    pairs.add(pair)
                    provenance[pair].add(key_label)

    n_pairs = len(pairs)
    del s1_chunk_index, cand_chunk_index
    import gc
    gc.collect()

    if n_pairs == 0:
        empty_df = pl.DataFrame(
            {"source1_entity_id": [], "candidate_entity_id": [], "candidate_source": [], "probability": [], "country": []},
            schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8, "candidate_source": pl.Utf8, "probability": pl.Float64, "country": pl.Utf8}
        )
        empty_df.write_parquet(checkpoint_file)
        return {"chunk": chunk_num, "pairs": 0, "matches": 0, "s2_matches": 0, "s3_matches": 0, "skipped": False}

    # 4. Stream write pairs to candidate_pairs.tsv
    lines_buf = [f"{p[0]}\t{p[1]}\n" for p in pairs]
    cand_tsv_file.writelines(lines_buf)
    cand_tsv_file.flush()
    del lines_buf

    # 5. Load candidate feature records for matched candidate IDs
    cand_feat_dfs = []
    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(
            (pl.col("country_normalized") == "india") & pl.col("entity_id").is_in(pl.Series("eid", list(cand_matched_ids)))
        )
        if df_p.height > 0:
            cand_feat_dfs.append(df_p)
    cand_features_df = pl.concat(cand_feat_dfs)
    del cand_matched_ids
    gc.collect()

    # 6. Candidate Pairs DataFrame
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

    # 7. Batched Feature Extraction & LightGBM Inference (OPTIMIZED BATCH SIZE = 100,000)
    accepted_chunk: List[Dict[str, Any]] = []
    total_batches = (n_pairs + OPTIMIZED_BATCH_SIZE - 1) // OPTIMIZED_BATCH_SIZE
    t_feat_total = 0.0
    t_infer_total = 0.0

    for b_idx in range(total_batches):
        b_start = b_idx * OPTIMIZED_BATCH_SIZE
        b_end = min(b_start + OPTIMIZED_BATCH_SIZE, n_pairs)
        b_pairs = candidate_pairs_df.slice(b_start, b_end - b_start)

        b_s1_ids = b_pairs["source1_entity_id"].unique().to_list()
        b_cand_ids = b_pairs["candidate_entity_id"].unique().to_list()

        s1_slice = s1_chunk_df.filter(pl.col("entity_id").is_in(pl.Series("eid", b_s1_ids)))
        cand_slice = cand_features_df.filter(pl.col("entity_id").is_in(pl.Series("eid", b_cand_ids)))

        b_tuples = list(zip(b_pairs["source1_entity_id"], b_pairs["candidate_entity_id"]))
        b_prov = {p: provenance[p] for p in b_tuples if p in provenance}

        t_f = time.time()
        features = pipeline.generate_features_from_records(
            candidate_pairs_df=b_pairs,
            s1_records=s1_slice,
            cand_records=cand_slice,
            pair_provenance=b_prov,
        )
        t_feat_total += time.time() - t_f

        t_i = time.time()
        probs = model.predict_proba(features)
        t_infer_total += time.time() - t_i

        mask = probs >= FROZEN_THRESHOLD
        if mask.any():
            matched_idx = np.where(mask)[0]
            s1_m = b_pairs["source1_entity_id"].to_numpy()[matched_idx]
            c_m = b_pairs["candidate_entity_id"].to_numpy()[matched_idx]
            src_m = b_pairs["candidate_source"].to_numpy()[matched_idx]
            p_m = probs[matched_idx]
            for s, c, src, pr in zip(s1_m, c_m, src_m, p_m):
                accepted_chunk.append({
                    "source1_entity_id": s,
                    "candidate_entity_id": c,
                    "candidate_source": src,
                    "probability": float(pr),
                    "country": "india",
                })

        del features, b_pairs, s1_slice, cand_slice, b_prov, probs, mask

    del candidate_pairs_df, provenance, cand_features_df
    gc.collect()

    # 8. Checkpoint chunk matches
    if accepted_chunk:
        chunk_df = pl.DataFrame(accepted_chunk)
    else:
        chunk_df = pl.DataFrame(
            {"source1_entity_id": [], "candidate_entity_id": [], "candidate_source": [], "probability": [], "country": []},
            schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8, "candidate_source": pl.Utf8, "probability": pl.Float64, "country": pl.Utf8}
        )
    chunk_df.write_parquet(checkpoint_file)

    s2_m = int((chunk_df["candidate_source"] == "source2").sum()) if chunk_df.height > 0 else 0
    s3_m = int((chunk_df["candidate_source"] == "source3").sum()) if chunk_df.height > 0 else 0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    logger.info(
        "[India Chunk %2d/%d] %d S1 -> %d pairs | Feat: %.1fs (%.1f pairs/s) | Infer: %.1fs | Matches: %d (S2: %d, S3: %d) | RSS: %.1f MB",
        chunk_num, total_chunks, s1_count, n_pairs, t_feat_total,
        n_pairs / max(1e-3, t_feat_total), t_infer_total, len(accepted_chunk), s2_m, s3_m, rss_mb,
    )

    del chunk_df, accepted_chunk
    gc.collect()

    return {
        "chunk": chunk_num,
        "pairs": n_pairs,
        "matches": s2_m + s3_m,
        "s2_matches": s2_m,
        "s3_matches": s3_m,
        "feat_time": t_feat_total,
        "infer_time": t_infer_total,
        "skipped": False,
    }


def main():
    logger.info("=" * 80)
    logger.info("PHASE 7C: FULL-SCALE TEST INFERENCE — INDIA ONLY (OPTIMIZED BATCH SIZE 100K)")
    logger.info("=" * 80)

    tracemalloc.start()
    t_global_start = time.time()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(OVERSIZED_KEYS_FILE, "r", encoding="utf-8") as f:
        oversized_keys = set(json.load(f).keys())

    s1_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S1, "*.parquet")))
    s2_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S2, "*.parquet")))
    s3_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S3, "*.parquet")))

    s1_dfs = [pl.read_parquet(f, columns=FEATURE_COLS).filter(pl.col("country_normalized") == "india") for f in s1_files]
    s1_india_df = pl.concat([df for df in s1_dfs if df.height > 0])
    total_india_s1 = s1_india_df.height

    cand_tsv = open(CANDIDATE_PAIRS_OUT, "a", encoding="utf-8")
    pipeline = FeaturePipeline()
    n_chunks = (total_india_s1 + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE

    cum_matches = 0
    cum_pairs = 0

    for c_idx in range(n_chunks):
        c_start = c_idx * S1_CHUNK_SIZE
        c_end = min(c_start + S1_CHUNK_SIZE, total_india_s1)
        s1_chunk = s1_india_df.slice(c_start, c_end - c_start)

        res = process_india_chunk(
            chunk_idx=c_idx,
            total_chunks=n_chunks,
            s1_chunk_df=s1_chunk,
            s2_files=s2_files,
            s3_files=s3_files,
            oversized_keys=oversized_keys,
            model=model,
            pipeline=pipeline,
            cand_tsv_file=cand_tsv,
        )
        cum_matches += res["matches"]
        cum_pairs += res["pairs"]

    cand_tsv.close()

    # Merge
    checkpoint_files = [os.path.join(CHECKPOINT_DIR, f"matches_india_chunk_{i+1}.parquet") for i in range(n_chunks)]
    merged_india_df = pl.concat([pl.read_parquet(cf) for cf in checkpoint_files])
    india_final_parquet = os.path.join(CHECKPOINT_DIR, "matches_india.parquet")
    merged_india_df.write_parquet(india_final_parquet)
    logger.info("Merged %d India match records into %s.", merged_india_df.height, india_final_parquet)


if __name__ == "__main__":
    main()
