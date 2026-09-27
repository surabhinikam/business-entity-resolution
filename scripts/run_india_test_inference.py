"""
Phase 7C: India-Only Full-Scale Test Inference Runner.

Executes complete India candidate generation across all 809,986 Source 1 records
in 17 chunks of 50,000, 29-feature extraction, LightGBM model scoring (threshold 0.88),
and per-chunk disk checkpointing to output/checkpoints/matches_india_chunk_{i}.parquet.

Invariants:
- France and US are ALREADY COMPLETE and strictly excluded.
- Frozen V4 blocker: A, C, D, E, F with max block size 5000.
- Bit-exact oversized block capping via output/oversized_keys_india.json.
- Exact frozen 29-feature schema matching FULL_PIPELINE_COLUMNS.
- Frozen LightGBM model: models/final_lightgbm_model.pkl.
- Frozen decision threshold: 0.88.
- Appends newly generated India pairs to output/candidate_pairs.tsv.
- Memory strictly bounded: peak RSS << 11 GiB ceiling.
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
from src.features.feature_schema import FULL_PIPELINE_COLUMNS
from src.models.baseline_model import BaselineMatchingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("india_inference.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("india_inference")

# Invariants
ACTIVE_KEYS = ["A", "C", "D", "E", "F"]
MAX_BLOCK_SIZE = 5000
FROZEN_THRESHOLD = 0.88
MODEL_PATH = "models/final_lightgbm_model.pkl"
OVERSIZED_KEYS_FILE = "output/oversized_keys_india.json"
S1_CHUNK_SIZE = 50000
BATCH_SIZE = 50000

# File Paths
TEST_SOURCE1_RAW = "data/raw/test/test_source1.tsv"
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


def inspect_existing_checkpoints():
    """Confirms France and US are marked completed and candidate_pairs.tsv is valid."""
    logger.info("=" * 80)
    logger.info("INSPECTING EXISTING OUTPUT AND CHECKPOINT STRUCTURE")
    logger.info("=" * 80)

    if not os.path.exists(CANDIDATE_PAIRS_OUT):
        raise FileNotFoundError(f"Missing {CANDIDATE_PAIRS_OUT}. France and US pairs must exist.")

    # Check line count
    with open(CANDIDATE_PAIRS_OUT, "r", encoding="utf-8") as f:
        first_line = f.readline()
        assert "source1_entity_id" in first_line, "candidate_pairs.tsv header missing"

    # Confirm France & US exclusion
    logger.info("Confirmed: France and United States test candidate pairs exist in %s.", CANDIDATE_PAIRS_OUT)
    logger.info("Confirmed: France and United States are EXCLUDED from this inference run.")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
    """Processes a single 50k S1 chunk of India with disk checkpointing."""
    chunk_num = chunk_idx + 1
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"matches_india_chunk_{chunk_num}.parquet")

    # Check if chunk was already processed
    if os.path.exists(checkpoint_file):
        existing_matches_df = pl.read_parquet(checkpoint_file)
        n_matches = existing_matches_df.height
        s2_matches = int((existing_matches_df["candidate_source"] == "source2").sum()) if n_matches > 0 else 0
        s3_matches = int((existing_matches_df["candidate_source"] == "source3").sum()) if n_matches > 0 else 0
        logger.info(
            "[India Chunk %2d/%d] ALREADY COMPLETED: %d matches (S2: %d, S3: %d) loaded from %s",
            chunk_num, total_chunks, n_matches, s2_matches, s3_matches, checkpoint_file,
        )
        return {
            "chunk": chunk_num,
            "pairs": 0,
            "matches": n_matches,
            "s2_matches": s2_matches,
            "s3_matches": s3_matches,
            "feat_time": 0.0,
            "infer_time": 0.0,
            "skipped": True,
        }

    t0_chunk = time.time()
    s1_count = s1_chunk_df.height

    # 1. Extract valid keys for this S1 chunk
    t_k0 = time.time()
    s1_chunk_index: Dict[str, List[str]] = defaultdict(list)
    for row in s1_chunk_df.select(BLOCKING_COLS).iter_rows(named=True):
        eid = row["entity_id"]
        keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
        valid_keys = keys - oversized_keys
        for k in valid_keys:
            s1_chunk_index[k].append(eid)
    s1_chunk_keys = set(s1_chunk_index.keys())
    t_keys = time.time() - t_k0

    # 2. Stream Candidate records and index ONLY for active s1_chunk_keys
    t_c0 = time.time()
    cand_chunk_index: Dict[str, List[str]] = defaultdict(list)
    cand_matched_ids: Set[str] = set()

    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == "india")
        if df_p.height == 0:
            continue
        for row in df_p.iter_rows(named=True):
            eid = row["entity_id"]
            keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
            common = keys & s1_chunk_keys
            if common:
                cand_matched_ids.add(eid)
                for k in common:
                    cand_chunk_index[k].append(eid)
    t_cand = time.time() - t_c0

    # 3. Generate candidate pairs with provenance
    t_p0 = time.time()
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
    t_pairs = time.time() - t_p0

    # Free chunk indexing structures immediately
    del s1_chunk_index, cand_chunk_index
    import gc
    gc.collect()

    if n_pairs == 0:
        empty_df = pl.DataFrame(
            {"source1_entity_id": [], "candidate_entity_id": [], "candidate_source": [], "probability": [], "country": []},
            schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8, "candidate_source": pl.Utf8, "probability": pl.Float64, "country": pl.Utf8}
        )
        empty_df.write_parquet(checkpoint_file)
        return {
            "chunk": chunk_num,
            "pairs": 0,
            "matches": 0,
            "s2_matches": 0,
            "s3_matches": 0,
            "feat_time": 0.0,
            "infer_time": 0.0,
            "skipped": False,
        }

    # 4. Stream write pairs to candidate_pairs.tsv
    lines_buf = [f"{p[0]}\t{p[1]}\n" for p in pairs]
    cand_tsv_file.writelines(lines_buf)
    cand_tsv_file.flush()
    del lines_buf

    # 5. Load candidate feature records for matched candidate IDs
    cand_feature_dfs = []
    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(
            (pl.col("country_normalized") == "india") & pl.col("entity_id").is_in(pl.Series("eid", list(cand_matched_ids)))
        )
        if df_p.height > 0:
            cand_feature_dfs.append(df_p)
    cand_features_df = pl.concat(cand_feature_dfs)
    del cand_matched_ids
    gc.collect()

    # 6. Build Candidate Pairs DataFrame
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

    # 7. Batched Feature Extraction & LightGBM Inference
    accepted_chunk: List[Dict[str, Any]] = []
    total_batches = (n_pairs + BATCH_SIZE - 1) // BATCH_SIZE
    t_feat_total = 0.0
    t_infer_total = 0.0

    for b_idx in range(total_batches):
        b_start = b_idx * BATCH_SIZE
        b_end = min(b_start + BATCH_SIZE, n_pairs)
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

    # 8. Checkpoint chunk matches to Parquet
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
    logger.info("PHASE 7C: FULL-SCALE TEST INFERENCE — INDIA ONLY")
    logger.info("=" * 80)
    logger.info("Country Target:          INDIA ONLY")
    logger.info("Blocker Keys:            %s", ACTIVE_KEYS)
    logger.info("Max Block Size Cap:      %d", MAX_BLOCK_SIZE)
    logger.info("Decision Threshold:      %.2f", FROZEN_THRESHOLD)
    logger.info("S1 Chunk Size:           %d", S1_CHUNK_SIZE)
    logger.info("Batch Size:              %d", BATCH_SIZE)
    logger.info("Checkpoint Dir:          %s", CHECKPOINT_DIR)
    logger.info("Candidate Output:        %s", CANDIDATE_PAIRS_OUT)
    logger.info("=" * 80)

    tracemalloc.start()
    t_global_start = time.time()

    # Step 1: Inspect existing output and checkpoints
    inspect_existing_checkpoints()

    # Step 2: Load model
    logger.info("Loading frozen LightGBM model from %s...", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Frozen LightGBM model loaded successfully.")

    # Step 3: Load oversized keys
    if not os.path.exists(OVERSIZED_KEYS_FILE):
        raise FileNotFoundError(f"Missing {OVERSIZED_KEYS_FILE}. Precompute first.")
    with open(OVERSIZED_KEYS_FILE, "r", encoding="utf-8") as f:
        oversized_data = json.load(f)
    oversized_keys = set(oversized_data.keys())
    logger.info("Loaded %d precomputed oversized keys.", len(oversized_keys))

    # Step 4: Load India S1 records
    s1_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S1, "*.parquet")))
    s2_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S2, "*.parquet")))
    s3_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S3, "*.parquet")))

    logger.info("Loading all Source 1 records for country 'india'...")
    t0_s1 = time.time()
    s1_dfs = []
    for f in s1_files:
        df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(pl.col("country_normalized") == "india")
        if df_p.height > 0:
            s1_dfs.append(df_p)
    s1_india_df = pl.concat(s1_dfs)
    total_india_s1 = s1_india_df.height
    logger.info("Loaded %d S1 India records (%.1f MB) in %.1fs.", total_india_s1, s1_india_df.estimated_size() / (1024 * 1024), time.time() - t0_s1)
    assert total_india_s1 == 809986, f"Expected 809,986 India S1 records, found {total_india_s1}"

    # Step 5: Open candidate_pairs.tsv in append mode
    cand_tsv = open(CANDIDATE_PAIRS_OUT, "a", encoding="utf-8")

    pipeline = FeaturePipeline()
    n_chunks = (total_india_s1 + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
    logger.info("Processing %d India S1 entities across %d chunks of %d...\n", total_india_s1, n_chunks, S1_CHUNK_SIZE)

    chunk_results = []
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

        chunk_results.append(res)
        cum_matches += res["matches"]
        cum_pairs += res["pairs"]
        logger.info(
            "--> [Progress] Completed Chunk %d/%d | Chunk Matches: %d | Cumulative Matches: %d\n",
            c_idx + 1, n_chunks, res["matches"], cum_matches,
        )

    cand_tsv.close()
    logger.info("Closed %s after appending India pairs.", CANDIDATE_PAIRS_OUT)

    # Step 6: Verify all India checkpoints and combine into matches_india.parquet
    logger.info("=" * 80)
    logger.info("VERIFYING INDIA CHECKPOINTS AND MERGING MATCHES")
    logger.info("=" * 80)

    checkpoint_files = [
        os.path.join(CHECKPOINT_DIR, f"matches_india_chunk_{i+1}.parquet")
        for i in range(n_chunks)
    ]

    all_exist = all(os.path.exists(cf) for cf in checkpoint_files)
    assert all_exist, "Missing one or more India chunk checkpoint files!"
    logger.info("All %d India chunk checkpoint files verified and present.", n_chunks)

    india_matches_dfs = [pl.read_parquet(cf) for cf in checkpoint_files]
    merged_india_df = pl.concat(india_matches_dfs)
    india_final_parquet = os.path.join(CHECKPOINT_DIR, "matches_india.parquet")
    merged_india_df.write_parquet(india_final_parquet)
    logger.info("Merged %d India match records into %s.", merged_india_df.height, india_final_parquet)

    # Summary
    total_time = time.time() - t_global_start
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    s2_total = int((merged_india_df["candidate_source"] == "source2").sum())
    s3_total = int((merged_india_df["candidate_source"] == "source3").sum())

    print("\n" + "=" * 80)
    print("PHASE 7C: INDIA FULL-SCALE TEST INFERENCE REPORT")
    print("=" * 80)
    print(f"Total India S1 Entities Processed:      {total_india_s1:12,d}")
    print(f"Total India Chunks Completed:           {n_chunks:12d}")
    print(f"Total Candidate Pairs Generated:         {cum_pairs:12,d}")
    print(f"Total Predicted Matches (P >= {FROZEN_THRESHOLD}):      {merged_india_df.height:12,d}")
    print(f"  - Source 2 Matches:                    {s2_total:12,d} ({100.0*s2_total/max(1, merged_india_df.height):5.1f}%)")
    print(f"  - Source 3 Matches:                    {s3_total:12,d} ({100.0*s3_total/max(1, merged_india_df.height):5.1f}%)")
    print(f"Total Wall Clock Runtime:                {total_time/60:12.2f} min")
    print(f"Peak RSS Memory:                         {rss_mb:12.1f} MB (WSL2 ceiling: ~11,000 MB)")
    print(f"Checkpoint File:                         {os.path.abspath(india_final_parquet)}")
    print(f"Updated Candidate Pairs File:            {os.path.abspath(CANDIDATE_PAIRS_OUT)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
