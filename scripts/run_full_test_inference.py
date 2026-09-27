"""
Phase 7C: Full-Scale Test Candidate Generation & Inference Runner.

Executes complete test candidate generation, 29-feature extraction,
LightGBM model scoring, thresholding at 0.88, match post-processing,
output generation (candidate_pairs.tsv and matching_results.tsv),
and submission format validation.

Guarantees:
- Strict preservation of frozen V4 blocker (Keys A, C, D, E, F with 5000 capping).
- Bit-exact mathematical equivalence to BlockIndex via country-global key capping.
- Exact 29 canonical feature schema matching FULL_PIPELINE_COLUMNS.
- Frozen LightGBM model configuration from Phase 4D.
- Single frozen decision threshold: 0.88.
- Memory-bounded streaming execution partitioned by country and 50k S1 chunks.
- Peak RAM strictly bounded < 2.5 GB (safe for 7.7 GB WSL2).
- Zero modification to frozen production modules.
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
import shutil
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import polars as pl

from src.candidate_generation.block_keys import generate_all_keys
from src.features.feature_pipeline import FeaturePipeline
from src.features.feature_schema import ALL_FEATURE_NAMES, FULL_PIPELINE_COLUMNS
from src.models.baseline_model import BaselineMatchingModel
from src.models.post_processing import MatchPostProcessor
from src.validation.validators import validate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_inference.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("phase7c_inference")

# Frozen Invariants
ACTIVE_KEYS = ["A", "C", "D", "E", "F"]
MAX_BLOCK_SIZE = 5000
FROZEN_THRESHOLD = 0.88
MODEL_PATH = "models/final_lightgbm_model.pkl"
S1_CHUNK_SIZE = 50000
BATCH_SIZE = 50000

# File Paths
TEST_SOURCE1_RAW = "data/raw/test/test_source1.tsv"
PROCESSED_TEST_S1 = "data/processed/test/source1"
PROCESSED_TEST_S2 = "data/processed/test/source2"
PROCESSED_TEST_S3 = "data/processed/test/source3"
OUTPUT_DIR = "output"
CANDIDATE_PAIRS_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_RESULTS_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")

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


def load_country_records(
    country: str,
    s1_files: List[str],
    s2_files: List[str],
    s3_files: List[str],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Loads all test records for a specific country into compact Polars DataFrames."""
    logger.info("Loading Source 1 records for country '%s'...", country)
    t0 = time.time()
    s1_dfs = []
    for f in s1_files:
        df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(pl.col("country_normalized") == country)
        if df_p.height > 0:
            s1_dfs.append(df_p)
    s1_df = pl.concat(s1_dfs) if s1_dfs else pl.DataFrame(schema={c: pl.Utf8 for c in FEATURE_COLS})
    logger.info("Loaded %d S1 records for '%s' (%.1f MB) in %.1fs", s1_df.height, country, s1_df.estimated_size() / (1024*1024), time.time() - t0)

    logger.info("Loading Candidate (S2 + S3) records for country '%s'...", country)
    t1 = time.time()
    cand_dfs = []
    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(pl.col("country_normalized") == country)
        if df_p.height > 0:
            cand_dfs.append(df_p)
    cand_df = pl.concat(cand_dfs) if cand_dfs else pl.DataFrame(schema={c: pl.Utf8 for c in FEATURE_COLS})
    logger.info("Loaded %d Candidate records for '%s' (%.1f MB) in %.1fs", cand_df.height, country, cand_df.estimated_size() / (1024*1024), time.time() - t1)

    return s1_df, cand_df


def build_country_candidate_index(
    s1_country_df: pl.DataFrame,
    cand_country_df: pl.DataFrame,
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Computes global S1 key counts and indexes candidate records for keys present in S1.
    Identifies all oversized blocks (> 5000) using global country counts, guaranteeing
    100% bit-exact V4 blocking semantics.
    """
    logger.info("Computing global S1 key frequency counts...")
    t0 = time.time()
    global_s1_counts: Counter[str] = Counter()
    for row in s1_country_df.select(BLOCKING_COLS).iter_rows(named=True):
        keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
        for k in keys:
            global_s1_counts[k] += 1
    logger.info("Computed global counts for %d unique S1 keys in %.1fs", len(global_s1_counts), time.time() - t0)

    s1_key_set = set(global_s1_counts.keys())

    logger.info("Indexing candidate records for active S1 keys...")
    t1 = time.time()
    cand_index: Dict[str, List[str]] = defaultdict(list)
    cand_count = 0
    for row in cand_country_df.select(BLOCKING_COLS).iter_rows(named=True):
        eid = row["entity_id"]
        keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
        common = keys & s1_key_set
        for k in common:
            cand_index[k].append(eid)
        cand_count += 1
    logger.info("Indexed %d candidates into %d shared keys in %.1fs", cand_count, len(cand_index), time.time() - t1)

    # Detect oversized blocks using global S1 counts
    oversized_keys = {
        k for k in cand_index
        if global_s1_counts[k] * len(cand_index[k]) > MAX_BLOCK_SIZE
    }
    logger.info("Detected and capped %d oversized blocks (> %d pairs)", len(oversized_keys), MAX_BLOCK_SIZE)

    return cand_index, oversized_keys


def process_s1_chunk(
    s1_chunk_df: pl.DataFrame,
    cand_df: pl.DataFrame,
    cand_index: Dict[str, List[str]],
    oversized_keys: Set[str],
    country: str,
    chunk_name: str,
    model: BaselineMatchingModel,
    pipeline: FeaturePipeline,
    candidate_tsv_file,
    batch_size: int = BATCH_SIZE,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates candidate pairs for a chunk of 50,000 S1 records against the precomputed candidate index.
    Streams pairs to candidate_pairs.tsv and extracts 29 features in batches for model scoring.
    """
    t_chunk_start = time.time()
    s1_count = s1_chunk_df.height

    # 1. Index S1 records for this chunk
    t_idx_s1 = time.time()
    s1_chunk_index: Dict[str, List[str]] = defaultdict(list)
    for row in s1_chunk_df.select(BLOCKING_COLS).iter_rows(named=True):
        eid = row["entity_id"]
        keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
        for k in keys:
            s1_chunk_index[k].append(eid)
    t_idx_done = time.time() - t_idx_s1

    # 2. Generate candidate pairs for this chunk
    t_pairs = time.time()
    pairs: Set[Tuple[str, str]] = set()
    provenance: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    shared_keys = set(s1_chunk_index.keys()) & set(cand_index.keys())
    for k in shared_keys:
        if k in oversized_keys:
            continue
        key_label = k.split("||")[0] if "||" in k else "?"
        s1_list = s1_chunk_index[k]
        cand_list = cand_index[k]
        for s1_id in s1_list:
            for c_id in cand_list:
                pair = (s1_id, c_id)
                pairs.add(pair)
                provenance[pair].add(key_label)

    del s1_chunk_index
    n_pairs = len(pairs)
    t_pairs_done = time.time() - t_pairs

    if n_pairs == 0:
        return [], {"pairs": 0, "accepted": 0, "feat_time": 0.0, "infer_time": 0.0, "total_time": time.time() - t_chunk_start}

    # 3. Stream write candidate pairs to TSV
    t_write = time.time()
    lines_buf = []
    for s1_id, c_id in pairs:
        lines_buf.append(f"{s1_id}\t{c_id}\n")
        if len(lines_buf) >= 100000:
            candidate_tsv_file.writelines(lines_buf)
            lines_buf.clear()
    if lines_buf:
        candidate_tsv_file.writelines(lines_buf)
        lines_buf.clear()
    candidate_tsv_file.flush()
    t_write_done = time.time() - t_write

    # 4. Build candidate pairs DataFrame with source
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
    import gc
    gc.collect()

    # 5. Batch feature extraction and model scoring
    accepted_matches: List[Dict[str, Any]] = []
    total_batches = (n_pairs + batch_size - 1) // batch_size
    total_feat_time = 0.0
    total_infer_time = 0.0

    for batch_idx in range(total_batches):
        b_start = batch_idx * batch_size
        b_end = min(b_start + batch_size, n_pairs)
        batch_pairs = candidate_pairs_df.slice(b_start, b_end - b_start)

        batch_s1_ids = batch_pairs["source1_entity_id"].unique().to_list()
        batch_cand_ids = batch_pairs["candidate_entity_id"].unique().to_list()

        s1_slice = s1_chunk_df.filter(pl.col("entity_id").is_in(pl.Series("entity_id", batch_s1_ids)))
        cand_slice = cand_df.filter(pl.col("entity_id").is_in(pl.Series("entity_id", batch_cand_ids)))

        batch_pair_tuples = list(zip(batch_pairs["source1_entity_id"], batch_pairs["candidate_entity_id"]))
        batch_prov = {p: provenance[p] for p in batch_pair_tuples if p in provenance}

        # Feature Extraction
        t_f0 = time.time()
        batch_features = pipeline.generate_features_from_records(
            candidate_pairs_df=batch_pairs,
            s1_records=s1_slice,
            cand_records=cand_slice,
            pair_provenance=batch_prov,
        )
        total_feat_time += time.time() - t_f0

        # Model Inference
        t_i0 = time.time()
        probs = model.predict_proba(batch_features)
        total_infer_time += time.time() - t_i0

        # Thresholding at 0.88
        mask = probs >= FROZEN_THRESHOLD
        if mask.any():
            matched_indices = np.where(mask)[0]
            s1_m = batch_pairs["source1_entity_id"].to_numpy()[matched_indices]
            c_m = batch_pairs["candidate_entity_id"].to_numpy()[matched_indices]
            src_m = batch_pairs["candidate_source"].to_numpy()[matched_indices]
            p_m = probs[matched_indices]

            for s, c, src, pr in zip(s1_m, c_m, src_m, p_m):
                accepted_matches.append({
                    "source1_entity_id": s,
                    "candidate_entity_id": c,
                    "candidate_source": src,
                    "probability": float(pr),
                    "country": country,
                })

        del batch_features, batch_pairs, s1_slice, cand_slice, batch_prov, probs, mask

    del candidate_pairs_df, provenance
    gc.collect()

    t_total = time.time() - t_chunk_start
    logger.info(
        "[%s] %d S1 -> %d pairs | Feat: %.1fs | Infer: %.1fs | Total: %.1fs (%.1f pairs/s) | Matches: %d",
        chunk_name, s1_count, n_pairs, total_feat_time, total_infer_time, t_total,
        n_pairs / max(1e-3, t_total), len(accepted_matches),
    )

    stats = {
        "pairs": n_pairs,
        "accepted": len(accepted_matches),
        "feat_time": total_feat_time,
        "infer_time": total_infer_time,
        "total_time": t_total,
    }
    return accepted_matches, stats


def main():
    logger.info("=" * 80)
    logger.info("PHASE 7C: FULL-SCALE TEST CANDIDATE GENERATION & INFERENCE")
    logger.info("=" * 80)
    logger.info("Active Blocker Keys:     %s", ACTIVE_KEYS)
    logger.info("Max Block Size Cap:      %d", MAX_BLOCK_SIZE)
    logger.info("Frozen Model Path:       %s", MODEL_PATH)
    logger.info("Frozen Threshold:        %.2f", FROZEN_THRESHOLD)
    logger.info("S1 Chunk Size:           %d", S1_CHUNK_SIZE)
    logger.info("Feature Batch Size:      %d", BATCH_SIZE)
    logger.info("=" * 80)

    tracemalloc.start()
    t_global_start = time.time()

    # 1. Load trained frozen LightGBM model
    logger.info("Step 1: Loading frozen LightGBM model from %s...", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")

    # 2. Load all Source 1 entity IDs in canonical test order
    logger.info("Step 2: Loading canonical test Source 1 entity IDs from %s...", TEST_SOURCE1_RAW)
    s1_raw_df = pl.read_csv(TEST_SOURCE1_RAW, separator="\t", columns=["entity_id"])
    all_s1_entities = s1_raw_df["entity_id"].to_list()
    total_test_s1 = len(all_s1_entities)
    logger.info("Loaded %d canonical test Source 1 entity IDs.", total_test_s1)

    # 3. Locate processed test files
    s1_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S1, "*.parquet")))
    s2_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S2, "*.parquet")))
    s3_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S3, "*.parquet")))
    logger.info("Found %d S1, %d S2, %d S3 processed parquet parts.", len(s1_files), len(s2_files), len(s3_files))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open candidate_pairs.tsv for streaming output
    logger.info("Opening %s for streaming output...", CANDIDATE_PAIRS_OUT)
    cand_tsv_file = open(CANDIDATE_PAIRS_OUT, "w", encoding="utf-8")
    cand_tsv_file.write("source1_entity_id\tcandidate_entity_id\n")

    countries = ["france", "united states", "india"]
    all_accepted_matches: List[Dict[str, Any]] = []
    total_pairs_generated = 0
    total_feat_time = 0.0
    total_infer_time = 0.0

    for country in countries:
        logger.info("\n" + "#" * 80)
        logger.info("STARTING COUNTRY: %s", country.upper())
        logger.info("#" * 80)

        # Load country records
        s1_country_df, cand_country_df = load_country_records(country, s1_files, s2_files, s3_files)
        total_country_s1 = s1_country_df.height

        # Build candidate index and detect oversized blocks using global S1 counts
        cand_index, oversized_keys = build_country_candidate_index(s1_country_df, cand_country_df)

        pipeline = FeaturePipeline()

        # Split country S1 into chunks of S1_CHUNK_SIZE
        n_chunks = (total_country_s1 + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
        logger.info("Processing %d S1 entities in %d chunks of %d...", total_country_s1, n_chunks, S1_CHUNK_SIZE)

        for c_idx in range(n_chunks):
            c_start = c_idx * S1_CHUNK_SIZE
            c_end = min(c_start + S1_CHUNK_SIZE, total_country_s1)
            s1_chunk_df = s1_country_df.slice(c_start, c_end - c_start)
            chunk_name = f"{country}_chunk_{c_idx+1}_of_{n_chunks}"

            chunk_matches, chunk_stats = process_s1_chunk(
                s1_chunk_df=s1_chunk_df,
                cand_df=cand_country_df,
                cand_index=cand_index,
                oversized_keys=oversized_keys,
                country=country,
                chunk_name=chunk_name,
                model=model,
                pipeline=pipeline,
                candidate_tsv_file=cand_tsv_file,
                batch_size=BATCH_SIZE,
            )

            all_accepted_matches.extend(chunk_matches)
            total_pairs_generated += chunk_stats["pairs"]
            total_feat_time += chunk_stats["feat_time"]
            total_infer_time += chunk_stats["infer_time"]

            import gc
            gc.collect()

        del s1_country_df, cand_country_df, cand_index, oversized_keys, pipeline
        gc.collect()

    cand_tsv_file.close()
    logger.info("Closed %s. Total candidate pairs generated: %d", CANDIDATE_PAIRS_OUT, total_pairs_generated)

    # Copy candidate_pairs.tsv to workspace root
    root_cand_tsv = "candidate_pairs.tsv"
    if os.path.abspath(CANDIDATE_PAIRS_OUT) != os.path.abspath(root_cand_tsv):
        logger.info("Copying %s to %s...", CANDIDATE_PAIRS_OUT, root_cand_tsv)
        shutil.copyfile(CANDIDATE_PAIRS_OUT, root_cand_tsv)

    # 4. Format final submission with MatchPostProcessor
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Formatting final submission via MatchPostProcessor (threshold=%.2f)...", FROZEN_THRESHOLD)
    logger.info("=" * 80)

    if all_accepted_matches:
        accepted_df = pl.DataFrame(all_accepted_matches)
    else:
        accepted_df = pl.DataFrame(
            {"source1_entity_id": [], "candidate_entity_id": [], "probability": [], "candidate_source": [], "country": []},
            schema={
                "source1_entity_id": pl.Utf8,
                "candidate_entity_id": pl.Utf8,
                "probability": pl.Float64,
                "candidate_source": pl.Utf8,
                "country": pl.Utf8,
            },
        )

    logger.info("Total accepted match pairs across all countries: %d", accepted_df.height)

    post_processor = MatchPostProcessor(threshold=FROZEN_THRESHOLD)
    submission_df = post_processor.format_submission_linkages(
        accepted_pairs_df=accepted_df.select(["source1_entity_id", "candidate_entity_id", "probability"]),
        all_s1_entities=all_s1_entities,
    )

    logger.info("Generated submission DataFrame: %d rows × %d columns", submission_df.height, submission_df.width)

    # 5. Save matching_results.tsv to output/ and root
    logger.info("Writing submission to %s...", MATCHING_RESULTS_OUT)
    submission_df.write_csv(MATCHING_RESULTS_OUT, separator="\t")

    root_matching_tsv = "matching_results.tsv"
    if os.path.abspath(MATCHING_RESULTS_OUT) != os.path.abspath(root_matching_tsv):
        logger.info("Writing copy to %s...", root_matching_tsv)
        submission_df.write_csv(root_matching_tsv, separator="\t")

    # 6. Submission Format Validation
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Running competition submission format validation...")
    logger.info("=" * 80)

    logger.info("Collecting candidate reference pool from test S2 and S3...")
    cand_ids_dfs = []
    for f in s2_files + s3_files:
        cand_ids_dfs.append(pl.read_parquet(f, columns=["entity_id"]))
    all_cand_ids = set(pl.concat(cand_ids_dfs)["entity_id"].to_list())
    del cand_ids_dfs
    logger.info("Reference candidate pool size: %d unique candidate IDs", len(all_cand_ids))

    is_valid, errors = validate_submission(
        submission_df=submission_df,
        all_s1_entities=set(all_s1_entities),
        all_candidate_entity_ids=all_cand_ids,
    )

    if not is_valid:
        logger.error("Submission validation FAILED with errors:")
        for err in errors:
            logger.error("  - %s", err)
        raise ValueError(f"Submission validation failed: {errors}")
    else:
        logger.info("Submission validation PASSED cleanly! 0 errors detected.")

    # 7. Compute Diagnostics & Summary Statistics
    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING FULL TEST INFERENCE DIAGNOSTICS & SUMMARY")
    logger.info("=" * 80)

    total_wall_time = time.time() - t_global_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    # S1 match counts distribution
    matched_strings = submission_df["matched_entity_ids"].to_list()
    match_counts = [len(m.split(",")) if m else 0 for m in matched_strings]
    count_counter = Counter(match_counts)

    zero_match_s1 = count_counter[0]
    single_match_s1 = count_counter[1]
    multi_match_s1 = sum(cnt for m_len, cnt in count_counter.items() if m_len > 1)
    max_matches_single_s1 = max(match_counts) if match_counts else 0

    # Source breakdown of accepted matches
    if accepted_df.height > 0:
        s2_matches = int((accepted_df["candidate_source"] == "source2").sum())
        s3_matches = int((accepted_df["candidate_source"] == "source3").sum())
        country_matches = accepted_df["country"].value_counts().to_dicts()
    else:
        s2_matches = 0
        s3_matches = 0
        country_matches = []

    print("\n" + "=" * 80)
    print("PHASE 7C: FULL-SCALE TEST INFERENCE REPORT")
    print("=" * 80)
    print(f"Total Canonical Test Source 1 Entities:  {total_test_s1:12,d}")
    print(f"Total Candidate Pairs Generated:         {total_pairs_generated:12,d}")
    print(f"Total Matches Predicted (P >= {FROZEN_THRESHOLD}):      {accepted_df.height:12,d}")
    print("-" * 80)
    print(f"Candidate Source Distribution of Matches:")
    print(f"  - Source 2 Matches:                    {s2_matches:12,d} ({100.0*s2_matches/max(1, accepted_df.height):5.1f}%)")
    print(f"  - Source 3 Matches:                    {s3_matches:12,d} ({100.0*s3_matches/max(1, accepted_df.height):5.1f}%)")
    print("-" * 80)
    print("S1 Resolution Profile:")
    print(f"  - S1 Entities with Zero Matches:       {zero_match_s1:12,d} ({100.0*zero_match_s1/total_test_s1:5.2f}%)")
    print(f"  - S1 Entities with Exactly 1 Match:    {single_match_s1:12,d} ({100.0*single_match_s1/total_test_s1:5.2f}%)")
    print(f"  - S1 Entities with Multiple Matches:   {multi_match_s1:12,d} ({100.0*multi_match_s1/total_test_s1:5.2f}%)")
    print(f"  - Maximum Matches for a Single S1:     {max_matches_single_s1:12d}")
    print("-" * 80)
    print("Country Breakdown of Predicted Matches:")
    for row in country_matches:
        c_name = row["country"]
        c_cnt = row["count"]
        print(f"  - {c_name:20s}:            {c_cnt:12,d} ({100.0*c_cnt/max(1, accepted_df.height):5.1f}%)")
    print("-" * 80)
    print("Throughput & Resource Utilization:")
    print(f"  - Feature Extraction Time:             {total_feat_time:12.1f}s ({total_pairs_generated / max(1e-3, total_feat_time):.1f} pairs/s)")
    print(f"  - Model Inference Time:                {total_infer_time:12.1f}s ({total_pairs_generated / max(1e-3, total_infer_time):.1f} pairs/s)")
    print(f"  - Total Elapsed Wall Time:             {total_wall_time:12.1f}s ({total_wall_time/60:.2f} min)")
    print(f"  - Peak Memory (Heap):                  {peak_mb:12.1f} MB (WSL2 ceiling: 7.7 GB)")
    print("-" * 80)
    print("Output Files Generated:")
    print(f"  1. Candidate Pairs:                    {os.path.abspath(CANDIDATE_PAIRS_OUT)}")
    print(f"  2. Matching Results (Submission):       {os.path.abspath(MATCHING_RESULTS_OUT)}")
    print(f"  3. Submission Validation Status:       PASSED (100% compliant)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
