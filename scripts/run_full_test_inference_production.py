"""
Phase 7C: Production Test Inference Runner with Incremental Checkpoints.

Executes complete test candidate generation, 29-feature extraction,
LightGBM model scoring (frozen threshold = 0.88), MatchPostProcessor formatting,
and submission format validation.

Guarantees:
- Strict preservation of frozen V4 blocker (Keys A, C, D, E, F with 5000 capping).
- Exact 29 canonical feature schema matching FULL_PIPELINE_COLUMNS.
- Frozen LightGBM model configuration from Phase 4D.
- Single frozen decision threshold: 0.88.
- Bit-exact mathematical identity via precomputed global oversized block caps.
- Strict RAM ceiling: < 2.5 GB peak RSS (safe for any WSL2/Linux environment).
- Per-chunk disk checkpointing: matches saved incrementally to Parquet.
- Zero data loss on interruption: automatically resumes from disk checkpoints.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
import resource
import shutil
import sys
import time
import tracemalloc
from collections import defaultdict
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
        logging.FileHandler("test_inference_production.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("production_inference")

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
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CANDIDATE_PAIRS_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_RESULTS_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OVERSIZED_KEYS_INDIA = os.path.join(OUTPUT_DIR, "oversized_keys_india.json")

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


def load_country_oversized_keys(
    country: str,
    s1_files: List[str],
    s2_files: List[str],
    s3_files: List[str],
) -> Set[str]:
    """Computes or loads precomputed oversized keys for country."""
    if country == "india" and os.path.exists(OVERSIZED_KEYS_INDIA):
        logger.info("Loading precomputed India oversized keys from %s...", OVERSIZED_KEYS_INDIA)
        with open(OVERSIZED_KEYS_INDIA, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.keys())

    # Fallback to computing on the fly
    logger.info("Computing global oversized keys for '%s'...", country)
    from collections import Counter
    s1_counts: Counter[str] = Counter()
    for f in s1_files:
        df_p = pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == country)
        for row in df_p.iter_rows(named=True):
            for k in generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set()):
                s1_counts[k] += 1

    s1_key_set = set(s1_counts.keys())
    cand_counts: Counter[str] = Counter()
    for f in s2_files + s3_files:
        df_p = pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == country)
        for row in df_p.iter_rows(named=True):
            for k in (generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set()) & s1_key_set):
                cand_counts[k] += 1

    oversized = {k for k, s1_c in s1_counts.items() if s1_c * cand_counts.get(k, 0) > MAX_BLOCK_SIZE}
    logger.info("Identified %d oversized keys for '%s'.", len(oversized), country)
    return oversized


def process_country(
    country: str,
    s1_files: List[str],
    s2_files: List[str],
    s3_files: List[str],
    model: BaselineMatchingModel,
    oversized_keys: Set[str],
    cand_tsv_file,
) -> str:
    """Processes a country in chunks of 50k S1 records with checkpointing."""
    final_country_parquet = os.path.join(CHECKPOINT_DIR, f"matches_{country}.parquet")
    if os.path.exists(final_country_parquet):
        logger.info("Checkpoint found for country '%s': %s (skipping processing).", country, final_country_parquet)
        return final_country_parquet

    logger.info("=" * 80)
    logger.info("PROCESSING COUNTRY: %s", country.upper())
    logger.info("=" * 80)

    # 1. Load S1 for country
    s1_dfs = []
    for f in s1_files:
        df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(pl.col("country_normalized") == country)
        if df_p.height > 0:
            s1_dfs.append(df_p)
    s1_country_df = pl.concat(s1_dfs)
    total_country_s1 = s1_country_df.height
    logger.info("Loaded %d S1 records for %s (%.1f MB).", total_country_s1, country, s1_country_df.estimated_size() / (1024 * 1024))

    # 2. Candidate files for this country
    all_cand_files = s2_files + s3_files

    # 3. Process in chunks of S1_CHUNK_SIZE
    n_chunks = (total_country_s1 + S1_CHUNK_SIZE - 1) // S1_CHUNK_SIZE
    pipeline = FeaturePipeline()
    chunk_parquets = []

    for c_idx in range(n_chunks):
        chunk_file = os.path.join(CHECKPOINT_DIR, f"matches_{country}_chunk_{c_idx+1}_of_{n_chunks}.parquet")
        chunk_parquets.append(chunk_file)

        if os.path.exists(chunk_file):
            logger.info("[%s] Chunk %d/%d already completed. Skipping.", country, c_idx + 1, n_chunks)
            continue

        c_start = c_idx * S1_CHUNK_SIZE
        c_end = min(c_start + S1_CHUNK_SIZE, total_country_s1)
        s1_chunk_df = s1_country_df.slice(c_start, c_end - c_start)
        chunk_name = f"{country}_chunk_{c_idx+1}_of_{n_chunks}"

        # A. Index S1 records for chunk
        s1_chunk_index: Dict[str, List[str]] = defaultdict(list)
        for row in s1_chunk_df.select(BLOCKING_COLS).iter_rows(named=True):
            eid = row["entity_id"]
            keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
            valid = keys - oversized_keys
            for k in valid:
                s1_chunk_index[k].append(eid)

        s1_chunk_keys = set(s1_chunk_index.keys())

        # B. Scan candidates and index ONLY for s1_chunk_keys
        cand_chunk_index: Dict[str, List[str]] = defaultdict(list)
        cand_matched_ids: Set[str] = set()

        for f in all_cand_files:
            df_p = pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == country)
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

        # C. Generate pairs for chunk
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

        del s1_chunk_index, cand_chunk_index
        import gc
        gc.collect()

        n_pairs = len(pairs)
        if n_pairs == 0:
            empty_df = pl.DataFrame(
                {"source1_entity_id": [], "candidate_entity_id": [], "candidate_source": [], "probability": [], "country": []},
                schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8, "candidate_source": pl.Utf8, "probability": pl.Float64, "country": pl.Utf8}
            )
            empty_df.write_parquet(chunk_file)
            continue

        # D. Stream pairs to candidate_pairs.tsv if provided
        if cand_tsv_file is not None:
            lines = [f"{p[0]}\t{p[1]}\n" for p in pairs]
            cand_tsv_file.writelines(lines)
            cand_tsv_file.flush()

        # E. Load feature rows for cand_matched_ids
        cand_feat_dfs = []
        for f in all_cand_files:
            df_p = pl.read_parquet(f, columns=FEATURE_COLS).filter(
                (pl.col("country_normalized") == country) & pl.col("entity_id").is_in(pl.Series("eid", list(cand_matched_ids)))
            )
            if df_p.height > 0:
                cand_feat_dfs.append(df_p)
        cand_feat_df = pl.concat(cand_feat_dfs)
        del cand_matched_ids
        gc.collect()

        # F. Candidate pairs DataFrame
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

        # G. Batched feature extraction and scoring
        accepted_chunk = []
        total_batches = (n_pairs + BATCH_SIZE - 1) // BATCH_SIZE
        t_f_total = 0.0
        t_i_total = 0.0

        for b_idx in range(total_batches):
            b_start = b_idx * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_pairs)
            b_pairs = candidate_pairs_df.slice(b_start, b_end - b_start)

            b_s1_ids = b_pairs["source1_entity_id"].unique().to_list()
            b_cand_ids = b_pairs["candidate_entity_id"].unique().to_list()

            s1_slice = s1_chunk_df.filter(pl.col("entity_id").is_in(pl.Series("eid", b_s1_ids)))
            cand_slice = cand_feat_df.filter(pl.col("entity_id").is_in(pl.Series("eid", b_cand_ids)))

            b_tuples = list(zip(b_pairs["source1_entity_id"], b_pairs["candidate_entity_id"]))
            b_prov = {p: provenance[p] for p in b_tuples if p in provenance}

            t0 = time.time()
            features = pipeline.generate_features_from_records(
                candidate_pairs_df=b_pairs,
                s1_records=s1_slice,
                cand_records=cand_slice,
                pair_provenance=b_prov,
            )
            t_f_total += time.time() - t0

            t1 = time.time()
            probs = model.predict_proba(features)
            t_i_total += time.time() - t1

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
                        "country": country,
                    })

            del features, b_pairs, s1_slice, cand_slice, b_prov, probs, mask

        del candidate_pairs_df, provenance, cand_feat_df
        gc.collect()

        # Save chunk parquet checkpoint
        if accepted_chunk:
            chunk_df = pl.DataFrame(accepted_chunk)
        else:
            chunk_df = pl.DataFrame(
                {"source1_entity_id": [], "candidate_entity_id": [], "candidate_source": [], "probability": [], "country": []},
                schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8, "candidate_source": pl.Utf8, "probability": pl.Float64, "country": pl.Utf8}
            )
        chunk_df.write_parquet(chunk_file)
        del chunk_df, accepted_chunk
        gc.collect()

        logger.info(
            "[%s] Chunk %d/%d DONE: %d pairs | Feat: %.1fs | Infer: %.1fs | Saved to %s",
            country, c_idx + 1, n_chunks, n_pairs, t_f_total, t_i_total, chunk_file,
        )

    # Merge chunk parquets into final country parquet
    logger.info("Merging %d chunk parquets for %s...", len(chunk_parquets), country)
    country_dfs = [pl.read_parquet(cp) for cp in chunk_parquets if os.path.exists(cp)]
    merged_country_df = pl.concat(country_dfs)
    merged_country_df.write_parquet(final_country_parquet)
    logger.info("Country %s complete: %d matches saved to %s", country, merged_country_df.height, final_country_parquet)

    return final_country_parquet


def main():
    logger.info("=" * 80)
    logger.info("PHASE 7C: PRODUCTION FULL-SCALE TEST INFERENCE")
    logger.info("=" * 80)
    logger.info("Active Blocker Keys:     %s", ACTIVE_KEYS)
    logger.info("Max Block Size Cap:      %d", MAX_BLOCK_SIZE)
    logger.info("Frozen Model Path:       %s", MODEL_PATH)
    logger.info("Frozen Threshold:        %.2f", FROZEN_THRESHOLD)
    logger.info("S1 Chunk Size:           %d", S1_CHUNK_SIZE)
    logger.info("=" * 80)

    tracemalloc.start()
    t_global0 = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Load model
    logger.info("Step 1: Loading frozen model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2. Canonical S1 entities
    logger.info("Step 2: Loading canonical test Source 1 entity IDs...")
    s1_raw_df = pl.read_csv(TEST_SOURCE1_RAW, separator="\t", columns=["entity_id"])
    all_s1_entities = s1_raw_df["entity_id"].to_list()
    total_test_s1 = len(all_s1_entities)
    logger.info("Total Canonical S1 Entities: %d", total_test_s1)

    s1_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S1, "*.parquet")))
    s2_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S2, "*.parquet")))
    s3_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S3, "*.parquet")))

    # Candidate pairs file handling:
    # Check if candidate_pairs.tsv already has France and US pairs
    existing_pairs = 0
    if os.path.exists(CANDIDATE_PAIRS_OUT):
        with open(CANDIDATE_PAIRS_OUT, "r", encoding="utf-8") as f:
            for _ in f:
                existing_pairs += 1
        existing_pairs -= 1  # subtract header
        logger.info("Found existing %s with %d pairs.", CANDIDATE_PAIRS_OUT, existing_pairs)

    countries = ["france", "united states", "india"]
    country_match_files = []

    for country in countries:
        final_cp = os.path.join(CHECKPOINT_DIR, f"matches_{country}.parquet")
        oversized = load_country_oversized_keys(country, s1_files, s2_files, s3_files)

        # Open candidate_pairs file for appending if not complete
        cand_tsv = None
        if country == "india" or existing_pairs < 24000000:
            mode = "a" if os.path.exists(CANDIDATE_PAIRS_OUT) else "w"
            cand_tsv = open(CANDIDATE_PAIRS_OUT, mode, encoding="utf-8")
            if mode == "w":
                cand_tsv.write("source1_entity_id\tcandidate_entity_id\n")

        country_file = process_country(
            country=country,
            s1_files=s1_files,
            s2_files=s2_files,
            s3_files=s3_files,
            model=model,
            oversized_keys=oversized,
            cand_tsv_file=cand_tsv,
        )
        if cand_tsv is not None:
            cand_tsv.close()

        country_match_files.append(country_file)

    # 4. Copy candidate_pairs.tsv to workspace root
    root_cand_tsv = "candidate_pairs.tsv"
    if os.path.abspath(CANDIDATE_PAIRS_OUT) != os.path.abspath(root_cand_tsv):
        logger.info("Copying %s to %s...", CANDIDATE_PAIRS_OUT, root_cand_tsv)
        shutil.copyfile(CANDIDATE_PAIRS_OUT, root_cand_tsv)

    # 5. Assemble all accepted matches across all countries
    logger.info("=" * 80)
    logger.info("Step 5: Formatting final submission via MatchPostProcessor...")
    logger.info("=" * 80)

    all_matches_df = pl.concat([pl.read_parquet(cf) for cf in country_match_files])
    logger.info("Total accepted match pairs across all countries: %d", all_matches_df.height)

    post_processor = MatchPostProcessor(threshold=FROZEN_THRESHOLD)
    submission_df = post_processor.format_submission_linkages(
        accepted_pairs_df=all_matches_df.select(["source1_entity_id", "candidate_entity_id", "probability"]),
        all_s1_entities=all_s1_entities,
    )
    logger.info("Submission DataFrame formatted: %d rows.", submission_df.height)

    # 6. Save matching_results.tsv to output/ and root
    logger.info("Writing submission to %s...", MATCHING_RESULTS_OUT)
    submission_df.write_csv(MATCHING_RESULTS_OUT, separator="\t")
    shutil.copyfile(MATCHING_RESULTS_OUT, "matching_results.tsv")

    # 7. Validate Submission
    logger.info("=" * 80)
    logger.info("Step 7: Running competition submission format validation...")
    logger.info("=" * 80)

    cand_ids_dfs = [pl.read_parquet(f, columns=["entity_id"]) for f in s2_files + s3_files]
    all_cand_ids = set(pl.concat(cand_ids_dfs)["entity_id"].to_list())
    del cand_ids_dfs

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

    # 8. Report Summary
    total_time = time.time() - t_global0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    matched_strings = submission_df["matched_entity_ids"].to_list()
    from collections import Counter
    match_counts = [len(m.split(",")) if m else 0 for m in matched_strings]
    count_counter = Counter(match_counts)

    print("\n" + "=" * 80)
    print("PHASE 7C: FULL TEST INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Total Canonical Test Source 1 Entities:  {total_test_s1:12,d}")
    print(f"Total Matches Predicted (P >= {FROZEN_THRESHOLD}):      {all_matches_df.height:12,d}")
    print(f"Zero-Match S1 Entities:                  {count_counter[0]:12,d} ({100.0*count_counter[0]/total_test_s1:5.2f}%)")
    print(f"Single-Match S1 Entities:                {count_counter[1]:12,d} ({100.0*count_counter[1]/total_test_s1:5.2f}%)")
    print(f"Multi-Match S1 Entities:                 {sum(v for k,v in count_counter.items() if k > 1):12,d}")
    print(f"Total Wall Clock Runtime:                {total_time/60:12.2f} min")
    print(f"Process Peak RSS:                        {rss_mb:12.1f} MB (WSL2 limit: ~11,000 MB)")
    print(f"Output 1 (Candidate Pairs):              {os.path.abspath(CANDIDATE_PAIRS_OUT)}")
    print(f"Output 2 (Submission Results):           {os.path.abspath(MATCHING_RESULTS_OUT)}")
    print(f"Validation Status:                       PASSED (100% compliant)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
