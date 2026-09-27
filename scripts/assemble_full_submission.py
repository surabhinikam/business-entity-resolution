"""
Phase 7C: Submission Assembler and Validator.

Combines France, United States, and India match checkpoints, formats the complete
final matching_results.tsv for all 1,732,544 canonical test S1 entities via MatchPostProcessor,
verifies candidate_pairs.tsv integrity, and executes competition submission format validation.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import sys
import time
from collections import Counter
from typing import List, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import polars as pl
from src.models.post_processing import MatchPostProcessor
from src.validation.validators import validate_submission

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("assemble_submission")

FROZEN_THRESHOLD = 0.88
TEST_SOURCE1_RAW = "data/raw/test/test_source1.tsv"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CANDIDATE_PAIRS_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_RESULTS_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
PROCESSED_TEST_S2 = "data/processed/test/source2"
PROCESSED_TEST_S3 = "data/processed/test/source3"


def main():
    logger.info("=" * 80)
    logger.info("PHASE 7C: FINAL SUBMISSION ASSEMBLY & VALIDATION")
    logger.info("=" * 80)

    # 1. Load canonical test S1 entities
    logger.info("Step 1: Loading canonical test Source 1 entity IDs...")
    s1_raw_df = pl.read_csv(TEST_SOURCE1_RAW, separator="\t", columns=["entity_id"])
    all_s1_entities = s1_raw_df["entity_id"].to_list()
    total_test_s1 = len(all_s1_entities)
    logger.info("Loaded %d canonical test S1 entities.", total_test_s1)
    assert total_test_s1 == 1732544, f"Expected 1,732,544 S1 entities, found {total_test_s1}"

    # 2. Check match checkpoints
    logger.info("Step 2: Locating match checkpoints in %s...", CHECKPOINT_DIR)
    match_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "matches_*.parquet")))
    # Exclude chunk files if merged country file exists
    merged_files = [f for f in match_files if not "_chunk_" in f]
    if merged_files:
        match_files = merged_files

    logger.info("Found %d match parquet files: %s", len(match_files), [os.path.basename(f) for f in match_files])
    if not match_files:
        raise FileNotFoundError(f"No match files found in {CHECKPOINT_DIR}")

    # Load and combine all accepted matches
    match_dfs = [pl.read_parquet(f) for f in match_files]
    all_matches_df = pl.concat(match_dfs)
    logger.info("Total combined accepted matches: %d rows.", all_matches_df.height)

    # 3. Format complete submission via MatchPostProcessor
    logger.info("Step 3: Formatting submission linkages via MatchPostProcessor (threshold=%.2f)...", FROZEN_THRESHOLD)
    post_processor = MatchPostProcessor(threshold=FROZEN_THRESHOLD)
    submission_df = post_processor.format_submission_linkages(
        accepted_pairs_df=all_matches_df.select(["source1_entity_id", "candidate_entity_id", "probability"]),
        all_s1_entities=all_s1_entities,
    )
    logger.info("Generated submission DataFrame: %d rows x %d cols.", submission_df.height, submission_df.width)

    # 4. Save matching_results.tsv to output/ and root
    logger.info("Step 4: Writing submission to %s...", MATCHING_RESULTS_OUT)
    submission_df.write_csv(MATCHING_RESULTS_OUT, separator="\t")
    shutil.copyfile(MATCHING_RESULTS_OUT, "matching_results.tsv")
    logger.info("Written matching_results.tsv to output/ and root workspace.")

    # 5. Verify candidate_pairs.tsv
    logger.info("Step 5: Verifying candidate_pairs.tsv...")
    if not os.path.exists(CANDIDATE_PAIRS_OUT):
        raise FileNotFoundError(f"Missing {CANDIDATE_PAIRS_OUT}")
    shutil.copyfile(CANDIDATE_PAIRS_OUT, "candidate_pairs.tsv")
    logger.info("Copied %s to root workspace candidate_pairs.tsv.", CANDIDATE_PAIRS_OUT)

    # 6. Validate Submission Format
    logger.info("Step 6: Running official submission format validation...")
    s2_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S2, "*.parquet")))
    s3_files = sorted(glob.glob(os.path.join(PROCESSED_TEST_S3, "*.parquet")))
    cand_ids_dfs = [pl.read_parquet(f, columns=["entity_id"]) for f in s2_files + s3_files]
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

    # 7. Summary Diagnostics
    matched_strings = submission_df["matched_entity_ids"].to_list()
    match_counts = [len(m.split(",")) if m else 0 for m in matched_strings]
    count_counter = Counter(match_counts)

    print("\n" + "=" * 80)
    print("PHASE 7C: COMPLETE SUBMISSION VERIFICATION REPORT")
    print("=" * 80)
    print(f"Total Canonical Test S1 Entities:        {total_test_s1:12,d}")
    print(f"Total Predicted Matches:                 {all_matches_df.height:12,d}")
    print(f"S1 Entities with 0 Matches:              {count_counter[0]:12,d} ({100.0*count_counter[0]/total_test_s1:5.2f}%)")
    print(f"S1 Entities with Exactly 1 Match:        {count_counter[1]:12,d} ({100.0*count_counter[1]/total_test_s1:5.2f}%)")
    print(f"S1 Entities with Multiple Matches:       {sum(v for k,v in count_counter.items() if k > 1):12,d} ({100.0*sum(v for k,v in count_counter.items() if k > 1)/total_test_s1:5.2f}%)")
    print(f"Maximum Matches for a Single S1:         {max(match_counts):12d}")
    print("-" * 80)
    print("Output Files:")
    print(f"  1. candidate_pairs.tsv:                {os.path.abspath('candidate_pairs.tsv')}")
    print(f"  2. matching_results.tsv:               {os.path.abspath('matching_results.tsv')}")
    print(f"  Validation Status:                     PASSED (100% compliant, 0 errors)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
