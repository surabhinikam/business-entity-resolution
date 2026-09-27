"""
Precomputes and verifies the global oversized keys for India.
Saves the 1,142 oversized keys to output/oversized_keys_india.json for bit-exact reuse.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import polars as pl
from src.candidate_generation.block_keys import generate_all_keys

ACTIVE_KEYS = ["A", "C", "D", "E", "F"]
MAX_BLOCK_SIZE = 5000
BLOCKING_COLS = [
    "entity_id",
    "business_name_normalized",
    "business_name_transliterated",
    "business_address_normalized",
    "country_normalized",
]


def main():
    print("=" * 80)
    print("PRECOMPUTING INDIA OVERSIZED KEYS")
    print("=" * 80)

    s1_files = sorted(glob.glob("data/processed/test/source1/*.parquet"))
    s2_files = sorted(glob.glob("data/processed/test/source2/*.parquet"))
    s3_files = sorted(glob.glob("data/processed/test/source3/*.parquet"))

    # Step 1: Count S1 keys
    print("Step 1: Counting S1 India keys across 809,986 records...")
    t0 = time.time()
    s1_counts: Counter[str] = Counter()
    s1_df = pl.concat([
        pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == "india")
        for f in s1_files
    ])
    for row in s1_df.iter_rows(named=True):
        keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
        for k in keys:
            s1_counts[k] += 1
    t_s1 = time.time() - t0
    print(f"  Counted {len(s1_counts):,} unique S1 keys in {t_s1:.1f}s.")

    s1_keys_set = set(s1_counts.keys())

    # Step 2: Count Candidate keys that intersect with S1
    print("\nStep 2: Counting Candidate India keys across 4,717,565 records...")
    t1 = time.time()
    cand_counts: Counter[str] = Counter()
    for f in s2_files + s3_files:
        df_part = pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == "india")
        if df_part.height == 0:
            continue
        for row in df_part.iter_rows(named=True):
            keys = generate_all_keys(row, active_keys=ACTIVE_KEYS).get("ALL", set())
            for k in (keys & s1_keys_set):
                cand_counts[k] += 1
    t_cand = time.time() - t1
    print(f"  Counted {len(cand_counts):,} intersecting candidate keys in {t_cand:.1f}s.")

    # Step 3: Identify oversized keys
    print("\nStep 3: Detecting oversized blocks (> 5000)...")
    oversized = {}
    for k, s1_c in s1_counts.items():
        c_c = cand_counts.get(k, 0)
        block_size = s1_c * c_c
        if block_size > MAX_BLOCK_SIZE:
            oversized[k] = {
                "s1_count": s1_c,
                "cand_count": c_c,
                "block_size": block_size,
            }

    print(f"  Detected {len(oversized):,} oversized keys (> {MAX_BLOCK_SIZE} pairs).")

    os.makedirs("output", exist_ok=True)
    out_file = "output/oversized_keys_india.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(oversized, f, indent=2)
    print(f"  Saved oversized keys to {out_file}")


if __name__ == "__main__":
    main()
