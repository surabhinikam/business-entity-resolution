"""
Deterministic Equivalence Test for India Candidate Generation.

Validates that the memory-bounded candidate blocker produces 100% bit-exact
identical candidate pairs, key provenance sets, and oversized block decisions
as the reference production BlockIndex implementation on real India test data.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import polars as pl
from src.candidate_generation.block_index import BlockIndex
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


class MemoryBoundedIndiaBlocker:
    """
    Two-pass memory-bounded blocker for large-scale candidate generation.
    Strictly preserves V4 blocking semantics (A, C, D, E, F; max block size 5000).
    """

    def __init__(self, max_block_size: int = MAX_BLOCK_SIZE):
        self.max_block_size = max_block_size
        self.oversized_keys: Set[str] = set()

    def detect_oversized_keys(
        self,
        s1_records: List[Dict[str, Any]],
        cand_records: List[Dict[str, Any]],
    ) -> Set[str]:
        """
        Pass 1: Computes global key counts for S1 and Candidates without storing
        candidate entity lists in memory. Identifies oversized blocks (> 5000).
        """
        s1_counts: Counter[str] = Counter()
        for rec in s1_records:
            keys = generate_all_keys(rec, active_keys=ACTIVE_KEYS).get("ALL", set())
            for k in keys:
                s1_counts[k] += 1

        cand_counts: Counter[str] = Counter()
        s1_keys_set = set(s1_counts.keys())
        for rec in cand_records:
            keys = generate_all_keys(rec, active_keys=ACTIVE_KEYS).get("ALL", set())
            for k in (keys & s1_keys_set):
                cand_counts[k] += 1

        self.oversized_keys = {
            k for k, s1_c in s1_counts.items()
            if s1_c * cand_counts.get(k, 0) > self.max_block_size
        }
        return self.oversized_keys

    def generate_pairs_chunked(
        self,
        s1_records: List[Dict[str, Any]],
        cand_records: List[Dict[str, Any]],
        s1_chunk_size: int = 500,
    ) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], Set[str]]]:
        """
        Pass 2: Builds candidate index ONLY for valid (non-oversized, active S1) keys.
        Generates pairs chunk-by-chunk for S1 records.
        """
        # 1. Determine all valid S1 keys (present in S1, not oversized)
        s1_all_keys: Set[str] = set()
        for rec in s1_records:
            keys = generate_all_keys(rec, active_keys=ACTIVE_KEYS).get("ALL", set())
            s1_all_keys.update(keys)

        valid_keys = s1_all_keys - self.oversized_keys

        # 2. Build candidate index ONLY for valid_keys
        cand_index: Dict[str, List[str]] = defaultdict(list)
        for rec in cand_records:
            eid = rec["entity_id"]
            keys = generate_all_keys(rec, active_keys=ACTIVE_KEYS).get("ALL", set())
            for k in (keys & valid_keys):
                cand_index[k].append(eid)

        # 3. Generate candidate pairs chunk-by-chunk
        all_pairs: Set[Tuple[str, str]] = set()
        all_provenance: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        n_s1 = len(s1_records)
        for start_idx in range(0, n_s1, s1_chunk_size):
            chunk = s1_records[start_idx : start_idx + s1_chunk_size]
            s1_chunk_index: Dict[str, List[str]] = defaultdict(list)
            for rec in chunk:
                eid = rec["entity_id"]
                keys = generate_all_keys(rec, active_keys=ACTIVE_KEYS).get("ALL", set())
                for k in (keys & valid_keys):
                    s1_chunk_index[k].append(eid)

            for k, s1_ids in s1_chunk_index.items():
                if k in cand_index:
                    key_label = k.split("||")[0] if "||" in k else "?"
                    cand_ids = cand_index[k]
                    for s1_id in s1_ids:
                        for c_id in cand_ids:
                            pair = (s1_id, c_id)
                            all_pairs.add(pair)
                            all_provenance[pair].add(key_label)

        return all_pairs, dict(all_provenance)


def run_equivalence_test():
    print("=" * 80)
    print("RUNNING DETERMINISTIC EQUIVALENCE TEST FOR INDIA BLOCKING")
    print("=" * 80)

    # Load 2,000 S1 India and 10,000 Candidate India records
    s1_files = sorted(glob.glob("data/processed/test/source1/*.parquet"))
    s2_files = sorted(glob.glob("data/processed/test/source2/*.parquet"))
    s3_files = sorted(glob.glob("data/processed/test/source3/*.parquet"))

    print("Loading test sample for India...")
    s1_df = pl.concat([
        pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == "india")
        for f in s1_files
    ]).head(2000)

    cand_df = pl.concat([
        pl.read_parquet(f, columns=BLOCKING_COLS).filter(pl.col("country_normalized") == "india")
        for f in s2_files + s3_files
    ]).head(10000)

    print(f"Sample loaded: {s1_df.height} S1 records, {cand_df.height} Candidate records.")

    s1_records = s1_df.to_dicts()
    cand_records = cand_df.to_dicts()

    # 1. Reference Implementation: Production BlockIndex
    print("\n[1] Running Reference Production BlockIndex...")
    t0 = time.time()
    ref_blocker = BlockIndex(max_block_size=MAX_BLOCK_SIZE)
    for rec in s1_records:
        ref_blocker.add_s1_record(rec["entity_id"], rec, active_keys=ACTIVE_KEYS)
    for rec in cand_records:
        ref_blocker.add_candidate_record(rec["entity_id"], rec, active_keys=ACTIVE_KEYS)
    ref_pairs, ref_prov = ref_blocker.generate_pairs(cap_blocks=True)
    t_ref = time.time() - t0
    ref_oversized = {b["key_string"] for b in ref_blocker.oversized_blocks}
    print(f"  Reference generated {len(ref_pairs)} pairs, {len(ref_oversized)} oversized blocks in {t_ref:.2f}s")

    # 2. Optimized Implementation: MemoryBoundedIndiaBlocker
    print("\n[2] Running MemoryBoundedIndiaBlocker (chunk_size=500)...")
    t1 = time.time()
    opt_blocker = MemoryBoundedIndiaBlocker(max_block_size=MAX_BLOCK_SIZE)
    opt_oversized = opt_blocker.detect_oversized_keys(s1_records, cand_records)
    opt_pairs, opt_prov = opt_blocker.generate_pairs_chunked(s1_records, cand_records, s1_chunk_size=500)
    t_opt = time.time() - t1
    print(f"  Optimized generated {len(opt_pairs)} pairs, {len(opt_oversized)} oversized blocks in {t_opt:.2f}s")

    # 3. Assert Equivalence
    print("\n[3] Verifying Equivalence (Standard Cap=5000)...")
    assert ref_oversized == opt_oversized, f"Oversized keys mismatch! Ref: {len(ref_oversized)}, Opt: {len(opt_oversized)}"
    print(f"  ✓ Oversized keys match 100%: exactly {len(ref_oversized)} oversized blocks.")

    assert ref_pairs == opt_pairs, f"Candidate pairs mismatch! Ref: {len(ref_pairs)}, Opt: {len(opt_pairs)}"
    print(f"  ✓ Candidate pairs match 100%: exactly {len(ref_pairs)} pairs.")

    assert ref_prov == opt_prov, f"Provenance mismatch! Ref keys: {len(ref_prov)}, Opt keys: {len(opt_prov)}"
    print(f"  ✓ Key provenance matches 100% across all pairs.")

    # 4. Stress Test Capping Equivalence with Tight Cap (max_block_size=10)
    print("\n[4] Stress Testing Capping with tight threshold (max_block_size=10)...")
    ref_blocker_tight = BlockIndex(max_block_size=10)
    for rec in s1_records:
        ref_blocker_tight.add_s1_record(rec["entity_id"], rec, active_keys=ACTIVE_KEYS)
    for rec in cand_records:
        ref_blocker_tight.add_candidate_record(rec["entity_id"], rec, active_keys=ACTIVE_KEYS)
    ref_pairs_tight, ref_prov_tight = ref_blocker_tight.generate_pairs(cap_blocks=True)
    ref_oversized_tight = {b["key_string"] for b in ref_blocker_tight.oversized_blocks}

    opt_blocker_tight = MemoryBoundedIndiaBlocker(max_block_size=10)
    opt_oversized_tight = opt_blocker_tight.detect_oversized_keys(s1_records, cand_records)
    opt_pairs_tight, opt_prov_tight = opt_blocker_tight.generate_pairs_chunked(s1_records, cand_records, s1_chunk_size=500)

    assert ref_oversized_tight == opt_oversized_tight, f"Tight cap oversized keys mismatch: {len(ref_oversized_tight)} vs {len(opt_oversized_tight)}"
    print(f"  ✓ Tight cap oversized keys match 100%: exactly {len(ref_oversized_tight)} oversized blocks detected.")

    assert ref_pairs_tight == opt_pairs_tight, f"Tight cap pairs mismatch: {len(ref_pairs_tight)} vs {len(opt_pairs_tight)}"
    print(f"  ✓ Tight cap pairs match 100%: exactly {len(ref_pairs_tight)} pairs.")

    assert ref_prov_tight == opt_prov_tight, f"Tight cap provenance mismatch: {len(ref_prov_tight)} vs {len(opt_prov_tight)}"
    print(f"  ✓ Tight cap provenance matches 100% across all pairs.")

    print("\n" + "=" * 80)
    print("ALL EQUIVALENCE CHECKS PASSED CLEANLY! MATHEMATICAL IDENTITY PROVEN.")
    print("=" * 80)


if __name__ == "__main__":
    run_equivalence_test()
