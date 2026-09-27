"""
Phase 7B: Test Data and France Audit Script.

Audits:
1. Total record counts per source (Source 1, Source 2, Source 3).
2. Country distribution per source, specifically quantifying France vs US vs India.
3. Missing/null rates for business_name, business_address, country.
4. Entity ID namespace integrity (S1-, S2-, S3- prefixes, uniqueness).
5. Script detection, transliteration, and normalization behaviors on France records.
6. Sample blocking keys generated for France entities under frozen V4 blocker (A, C, D, E, F).
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import polars as pl

from src.candidate_generation.block_keys import generate_all_keys
from src.normalization.country_normalizer import normalize_country
from src.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_TEST_DIR = "data/raw/test"


def audit_test_files() -> Dict[str, Any]:
    """Scans all three raw test TSV files and gathers descriptive statistics."""
    sources = [
        ("source1", os.path.join(RAW_TEST_DIR, "test_source1.tsv"), "S1-"),
        ("source2", os.path.join(RAW_TEST_DIR, "test_source2.tsv"), "S2-"),
        ("source3", os.path.join(RAW_TEST_DIR, "test_source3.tsv"), "S3-"),
    ]

    stats: Dict[str, Any] = {}

    for src_name, file_path, expected_prefix in sources:
        logger.info("Auditing %s: %s", src_name.upper(), file_path)
        df = pl.read_csv(file_path, separator="\t")

        total_rows = df.height
        cols = df.columns
        unique_ids = df["entity_id"].n_unique()

        # Prefix check
        prefix_matches = df["entity_id"].str.starts_with(expected_prefix).sum()
        bad_prefixes = total_rows - prefix_matches

        # Null checks
        null_names = df["business_name"].null_count()
        null_addrs = df["business_address"].null_count()
        null_countries = df["country"].null_count()

        # Country distribution
        country_counts = df["country"].value_counts().to_dicts()
        country_dist = {row["country"]: row["count"] for row in country_counts}

        stats[src_name] = {
            "file": file_path,
            "total_records": total_rows,
            "unique_entity_ids": unique_ids,
            "bad_namespaces": bad_prefixes,
            "null_names": null_names,
            "null_addrs": null_addrs,
            "null_countries": null_countries,
            "country_dist": country_dist,
        }

    return stats


def audit_france_normalization() -> Dict[str, Any]:
    """Audits normalization, tokenization, and blocking keys on France test records."""
    logger.info("Sampling France records from test_source1.tsv...")
    s1_path = os.path.join(RAW_TEST_DIR, "test_source1.tsv")
    df_fr = (
        pl.scan_csv(s1_path, separator="\t")
        .filter(pl.col("country") == "France")
        .limit(100)
        .collect()
    )

    pipeline = PreprocessingPipeline()
    sample_audits = []

    for row in df_fr.iter_rows(named=True):
        from src.preprocessing.schema import create_canonical_record
        rec = create_canonical_record(
            source="source1",
            entity_id=row["entity_id"],
            business_name=row["business_name"],
            business_address=row["business_address"],
            country=row["country"],
        )
        processed = pipeline.process_record(rec)
        keys = generate_all_keys(processed, active_keys=["A", "C", "D", "E", "F"])

        sample_audits.append({
            "entity_id": row["entity_id"],
            "raw_name": row["business_name"],
            "norm_name": processed["business_name_normalized"],
            "raw_addr": row["business_address"],
            "norm_addr": processed["business_address_normalized"],
            "country_norm": processed["country_normalized"],
            "keys_generated": len(keys),
            "sample_keys": list(keys)[:3],
        })

    return {
        "france_sample_size": len(sample_audits),
        "sample_audits": sample_audits,
    }


def main():
    logger.info("Starting Phase 7B Test Data & France Audit...")
    stats = audit_test_files()
    fr_audit = audit_france_normalization()

    print("\n" + "=" * 80)
    print("PHASE 7B: TEST DATA & FRANCE AUDIT SUMMARY")
    print("=" * 80)

    for src, s in stats.items():
        print(f"\n--- {src.upper()} ---")
        print(f"File:               {s['file']}")
        print(f"Total Records:      {s['total_records']:,}")
        print(f"Unique Entity IDs:  {s['unique_entity_ids']:,}")
        print(f"Namespace Errors:   {s['bad_namespaces']}")
        print(f"Null Names:         {s['null_names']}")
        print(f"Null Addresses:     {s['null_addrs']}")
        print(f"Null Countries:     {s['null_countries']}")
        print("Country Breakdown:")
        for c, count in s["country_dist"].items():
            pct = 100.0 * count / s["total_records"]
            print(f"  - {c:15s}: {count:10,d} ({pct:6.2f}%)")

    print("\n" + "=" * 80)
    print("FRANCE NORMALIZATION & BLOCKING AUDIT (First 5 Samples)")
    print("=" * 80)
    for i, ex in enumerate(fr_audit["sample_audits"][:5]):
        print(f"\n[Sample {i+1}] Entity ID: {ex['entity_id']}")
        print(f"  Raw Name:      {ex['raw_name']}")
        print(f"  Norm Name:     {ex['norm_name']}")
        print(f"  Raw Addr:      {ex['raw_addr']}")
        print(f"  Norm Addr:     {ex['norm_addr']}")
        print(f"  Norm Country:  {ex['country_norm']}")
        print(f"  Total Keys:    {ex['keys_generated']}")
        print(f"  Sample Keys:   {ex['sample_keys']}")


if __name__ == "__main__":
    main()
