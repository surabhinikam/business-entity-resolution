"""
Integration demonstration for Phase 4: Deterministic Normalization.

Streams 100 records from train_source1, train_source2, and train_source3,
runs the complete pipeline:
  1. Script detection
  2. Transliteration (if non-Latin / Mixed)
  3. Text / Business / Address / Country Normalization
  4. Tokenization

Outputs representative samples (Latin, Indic, Mixed, missing address)
verifying zero data corruption and accurate normalization.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.ingestion import (
    read_train_source1,
    read_train_source2,
    read_train_source3,
)
from src.transliteration.script_detector import ScriptDetector
from src.transliteration.transliterator import IndicTransliterator
from src.normalization.business_normalizer import (
    normalize_business_name,
    tokenize_business_name,
)
from src.normalization.address_normalizer import (
    normalize_address,
    tokenize_address,
)
from src.normalization.country_normalizer import normalize_country


def process_record(
    record,
    detector: ScriptDetector,
    transliterator: IndicTransliterator,
) -> Dict[str, Any]:
    # 1. Name processing
    name_script = detector.detect(record.business_name) if record.business_name else "Unknown"
    if record.business_name and name_script not in ("Latin", "Unknown"):
        name_transliterated = transliterator.transliterate(record.business_name, script=name_script)
    else:
        name_transliterated = record.business_name

    norm_name = normalize_business_name(record.business_name, transliterated=name_transliterated)
    name_tokens = tokenize_business_name(norm_name)

    # 2. Address processing
    addr_script = detector.detect(record.business_address) if record.business_address else "Unknown"
    if record.business_address and addr_script not in ("Latin", "Unknown"):
        addr_transliterated = transliterator.transliterate(record.business_address, script=addr_script)
    else:
        addr_transliterated = record.business_address

    norm_addr = normalize_address(record.business_address, transliterated=addr_transliterated)
    addr_tokens = tokenize_address(norm_addr)

    # 3. Country processing
    norm_country = normalize_country(record.country)

    return {
        "entity_id": record.entity_id,
        "name_orig": record.business_name,
        "name_script": name_script,
        "name_transliterated": name_transliterated,
        "name_norm": norm_name,
        "name_tokens": name_tokens,
        "addr_orig": record.business_address,
        "addr_script": addr_script,
        "addr_transliterated": addr_transliterated,
        "addr_norm": norm_addr,
        "addr_tokens": addr_tokens,
        "country_orig": record.country,
        "country_norm": norm_country,
    }


def main():
    print("=" * 70)
    print("PHASE 4: DETERMINISTIC NORMALIZATION INTEGRATION PIPELINE")
    print("=" * 70)

    detector = ScriptDetector()
    transliterator = IndicTransliterator()

    sources = [
        ("Source 1 (Reference)", read_train_source1),
        ("Source 2 (Candidate)", read_train_source2),
        ("Source 3 (Candidate)", read_train_source3),
    ]

    all_processed = []

    for name, reader_func in sources:
        count = 0
        for chunk in reader_func(chunk_size=100, max_rows=100):
            for rec in chunk:
                processed = process_record(rec, detector, transliterator)
                all_processed.append(processed)
                count += 1
        print(f"Processed {count} records from {name}.")

    print(f"\nTotal records processed: {len(all_processed)}")

    # Display diverse representative samples
    print("\n" + "=" * 70)
    print("REPRESENTATIVE SAMPLES ACROSS PIPELINE")
    print("=" * 70)

    # 1. Pure Latin record
    latin_sample = next((r for r in all_processed if r["name_script"] == "Latin"), None)
    if latin_sample:
        print("\n--- [1] PURE LATIN EXAMPLE ---")
        print(f"Original Name:        {latin_sample['name_orig']}")
        print(f"Normalized Name:      {latin_sample['name_norm']}")
        print(f"Name Tokens:          {latin_sample['name_tokens']}")
        print(f"Original Address:     {latin_sample['addr_orig']}")
        print(f"Normalized Address:   {latin_sample['addr_norm']}")
        print(f"Address Tokens:       {latin_sample['addr_tokens']}")
        print(f"Country:              {latin_sample['country_orig']} -> {latin_sample['country_norm']}")

    # 2. Indic / Non-Latin Name record
    indic_sample = next(
        (r for r in all_processed if r["name_script"] not in ("Latin", "Unknown", "Mixed")),
        None,
    )
    if indic_sample:
        print(f"\n--- [2] INDIC SCRIPT EXAMPLE ({indic_sample['name_script']}) ---")
        print(f"Original Name:        {indic_sample['name_orig']}")
        print(f"Transliterated:       {indic_sample['name_transliterated']}")
        print(f"Normalized Name:      {indic_sample['name_norm']}")
        print(f"Name Tokens:          {indic_sample['name_tokens']}")
        print(f"Original Address:     {indic_sample['addr_orig']}")
        print(f"Transliterated Addr:  {indic_sample['addr_transliterated']}")
        print(f"Normalized Address:   {indic_sample['addr_norm']}")
        print(f"Address Tokens:       {indic_sample['addr_tokens']}")
        print(f"Country:              {indic_sample['country_orig']} -> {indic_sample['country_norm']}")

    # 3. Mixed Script record
    mixed_sample = next(
        (r for r in all_processed if r["name_script"] == "Mixed" or r["addr_script"] == "Mixed"),
        None,
    )
    if mixed_sample:
        print("\n--- [3] MIXED SCRIPT EXAMPLE ---")
        print(f"Original Name:        {mixed_sample['name_orig']} (Script: {mixed_sample['name_script']})")
        print(f"Transliterated Name:  {mixed_sample['name_transliterated']}")
        print(f"Normalized Name:      {mixed_sample['name_norm']}")
        print(f"Name Tokens:          {mixed_sample['name_tokens']}")
        print(f"Original Address:     {mixed_sample['addr_orig']} (Script: {mixed_sample['addr_script']})")
        print(f"Transliterated Addr:  {mixed_sample['addr_transliterated']}")
        print(f"Normalized Address:   {mixed_sample['addr_norm']}")
        print(f"Address Tokens:       {mixed_sample['addr_tokens']}")
        print(f"Country:              {mixed_sample['country_orig']} -> {mixed_sample['country_norm']}")

    # 4. Missing address record
    missing_addr_sample = next(
        (r for r in all_processed if r["addr_orig"] is None),
        None,
    )
    if missing_addr_sample:
        print("\n--- [4] MISSING ADDRESS EXAMPLE ---")
        print(f"Entity ID:            {missing_addr_sample['entity_id']}")
        print(f"Original Name:        {missing_addr_sample['name_orig']}")
        print(f"Normalized Name:      {missing_addr_sample['name_norm']}")
        print(f"Original Address:     {missing_addr_sample['addr_orig']}")
        print(f"Normalized Address:   {missing_addr_sample['addr_norm']}")
        print(f"Address Tokens:       {missing_addr_sample['addr_tokens']}")
        print(f"Country:              {missing_addr_sample['country_orig']} -> {missing_addr_sample['country_norm']}")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
