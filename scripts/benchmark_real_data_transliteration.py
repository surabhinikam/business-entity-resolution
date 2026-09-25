"""Real-data benchmark for script detection and transliteration.

Streams the first 1,000 records from each training source (source1, source2, source3),
runs script detection and transliteration on business_name and business_address,
produces summary statistics, prints 30+ representative non-Latin/mixed examples,
and flags suspicious cases.
"""

import os
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

from src.preprocessing.ingestion import (
    read_train_source1,
    read_train_source2,
    read_train_source3,
)
from src.transliteration.script_detector import ScriptDetector
from src.transliteration.transliterator import IndicTransliterator

SAMPLE_SIZE = 1000  # records per source
SEPARATOR = "-" * 70

detector = ScriptDetector()
transliterator = IndicTransliterator()


def has_indic_chars(text: str) -> bool:
    """Return True if any Indic Unicode characters remain in text."""
    return any(0x0900 <= ord(c) <= 0x0D7F for c in text)


def flag_suspicious(field: str, original: str, detected_script: str, transliterated: str) -> Optional[str]:
    """Return a warning string if a suspicious case is detected, else None."""
    if not original:
        return None

    # Case 1: Indic chars remain after transliteration
    if detected_script not in ("Latin", "Unknown") and transliterated and has_indic_chars(transliterated):
        return "⚠ INDIC CHARS REMAIN AFTER TRANSLITERATION"

    # Case 2: Transliteration output unexpectedly empty when input is non-empty
    if original.strip() and not transliterated.strip():
        return "⚠ UNEXPECTED EMPTY TRANSLITERATION"

    # Case 3: Original Indic field unchanged after transliteration (i.e. transliteration silently skipped)
    if detected_script not in ("Latin", "Unknown", "Mixed") and original == transliterated:
        return "⚠ INDIC INPUT UNCHANGED IN OUTPUT"

    return None


def process_source(reader_fn, source_label):
    """Process records from a source reader and return stats + examples."""
    stats = {
        "total": 0,
        "name_scripts": Counter(),
        "addr_scripts": Counter(),
        "suspicious": [],
    }
    non_latin_examples = []

    all_chunks = reader_fn(chunk_size=SAMPLE_SIZE, max_rows=SAMPLE_SIZE)
    for chunk in all_chunks:
        for record in chunk:
            stats["total"] += 1

            name = record.business_name or ""
            addr = record.business_address or ""

            name_script = detector.detect(name)
            addr_script = detector.detect(addr)

            name_trans = transliterator.transliterate(name, script=name_script)
            addr_trans = transliterator.transliterate(addr, script=addr_script)

            stats["name_scripts"][name_script] += 1
            stats["addr_scripts"][addr_script] += 1

            # Flag suspicious cases
            for field, orig, scr, trans in [
                ("business_name", name, name_script, name_trans),
                ("business_address", addr, addr_script, addr_trans),
            ]:
                warning = flag_suspicious(field, orig, scr, trans)
                if warning:
                    stats["suspicious"].append({
                        "source": source_label,
                        "entity_id": record.entity_id,
                        "field": field,
                        "original": orig,
                        "detected_script": scr,
                        "transliterated": trans,
                        "warning": warning,
                    })

            # Collect non-Latin/mixed examples
            if name_script not in ("Latin", "Unknown") and len(non_latin_examples) < 20:
                non_latin_examples.append({
                    "source": source_label,
                    "entity_id": record.entity_id,
                    "field": "business_name",
                    "original": name,
                    "detected_script": name_script,
                    "transliterated": name_trans,
                })
            if addr_script not in ("Latin", "Unknown") and len(non_latin_examples) < 20:
                non_latin_examples.append({
                    "source": source_label,
                    "entity_id": record.entity_id,
                    "field": "business_address",
                    "original": addr,
                    "detected_script": addr_script,
                    "transliterated": addr_trans,
                })

    return stats, non_latin_examples


def print_stats(source_label, stats):
    """Print summary statistics for a source."""
    total = stats["total"]
    print(f"\n{'='*70}")
    print(f"SOURCE: {source_label}  ({total} records sampled)")
    print(f"{'='*70}")

    print("\n--- business_name script distribution ---")
    for script, count in stats["name_scripts"].most_common():
        pct = 100 * count / total if total else 0
        category = (
            "Latin-only" if script == "Latin" else
            "Indic" if script not in ("Latin", "Mixed", "Unknown") else
            script
        )
        print(f"  {script:<14}: {count:>5}  ({pct:5.1f}%)  [{category}]")

    print("\n--- business_address script distribution ---")
    for script, count in stats["addr_scripts"].most_common():
        pct = 100 * count / total if total else 0
        category = (
            "Latin-only" if script == "Latin" else
            "Indic" if script not in ("Latin", "Mixed", "Unknown") else
            script
        )
        print(f"  {script:<14}: {count:>5}  ({pct:5.1f}%)  [{category}]")


def print_examples(examples, max_show=15):
    """Print representative non-Latin transliteration examples."""
    shown = 0
    for ex in examples[:max_show]:
        print(SEPARATOR)
        print(f"SOURCE          : {ex['source']}")
        print(f"ENTITY_ID       : {ex['entity_id']}")
        print(f"FIELD           : {ex['field']}")
        print(f"ORIGINAL        : {ex['original']}")
        print(f"DETECTED_SCRIPT : {ex['detected_script']}")
        print(f"TRANSLITERATED  : {ex['transliterated']}")
        shown += 1
    return shown


def print_suspicious(suspicious_list):
    """Print all suspicious cases."""
    if not suspicious_list:
        print("  ✓ No suspicious cases detected.")
        return
    for s in suspicious_list:
        print(SEPARATOR)
        print(f"  {s['warning']}")
        print(f"  SOURCE    : {s['source']}")
        print(f"  ENTITY_ID : {s['entity_id']}")
        print(f"  FIELD     : {s['field']}")
        print(f"  ORIGINAL  : {s['original'][:120]}")
        print(f"  DETECTED  : {s['detected_script']}")
        print(f"  TRANS OUT : {s['transliterated'][:120]}")


def main():
    print("=" * 70)
    print("REAL-DATA TRANSLITERATION BENCHMARK")
    print(f"Sampling {SAMPLE_SIZE} records from each of: source1, source2, source3")
    print("=" * 70)

    sources = [
        ("train_source1", read_train_source1),
        ("train_source2", read_train_source2),
        ("train_source3", read_train_source3),
    ]

    all_examples = []
    all_suspicious = []
    global_name_scripts = Counter()
    global_addr_scripts = Counter()
    grand_total = 0

    for source_label, reader_fn in sources:
        stats, examples = process_source(reader_fn, source_label)
        print_stats(source_label, stats)
        all_examples.extend(examples)
        all_suspicious.extend(stats["suspicious"])
        global_name_scripts.update(stats["name_scripts"])
        global_addr_scripts.update(stats["addr_scripts"])
        grand_total += stats["total"]

    # Global summary
    print(f"\n{'='*70}")
    print(f"GLOBAL SUMMARY  (all 3 sources, {grand_total} total records)")
    print(f"{'='*70}")
    print("\n--- business_name (combined) ---")
    for script, count in global_name_scripts.most_common():
        pct = 100 * count / grand_total if grand_total else 0
        print(f"  {script:<14}: {count:>5}  ({pct:5.1f}%)")
    print("\n--- business_address (combined) ---")
    for script, count in global_addr_scripts.most_common():
        pct = 100 * count / grand_total if grand_total else 0
        print(f"  {script:<14}: {count:>5}  ({pct:5.1f}%)")

    # Representative non-Latin examples (aim for 30+)
    print(f"\n{'='*70}")
    print("REPRESENTATIVE NON-LATIN / MIXED-SCRIPT EXAMPLES (up to 35 shown)")
    print(f"{'='*70}")
    shown = print_examples(all_examples, max_show=35)
    print(f"\n  (Showed {shown} of {len(all_examples)} non-Latin examples collected)")

    # Suspicious cases
    print(f"\n{'='*70}")
    print(f"SUSPICIOUS CASES ({len(all_suspicious)} found)")
    print(f"{'='*70}")
    print_suspicious(all_suspicious)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
