"""
Production streaming dataset processing pipeline for Business Entity Resolution.

Reads raw TSV in chunks via canonical ingestion, runs:
- Unicode script detection
- Language inference
- Indic-to-Latin transliteration
- Text, business name, address, and country normalization
- Tokenization
- Incremental Parquet writing with checkpointing and error logging
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
import sys
import time
import tracemalloc
from typing import Optional, List, Dict, Any, Set

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pyarrow.parquet as pq

from src.preprocessing.ingestion import stream_source_tsv, get_dataset_path
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.writer import ParquetPartWriter, OUTPUT_SCHEMA
from src.preprocessing.checkpoint import CheckpointManager


def infer_source_from_filename(filename: str) -> str:
    """Infer source identifier ('source1', 'source2', 'source3') from filename."""
    lower = os.path.basename(filename).lower()
    if "source1" in lower:
        return "source1"
    elif "source2" in lower:
        return "source2"
    elif "source3" in lower:
        return "source3"
    raise ValueError(f"Could not infer source from filename: {filename}. Please specify --source.")


def log_error(
    error_csv_path: str,
    source: str,
    entity_id: str,
    field: str,
    error_msg: str,
    exc_type: str,
) -> None:
    """Append row processing error to error log CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(error_csv_path)), exist_ok=True)
    file_exists = os.path.exists(error_csv_path)
    with open(error_csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "source", "entity_id", "field", "error_message", "exception_type"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            source,
            entity_id,
            field,
            error_msg,
            exc_type,
        ])


def process_dataset(
    input_file: str,
    source: Optional[str] = None,
    output_dir: str = "data/processed/train",
    chunk_size: int = 10000,
    limit: Optional[int] = None,
    resume: bool = True,
    checkpoint_dir: str = "data/processed/checkpoints",
    error_log_path: str = "reports/processing_errors.csv",
    progress_interval: int = 10000,
    total_expected_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stream and process a dataset TSV, write incremental Parquet part files,
    and return validation summary statistics.
    """
    tracemalloc.start()
    start_time = time.time()

    if not source:
        source = infer_source_from_filename(input_file)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    file_stem = os.path.splitext(os.path.basename(input_file))[0]
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{file_stem}.json")
    checkpoint_mgr = CheckpointManager(checkpoint_file)

    # Check for existing checkpoint
    rows_skipped = 0
    part_num = 1
    existing_part_files: List[str] = []
    error_count = 0

    if resume and checkpoint_mgr.exists():
        state = checkpoint_mgr.load()
        if state and state.get("status") in ("in_progress", "completed"):
            rows_skipped = state.get("rows_processed", 0)
            existing_part_files = state.get("part_files", [])
            part_num = len(existing_part_files) + 1
            error_count = state.get("error_count", 0)
            print(f"[Checkpoint] Resuming '{file_stem}' from row {rows_skipped} (next part: {part_num})")

    pipeline = PreprocessingPipeline()
    writer = ParquetPartWriter(output_dir=output_dir, file_prefix=file_stem)

    current_chunk_records: List[Dict[str, Any]] = []
    part_files_written = list(existing_part_files)
    total_processed = rows_skipped
    rows_streamed = 0

    print("=" * 70)
    print(f"STARTING DATASET PROCESSING: {file_stem}")
    print(f"Input:       {input_file}")
    print(f"Source:      {source}")
    print(f"Output dir:  {output_dir}")
    print(f"Chunk size:  {chunk_size}")
    print(f"Limit:       {limit or 'None (Full dataset)'}")
    print("=" * 70)

    # Use stream_source_tsv from canonical ingestion
    # Note: canonical ingestion reads chunks from TSV.
    # We pass chunk_size to match our batch writing granularity.
    stream = stream_source_tsv(
        file_path=input_file,
        source=source,
        chunk_size=chunk_size,
        max_rows=limit,
    )

    for chunk in stream:
        for record in chunk:
            rows_streamed += 1

            # Skip already checkpointed rows
            if rows_streamed <= rows_skipped:
                continue

            try:
                processed_dict = pipeline.process_record(record)
            except Exception as e:
                error_count += 1
                log_error(
                    error_csv_path=error_log_path,
                    source=source,
                    entity_id=record.entity_id,
                    field="record",
                    error_msg=str(e),
                    exc_type=type(e).__name__,
                )
                # Fallback record: preserve raw values, set transformed to None
                processed_dict = {
                    "source": record.source,
                    "entity_id": record.entity_id,
                    "business_name": record.business_name,
                    "business_name_script": None,
                    "business_name_language": None,
                    "business_name_transliterated": record.business_name,
                    "business_name_normalized": None,
                    "business_name_tokens": None,
                    "business_address": record.business_address,
                    "business_address_script": None,
                    "business_address_language": None,
                    "business_address_transliterated": record.business_address,
                    "business_address_normalized": None,
                    "business_address_tokens": None,
                    "country": record.country,
                    "country_normalized": None,
                }

            current_chunk_records.append(processed_dict)
            total_processed += 1

            # When chunk size is reached, write Parquet part file
            if len(current_chunk_records) >= chunk_size:
                part_path = writer.write_chunk(current_chunk_records, part_num=part_num)
                part_files_written.append(part_path)
                checkpoint_mgr.save(
                    input_file=input_file,
                    source=source,
                    rows_processed=total_processed,
                    current_part=part_num,
                    part_files=part_files_written,
                    status="in_progress",
                    error_count=error_count,
                )
                part_num += 1
                current_chunk_records = []

            # Progress reporting
            if total_processed % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = (total_processed - rows_skipped) / elapsed if elapsed > 0 else 0
                if total_expected_rows:
                    pct = (total_processed / total_expected_rows) * 100
                    rem_rows = max(0, total_expected_rows - total_processed)
                    eta_sec = rem_rows / rate if rate > 0 else 0
                    print(
                        f"[{source.upper()}] {total_processed:,} / {total_expected_rows:,} "
                        f"({pct:.1f}%) | Elapsed: {elapsed:.1f}s | Rate: {rate:.0f} rows/s | ETA: {eta_sec:.1f}s"
                    )
                else:
                    print(
                        f"[{source.upper()}] {total_processed:,} rows processed | "
                        f"Elapsed: {elapsed:.1f}s | Rate: {rate:.0f} rows/s"
                    )

    # Write any remaining records in final part
    if current_chunk_records:
        part_path = writer.write_chunk(current_chunk_records, part_num=part_num)
        part_files_written.append(part_path)
        checkpoint_mgr.save(
            input_file=input_file,
            source=source,
            rows_processed=total_processed,
            current_part=part_num,
            part_files=part_files_written,
            status="completed",
            error_count=error_count,
        )

    checkpoint_mgr.mark_completed(
        rows_processed=total_processed,
        current_part=len(part_files_written),
        part_files=part_files_written,
        error_count=error_count,
    )

    elapsed_total = time.time() - start_time
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    overall_rate = (total_processed - rows_skipped) / elapsed_total if elapsed_total > 0 else 0

    print("=" * 70)
    print(f"PROCESSING COMPLETED FOR {file_stem}")
    print(f"Total Rows Processed: {total_processed:,}")
    print(f"Parquet Part Files:   {len(part_files_written)}")
    print(f"Errors Logged:        {error_count}")
    print(f"Elapsed Time:         {elapsed_total:.2f} s")
    print(f"Throughput Rate:      {overall_rate:.0f} rows/s")
    print(f"Peak Memory:          {peak_mem / (1024 * 1024):.2f} MB")
    print("=" * 70)

    # Run validation on written files
    validation_stats = validate_processed_output(
        part_files=part_files_written,
        expected_source=source,
        expected_count=total_processed,
    )

    return {
        "source": source,
        "input_file": input_file,
        "total_processed": total_processed,
        "part_files": part_files_written,
        "error_count": error_count,
        "elapsed_sec": elapsed_total,
        "rate_rows_per_sec": overall_rate,
        "peak_mem_mb": peak_mem / (1024 * 1024),
        "validation": validation_stats,
    }


from src.preprocessing.schema import NAMESPACE_PREFIX_MAP

def validate_processed_output(
    part_files: List[str],
    expected_source: str,
    expected_count: int,
) -> Dict[str, Any]:
    """
    Verify processed Parquet part files:
    - Total rows match
    - Schema conformity
    - Unique entity_id count
    - Namespace consistency
    - Missing entity_id checks
    - Sample check for null representation
    """
    print("\n--- VALIDATING PARQUET OUTPUT ---")
    total_rows = 0
    seen_entity_ids: Set[str] = set()
    bad_namespace_count = 0
    null_entity_ids = 0
    string_nan_count = 0
    null_address_count = 0

    expected_prefix = NAMESPACE_PREFIX_MAP.get(expected_source, "")

    for pfile in part_files:
        if not os.path.exists(pfile):
            raise FileNotFoundError(f"Part file missing: {pfile}")

        table = pq.read_table(pfile)
        num_rows = table.num_rows
        total_rows += num_rows

        # Check schema
        for col_name in OUTPUT_SCHEMA.names:
            if col_name not in table.column_names:
                raise ValueError(f"Missing column in schema: {col_name}")

        # Scan batch values
        sources = table["source"].to_pylist()
        entity_ids = table["entity_id"].to_pylist()
        addresses = table["business_address"].to_pylist()

        for s, eid, addr in zip(sources, entity_ids, addresses):
            if s != expected_source:
                bad_namespace_count += 1
            if eid is None or eid == "":
                null_entity_ids += 1
            else:
                seen_entity_ids.add(eid)
                if expected_prefix and not eid.startswith(expected_prefix):
                    bad_namespace_count += 1

            if addr is None:
                null_address_count += 1
            elif addr in ("nan", "None", "null", "NaN"):
                string_nan_count += 1

    print(f"Verified Part Files:       {len(part_files)}")
    print(f"Total Output Rows:         {total_rows:,} (Expected: {expected_count:,})")
    print(f"Unique Entity IDs:         {len(seen_entity_ids):,}")
    print(f"Namespace Inconsistencies: {bad_namespace_count}")
    print(f"Missing/Null Entity IDs:   {null_entity_ids}")
    print(f"String 'nan'/'None' count: {string_nan_count}")
    print(f"Null Business Addresses:   {null_address_count}")

    is_valid = (
        total_rows == expected_count
        and len(seen_entity_ids) == expected_count
        and bad_namespace_count == 0
        and null_entity_ids == 0
        and string_nan_count == 0
    )
    print(f"Validation Status:         {'PASSED' if is_valid else 'FAILED'}\n")

    return {
        "is_valid": is_valid,
        "total_rows": total_rows,
        "unique_entity_ids": len(seen_entity_ids),
        "bad_namespaces": bad_namespace_count,
        "null_entity_ids": null_entity_ids,
        "string_nan_count": string_nan_count,
        "null_address_count": null_address_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Process dataset TSV into normalized Parquet files.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset TSV.")
    parser.add_argument("--source", type=str, default=None, choices=["source1", "source2", "source3"], help="Source identifier.")
    parser.add_argument("--output-dir", type=str, default="data/processed/train", help="Directory to save Parquet part files.")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of records per Parquet part file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process (dry-run).")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing checkpoint and start fresh.")
    parser.add_argument("--progress-interval", type=int, default=10000, help="Row interval for progress updates.")
    parser.add_argument("--expected-rows", type=int, default=None, help="Expected total rows for percentage & ETA.")

    args = parser.parse_args()

    process_dataset(
        input_file=args.input,
        source=args.source,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        limit=args.limit,
        resume=args.resume,
        progress_interval=args.progress_interval,
        total_expected_rows=args.expected_rows,
    )


if __name__ == "__main__":
    main()
