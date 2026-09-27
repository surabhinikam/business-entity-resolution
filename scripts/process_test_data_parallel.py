"""
Fast parallel test dataset processing script for Business Entity Resolution.

Processes raw test TSVs into standardized Parquet part files using the frozen PreprocessingPipeline.
Leverages Python multiprocessing across CPU cores for maximum throughput.
"""

from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.schema import CanonicalRecord, create_canonical_record
from src.preprocessing.writer import OUTPUT_SCHEMA, ParquetPartWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

worker_pipeline: Optional[PreprocessingPipeline] = None


def init_worker():
    global worker_pipeline
    worker_pipeline = PreprocessingPipeline()


def process_single_record(record_dict: Dict[str, Any]) -> Dict[str, Any]:
    global worker_pipeline
    assert worker_pipeline is not None
    rec = create_canonical_record(**record_dict)
    return worker_pipeline.process_record(rec)


def process_source_file(
    input_file: str,
    source: str,
    output_dir: str,
    n_workers: int = 8,
    chunk_size: int = 50000,
) -> Dict[str, Any]:
    """
    Reads a raw TSV in chunks, processes records across worker processes,
    and writes standardized Parquet files.
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PROCESSING %s: %s", source.upper(), input_file)
    logger.info("Output dir: %s | Workers: %d | Chunk size: %d", output_dir, n_workers, chunk_size)
    logger.info("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    # Clean any previous part files in output_dir
    existing_parts = glob.glob(os.path.join(output_dir, "*.parquet"))
    for f in existing_parts:
        try:
            os.remove(f)
        except OSError:
            pass

    file_prefix = f"test_{source}"
    writer = ParquetPartWriter(output_dir=output_dir, file_prefix=file_prefix)

    # Scan TSV
    reader = pl.read_csv_batched(
        input_file,
        separator="\t",
        batch_size=chunk_size,
    )

    total_rows = 0
    part_num = 1
    part_files: List[str] = []

    with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch_df = batches[0]
            n_batch = batch_df.height
            if n_batch == 0:
                break

            # Add source column
            raw_dicts = batch_df.with_columns(pl.lit(source).alias("source")).to_dicts()

            # Parallel transformation
            processed = pool.map(process_single_record, raw_dicts, chunksize=1000)

            # Write parquet part
            part_path = writer.write_chunk(processed, part_num=part_num)
            part_files.append(part_path)
            total_rows += len(processed)

            elapsed = time.time() - t0
            rate = total_rows / elapsed if elapsed > 0 else 0
            logger.info(
                "[%s] Part %d written (%d rows) | Total: %d rows | %.1fs | %.0f rows/s",
                source.upper(), part_num, len(processed), total_rows, elapsed, rate
            )
            part_num += 1

    total_time = time.time() - t0
    avg_rate = total_rows / total_time if total_time > 0 else 0
    logger.info(
        "FINISHED %s: %d total rows in %d parts | Elapsed: %.1fs | Rate: %.0f rows/s",
        source.upper(), total_rows, len(part_files), total_time, avg_rate
    )

    return {
        "source": source,
        "input_file": input_file,
        "output_dir": output_dir,
        "total_rows": total_rows,
        "part_files": len(part_files),
        "elapsed_seconds": round(total_time, 2),
        "rows_per_second": round(avg_rate, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel test data preprocessor.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Records per parquet chunk.")
    args = parser.parse_args()

    sources = [
        ("source1", "data/raw/test/test_source1.tsv", "data/processed/test/source1"),
        ("source2", "data/raw/test/test_source2.tsv", "data/processed/test/source2"),
        ("source3", "data/raw/test/test_source3.tsv", "data/processed/test/source3"),
    ]

    summaries = []
    t_start = time.time()
    for src, in_file, out_dir in sources:
        res = process_source_file(
            input_file=in_file,
            source=src,
            output_dir=out_dir,
            n_workers=args.workers,
            chunk_size=args.chunk_size,
        )
        summaries.append(res)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"ALL TEST SOURCES PROCESSED IN {total_time:.1f}s ({total_time/60:.2f} min)")
    print("=" * 70)
    for s in summaries:
        print(f"Source: {s['source']:7s} | Rows: {s['total_rows']:10,d} | Parts: {s['part_files']:3d} | Time: {s['elapsed_seconds']:6.1f}s | Rate: {s['rows_per_second']:7.1f} rows/s")


if __name__ == "__main__":
    main()
