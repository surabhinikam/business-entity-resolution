"""
Streaming Parquet writer for processed business entity records.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import pyarrow as pa
import pyarrow.parquet as pq

# Canonical 16-field Parquet output schema
OUTPUT_SCHEMA = pa.schema([
    pa.field("source", pa.string(), nullable=False),
    pa.field("entity_id", pa.string(), nullable=False),
    pa.field("business_name", pa.string(), nullable=True),
    pa.field("business_name_script", pa.string(), nullable=True),
    pa.field("business_name_language", pa.string(), nullable=True),
    pa.field("business_name_transliterated", pa.string(), nullable=True),
    pa.field("business_name_normalized", pa.string(), nullable=True),
    pa.field("business_name_tokens", pa.list_(pa.string()), nullable=True),
    pa.field("business_address", pa.string(), nullable=True),
    pa.field("business_address_script", pa.string(), nullable=True),
    pa.field("business_address_language", pa.string(), nullable=True),
    pa.field("business_address_transliterated", pa.string(), nullable=True),
    pa.field("business_address_normalized", pa.string(), nullable=True),
    pa.field("business_address_tokens", pa.list_(pa.string()), nullable=True),
    pa.field("country", pa.string(), nullable=True),
    pa.field("country_normalized", pa.string(), nullable=True),
])


class ParquetPartWriter:
    """Writes processed records incrementally into numbered Parquet part files."""

    def __init__(self, output_dir: str, file_prefix: str, compression: str = "snappy"):
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.compression = compression
        os.makedirs(self.output_dir, exist_ok=True)

    def get_part_filename(self, part_num: int) -> str:
        """Get formatted part filename."""
        return f"{self.file_prefix}_part_{part_num:05d}.parquet"

    def get_part_path(self, part_num: int) -> str:
        """Get absolute path to part file."""
        return os.path.join(self.output_dir, self.get_part_filename(part_num))

    def write_chunk(self, records: List[Dict[str, Any]], part_num: int) -> str:
        """
        Convert chunk records to Arrow Table and write to a Parquet part file.
        Returns the path to the written part file.
        """
        if not records:
            return ""

        part_path = self.get_part_path(part_num)
        table = pa.Table.from_pylist(records, schema=OUTPUT_SCHEMA)
        pq.write_table(table, part_path, compression=self.compression)
        return part_path
