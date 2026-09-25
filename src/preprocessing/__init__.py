"""Preprocessing package exports."""

from src.preprocessing.schema import (
    CanonicalRecord,
    GroundTruthRecord,
    create_canonical_record,
    CANONICAL_FIELDS,
    REQUIRED_SOURCE_COLUMNS,
    ALLOWED_SOURCES,
    NAMESPACE_PREFIX_MAP,
)
from src.preprocessing.ingestion import (
    DEFAULT_DATASET_DIR,
    get_dataset_path,
    stream_source_tsv,
    stream_ground_truth_tsv,
    read_train_source1,
    read_train_source2,
    read_train_source3,
    read_test_source1,
    read_test_source2,
    read_test_source3,
    read_train_ground_truth,
)

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.writer import ParquetPartWriter, OUTPUT_SCHEMA
from src.preprocessing.checkpoint import CheckpointManager

__all__ = [
    "CanonicalRecord",
    "GroundTruthRecord",
    "create_canonical_record",
    "CANONICAL_FIELDS",
    "REQUIRED_SOURCE_COLUMNS",
    "ALLOWED_SOURCES",
    "NAMESPACE_PREFIX_MAP",
    "DEFAULT_DATASET_DIR",
    "get_dataset_path",
    "stream_source_tsv",
    "stream_ground_truth_tsv",
    "read_train_source1",
    "read_train_source2",
    "read_train_source3",
    "read_test_source1",
    "read_test_source2",
    "read_test_source3",
    "read_train_ground_truth",
    "PreprocessingPipeline",
    "ParquetPartWriter",
    "OUTPUT_SCHEMA",
    "CheckpointManager",
]
