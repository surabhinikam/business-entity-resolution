"""Unit tests for the canonical data ingestion layer and schema validation."""

import os
import unittest
from typing import List

from src.preprocessing.schema import (
    CanonicalRecord,
    GroundTruthRecord,
    CANONICAL_FIELDS,
    REQUIRED_SOURCE_COLUMNS,
    create_canonical_record,
)
from src.validation.validators import (
    ValidationError,
    ReadOnlyViolationError,
    validate_file_open_mode,
    validate_source_headers,
    validate_source_namespace,
    validate_source_record,
    validate_canonical_record,
)
from src.preprocessing.ingestion import (
    DEFAULT_DATASET_DIR,
    read_train_source1,
    read_train_source2,
    read_train_source3,
    read_test_source1,
    read_test_source2,
    read_test_source3,
    read_train_ground_truth,
)


class TestCanonicalSchemaAndValidation(unittest.TestCase):
    """Test schema definitions, field initializations, and validation rules."""

    def test_canonical_fields_completeness(self):
        """Ensure all required canonical and metadata fields are defined."""
        expected_fields = [
            "source",
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "business_name_script",
            "business_name_language",
            "business_name_transliterated",
            "business_name_normalized",
            "business_name_tokens",
            "business_address_script",
            "business_address_language",
            "business_address_transliterated",
            "business_address_normalized",
            "business_address_tokens",
            "country_normalized",
        ]
        self.assertEqual(CANONICAL_FIELDS, expected_fields)

    def test_canonical_record_initialization(self):
        """Verify CanonicalRecord factory initializes future fields to None."""
        rec = create_canonical_record(
            source="source1",
            entity_id="S1-12345",
            business_name="Acme Corp",
            business_address="123 Main St",
            country="US",
        )
        self.assertEqual(rec.source, "source1")
        self.assertEqual(rec.entity_id, "S1-12345")
        self.assertEqual(rec.business_name, "Acme Corp")
        self.assertEqual(rec.business_address, "123 Main St")
        self.assertEqual(rec.country, "US")

        # Future fields must be None
        self.assertIsNone(rec.business_name_script)
        self.assertIsNone(rec.business_name_language)
        self.assertIsNone(rec.business_name_transliterated)
        self.assertIsNone(rec.business_name_normalized)
        self.assertIsNone(rec.business_name_tokens)
        self.assertIsNone(rec.business_address_script)
        self.assertIsNone(rec.business_address_language)
        self.assertIsNone(rec.business_address_transliterated)
        self.assertIsNone(rec.business_address_normalized)
        self.assertIsNone(rec.business_address_tokens)
        self.assertIsNone(rec.country_normalized)

        # Validate canonical record
        validate_canonical_record(rec)

    def test_header_validation_success(self):
        """Header validation should pass when all required columns are present."""
        headers = ["entity_id", "business_name", "business_address", "country", "extra_col"]
        validate_source_headers(headers)

    def test_header_validation_failure(self):
        """Header validation should raise ValidationError when required columns are missing."""
        headers = ["entity_id", "business_name", "country"]  # missing business_address
        with self.assertRaises(ValidationError) as ctx:
            validate_source_headers(headers, file_name="sample.tsv")
        self.assertIn("Missing required column", str(ctx.exception))
        self.assertIn("business_address", str(ctx.exception))

    def test_namespace_validation(self):
        """Validate strict S1, S2, S3 namespace enforcement."""
        validate_source_namespace("source1", "S1-999")
        validate_source_namespace("source2", "S2-888")
        validate_source_namespace("source3", "S3-777")

        # Mismatches
        with self.assertRaises(ValidationError):
            validate_source_namespace("source1", "S2-123")

        with self.assertRaises(ValidationError):
            validate_source_namespace("source2", "S1-123")

        with self.assertRaises(ValidationError):
            validate_source_namespace("source3", "INVALID-123")

        with self.assertRaises(ValidationError):
            validate_source_namespace("unknown_source", "S1-123")

    def test_silent_entity_id_modification_detection(self):
        """Ensure attempting to silently mutate an entity_id triggers an error."""
        raw_rec = {
            "entity_id": "S1-99999",
            "business_name": "Test",
            "business_address": "Addr",
            "country": "US",
        }
        with self.assertRaises(ValidationError) as ctx:
            validate_source_record("source1", raw_rec, original_entity_id="S1-00000")
        self.assertIn("Entity ID modification detected", str(ctx.exception))

    def test_read_only_mode_enforcement(self):
        """Enforce that files cannot be opened with write/append flags."""
        validate_file_open_mode("r")
        validate_file_open_mode("rt")
        validate_file_open_mode("rb")

        with self.assertRaises(ReadOnlyViolationError):
            validate_file_open_mode("w")

        with self.assertRaises(ReadOnlyViolationError):
            validate_file_open_mode("a")

        with self.assertRaises(ReadOnlyViolationError):
            validate_file_open_mode("r+")


class TestDatasetStreamingIngestion(unittest.TestCase):
    """Test chunked ingestion against small samples of the real organizer datasets."""

    def setUp(self):
        self.dataset_dir = DEFAULT_DATASET_DIR

    def _verify_canonical_chunk(self, chunk: List[CanonicalRecord], expected_source: str, expected_prefix: str):
        self.assertGreater(len(chunk), 0)
        for rec in chunk:
            self.assertIsInstance(rec, CanonicalRecord)
            self.assertEqual(rec.source, expected_source)
            self.assertTrue(rec.entity_id.startswith(expected_prefix))
            self.assertIsInstance(rec.business_name, str)
            self.assertIsInstance(rec.business_address, str)
            self.assertIsInstance(rec.country, str)
            # Future fields are None
            self.assertIsNone(rec.business_name_script)
            self.assertIsNone(rec.business_name_language)
            self.assertIsNone(rec.business_name_transliterated)
            self.assertIsNone(rec.business_name_normalized)
            self.assertIsNone(rec.business_name_tokens)
            self.assertIsNone(rec.business_address_script)
            self.assertIsNone(rec.business_address_language)
            self.assertIsNone(rec.business_address_transliterated)
            self.assertIsNone(rec.business_address_normalized)
            self.assertIsNone(rec.business_address_tokens)
            self.assertIsNone(rec.country_normalized)

    def test_read_train_source1_sample(self):
        chunks = list(read_train_source1(self.dataset_dir, chunk_size=3, max_rows=5))
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 5)
        self._verify_canonical_chunk(chunks[0], expected_source="source1", expected_prefix="S1-")

    def test_read_train_source2_sample(self):
        chunks = list(read_train_source2(self.dataset_dir, chunk_size=3, max_rows=5))
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 5)
        self._verify_canonical_chunk(chunks[0], expected_source="source2", expected_prefix="S2-")

    def test_read_train_source3_sample(self):
        chunks = list(read_train_source3(self.dataset_dir, chunk_size=3, max_rows=5))
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 5)
        self._verify_canonical_chunk(chunks[0], expected_source="source3", expected_prefix="S3-")

    def test_read_test_source1_sample(self):
        chunks = list(read_test_source1(self.dataset_dir, chunk_size=3, max_rows=5))
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 5)
        self._verify_canonical_chunk(chunks[0], expected_source="source1", expected_prefix="S1-")

    def test_read_test_source2_sample(self):
        chunks = list(read_test_source2(self.dataset_dir, chunk_size=3, max_rows=5))
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 5)
        self._verify_canonical_chunk(chunks[0], expected_source="source2", expected_prefix="S2-")

    def test_read_test_source3_sample(self):
        chunks = list(read_test_source3(self.dataset_dir, chunk_size=3, max_rows=5))
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 5)
        self._verify_canonical_chunk(chunks[0], expected_source="source3", expected_prefix="S3-")

    def test_read_train_ground_truth_sample(self):
        chunks = list(read_train_ground_truth(self.dataset_dir, chunk_size=3, max_rows=5))
        total_rows = sum(len(c) for c in chunks)
        self.assertEqual(total_rows, 5)
        
        first_record = chunks[0][0]
        self.assertIsInstance(first_record, GroundTruthRecord)
        self.assertTrue(first_record.source1_entity_id.startswith("S1-"))
        self.assertIsInstance(first_record.matched_entity_ids, str)
        self.assertIsInstance(first_record.matched_entity_list, list)
        
        # Verify matched entities adhere to S2 or S3
        for match_id in first_record.matched_entity_list:
            self.assertTrue(match_id.startswith("S2-") or match_id.startswith("S3-"))


if __name__ == "__main__":
    unittest.main()
