import os
import unittest
import tempfile
import pyarrow.parquet as pq

from src.preprocessing.schema import CanonicalRecord
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.checkpoint import CheckpointManager
from src.preprocessing.writer import ParquetPartWriter, OUTPUT_SCHEMA
from src.language_detection.detector import LanguageDetector


class TestProcessingPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = PreprocessingPipeline()

    def test_pure_latin_record(self):
        rec = CanonicalRecord(
            source="source1",
            entity_id="S1-1001",
            business_name="Acme Corporation",
            business_address="123 Main St, New York, NY",
            country="USA",
        )
        res = self.pipeline.process_record(rec)
        self.assertEqual(res["source"], "source1")
        self.assertEqual(res["entity_id"], "S1-1001")
        self.assertEqual(res["business_name_script"], "Latin")
        self.assertEqual(res["business_name_normalized"], "acme corp")
        self.assertEqual(res["business_name_tokens"], ["acme", "corp"])
        self.assertEqual(res["country_normalized"], "united states")

    def test_indic_record(self):
        rec = CanonicalRecord(
            source="source2",
            entity_id="S2-2001",
            business_name="राम मार्केटिंग प्राइवेट लिमिटेड",
            business_address="KH NO. -570/13, नई दिल्ली",
            country="India",
        )
        res = self.pipeline.process_record(rec)
        self.assertEqual(res["business_name_script"], "Devanagari")
        self.assertEqual(res["business_name_language"], "hindi")
        self.assertIn("pvt ltd", res["business_name_normalized"])
        self.assertIn("-570/13", res["business_address_normalized"])
        self.assertEqual(res["country_normalized"], "india")

    def test_null_preservation_not_string_nan(self):
        rec = CanonicalRecord(
            source="source3",
            entity_id="S3-3001",
            business_name="Global Solutions",
            business_address=None,
            country="FR",
        )
        res = self.pipeline.process_record(rec)
        self.assertIsNone(res["business_address"])
        self.assertIsNone(res["business_address_script"])
        self.assertIsNone(res["business_address_normalized"])
        self.assertIsNone(res["business_address_tokens"])
        self.assertNotEqual(res["business_address"], "nan")
        self.assertNotEqual(res["business_address"], "None")
        self.assertEqual(res["country_normalized"], "france")


class TestCheckpointAndWriter(unittest.TestCase):
    def test_checkpoint_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = os.path.join(tmpdir, "checkpoint.json")
            mgr = CheckpointManager(cp_path)
            self.assertFalse(mgr.exists())

            mgr.save(
                input_file="test.tsv",
                source="source2",
                rows_processed=1000,
                current_part=1,
                part_files=["part_00001.parquet"],
                status="in_progress",
            )
            self.assertTrue(mgr.exists())
            state = mgr.load()
            self.assertEqual(state["rows_processed"], 1000)
            self.assertEqual(state["current_part"], 1)

    def test_parquet_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetPartWriter(output_dir=tmpdir, file_prefix="test_source")
            records = [
                {
                    "source": "source2",
                    "entity_id": "s2_1",
                    "business_name": "Test Co",
                    "business_name_script": "Latin",
                    "business_name_language": "english",
                    "business_name_transliterated": "Test Co",
                    "business_name_normalized": "test co",
                    "business_name_tokens": ["test", "co"],
                    "business_address": None,
                    "business_address_script": None,
                    "business_address_language": None,
                    "business_address_transliterated": None,
                    "business_address_normalized": None,
                    "business_address_tokens": None,
                    "country": "India",
                    "country_normalized": "india",
                }
            ]
            part_path = writer.write_chunk(records, part_num=1)
            self.assertTrue(os.path.exists(part_path))

            table = pq.read_table(part_path)
            self.assertEqual(table.num_rows, 1)
            self.assertEqual(table.column_names, OUTPUT_SCHEMA.names)


if __name__ == "__main__":
    unittest.main()
