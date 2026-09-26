"""Unit tests for data loading and partition discovery utilities."""

import unittest
import tempfile
import os
import polars as pl

from src.analysis.data_loader import (
    discover_parquet_files,
    explode_ground_truth,
)


class TestDataLoading(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discover_parquet_files_filtering(self):
        """Test that discover_parquet_files correctly isolates source partitions."""
        d = self.temp_dir.name
        # Create fake parquet files
        s1_file = os.path.join(d, "train_source1_part_00001.parquet")
        s2_file = os.path.join(d, "train_source2_part_00001.parquet")
        with open(s1_file, "wb") as f:
            f.write(b"fake parquet 1")
        with open(s2_file, "wb") as f:
            f.write(b"fake parquet 2")

        # Discover source2 only
        files = discover_parquet_files(d, source="source2", warn_on_misplaced=False)
        self.assertEqual(len(files), 1)
        self.assertIn("train_source2_part_00001.parquet", files[0])

        # Discover all
        all_files = discover_parquet_files(d, source=None)
        self.assertEqual(len(all_files), 2)

    def test_explode_ground_truth(self):
        """Test explode_ground_truth unpacks comma-separated matches correctly."""
        gt_data = {
            "source1_entity_id": ["S1-100", "S1-200", "S1-300"],
            "matched_entity_ids": ["S2-1,S3-2", "", None],
        }
        gt_df = pl.DataFrame(gt_data)
        exploded = explode_ground_truth(gt_df)

        self.assertEqual(exploded.height, 2)
        self.assertListEqual(
            exploded["matched_entity_id"].to_list(), ["S2-1", "S3-2"]
        )
        self.assertListEqual(
            exploded["target_source"].to_list(), ["source2", "source3"]
        )


if __name__ == "__main__":
    unittest.main()
