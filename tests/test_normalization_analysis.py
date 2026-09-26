"""Unit tests for blocking evaluation and normalization collision analysis."""

import unittest

from src.analysis.blocking_evaluator import compute_block_statistics


class TestNormalizationAnalysis(unittest.TestCase):

    def test_compute_block_statistics_perfect_recall(self):
        s1_keys = {
            "S1-1": {"block_A"},
            "S1-2": {"block_B"},
        }
        cand_keys = {
            "S2-1": {"block_A"},
            "S2-2": {"block_B"},
            "S2-3": {"block_C"},
        }
        gt_pairs = {("S1-1", "S2-1"), ("S1-2", "S2-2")}
        total_possible = 2 * 3

        stats = compute_block_statistics(s1_keys, cand_keys, gt_pairs, total_possible_pairs=total_possible)

        self.assertEqual(stats["ground_truth_recovered"], 2)
        self.assertEqual(stats["blocking_recall"], 1.0)
        self.assertEqual(stats["total_candidate_pairs"], 2)
        self.assertAlmostEqual(stats["reduction_ratio"], 1.0 - (2 / 6))

    def test_compute_block_statistics_partial_recall(self):
        s1_keys = {
            "S1-1": {"block_A"},
            "S1-2": {"block_B"},
        }
        cand_keys = {
            "S2-1": {"block_A"},
            "S2-2": {"block_X"},  # Mismatched key
        }
        gt_pairs = {("S1-1", "S2-1"), ("S1-2", "S2-2")}

        stats = compute_block_statistics(s1_keys, cand_keys, gt_pairs, total_possible_pairs=4)

        self.assertEqual(stats["ground_truth_recovered"], 1)
        self.assertEqual(stats["blocking_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
