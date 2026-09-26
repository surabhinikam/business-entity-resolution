"""
Block Index: builds and manages an inverted index from blocking keys to entity IDs.

Supports:
- Building indexes from entity records
- Configurable block-size capping with logging/reporting of oversized blocks
- Candidate pair generation with key provenance tracking
- Deduplication of candidate pairs
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.candidate_generation.block_keys import generate_all_keys, BLOCKING_KEY_NAMES

logger = logging.getLogger(__name__)


class BlockIndex:
    """
    Inverted index mapping blocking keys to entity IDs.

    Supports separate indexes for S1 (source1) and candidates (source2/source3),
    with configurable block-size capping.
    """

    def __init__(self, max_block_size: Optional[int] = None):
        """
        Initialize the block index.

        Parameters:
            max_block_size: If set, blocks larger than this (measured as
                |S1_entities| * |candidate_entities| for a given key)
                are detected and reported. NOT silently discarded.
        """
        self.max_block_size = max_block_size

        # key_string -> list of entity_ids
        self.s1_index: Dict[str, List[str]] = defaultdict(list)
        self.cand_index: Dict[str, List[str]] = defaultdict(list)

        # entity_id -> set of key_strings (for recall computation)
        self.s1_entity_keys: Dict[str, Set[str]] = defaultdict(set)
        self.cand_entity_keys: Dict[str, Set[str]] = defaultdict(set)

        # Per-key-label tracking
        self.s1_keys_by_label: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.cand_keys_by_label: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )

        # Stats
        self.s1_count = 0
        self.cand_count = 0
        self.oversized_blocks: List[Dict[str, Any]] = []

    def add_s1_record(
        self,
        entity_id: str,
        record: Dict[str, Any],
        active_keys: Optional[List[str]] = None,
    ) -> None:
        """Add a Source 1 record to the index."""
        keys_by_label = generate_all_keys(record, active_keys=active_keys)
        all_keys = keys_by_label.pop("ALL", set())

        for key_str in all_keys:
            self.s1_index[key_str].append(entity_id)
        self.s1_entity_keys[entity_id] |= all_keys

        for label, keys in keys_by_label.items():
            self.s1_keys_by_label[label][entity_id] |= keys

        self.s1_count += 1

    def add_candidate_record(
        self,
        entity_id: str,
        record: Dict[str, Any],
        active_keys: Optional[List[str]] = None,
    ) -> None:
        """Add a Source 2 or Source 3 record to the index."""
        keys_by_label = generate_all_keys(record, active_keys=active_keys)
        all_keys = keys_by_label.pop("ALL", set())

        for key_str in all_keys:
            self.cand_index[key_str].append(entity_id)
        self.cand_entity_keys[entity_id] |= all_keys

        for label, keys in keys_by_label.items():
            self.cand_keys_by_label[label][entity_id] |= keys

        self.cand_count += 1

    def _detect_oversized_blocks(self) -> List[Dict[str, Any]]:
        """Detect blocks exceeding max_block_size."""
        if self.max_block_size is None:
            return []

        oversized = []
        common_keys = set(self.s1_index.keys()) & set(self.cand_index.keys())

        for key_str in common_keys:
            n_s1 = len(self.s1_index[key_str])
            n_cand = len(self.cand_index[key_str])
            block_size = n_s1 * n_cand

            if block_size > self.max_block_size:
                # Determine which key label this belongs to
                key_label = key_str.split("||")[0] if "||" in key_str else "?"
                oversized.append({
                    "key_string": key_str[:100],  # Truncate for logging
                    "key_label": key_label,
                    "n_s1": n_s1,
                    "n_cand": n_cand,
                    "block_size": block_size,
                })

        self.oversized_blocks = sorted(
            oversized, key=lambda x: x["block_size"], reverse=True
        )

        if self.oversized_blocks:
            logger.warning(
                f"Detected {len(self.oversized_blocks)} oversized blocks "
                f"(cap={self.max_block_size}). "
                f"Largest: {self.oversized_blocks[0]['block_size']:,} pairs"
            )

        return self.oversized_blocks

    def generate_pairs(
        self,
        cap_blocks: bool = False,
    ) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], Set[str]]]:
        """
        Generate deduplicated candidate pairs from the block index.

        Parameters:
            cap_blocks: If True and max_block_size is set, skip pairs
                from oversized blocks. These are logged, not silently discarded.

        Returns:
            Tuple of:
            - Set of (s1_id, cand_id) candidate pairs
            - Dict mapping each pair to set of key labels that produced it
        """
        self._detect_oversized_blocks()
        oversized_keys = set()
        if cap_blocks and self.max_block_size is not None:
            oversized_keys = {
                b["key_string"] for b in self.oversized_blocks
                # Match full key_string
            }
            # Need to rebuild from full key strings
            common_keys = set(self.s1_index.keys()) & set(self.cand_index.keys())
            for key_str in common_keys:
                n_s1 = len(self.s1_index[key_str])
                n_cand = len(self.cand_index[key_str])
                if n_s1 * n_cand > self.max_block_size:
                    oversized_keys.add(key_str)

        pairs: Set[Tuple[str, str]] = set()
        pair_provenance: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        common_keys = set(self.s1_index.keys()) & set(self.cand_index.keys())

        for key_str in common_keys:
            if cap_blocks and key_str in oversized_keys:
                continue

            key_label = key_str.split("||")[0] if "||" in key_str else "?"
            s1_entities = self.s1_index[key_str]
            cand_entities = self.cand_index[key_str]

            for s1_id in s1_entities:
                for cand_id in cand_entities:
                    pair = (s1_id, cand_id)
                    pairs.add(pair)
                    pair_provenance[pair].add(key_label)

        return pairs, dict(pair_provenance)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        common_keys = set(self.s1_index.keys()) & set(self.cand_index.keys())

        total_s1_keys = sum(len(keys) for keys in self.s1_entity_keys.values())
        total_cand_keys = sum(len(keys) for keys in self.cand_entity_keys.values())

        return {
            "s1_entities": self.s1_count,
            "candidate_entities": self.cand_count,
            "total_unique_s1_keys": len(self.s1_index),
            "total_unique_cand_keys": len(self.cand_index),
            "common_blocks": len(common_keys),
            "avg_keys_per_s1": total_s1_keys / self.s1_count if self.s1_count else 0,
            "avg_keys_per_cand": total_cand_keys / self.cand_count if self.cand_count else 0,
            "max_block_size_setting": self.max_block_size,
            "oversized_blocks_count": len(self.oversized_blocks),
        }

    def evaluate_recall(
        self,
        gt_pairs: Set[Tuple[str, str]],
        cap_blocks: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate blocking recall against ground truth.

        Parameters:
            gt_pairs: Set of (s1_id, cand_id) true match pairs.
            cap_blocks: Whether to apply block size capping.

        Returns:
            Dict with recall, candidate count, reduction ratio, etc.
        """
        total_gt = len(gt_pairs)
        if total_gt == 0:
            return {"error": "Empty ground truth"}

        # Get candidate pairs
        candidate_pairs, provenance = self.generate_pairs(cap_blocks=cap_blocks)

        # Count recovered GT pairs
        recovered = candidate_pairs & gt_pairs
        recall = len(recovered) / total_gt

        # Reduction ratio
        total_possible = self.s1_count * self.cand_count
        reduction_ratio = (
            1.0 - (len(candidate_pairs) / total_possible) if total_possible > 0 else 0.0
        )

        return {
            "ground_truth_total": total_gt,
            "ground_truth_recovered": len(recovered),
            "blocking_recall": recall,
            "total_candidate_pairs": len(candidate_pairs),
            "reduction_ratio": reduction_ratio,
            "total_possible_pairs": total_possible,
            "oversized_blocks_count": len(self.oversized_blocks),
        }
