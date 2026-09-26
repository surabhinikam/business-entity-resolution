"""
Blocking strategy evaluation utilities for Business Entity Resolution.
Evaluates candidate blocking keys, block size distributions, ground-truth recall,
and reduction ratio.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Set, Dict, Any, Callable, Optional, Tuple, Iterable
import logging

logger = logging.getLogger(__name__)


def compute_block_statistics(
    s1_keys: Dict[str, Set[str]],
    cand_keys: Dict[str, Set[str]],
    gt_pairs: Set[Tuple[str, str]],
    total_possible_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute blocking statistics for a given blocking key assignment.

    Parameters:
      - s1_keys: mapping entity_id -> set of blocking keys for Source 1
      - cand_keys: mapping entity_id -> set of blocking keys for Candidates (Source 2 / 3)
      - gt_pairs: set of (s1_id, candidate_id) ground-truth match pairs
      - total_possible_pairs: total Cartesian space |S1| * (|S2| + |S3|)
    """
    total_gt = len(gt_pairs)
    if total_gt == 0:
        return {"error": "Ground truth pairs set is empty"}

    s1_index: Dict[str, List[str]] = defaultdict(list)
    for eid, keys in s1_keys.items():
        for k in keys:
            if k:
                s1_index[k].append(eid)

    cand_index: Dict[str, List[str]] = defaultdict(list)
    for eid, keys in cand_keys.items():
        for k in keys:
            if k:
                cand_index[k].append(eid)

    common_keys = set(s1_index.keys()) & set(cand_index.keys())

    total_candidates = 0
    max_block_size = 0
    block_size_distribution: Dict[str, int] = {
        "<= 10": 0,
        "11 - 100": 0,
        "101 - 1000": 0,
        "1001 - 10000": 0,
        "> 10000": 0,
    }

    for k in common_keys:
        n_s1 = len(s1_index[k])
        n_cand = len(cand_index[k])
        block_candidates = n_s1 * n_cand
        total_candidates += block_candidates
        if block_candidates > max_block_size:
            max_block_size = block_candidates

        if block_candidates <= 10:
            block_size_distribution["<= 10"] += 1
        elif block_candidates <= 100:
            block_size_distribution["11 - 100"] += 1
        elif block_candidates <= 1000:
            block_size_distribution["101 - 1000"] += 1
        elif block_candidates <= 10000:
            block_size_distribution["1001 - 10000"] += 1
        else:
            block_size_distribution["> 10000"] += 1

    recovered_gt = 0
    for s1_id, cand_id in gt_pairs:
        keys_s1 = s1_keys.get(s1_id)
        keys_cand = cand_keys.get(cand_id)
        if keys_s1 and keys_cand and bool(keys_s1 & keys_cand):
            recovered_gt += 1

    recall = recovered_gt / total_gt if total_gt > 0 else 0.0

    reduction_ratio = 0.0
    if total_possible_pairs and total_possible_pairs > 0:
        reduction_ratio = max(0.0, 1.0 - (total_candidates / total_possible_pairs))

    return {
        "total_common_blocks": len(common_keys),
        "total_candidate_pairs": total_candidates,
        "max_block_size": max_block_size,
        "block_size_distribution": block_size_distribution,
        "ground_truth_total": total_gt,
        "ground_truth_recovered": recovered_gt,
        "blocking_recall": recall,
        "reduction_ratio": reduction_ratio,
    }
