"""
Blocking evaluation module for Business Entity Resolution.

High-performance, memory-efficient evaluation using partitioned
Polars data structures and streaming key lookups.
Evaluates keys A-F against the full ground truth (7,638,365 pairs).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from src.analysis.data_loader import (
    load_ground_truth,
    explode_ground_truth,
    load_processed_parquet,
)
from src.candidate_generation.block_keys import (
    generate_all_keys,
    BLOCKING_KEY_NAMES,
)
from src.candidate_generation.address_parser import extract_address_number
from src.candidate_generation.name_tokens import extract_meaningful_tokens

logger = logging.getLogger(__name__)

KEY_LABELS = ["A", "B", "C", "D", "E", "F"]


def _ensure_keys_parquet(
    source_name: str,
    parquet_path: str,
    data_dir: str,
) -> None:
    """Precompute blocking keys parquet for a source if not already present."""
    if os.path.exists(parquet_path):
        return

    logger.info(f"Precomputing {source_name} keys to {parquet_path}...")
    t0 = time.time()
    cols = [
        "entity_id",
        "business_name_normalized",
        "business_name_transliterated",
        "business_address_normalized",
        "country_normalized",
    ]
    df = load_processed_parquet(data_dir, source=source_name, columns=cols).collect()

    eids = df["entity_id"].to_list()
    names = df["business_name_normalized"].to_list()
    translits = df["business_name_transliterated"].to_list()
    addrs = df["business_address_normalized"].to_list()
    countries = df["country_normalized"].to_list()
    del df

    key_A = []
    key_B = []
    key_C = []
    key_D = []
    key_E = []
    key_F = []
    has_nums = []
    has_addrs = []
    clean_countries = []

    for eid, name, translit, addr, country in zip(eids, names, translits, addrs, countries):
        c = country.strip().lower() if country else None
        clean_countries.append(c)

        ad = addr.strip() if addr else None
        has_addr = bool(ad)
        has_addrs.append(has_addr)

        num, has_num = extract_address_number(ad) if has_addr else (None, False)
        has_nums.append(has_num)

        if not c:
            key_A.append(None)
            key_B.append(None)
            key_C.append(None)
            key_D.append(None)
            key_E.append(None)
            key_F.append(None)
            continue

        nm = name.strip() if name else None
        key_A.append(f"A||{c}||{nm}" if nm else None)

        tr = translit.strip().lower() if translit else None
        key_B.append(f"B||{c}||{tr}" if tr else None)

        tokens = extract_meaningful_tokens(nm) if nm else []
        key_C.append(f"C||{c}||{tokens[0]} {tokens[1]}" if len(tokens) >= 2 else None)

        if tokens and has_num:
            key_D.append(f"D||{c}||{tokens[0]}||{num}")
        else:
            key_D.append(None)

        if tokens and not has_num:
            key_E.append(f"E||{c}||{tokens[0]}")
        else:
            key_E.append(None)

        key_F.append(f"F||{c}||{ad}" if has_addr else None)

    out_df = pl.DataFrame({
        "entity_id": eids,
        "country": clean_countries,
        "has_addr_num": has_nums,
        "has_addr": has_addrs,
        "key_A": key_A,
        "key_B": key_B,
        "key_C": key_C,
        "key_D": key_D,
        "key_E": key_E,
        "key_F": key_F,
    })

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    out_df.write_parquet(parquet_path, compression="snappy")
    logger.info(f"Finished {source_name}: {out_df.height:,} rows in {time.time()-t0:.1f}s")
    del out_df


def evaluate_blocking(
    s1_dir: str = "data/processed/train/source1",
    s2_dir: str = "data/processed/train/source2",
    s3_dir: str = "data/processed/train/source3",
    gt_path: str = "data/raw/train/train_ground_truth.tsv",
    output_dir: str = "reports/data_analysis",
    block_size_caps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive blocking evaluation using full ground truth.
    Memory-efficient, stream-based evaluation.
    """
    if block_size_caps is None:
        block_size_caps = [50, 100, 500, 1000, 5000, 10000, 50000]

    all_results: Dict[str, Any] = {}
    t_start = time.time()

    # 1. Ensure keys parquet files exist
    keys_dir = "data/processed/train/blocking_keys"
    s1_parquet = os.path.join(keys_dir, "s1_keys.parquet")
    s2_parquet = os.path.join(keys_dir, "s2_keys.parquet")
    s3_parquet = os.path.join(keys_dir, "s3_keys.parquet")

    _ensure_keys_parquet("source1", s1_parquet, s1_dir)
    _ensure_keys_parquet("source2", s2_parquet, s2_dir)
    _ensure_keys_parquet("source3", s3_parquet, s3_dir)

    # 2. Get dataset sizes and metadata
    total_s1 = pl.read_parquet(s1_parquet, columns=["entity_id"]).height
    total_s2 = pl.read_parquet(s2_parquet, columns=["entity_id"]).height
    total_s3 = pl.read_parquet(s3_parquet, columns=["entity_id"]).height
    total_cand = total_s2 + total_s3
    total_possible = total_s1 * total_cand

    # 3. Load Ground Truth
    logger.info("Loading ground truth...")
    t_gt = time.time()
    gt_raw = load_ground_truth(gt_path)
    gt_exploded = explode_ground_truth(gt_raw)
    del gt_raw

    total_gt = gt_exploded.height
    gt_s2_df = gt_exploded.filter(pl.col("target_source") == "source2")
    gt_s3_df = gt_exploded.filter(pl.col("target_source") == "source3")
    gt_s2_count = gt_s2_df.height
    gt_s3_count = gt_s3_df.height
    del gt_exploded

    s1_ids_s2 = gt_s2_df["source1_entity_id"].to_list()
    cand_ids_s2 = gt_s2_df["matched_entity_id"].to_list()
    del gt_s2_df

    s1_ids_s3 = gt_s3_df["source1_entity_id"].to_list()
    cand_ids_s3 = gt_s3_df["matched_entity_id"].to_list()
    del gt_s3_df

    all_gt_s1_ids = s1_ids_s2 + s1_ids_s3
    all_gt_cand_ids = cand_ids_s2 + cand_ids_s3
    s2_slice = slice(0, gt_s2_count)
    s3_slice = slice(gt_s2_count, total_gt)

    # Singletons
    unique_s1_in_gt = set(all_gt_s1_ids)
    singleton_s1_count = total_s1 - len(unique_s1_in_gt)

    all_results["ground_truth"] = {
        "total_pairs": total_gt,
        "s1_to_s2_pairs": gt_s2_count,
        "s1_to_s3_pairs": gt_s3_count,
    }
    all_results["data_stats"] = {
        "s1_entities": total_s1,
        "candidate_entities": total_cand,
        "total_possible_pairs": total_possible,
        "singleton_s1_entities": singleton_s1_count,
    }
    logger.info(
        f"  Ground truth pairs: {total_gt:,} (S1->S2: {gt_s2_count:,}, S1->S3: {gt_s3_count:,}) "
        f"loaded in {time.time()-t_gt:.1f}s"
    )

    # 4. Load metadata for stratifications
    logger.info("Loading entity metadata for stratifications...")
    s1_meta = pl.read_parquet(s1_parquet, columns=["entity_id", "country", "has_addr_num"])
    s1_country_map = dict(zip(s1_meta["entity_id"], s1_meta["country"]))
    s1_has_num_map = dict(zip(s1_meta["entity_id"], s1_meta["has_addr_num"]))
    del s1_meta

    s2_meta = pl.read_parquet(s2_parquet, columns=["entity_id", "has_addr_num"])
    cand_has_num_map_s2 = dict(zip(s2_meta["entity_id"], s2_meta["has_addr_num"]))
    del s2_meta

    s3_meta = pl.read_parquet(s3_parquet, columns=["entity_id", "has_addr_num"])
    cand_has_num_map_s3 = dict(zip(s3_meta["entity_id"], s3_meta["has_addr_num"]))
    del s3_meta

    # Boolean masks for GT pairs
    # Source masks
    gt_s2_mask = np.zeros(total_gt, dtype=bool)
    gt_s2_mask[s2_slice] = True
    gt_s3_mask = ~gt_s2_mask

    # Country masks (based on S1 entity country)
    gt_s1_countries = [s1_country_map.get(s, "") for s in all_gt_s1_ids]
    gt_us_mask = np.array([c == "united states" for c in gt_s1_countries], dtype=bool)
    gt_in_mask = np.array([c == "india" for c in gt_s1_countries], dtype=bool)
    del s1_country_map, gt_s1_countries

    # Address number masks
    s1_has_nums = np.array([bool(s1_has_num_map.get(s, False)) for s in all_gt_s1_ids], dtype=bool)
    del s1_has_num_map

    cand_has_nums_s2 = [bool(cand_has_num_map_s2.get(c, False)) for c in cand_ids_s2]
    del cand_has_num_map_s2
    cand_has_nums_s3 = [bool(cand_has_num_map_s3.get(c, False)) for c in cand_ids_s3]
    del cand_has_num_map_s3
    cand_has_nums = np.array(cand_has_nums_s2 + cand_has_nums_s3, dtype=bool)
    del cand_has_nums_s2, cand_has_nums_s3

    gt_both_num_mask = s1_has_nums & cand_has_nums
    gt_miss_num_mask = ~gt_both_num_mask
    del s1_has_nums, cand_has_nums

    # 5. Evaluate each key independently (block stats + GT match vector)
    logger.info("Evaluating keys A through F independently...")
    match_vectors: Dict[str, np.ndarray] = {}
    matched_keys_dict: Dict[str, Dict[int, str]] = {}  # key_label -> {pair_idx: key_string} for capped recall
    block_tables: Dict[str, pl.DataFrame] = {}
    key_block_stats: Dict[str, Dict[str, Any]] = {}
    s1_coverage_stats: Dict[str, int] = {}

    for k in KEY_LABELS:
        t_k = time.time()
        col_name = f"key_{k}"
        logger.info(f"\n--- Evaluating Key {k}: {BLOCKING_KEY_NAMES[k]} ---")

        # 5a. Block sizes and candidate counts
        s1_df_k = pl.read_parquet(s1_parquet, columns=["entity_id", col_name]).filter(pl.col(col_name).is_not_null())
        s1_coverage_stats[k] = s1_df_k.height

        s1_counts = s1_df_k.group_by(col_name).len().rename({col_name: "key", "len": "s1_count"})
        s1_map = dict(zip(s1_df_k["entity_id"], s1_df_k[col_name]))
        del s1_df_k

        s2_df_k = pl.read_parquet(s2_parquet, columns=["entity_id", col_name]).filter(pl.col(col_name).is_not_null())
        s2_counts = s2_df_k.group_by(col_name).len()
        s2_map = dict(zip(s2_df_k["entity_id"], s2_df_k[col_name]))
        del s2_df_k

        s3_df_k = pl.read_parquet(s3_parquet, columns=["entity_id", col_name]).filter(pl.col(col_name).is_not_null())
        s3_counts = s3_df_k.group_by(col_name).len()
        s3_map = dict(zip(s3_df_k["entity_id"], s3_df_k[col_name]))
        del s3_df_k

        cand_counts = (
            pl.concat([s2_counts, s3_counts])
            .group_by(col_name)
            .sum()
            .rename({col_name: "key", "len": "cand_count"})
        )
        del s2_counts, s3_counts

        blocks = (
            s1_counts.join(cand_counts, on="key", how="inner")
            .with_columns((pl.col("s1_count") * pl.col("cand_count")).alias("block_size"))
        )
        del s1_counts, cand_counts
        block_tables[k] = blocks

        common_b = blocks.height
        total_p = int(blocks["block_size"].sum()) if common_b > 0 else 0
        max_b = int(blocks["block_size"].max()) if common_b > 0 else 0
        rr = 1.0 - (total_p / total_possible) if total_possible > 0 else 0.0

        key_block_stats[k] = {
            "common_blocks": common_b,
            "candidate_pairs": total_p,
            "max_block_size": max_b,
            "reduction_ratio": rr,
        }

        # 5b. Match evaluation on Ground Truth
        # S2 pairs
        match_s2_list = []
        matched_keys_k: Dict[int, str] = {}
        for idx, (s, c) in enumerate(zip(s1_ids_s2, cand_ids_s2)):
            k_s1 = s1_map.get(s)
            k_c = s2_map.get(c)
            is_match = bool(k_s1 and k_s1 == k_c)
            match_s2_list.append(is_match)
            if is_match:
                matched_keys_k[idx] = k_s1
        del s2_map

        # S3 pairs
        match_s3_list = []
        offset = gt_s2_count
        for idx, (s, c) in enumerate(zip(s1_ids_s3, cand_ids_s3)):
            k_s1 = s1_map.get(s)
            k_c = s3_map.get(c)
            is_match = bool(k_s1 and k_s1 == k_c)
            match_s3_list.append(is_match)
            if is_match:
                matched_keys_k[offset + idx] = k_s1
        del s3_map, s1_map

        match_vec = np.array(match_s2_list + match_s3_list, dtype=bool)
        del match_s2_list, match_s3_list
        match_vectors[k] = match_vec
        matched_keys_dict[k] = matched_keys_k

        rec_count = int(match_vec.sum())
        recall_val = rec_count / total_gt
        logger.info(
            f"  Key {k}: Recall={recall_val:.4f} ({rec_count:,}/{total_gt:,}), "
            f"Candidates={total_p:,}, Blocks={common_b:,}, MaxBlock={max_b:,} ({time.time()-t_k:.1f}s)"
        )

    # 6. Build Individual Key Results
    ind_results: Dict[str, Dict[str, Any]] = {}
    for k in KEY_LABELS:
        m = match_vectors[k]
        b = key_block_stats[k]
        rec = int(m.sum())
        rec_s2 = int(m[gt_s2_mask].sum())
        rec_s3 = int(m[gt_s3_mask].sum())
        rec_both_num = int(m[gt_both_num_mask].sum())
        rec_miss_num = int(m[gt_miss_num_mask].sum())

        country_res = {}
        if gt_us_mask.sum() > 0:
            rec_us = int(m[gt_us_mask].sum())
            tot_us = int(gt_us_mask.sum())
            country_res["united states"] = {
                "gt_pairs": tot_us,
                "recovered": rec_us,
                "recall": round(rec_us / tot_us, 6),
            }
        if gt_in_mask.sum() > 0:
            rec_in = int(m[gt_in_mask].sum())
            tot_in = int(gt_in_mask.sum())
            country_res["india"] = {
                "gt_pairs": tot_in,
                "recovered": rec_in,
                "recall": round(rec_in / tot_in, 6),
            }

        s1_w_k = s1_coverage_stats[k]
        tot_both = int(gt_both_num_mask.sum())
        tot_miss = int(gt_miss_num_mask.sum())

        ind_results[k] = {
            "key_label": k,
            "key_name": BLOCKING_KEY_NAMES[k],
            "overall": {
                "gt_pairs": total_gt,
                "recovered": rec,
                "recall": round(rec / total_gt, 6),
                "candidate_pairs": b["candidate_pairs"],
                "common_blocks": b["common_blocks"],
                "max_block_size": b["max_block_size"],
                "reduction_ratio": round(b["reduction_ratio"], 8),
                "oversized_blocks": 0,
            },
            "by_source": {
                "s1_to_s2": {
                    "gt_pairs": gt_s2_count,
                    "recovered": rec_s2,
                    "recall": round(rec_s2 / gt_s2_count, 6),
                },
                "s1_to_s3": {
                    "gt_pairs": gt_s3_count,
                    "recovered": rec_s3,
                    "recall": round(rec_s3 / gt_s3_count, 6),
                },
            },
            "by_address_number": {
                "both_have_addr_num": {
                    "gt_pairs": tot_both,
                    "recall": round(rec_both_num / tot_both, 6) if tot_both > 0 else 0.0,
                },
                "missing_addr_num": {
                    "gt_pairs": tot_miss,
                    "recall": round(rec_miss_num / tot_miss, 6) if tot_miss > 0 else 0.0,
                },
            },
            "by_country": country_res,
            "s1_coverage": {
                "s1_with_keys": s1_w_k,
                "s1_total": total_s1,
                "coverage_pct": round(100.0 * s1_w_k / total_s1, 3),
            },
        }

    all_results["individual_keys"] = ind_results

    # 7. Incremental Combinations
    logger.info("\nComputing incremental combinations (A, A+B, ..., A+B+C+D+E+F)...")
    incremental_results: List[Dict[str, Any]] = []
    active_labels: List[str] = []
    cum_match = np.zeros(total_gt, dtype=bool)
    cum_candidates = 0
    cum_blocks = 0
    cum_max_block = 0

    for k in KEY_LABELS:
        t_inc = time.time()
        active_labels.append(k)
        combo_name = "+".join(active_labels)

        cum_match = cum_match | match_vectors[k]
        cum_candidates += key_block_stats[k]["candidate_pairs"]
        cum_blocks += key_block_stats[k]["common_blocks"]
        cum_max_block = max(cum_max_block, key_block_stats[k]["max_block_size"])
        cum_rr = 1.0 - (cum_candidates / total_possible) if total_possible > 0 else 0.0

        rec = int(cum_match.sum())
        rec_s2 = int(cum_match[gt_s2_mask].sum())
        rec_s3 = int(cum_match[gt_s3_mask].sum())

        inc_res = {
            "keys": combo_name,
            "key_labels": list(active_labels),
            "overall": {
                "gt_pairs": total_gt,
                "recovered": rec,
                "recall": round(rec / total_gt, 6),
                "candidate_pairs": cum_candidates,
                "common_blocks": cum_blocks,
                "max_block_size": cum_max_block,
                "reduction_ratio": round(cum_rr, 8),
            },
            "by_source": {
                "s1_to_s2": {"recovered": rec_s2, "recall": round(rec_s2 / gt_s2_count, 6)},
                "s1_to_s3": {"recovered": rec_s3, "recall": round(rec_s3 / gt_s3_count, 6)},
            },
            "eval_time_seconds": round(time.time() - t_inc, 2),
        }
        incremental_results.append(inc_res)
        logger.info(
            f"  Combo {combo_name}: Recall={rec / total_gt:.4f} ({rec:,}/{total_gt:,}), "
            f"Candidates={cum_candidates:,}, Blocks={cum_blocks:,}, MaxBlock={cum_max_block:,}"
        )

    all_results["incremental"] = incremental_results

    # 8. Block-Size Capping Experiments
    logger.info("\nRunning block-size capping experiments on full A+B+C+D+E+F union...")
    capping_results: List[Dict[str, Any]] = []

    rec_uncapped = int(cum_match.sum())
    recall_uncapped = rec_uncapped / total_gt
    pairs_uncapped = cum_candidates
    blocks_uncapped = cum_blocks

    capping_results.append({
        "cap": "none",
        "recall": round(recall_uncapped, 6),
        "recovered": rec_uncapped,
        "candidate_pairs": pairs_uncapped,
        "common_blocks": blocks_uncapped,
        "oversized_blocks": 0,
        "recall_lost_pct": 0.0,
        "reduction_ratio": round(1.0 - (pairs_uncapped / total_possible), 8),
    })

    for cap in block_size_caps:
        t_cap = time.time()
        oversized_count = 0
        cand_pairs_capped = 0
        common_blocks_capped = 0

        # For each key, determine which GT matches survived the cap
        survived_matches = []
        for k in KEY_LABELS:
            blocks = block_tables[k]
            over = blocks.filter(pl.col("block_size") > cap)
            under = blocks.filter(pl.col("block_size") <= cap)

            oversized_count += over.height
            cand_pairs_capped += int(under["block_size"].sum()) if under.height > 0 else 0
            common_blocks_capped += under.height

            if over.height > 0:
                over_set = set(over["key"].to_list())
                # A pair is valid under cap if it matched on k AND its key is not in over_set
                m_k = match_vectors[k].copy()
                # Unset matches that belonged to oversized blocks
                matched_pairs = matched_keys_dict[k]
                for p_idx, k_str in matched_pairs.items():
                    if k_str in over_set:
                        m_k[p_idx] = False
                survived_matches.append(m_k)
            else:
                survived_matches.append(match_vectors[k])

        cum_capped_match = survived_matches[0]
        for sm in survived_matches[1:]:
            cum_capped_match = cum_capped_match | sm

        rec_capped = int(cum_capped_match.sum())
        recall_capped = rec_capped / total_gt
        recall_lost = recall_uncapped - recall_capped
        recall_lost_pct = (recall_lost / recall_uncapped * 100.0) if recall_uncapped > 0 else 0.0
        rr_capped = 1.0 - (cand_pairs_capped / total_possible) if total_possible > 0 else 0.0

        cap_res = {
            "cap": cap,
            "recall": round(recall_capped, 6),
            "recovered": rec_capped,
            "candidate_pairs": cand_pairs_capped,
            "common_blocks": common_blocks_capped,
            "oversized_blocks": oversized_count,
            "recall_lost_vs_uncapped": round(recall_lost, 6),
            "recall_lost_pct": round(recall_lost_pct, 3),
            "reduction_ratio": round(rr_capped, 8),
            "eval_time_seconds": round(time.time() - t_cap, 2),
        }
        capping_results.append(cap_res)
        logger.info(
            f"  Cap={cap:,}: Recall={recall_capped:.4f} (lost {recall_lost_pct:.2f}%), "
            f"Candidates={cand_pairs_capped:,}, OversizedBlocks={oversized_count:,}"
        )

    all_results["block_size_experiments"] = capping_results

    # 9. Key Coverage Analysis
    logger.info("\nComputing multi-key and exclusive-key coverage...")
    n_covering = (
        match_vectors["A"].astype(int) +
        match_vectors["B"].astype(int) +
        match_vectors["C"].astype(int) +
        match_vectors["D"].astype(int) +
        match_vectors["E"].astype(int) +
        match_vectors["F"].astype(int)
    )

    counts = np.bincount(n_covering, minlength=7)
    pair_cov_dist = {str(i): int(counts[i]) for i in range(7)}
    uncovered = int(counts[0])
    uncovered_pct = round(100.0 * uncovered / total_gt, 3)

    key_exclusive = {}
    for k in KEY_LABELS:
        exc_mask = match_vectors[k] & (n_covering == 1)
        key_exclusive[k] = int(exc_mask.sum())

    all_results["coverage_analysis"] = {
        "pair_coverage_distribution": pair_cov_dist,
        "key_exclusive_pairs": key_exclusive,
        "uncovered_pairs": uncovered,
        "uncovered_pct": uncovered_pct,
    }

    all_results["total_eval_time_seconds"] = round(time.time() - t_start, 2)
    logger.info(f"\nFull evaluation completed in {all_results['total_eval_time_seconds']:.1f}s!")

    # 10. Save JSON output
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "blocking_v2_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Machine-readable results saved to {json_path}")

    return all_results
