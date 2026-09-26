#!/usr/bin/env python3
"""
Blocking V2 Evaluation Script.

Runs the comprehensive blocking evaluation with keys A-F against
the full ground truth. Generates reports and machine-readable results.

Usage:
    .venv/bin/python scripts/evaluate_blocking_v2.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.candidate_generation.evaluate_blocking import evaluate_blocking


def generate_report(results: dict, output_path: str) -> None:
    """Generate a markdown report from evaluation results."""

    lines = []
    lines.append("# Blocking V2 Evaluation Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Evaluation Time**: {results.get('total_eval_time_seconds', 0):.1f}s")
    lines.append("")

    # Data stats
    ds = results.get("data_stats", {})
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- S1 entities: {ds.get('s1_entities', 0):,}")
    lines.append(f"- Candidate entities (S2+S3): {ds.get('candidate_entities', 0):,}")
    lines.append(f"- Total possible pairs: {ds.get('total_possible_pairs', 0):,}")
    lines.append(f"- Singleton S1 entities: {ds.get('singleton_s1_entities', 0):,}")
    lines.append("")

    gt = results.get("ground_truth", {})
    lines.append(f"- **Total GT pairs**: {gt.get('total_pairs', 0):,}")
    lines.append(f"  - S1→S2: {gt.get('s1_to_s2_pairs', 0):,}")
    lines.append(f"  - S1→S3: {gt.get('s1_to_s3_pairs', 0):,}")
    lines.append("")

    # Individual key evaluation
    lines.append("## Individual Key Evaluation")
    lines.append("")
    lines.append("| Key | Name | Recall | Recovered | Candidate Pairs | Blocks | Max Block | RR | S1→S2 Recall | S1→S3 Recall |")
    lines.append("|-----|------|--------|-----------|-----------------|--------|-----------|-----|--------------|--------------|")

    ind = results.get("individual_keys", {})
    for label in ["A", "B", "C", "D", "E", "F"]:
        r = ind.get(label, {})
        o = r.get("overall", {})
        bs = r.get("by_source", {})
        lines.append(
            f"| **{label}** | {r.get('key_name', '')} | "
            f"{o.get('recall', 0):.4f} | {o.get('recovered', 0):,} | "
            f"{o.get('candidate_pairs', 0):,} | {o.get('common_blocks', 0):,} | "
            f"{o.get('max_block_size', 0):,} | {o.get('reduction_ratio', 0):.6f} | "
            f"{bs.get('s1_to_s2', {}).get('recall', 0):.4f} | "
            f"{bs.get('s1_to_s3', {}).get('recall', 0):.4f} |"
        )
    lines.append("")

    # Address number stratification
    lines.append("### Stratification by Address Number Presence")
    lines.append("")
    lines.append("| Key | Both Have Addr Num (Recall) | Missing Addr Num (Recall) | Addr Num GT Pairs | No Addr GT Pairs |")
    lines.append("|-----|---------------------------|--------------------------|-------------------|------------------|")
    for label in ["A", "B", "C", "D", "E", "F"]:
        r = ind.get(label, {})
        ba = r.get("by_address_number", {})
        wn = ba.get("both_have_addr_num", {})
        nn = ba.get("missing_addr_num", {})
        lines.append(
            f"| **{label}** | {wn.get('recall', 0):.4f} | {nn.get('recall', 0):.4f} | "
            f"{wn.get('gt_pairs', 0):,} | {nn.get('gt_pairs', 0):,} |"
        )
    lines.append("")

    # Country stratification
    lines.append("### Stratification by Country")
    lines.append("")
    # Collect all countries
    all_countries = set()
    for label in ["A", "B", "C", "D", "E", "F"]:
        r = ind.get(label, {})
        all_countries |= set(r.get("by_country", {}).keys())

    if all_countries:
        header = "| Key |"
        sep = "|-----|"
        for c in sorted(all_countries):
            header += f" {c.title()} Recall |"
            sep += "------------|"
        lines.append(header)
        lines.append(sep)

        for label in ["A", "B", "C", "D", "E", "F"]:
            r = ind.get(label, {})
            bc = r.get("by_country", {})
            row = f"| **{label}** |"
            for c in sorted(all_countries):
                cd = bc.get(c, {})
                row += f" {cd.get('recall', 0):.4f} |"
            lines.append(row)
        lines.append("")

    # S1 coverage
    lines.append("### S1 Entity Coverage")
    lines.append("")
    lines.append("| Key | S1 with Keys | S1 Total | Coverage % |")
    lines.append("|-----|-------------|----------|-----------|")
    for label in ["A", "B", "C", "D", "E", "F"]:
        r = ind.get(label, {})
        sc = r.get("s1_coverage", {})
        lines.append(
            f"| **{label}** | {sc.get('s1_with_keys', 0):,} | "
            f"{sc.get('s1_total', 0):,} | {sc.get('coverage_pct', 0):.2f}% |"
        )
    lines.append("")

    # Incremental evaluation
    lines.append("## Incremental Key Evaluation")
    lines.append("")
    lines.append("| Keys | Recall | Recovered | Candidate Pairs | Blocks | Max Block | RR | S1→S2 Recall | S1→S3 Recall |")
    lines.append("|------|--------|-----------|-----------------|--------|-----------|-----|--------------|--------------|")

    for inc in results.get("incremental", []):
        o = inc.get("overall", {})
        bs = inc.get("by_source", {})
        lines.append(
            f"| {inc.get('keys', '')} | {o.get('recall', 0):.4f} | "
            f"{o.get('recovered', 0):,} | {o.get('candidate_pairs', 0):,} | "
            f"{o.get('common_blocks', 0):,} | {o.get('max_block_size', 0):,} | "
            f"{o.get('reduction_ratio', 0):.6f} | "
            f"{bs.get('s1_to_s2', {}).get('recall', 0):.4f} | "
            f"{bs.get('s1_to_s3', {}).get('recall', 0):.4f} |"
        )
    lines.append("")

    # Block-size capping experiments
    lines.append("## Block-Size Capping Experiments")
    lines.append("")
    lines.append("Using full A+B+C+D+E+F union:")
    lines.append("")
    lines.append("| Cap | Recall | Candidate Pairs | Blocks | Oversized | Recall Lost % | RR |")
    lines.append("|-----|--------|-----------------|--------|-----------|--------------|-----|")

    for cap in results.get("block_size_experiments", []):
        lines.append(
            f"| {cap.get('cap', '')} | {cap.get('recall', 0):.4f} | "
            f"{cap.get('candidate_pairs', 0):,} | {cap.get('common_blocks', 0):,} | "
            f"{cap.get('oversized_blocks', 0):,} | {cap.get('recall_lost_pct', 0):.2f}% | "
            f"{cap.get('reduction_ratio', 0):.6f} |"
        )
    lines.append("")

    # Coverage analysis
    cov = results.get("coverage_analysis", {})
    lines.append("## Key Coverage Analysis")
    lines.append("")
    lines.append("### GT Pairs by Number of Covering Keys")
    lines.append("")
    lines.append("| # Keys | Count | % |")
    lines.append("|--------|-------|---|")
    total_gt = gt.get("total_pairs", 1)
    for n_keys, count in sorted(cov.get("pair_coverage_distribution", {}).items(), key=lambda x: int(x[0])):
        pct = 100.0 * count / total_gt if total_gt > 0 else 0
        lines.append(f"| {n_keys} | {count:,} | {pct:.2f}% |")
    lines.append("")

    lines.append(f"**Uncovered pairs**: {cov.get('uncovered_pairs', 0):,} ({cov.get('uncovered_pct', 0):.2f}%)")
    lines.append("")

    lines.append("### Exclusive Key Contributions (pairs covered by ONLY this key)")
    lines.append("")
    lines.append("| Key | Exclusive Pairs |")
    lines.append("|-----|----------------|")
    for label in ["A", "B", "C", "D", "E", "F"]:
        exc = cov.get("key_exclusive_pairs", {}).get(label, 0)
        lines.append(f"| **{label}** | {exc:,} |")
    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")

    # Find best incremental config
    best_inc = None
    for inc in results.get("incremental", []):
        if best_inc is None or inc["overall"]["recall"] > best_inc["overall"]["recall"]:
            best_inc = inc

    if best_inc:
        lines.append(f"**Best recall configuration**: {best_inc['keys']}")
        lines.append(f"- Recall: {best_inc['overall']['recall']:.4f}")
        lines.append(f"- Candidate pairs: {best_inc['overall']['candidate_pairs']:,}")
        lines.append(f"- Reduction ratio: {best_inc['overall']['reduction_ratio']:.6f}")
    lines.append("")

    # Find best cap
    caps = results.get("block_size_experiments", [])
    best_cap = None
    for cap in caps:
        if cap.get("cap") == "none":
            continue
        if cap.get("recall_lost_pct", 100) < 1.0:  # Less than 1% recall loss
            if best_cap is None or cap.get("candidate_pairs", float("inf")) < best_cap.get("candidate_pairs", float("inf")):
                best_cap = cap
    if best_cap:
        lines.append(f"**Recommended block size cap**: {best_cap['cap']:,}")
        lines.append(f"- Recall: {best_cap['recall']:.4f} (lost {best_cap['recall_lost_pct']:.2f}%)")
        lines.append(f"- Candidate pairs: {best_cap['candidate_pairs']:,}")
        lines.append(f"- Oversized blocks removed: {best_cap['oversized_blocks']:,}")

    report_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report saved to {output_path}")


def main():
    """Run the full blocking evaluation."""
    logger.info("=" * 70)
    logger.info("BLOCKING V2 EVALUATION — Full Ground Truth")
    logger.info("=" * 70)

    t0 = time.time()

    results = evaluate_blocking(
        s1_dir="data/processed/train/source1",
        s2_dir="data/processed/train/source2",
        s3_dir="data/processed/train/source3",
        gt_path="data/raw/train/train_ground_truth.tsv",
        output_dir="reports/data_analysis",
        block_size_caps=[100, 500, 1000, 5000, 10000, 50000],
    )

    # Generate markdown report
    report_path = os.path.join("reports", "data_analysis", "blocking_v2.md")
    generate_report(results, report_path)

    logger.info(f"\nTotal elapsed time: {time.time() - t0:.1f}s")
    logger.info("Done!")


if __name__ == "__main__":
    main()
