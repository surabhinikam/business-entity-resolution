"""
Audit script for analyzing transliterated vocabulary from the 100,000 processed Source 2 records.
Extracts:
1. Top 200 tokens from business_name_normalized for non-Latin records.
2. Candidate phonetically transliterated English/business terms.
3. Top legal suffix occurrences / variants.
"""

from __future__ import annotations

import collections
import os
import sys
from typing import Dict, List, Tuple

# Reconfigure stdout to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pyarrow.parquet as pq

DATA_DIR = "data/processed/train"


def run_audit():
    print(f"Reading Parquet dataset from: {DATA_DIR}")
    dataset = pq.read_table(DATA_DIR)
    print(f"Total rows in dataset: {dataset.num_rows}")

    scripts = dataset["business_name_script"].to_pylist()
    raw_names = dataset["business_name"].to_pylist()
    translit_names = dataset["business_name_transliterated"].to_pylist()
    norm_names = dataset["business_name_normalized"].to_pylist()
    tokens_col = dataset["business_name_tokens"].to_pylist()

    non_latin_count = 0
    token_freq = collections.Counter()
    token_examples = collections.defaultdict(list)
    script_counts = collections.Counter()
    suffix_counter = collections.Counter()

    for idx, script in enumerate(scripts):
        if script and script not in ("Latin", "Unknown"):
            non_latin_count += 1
            script_counts[script] += 1
            toks = tokens_col[idx] or []
            r_name = raw_names[idx]
            t_name = translit_names[idx]
            n_name = norm_names[idx]

            # Check suffix (last 1 or 2 tokens)
            if toks:
                if len(toks) >= 2 and f"{toks[-2]} {toks[-1]}" in ("pvt ltd", "public limited"):
                    suffix_counter[f"{toks[-2]} {toks[-1]}"] += 1
                else:
                    suffix_counter[toks[-1]] += 1

            for t in toks:
                token_freq[t] += 1
                if len(token_examples[t]) < 5:
                    token_examples[t].append((r_name, t_name, n_name))

    print(f"Non-Latin records found: {non_latin_count} ({non_latin_count / dataset.num_rows * 100:.2f}%)")
    print("\nNon-Latin script distribution:")
    for s, c in script_counts.most_common():
        print(f"  {s:15s}: {c:5d} ({c / non_latin_count * 100:.1f}%)")

    top_200 = token_freq.most_common(200)

    # Let's save a structured JSON and markdown report
    output_report_path = "reports/transliterated_vocabulary_audit.md"
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("# Transliterated Vocabulary Audit (100,000 Source 2 Records)\n\n")
        f.write(f"- Total records audited: **{dataset.num_rows:,}**\n")
        f.write(f"- Non-Latin business names: **{non_latin_count:,}** ({non_latin_count / dataset.num_rows * 100:.2f}%)\n")
        f.write(f"- Distinct tokens in non-Latin names: **{len(token_freq):,}**\n\n")

        f.write("## 1. Script Distribution in Non-Latin Records\n\n")
        f.write("| Script | Record Count | Percentage |\n|---|---|---|\n")
        for s, c in script_counts.most_common():
            f.write(f"| {s} | {c:,} | {c / non_latin_count * 100:.1f}% |\n")
        f.write("\n")

        f.write("## 2. Top Legal Suffix Variants Found\n\n")
        f.write("| Suffix / Ending Token | Count | Frequency % |\n|---|---|---|\n")
        for suf, count in suffix_counter.most_common(25):
            f.write(f"| `{suf}` | {count:,} | {count / non_latin_count * 100:.2f}% |\n")
        f.write("\n")

        f.write("## 3. Top 200 Transliterated Tokens\n\n")
        f.write("| Rank | Token | Frequency | Sample Original Business Names |\n")
        f.write("|---|---|---|---|\n")
        for rank, (tok, count) in enumerate(top_200, start=1):
            examples = "; ".join([f"`{r}`" for r, _, _ in token_examples[tok][:2]])
            f.write(f"| {rank} | `{tok}` | {count:,} | {examples} |\n")
        f.write("\n")

    print(f"\nAudit report saved to: {output_report_path}")
    return top_200, token_examples, suffix_counter


if __name__ == "__main__":
    run_audit()
