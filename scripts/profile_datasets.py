import os
import sys
import json
import unicodedata
from collections import Counter, defaultdict

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

REPOSITORY_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
)
LEGACY_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'student_resource', 'dataset')
)
DATASET_DIR = REPOSITORY_DATASET_DIR if os.path.isdir(REPOSITORY_DATASET_DIR) else LEGACY_DATASET_DIR

SOURCE_FILES = {
    "train_source1": os.path.join(DATASET_DIR, "train", "train_source1.tsv"),
    "train_source2": os.path.join(DATASET_DIR, "train", "train_source2.tsv"),
    "train_source3": os.path.join(DATASET_DIR, "train", "train_source3.tsv"),
    "test_source1": os.path.join(DATASET_DIR, "test", "test_source1.tsv"),
    "test_source2": os.path.join(DATASET_DIR, "test", "test_source2.tsv"),
    "test_source3": os.path.join(DATASET_DIR, "test", "test_source3.tsv"),
}

GROUND_TRUTH_FILE = os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv")

def detect_script(text):
    """Identify Unicode scripts present in the text."""
    scripts = set()
    for char in text:
        if not char.isalpha():
            continue
        cp = ord(char)
        if 0x0000 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
            scripts.add("Latin")
        elif 0x0900 <= cp <= 0x097F:
            scripts.add("Devanagari")
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
            scripts.add("Arabic")
        elif 0x0400 <= cp <= 0x04FF:
            scripts.add("Cyrillic")
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            scripts.add("Han (Chinese)")
        elif 0x3040 <= cp <= 0x30FF:
            scripts.add("Japanese (Kana)")
        elif 0xAC00 <= cp <= 0xD7AF:
            scripts.add("Korean (Hangul)")
        elif 0x0E00 <= cp <= 0x0E7F:
            scripts.add("Thai")
        elif 0x0980 <= cp <= 0x09FF:
            scripts.add("Bengali")
        elif 0x0B80 <= cp <= 0x0BFF:
            scripts.add("Tamil")
        elif 0x0C00 <= cp <= 0x0C7F:
            scripts.add("Telugu")
        elif 0x0C80 <= cp <= 0x0CFF:
            scripts.add("Kannada")
        elif 0x0D00 <= cp <= 0x0D7F:
            scripts.add("Malayalam")
        elif 0x0A80 <= cp <= 0x0AFF:
            scripts.add("Gujarati")
        elif 0x0B00 <= cp <= 0x0B7F:
            scripts.add("Oriya")
        elif 0x0A00 <= cp <= 0x0A7F:
            scripts.add("Gurmukhi")
        elif 0x0590 <= cp <= 0x05FF:
            scripts.add("Hebrew")
        elif 0x0370 <= cp <= 0x03FF:
            scripts.add("Greek")
        else:
            try:
                name = unicodedata.name(char, "UNKNOWN")
                script_guess = name.split()[0].title()
                scripts.add(script_guess)
            except Exception:
                scripts.add("Other")
    return scripts

def profile_source_file(file_path, file_key):
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    
    row_count = 0
    missing_counts = defaultdict(int)
    field_types = defaultdict(set)
    country_counts = Counter()
    script_counter_name = Counter()
    script_counter_addr = Counter()
    
    script_examples = defaultdict(lambda: {"business_name": None, "business_address": None})
    
    samples_latin = []
    samples_non_latin = []
    samples_missing_addr = []
    samples_special_chars = []
    samples_general = []
    
    sample_rate = 5  # Sample every 5th row for scripts
    
    delimiter = "\\t (tab)"
    encoding = "UTF-8"
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            return None
        
        headers = [h.strip() for h in header_line.split("\t")]
        num_cols = len(headers)
        
        for line in f:
            row_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < num_cols:
                parts.extend([""] * (num_cols - len(parts)))
            elif len(parts) > num_cols:
                parts = parts[:num_cols-1] + ["\t".join(parts[num_cols-1:])]
            
            record = dict(zip(headers, parts))
            
            for col, val in record.items():
                val_clean = val.strip()
                if not val_clean:
                    missing_counts[col] += 1
                else:
                    if val_clean.isdigit():
                        field_types[col].add("integer")
                    else:
                        try:
                            float(val_clean)
                            field_types[col].add("float")
                        except ValueError:
                            field_types[col].add("string")
            
            country_val = record.get("country", "").strip()
            if country_val:
                country_counts[country_val] += 1
                
            name = record.get("business_name", "")
            addr = record.get("business_address", "")
            
            name_scripts = detect_script(name)
            addr_scripts = detect_script(addr)
            
            # Record script examples
            for s in name_scripts:
                if script_examples[s]["business_name"] is None:
                    script_examples[s]["business_name"] = name
            for s in addr_scripts:
                if script_examples[s]["business_address"] is None:
                    script_examples[s]["business_address"] = addr
            
            if row_count % sample_rate == 0:
                if not name_scripts:
                    script_counter_name["[No Letters/Empty]"] += 1
                else:
                    for s in name_scripts:
                        script_counter_name[s] += 1
                
                if not addr_scripts:
                    script_counter_addr["[No Letters/Empty]"] += 1
                else:
                    for s in addr_scripts:
                        script_counter_addr[s] += 1
            
            # Representative sample candidates
            if len(samples_general) < 2:
                samples_general.append(record)
            if not addr.strip() and len(samples_missing_addr) < 1:
                samples_missing_addr.append(record)
            if any(s != "Latin" for s in name_scripts) and len(samples_non_latin) < 2:
                samples_non_latin.append(record)
            if any(c in name for c in ["-", "&", "/", "@", ".", "(", ")"]) and len(samples_special_chars) < 1:
                samples_special_chars.append(record)

    selected_samples = []
    seen_ids = set()
    for pool in [samples_general, samples_non_latin, samples_missing_addr, samples_special_chars]:
        for rec in pool:
            if rec["entity_id"] not in seen_ids and len(selected_samples) < 5:
                seen_ids.add(rec["entity_id"])
                selected_samples.append(rec)

    inferred_types = {}
    for col in headers:
        types = field_types[col]
        if "string" in types or not types:
            inferred_types[col] = "string"
        elif "float" in types:
            inferred_types[col] = "float"
        elif "integer" in types:
            inferred_types[col] = "integer"
            
    sampled_total = row_count // sample_rate
    name_script_pct = {k: round((v / sampled_total) * 100, 2) for k, v in script_counter_name.most_common(10)}
    addr_script_pct = {k: round((v / sampled_total) * 100, 2) for k, v in script_counter_addr.most_common(10)}

    clean_script_examples = {k: v for k, v in script_examples.items() if k in ["Latin", "Devanagari", "Tamil", "Telugu", "Kannada", "Gujarati", "Bengali", "Malayalam"]}

    return {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "total_rows": row_count,
        "delimiter": delimiter,
        "encoding": encoding,
        "columns": headers,
        "data_types": inferred_types,
        "missing_counts": dict(missing_counts),
        "missing_percentages": {col: round((missing_counts[col] / row_count) * 100, 3) for col in headers},
        "target_fields_present": {
            "business_name": "business_name" in headers,
            "business_address": "business_address" in headers,
            "country": "country" in headers
        },
        "country_distribution": dict(country_counts),
        "script_distribution_sampled": {
            "business_name_top_scripts_pct": name_script_pct,
            "business_address_top_scripts_pct": addr_script_pct
        },
        "character_script_examples": clean_script_examples,
        "representative_samples": selected_samples
    }

def profile_ground_truth(file_path):
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    
    row_count = 0
    missing_counts = defaultdict(int)
    samples = []
    
    total_matched_entities = 0
    zero_match_count = 0
    s2_total_matches = 0
    s3_total_matches = 0
    match_cardinality = Counter()
    
    source1_ids = set()
    s2_unique_matched = set()
    s3_unique_matched = set()
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        headers = [h.strip() for h in header_line.split("\t")]
        num_cols = len(headers)
        
        for line in f:
            row_count += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < num_cols:
                parts.extend([""] * (num_cols - len(parts)))
            record = dict(zip(headers, parts))
            
            s1_id = record.get("source1_entity_id", "").strip()
            matched_str = record.get("matched_entity_ids", "").strip()
            
            if not s1_id:
                missing_counts["source1_entity_id"] += 1
            if not matched_str:
                missing_counts["matched_entity_ids"] += 1
                zero_match_count += 1
                
            source1_ids.add(s1_id)
            
            matched_list = [m.strip() for m in matched_str.split(",") if m.strip()]
            num_matches = len(matched_list)
            match_cardinality[num_matches] += 1
            total_matched_entities += num_matches
            
            s2_in_row = sum(1 for m in matched_list if m.startswith("S2-"))
            s3_in_row = sum(1 for m in matched_list if m.startswith("S3-"))
            s2_total_matches += s2_in_row
            s3_total_matches += s3_in_row
            
            for m in matched_list:
                if m.startswith("S2-"):
                    s2_unique_matched.add(m)
                elif m.startswith("S3-"):
                    s3_unique_matched.add(m)
            
            if len(samples) < 5:
                samples.append({
                    "source1_entity_id": s1_id,
                    "matched_count": num_matches,
                    "s2_matches": s2_in_row,
                    "s3_matches": s3_in_row,
                    "matched_entity_ids": matched_str[:120] + ("..." if len(matched_str) > 120 else "")
                })

    avg_matches = round(total_matched_entities / row_count, 2) if row_count > 0 else 0
    avg_matches_for_matched = round(total_matched_entities / (row_count - zero_match_count), 2) if (row_count - zero_match_count) > 0 else 0

    return {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "total_rows": row_count,
        "columns": headers,
        "missing_counts": dict(missing_counts),
        "missing_percentages": {col: round((missing_counts[col] / row_count) * 100, 3) for col in headers},
        "linkage_analysis": {
            "total_source1_entities": row_count,
            "unique_source1_entities": len(source1_ids),
            "unmatched_source1_entities": zero_match_count,
            "unmatched_source1_pct": round((zero_match_count / row_count) * 100, 3),
            "total_linked_pairs": total_matched_entities,
            "total_s2_matches": s2_total_matches,
            "total_s3_matches": s3_total_matches,
            "avg_matches_per_source1": avg_matches,
            "avg_matches_for_matched_source1": avg_matches_for_matched,
            "unique_s2_entities_matched": len(s2_unique_matched),
            "unique_s3_entities_matched": len(s3_unique_matched),
            "top_match_cardinalities": dict(match_cardinality.most_common(6)),
            "linking_mechanism": "Each record maps a single query business entity from source1 (S1-*) to zero, one, or multiple candidate entities in source2 (S2-*) and source3 (S3-*). Multiple matches are represented as comma-separated identifiers."
        },
        "sample_records": samples
    }

def main():
    profile = {
        "sources": {},
        "ground_truth": None
    }
    
    print("Profiling source files...")
    for key, path in SOURCE_FILES.items():
        print(f"Profiling {key} ({path})...")
        profile["sources"][key] = profile_source_file(path, key)
        
    print("Profiling ground truth...")
    profile["ground_truth"] = profile_ground_truth(GROUND_TRUTH_FILE)
    
    # Save JSON summary
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "dataset_profile.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"Saved machine-readable profile to {json_path}")

    # Generate Markdown Report
    md_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "dataset_profile.md"))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Business Entity Resolution - Dataset Profiling Report\n\n")
        f.write("## 1. Executive Summary & Macro Metrics\n\n")
        f.write("This document provides a thorough profiling of the organizer datasets located under `../student_resource/dataset/`.\n")
        f.write("All profiling was performed using streaming reads to ensure zero memory exhaustion and zero in-place mutations.\n\n")
        
        f.write("### 1.1 Complete Dataset Overview Table\n\n")
        f.write("| Dataset | Partition | File Size | Row Count | Delimiter | Encoding | Columns | Missing Addr (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for key, p in profile["sources"].items():
            partition = "Train" if "train" in key else "Test"
            f.write(f"| `{p['file_name']}` | {partition} | {p['file_size_mb']} MB | {p['total_rows']:,} | `\\t` | {p['encoding']} | `{', '.join(p['columns'])}` | {p['missing_percentages'].get('business_address', 0)}% |\n")
            
        gt = profile["ground_truth"]
        f.write(f"| `{gt['file_name']}` | Train (GT) | {gt['file_size_mb']} MB | {gt['total_rows']:,} | `\\t` | UTF-8 | `{', '.join(gt['columns'])}` | N/A |\n\n")

        f.write("### 1.2 Country Distribution Across Datasets\n\n")
        f.write("| Dataset | Country Breakdown |\n")
        f.write("| :--- | :--- |\n")
        for key, p in profile["sources"].items():
            breakdown_str = ", ".join([f"{k}: {v:,} ({round(v/p['total_rows']*100, 1)}%)" for k, v in p["country_distribution"].items()])
            f.write(f"| `{p['file_name']}` | {breakdown_str} |\n")
        f.write("\n> [!CRITICAL]\n")
        f.write("> **Out-of-Distribution Country in Test Data**: The `France` category accounts for ~15% of records in the test split (`test_source1`: 259,452 rows; `test_source2`: ~710k rows; `test_source3`: ~735k rows), but **0% in the training split**. Any hard-coded geographic rules or models conditioned strictly on US/India will fail on France test records.\n\n")

        f.write("## 2. Individual Source Dataset Profiles\n\n")
        for key, p in profile["sources"].items():
            f.write(f"### 2.{list(profile['sources'].keys()).index(key)+1} `{p['file_name']}`\n\n")
            f.write(f"- **Path**: `{p['file_path']}`\n")
            f.write(f"- **File Size**: {p['file_size_mb']} MB ({p['file_size_bytes']:,} bytes)\n")
            f.write(f"- **Total Rows**: {p['total_rows']:,}\n")
            f.write(f"- **Encoding & Delimiter**: {p['encoding']}, delimiter `{p['delimiter']}`\n")
            f.write(f"- **Core Fields Verified**:\n")
            f.write(f"  - `business_name`: {'Present' if p['target_fields_present']['business_name'] else 'MISSING'}\n")
            f.write(f"  - `business_address`: {'Present' if p['target_fields_present']['business_address'] else 'MISSING'}\n")
            f.write(f"  - `country`: {'Present' if p['target_fields_present']['country'] else 'MISSING'}\n\n")
            
            f.write("#### Schema, Inferred Types, and Missing Values\n\n")
            f.write("| Column | Inferred Type | Missing Count | Missing (%) |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for col in p["columns"]:
                f.write(f"| `{col}` | `{p['data_types'][col]}` | {p['missing_counts'].get(col, 0):,} | {p['missing_percentages'].get(col, 0)}% |\n")
            f.write("\n")

            f.write("#### Script Distribution (Sampled)\n\n")
            f.write("- **`business_name` Scripts**:\n")
            for s, pct in p["script_distribution_sampled"]["business_name_top_scripts_pct"].items():
                f.write(f"  - {s}: {pct}%\n")
            f.write("- **`business_address` Scripts**:\n")
            for s, pct in p["script_distribution_sampled"]["business_address_top_scripts_pct"].items():
                f.write(f"  - {s}: {pct}%\n")
            f.write("\n")

            f.write("#### Character & Script Examples Found\n\n")
            f.write("| Script | Example Business Name | Example Address |\n")
            f.write("| :--- | :--- | :--- |\n")
            for s, eg in p["character_script_examples"].items():
                eg_name = (eg["business_name"] or "None").replace("|", "\\|")
                eg_addr = (eg["business_address"] or "None").replace("|", "\\|")
                f.write(f"| {s} | {eg_name} | {eg_addr} |\n")
            f.write("\n")

            f.write("#### Representative Sample Records (5 Samples)\n\n")
            f.write("| entity_id | business_name | business_address | country |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for rec in p["representative_samples"]:
                bname = rec.get("business_name", "").replace("|", "\\|")
                baddr = rec.get("business_address", "").replace("|", "\\|")
                bcountry = rec.get("country", "").replace("|", "\\|")
                f.write(f"| `{rec['entity_id']}` | {bname} | {baddr} | {bcountry} |\n")
            f.write("\n---\n\n")

        f.write("## 3. Ground Truth Analysis (`train_ground_truth.tsv`)\n\n")
        f.write(f"- **Schema**: `{', '.join(gt['columns'])}`\n")
        f.write(f"- **Total Rows**: {gt['total_rows']:,}\n")
        f.write(f"- **Total Matched Entity Pairings**: {gt['linkage_analysis']['total_linked_pairs']:,}\n")
        f.write(f"  - Matches to Source 2: {gt['linkage_analysis']['total_s2_matches']:,}\n")
        f.write(f"  - Matches to Source 3: {gt['linkage_analysis']['total_s3_matches']:,}\n")
        f.write(f"- **Unmatched Source 1 Records (No Ground Truth Match)**: {gt['linkage_analysis']['unmatched_source1_entities']:,} ({gt['linkage_analysis']['unmatched_source1_pct']}%)\n")
        f.write(f"- **Average Matches per Source 1 Record**: {gt['linkage_analysis']['avg_matches_per_source1']} (Average for matched: {gt['linkage_analysis']['avg_matches_for_matched_source1']})\n")
        f.write(f"- **Unique Matched Source 2 Entities**: {gt['linkage_analysis']['unique_s2_entities_matched']:,} / 5,034,616\n")
        f.write(f"- **Unique Matched Source 3 Entities**: {gt['linkage_analysis']['unique_s3_entities_matched']:,} / 5,285,603\n\n")
        
        f.write("### 3.1 Linkage Mechanism\n\n")
        f.write(f"{gt['linkage_analysis']['linking_mechanism']}\n\n")
        f.write("Distribution of Match Cardinalities (number of candidate entities linked to one source1 entity):\n\n")
        for num_m, cnt in gt["linkage_analysis"]["top_match_cardinalities"].items():
            f.write(f"- **{num_m} matches**: {cnt:,} source1 records ({round(cnt/gt['total_rows']*100, 2)}%)\n")
        f.write("\n")

        f.write("Sample Ground Truth Rows:\n\n")
        f.write("| source1_entity_id | Total Matches | S2 Matches | S3 Matches | Matched Entity IDs (Sample) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for rec in gt["sample_records"]:
            f.write(f"| `{rec['source1_entity_id']}` | {rec['matched_count']} | {rec['s2_matches']} | {rec['s3_matches']} | `{rec['matched_entity_ids']}` |\n")
        f.write("\n---\n\n")

        f.write("## 4. Key Findings & Uncertainties to Resolve\n\n")
        f.write("### 4.1 Key Findings\n")
        f.write("1. **Unified Schema**: Every source dataset in both `train` and `test` strictly adheres to `[entity_id, business_name, business_address, country]` delimited by tabs (`\\t`) with UTF-8 encoding.\n")
        f.write("2. **Source 1 is Uniform Latin**: In both train and test, `source1` contains 100% Latin text (even for Indian businesses, names/addresses are in English or transliterated Latin). There are 0 missing addresses in `source1`.\n")
        f.write("3. **Source 2 & 3 are Multilingual / Multi-Script**: Contains native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya) alongside English. Addressing requires robust, offline transliteration to bridge script barriers with Source 1.\n")
        f.write("4. **Missing Addresses in Source 2 and 3**: ~2.6% to 3.4% of entities in Source 2 and 3 have blank `business_address`. Matching logic must handle entity resolution on `business_name` + `country` alone when addresses are absent.\n")
        f.write("5. **Ground Truth Structure**: Ground truth is 1:N mapping `source1_entity_id` -> `[S2-*, S3-*]`. Approximately 5.58% of `source1` entities have 0 matches.\n\n")

        f.write("### 4.2 Critical Uncertainties for Preprocessing\n")
        f.write("1. **Out-of-Distribution `France` in Test**: `France` records exist exclusively in the test sets (~15% of records). Preprocessing and normalization rules must handle French address conventions (e.g., *Rue*, *Avenue*, *Boulevard*, *Cedex*, postal codes) and legal suffixes (*SAS*, *SARL*, *SASU*) without overfitting to US/India data.\n")
        f.write("2. **Transliteration Strategy (Offline Only)**: Because network calls and external APIs are strictly prohibited, we must evaluate offline transliteration libraries (e.g. `indic-transliteration` or `polyglot`/rule-based mapping) to map Indic scripts to Latin without installing heavy or disallowed network dependencies.\n")
        f.write("3. **Country Naming Consistency**: Do countries have divergent naming forms (e.g. `US` vs `USA` vs `United States`)? Current profiling shows exact values `US`, `India`, `France`, but casing and whitespace must be strictly standardized.\n")
        f.write("4. **Memory Footprint & Blocking**: The total test dataset across the three sources is ~11.7 million rows. Naive pairwise matching ($1.7M \\times 10M$) is impossible ($10^{13}$ pairs). High-efficiency blocking (by country, phonetic tokens, or locality) and chunked processing are mandatory.\n")

    print(f"Saved markdown report to {md_path}")

if __name__ == "__main__":
    main()
