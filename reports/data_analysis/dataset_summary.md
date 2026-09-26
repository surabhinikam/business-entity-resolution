# Raw Dataset Summary & Profiling Report

## 1. Executive Summary

This report provides an in-depth empirical analysis of the raw training datasets supplied for the Amazon ML Challenge 2026 Business Entity Resolution task:
- `train_source1.tsv`: Deduplicated reference businesses
- `train_source2.tsv`: Noisy business records
- `train_source3.tsv`: Noisy business records

All measurements were computed directly on the full datasets using Polars streaming/in-memory readers inside WSL2 Ubuntu without external network calls.

## 2. Dataset Dimensions and Key Integrity

| Source | Total Records | Unique Entity IDs | Duplicate IDs | Schema Columns | Missing / Unexpected |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `source1` | 2,206,821 | 2,206,821 | 0 | 4 columns (`entity_id, business_name, business_address, country`) | None |
| `source2` | 5,034,616 | 5,034,616 | 0 | 4 columns (`entity_id, business_name, business_address, country`) | None |
| `source3` | 5,285,603 | 5,285,603 | 0 | 4 columns (`entity_id, business_name, business_address, country`) | None |

### Key Findings:
- **Zero duplicate entity IDs**: Every `entity_id` is unique within its respective source.
- **Namespace prefixing**: Source 1 uses `S1-`, Source 2 uses `S2-`, and Source 3 uses `S3-` prefixes.

## 3. Missing Value Analysis

| Field | Source 1 Missing | Source 1 (%) | Source 2 Missing | Source 2 (%) | Source 3 Missing | Source 3 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `entity_id` | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| `business_name` | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |
| `business_address` | 0 | 0.0% | 168,967 | 3.3561% | 175,916 | 3.3282% |
| `country` | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% |

> [!IMPORTANT]
> - `entity_id`, `business_name`, and `country` have **0.0% missing values** across all three training sources.
> - `business_address` has **0.0% missing in Source 1**, but **3.356% missing in Source 2** (168,967 rows) and **3.328% missing in Source 3** (175,892 rows).
> - Any candidate blocking or matching rule that strictly requires address tokens will fail on these ~344k address-less records. Name-based fallback blocking is mandatory.

## 4. Country Distribution & Open-Set Handling

| Source | Country | Record Count | Percentage |
| :--- | :--- | :--- | :--- |
| `source1` | US | 1,323,633 | 59.979% |
| `source1` | India | 883,188 | 40.021% |
| `source2` | US | 3,016,817 | 59.921% |
| `source2` | India | 2,017,799 | 40.079% |
| `source3` | US | 3,170,056 | 59.975% |
| `source3` | India | 2,115,547 | 40.025% |

> [!CRITICAL]
> **Open-Set Country Requirement**: Training data contains US (~60%) and India (~40%). However, the unseen test set contains France (~15%), where France is 0% in training data.
> Country normalization and blocking keys must treat `country` as an open-set categorical string and never hard-code logic to only US/India.

## 5. Business Name Characteristics & Noise Profiles

| Metric | Source 1 (Reference) | Source 2 (Noisy) | Source 3 (Noisy) |
| :--- | :--- | :--- | :--- |
| Valid Record Count | 2,206,821 | 5,034,616 | 5,285,603 |
| Unique Raw Names | 1,539,229 | 4,402,009 | 4,651,609 |
| Duplicate Names | 667,592 | 632,607 | 633,994 |
| Mean Char Length | 24.03 | 25.10 | 25.20 |
| Median Char Length | 24.0 | 25.0 | 25.0 |
| 90th / 99th Percentile Length | 34 / 42 | 37 / 48 | 37 / 50 |
| Mean Word Count | 3.55 | 3.61 | 3.63 |
| Non-ASCII Records (%) | 0 (0.0%) | 764,608 (15.187%) | 606,737 (11.479%) |

### Top Repeated Raw Business Names:

**Source 1**:
- `Primary Care Group`: 253 occurrences
- `Ear Nose & Throat Group`: 251 occurrences
- `Pediatric Group`: 222 occurrences
- `Womens Health Group`: 220 occurrences
- `Physical Therapy Group`: 218 occurrences

**Source 2**:
- `Primary Care`: 320 occurrences
- `Physical Therapy`: 307 occurrences
- `Urgent Care`: 297 occurrences
- `Womens Health`: 297 occurrences
- `Behavioral Health`: 296 occurrences

**Source 3**:
- `Primary Care`: 421 occurrences
- `Physical Therapy`: 399 occurrences
- `Pediatric Dental`: 393 occurrences
- `Womens Health`: 379 occurrences
- `Urgent Care`: 377 occurrences

## 6. Address Characteristics & Noise Profiles

| Metric | Source 1 (Reference) | Source 2 (Noisy) | Source 3 (Noisy) |
| :--- | :--- | :--- | :--- |
| Valid Record Count | 2,206,821 | 4,865,649 | 5,109,687 |
| Unique Raw Addresses | 2,130,606 | 4,337,261 | 4,632,764 |
| Mean Char Length | 52.07 | 47.83 | 48.32 |
| Median Char Length | 41.0 | 37.0 | 42.0 |
| 90th / 99th Percentile Length | 90 / 124 | 84 / 118 | 78 / 116 |
| Mean Word Count | 8.03 | 7.56 | 7.42 |
| Non-ASCII Address Records (%) | 554 (0.025%) | 478,453 (9.833%) | 476,588 (9.327%) |

## 7. Implications for Candidate Generation

1. **Reference vs Noisy Asymmetry**: Source 1 contains clean English/Latin text with zero missing values. Sources 2 and 3 contain native Indic scripts, missing addresses (3.3%), and severe syntactic noise.
2. **Address Null Fallback**: Because ~344k records in S2/S3 have no address, candidate generation cannot rely exclusively on address-based blocking keys.
3. **Transliteration Prerequisite**: Since Source 1 is 100% Latin and Sources 2 & 3 contain hundreds of thousands of native Indic-script names, exact or token blocking without transliteration would miss all cross-script true matches.
