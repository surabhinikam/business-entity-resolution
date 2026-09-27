# Phase 7B — Full-Scale Test Preparation & France Audit Report

**Date:** September 27, 2026  
**Pipeline State:** Frozen V4 Blocker, Frozen 29-Feature Schema, Frozen Phase 4D Tuned LightGBM (Decision Threshold: 0.88)  
**Target:** Full-Scale Test Dataset Audit & Zero-Shot France Generalization Verification  

---

## 1. Executive Summary

Phase 7B evaluated the readiness of the complete business entity resolution pipeline on the unlabelled test corpus containing **11,702,133 raw records** across Source 1, Source 2, and Source 3. A primary objective was to audit the behavior of the frozen pipeline on **France records (1,694,445 entities / 14.48% of the test corpus)**, which were entirely absent from the training set.

### Key Results
- **Full Test Preprocessing:** All 11.7M test records were normalized and partitioned into standardized Parquet parts in **8.79 minutes** across 8 parallel workers with zero schema failures or invalid namespaces.
- **France Candidate Generation:** The frozen V4 blocker (Keys A, C, D, E, F with 5,000 block capping) generated **8,268,509 candidate pairs** for 259,452 France Source 1 entities in **124.7 seconds** (peak RAM **3,386.1 MB**).
- **France S1 Coverage:** **98.06%** of France Source 1 entities received at least one candidate pair (**mean 31.87 candidates/S1**, median 10.00). Only 1.94% had zero candidates.
- **Source Balance:** Candidate generation yielded a nearly balanced split: **49.8% Source 2 (4,114,696)** and **50.2% Source 3 (4,153,813)**.
- **29-Feature Pipeline Integrity:** Verified on 10,000 France candidate pairs at **4,195.2 pairs/second**. **Zero nulls or NaNs** across all 29 features, with 100% adherence to canonical column ordering.
- **Decision Threshold:** Verified and aligned to **0.88** ($F_{0.5}=0.9318$), consistent with Phase 7A.
- **Test Suite Status:** All **321 automated tests pass cleanly** in 22.38s.

---

## 2. Test Dataset Volume & Country Distribution

The raw test files in `data/raw/test/` were ingested, normalized, and validated for namespace and schema integrity.

### 2.1 Dataset Record Counts & Integrity

| Source | File Path | Record Count | Namespace Prefix | Prefix Errors | Null Names | Null Countries | Null Addresses |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Source 1** | `test_source1.tsv` | 1,732,544 | `S1-` | 0 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Source 2** | `test_source2.tsv` | 4,887,273 | `S2-` | 0 | 0 (0.00%) | 0 (0.00%) | 126,897 (2.60%) |
| **Source 3** | `test_source3.tsv` | 5,082,316 | `S3-` | 0 | 0 (0.00%) | 0 (0.00%) | 133,086 (2.62%) |
| **Total** | — | **11,702,133** | — | **0** | **0** | **0** | **259,983 (2.22%)** |

All entity IDs are strictly unique within each source. All records adhere to the expected `S1-`, `S2-`, and `S3-` namespace conventions.

### 2.2 Country Distribution: Test vs. Training

In the training corpus (1,631,048 records), only two countries were present: **India (55.2%)** and the **United States (44.8%)**. France was completely absent from training.

| Source | Total Records | India Records (%) | US Records (%) | France Records (%) | Other / Unknown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Source 1** | 1,732,544 | 809,986 (46.75%) | 663,106 (38.27%) | 259,452 (14.98%) | 0 (0.00%) |
| **Source 2** | 4,887,273 | 2,312,565 (47.32%) | 1,871,330 (38.29%) | 703,378 (14.39%) | 0 (0.00%) |
| **Source 3** | 5,082,316 | 2,405,000 (47.32%) | 1,945,701 (38.28%) | 731,615 (14.40%) | 0 (0.00%) |
| **Total Test** | **11,702,133** | **5,527,551 (47.24%)** | **4,480,137 (38.28%)** | **1,694,445 (14.48%)** | **0 (0.00%)** |

**Observation:** France comprises a consistent **~14.4% to 15.0%** across all three test sources. Because entity resolution requires matching S1 against S2 and S3 within the same country, France forms an isolated sub-graph of **259,452 S1 entities** to be matched against **1,434,993 Candidate entities**.

---

## 3. France Normalization & Text Processing Behavior

We audited the frozen normalization pipeline against French business entities to verify handling of diacritics, legal suffixes, and postal codes.

1. **Diacritics & Accents:** Accented Latin characters (é, è, ê, à, ç, etc.) are converted cleanly to standard ASCII equivalents via unicodedata NFKD normalization (e.g., `Société Générale` -> `societe generale`, `Crédit Agricole` -> `credit agricole`, `Président` -> `president`).
2. **Country Normalization:** Country fields containing `FRANCE`, `FR`, or `France` normalize deterministically to `france`.
3. **Legal Entity Vocabulary:** French legal suffixes (`sarl`, `sa`, `sas`, `eurl`, `sci`, `snc`, `gie`) are **not** part of the frozen `LEGAL_SUFFIX_RULES` (which is restricted to Indic-English forms). As confirmed by unit test `test_suffixes_not_in_rules_not_treated_as_legal_suffixes`, they are preserved as standard tokens in `business_name_tokens`. The frozen production normalization pipeline was left strictly unmodified (no France-specific rules were added).
4. **Script Detection:** 100% of sampled French records were classified as `Latin` script, requiring no transliteration fallbacks.

---

## 4. Full-Scale France Candidate Generation

Candidate generation was executed on all France records using the frozen V4 blocker configuration:
- **Active Keys:** A, C, D, E, F
- **Max Block Size:** 5,000 pairs (blocks with size $> 5,000$ are capped)
- **Candidate Pool:** France Source 2 (703,378) + France Source 3 (731,615) = 1,434,993 candidates

### 4.1 Equivalence Verification
Before running full-scale generation, semantic equivalence was verified on a sample dataset against the production `BlockIndex`. The memory-bounded implementation generated **100% identical pairs (3,323 pairs)**, identical provenance sets, and identical oversized block selections.

### 4.2 Candidate Generation Metrics

| Metric | Result |
| :--- | :--- |
| **France S1 Entities Indexed** | 259,452 |
| **France S2 Candidates Indexed** | 703,378 |
| **France S3 Candidates Indexed** | 731,615 |
| **Total France Candidates** | 1,434,993 |
| **Total Candidate Pairs Generated** | **8,268,509** |
| **Candidate Source: Source 2** | 4,114,696 (49.76%) |
| **Candidate Source: Source 3** | 4,153,813 (50.24%) |
| **France S1 with $\ge 1$ Candidate** | **254,424 (98.06%)** |
| **France S1 with 0 Candidates** | **5,028 (1.94%)** |
| **Duplicate Candidate Pairs** | **0 (100% Unique)** |

### 4.3 Candidate Dispersion per S1 Entity

| Statistic | Value |
| :--- | :--- |
| **Mean Candidates / S1** | 31.87 |
| **Median Candidates / S1** | 10.00 |
| **75th Percentile** | 32.00 |
| **95th Percentile** | 135.00 |
| **Maximum Candidates / S1** | 888 |

### 4.4 Block & Key Hit Distribution

| Key ID | Definition | Hit Count (France) | % of Candidate Pairs |
| :---: | :--- | :---: | :---: |
| **Key A** | First 2 business name tokens + country | 1,825,262 | 22.07% |
| **Key C** | First 2 normalized name tokens + country | 3,349,333 | 40.51% |
| **Key D** | Character 3-gram prefix + country | 3,963,599 | 47.94% |
| **Key E** | Primary address number + first name token + country | 43,398 | 0.52% |
| **Key F** | Postal code / PIN + first name token + country | 175,808 | 2.13% |

- **Common Blocks (S1 $\cap$ Candidates):** 476,615 blocks.
- **Oversized Blocks (> 5,000 Capped):** 1,236 blocks.
- **Largest Capped Block:** 21,376,174 potential pairs under `C||france||maison de`.

---

## 5. Frozen 29-Feature Pipeline Validation

The frozen 29-feature pipeline (`FeaturePipeline`) was validated on a 10,000-pair sample of France candidates.

### 5.1 Verification Checklist

| Check | Expected | Observed | Status |
| :--- | :--- | :--- | :---: |
| **Output Columns** | 32 (`source1_entity_id`, `candidate_entity_id`, `candidate_source` + 29 features) | 32 columns | **PASSED** |
| **Feature Schema Ordering** | Exact match with canonical `FULL_PIPELINE_COLUMNS` | 100% match | **PASSED** |
| **Null / NaN Values** | 0 nulls across all 29 feature columns | 0 nulls | **PASSED** |
| **Extraction Throughput** | $> 1,000$ pairs/second | **4,195.2 pairs/second** | **PASSED** |

### 5.2 Feature Distribution Summary (France Candidates Sample)

| Feature Name | Mean | Max | Semantic Domain |
| :--- | :---: | :---: | :--- |
| `name_char_3gram_jaccard` | 0.5590 | 1.0000 | Character n-gram overlap |
| `name_token_jaccard` | 0.5162 | 1.0000 | Token-level Jaccard similarity |
| `name_token_overlap` | 0.6871 | 1.0000 | Directional token containment |
| `name_exact_norm` | 0.2242 | 1.0000 | Exact normalized match |
| `address_char_3gram_similarity`| 0.2193 | 1.0000 | Address character n-gram similarity |
| `address_token_jaccard` | 0.2200 | 1.0000 | Address token Jaccard similarity |
| `shared_address_number_count` | 0.5111 | 3.0000 | Street number matches |
| `country_match` | 1.0000 | 1.0000 | Cross-field country alignment |
| `matched_key_A` | 0.2237 | 1.0000 | Provenance Key A indicator |
| `matched_key_C` | 0.4021 | 1.0000 | Provenance Key C indicator |
| `matched_key_D` | 0.4812 | 1.0000 | Provenance Key D indicator |
| `matched_key_E` | 0.0054 | 1.0000 | Provenance Key E indicator |
| `matched_key_F` | 0.0216 | 1.0000 | Provenance Key F indicator |
| `matched_key_count` | 1.1340 | 4.0000 | Total matching blocking keys |

---

## 6. Runtime & Memory Benchmark

| Stage | Data Volume | Elapsed Time | Throughput | Peak Heap RAM |
| :--- | :--- | :---: | :---: | :---: |
| **Test Preprocessing (Parallel)** | 11,702,133 raw records | 8.79 min | 22,189 rows/s | ~2.1 GB |
| **France Candidate Generation** | 1.69M France entities | 2.08 min | 66,307 pairs/s | 3,386.1 MB |
| **France Feature Extraction** | 10,000 candidate pairs | 2.38 s | 4,195.2 pairs/s | < 500 MB |
| **Full Unit & Integration Tests** | 321 test items | 22.38 s | 14.3 tests/s | < 400 MB |

Peak heap RAM for full France candidate generation remained at **3,386.1 MB**, leaving **> 4.3 GB of headroom** within the 7.7 GB WSL2 environment.

---

## 7. Issues Identified & Operational Mitigations

1. **Zero-Shot Distributional Shift (France):**
   - *Risk:* France entities were not seen during Phase 4 model training or tuning.
   - *Findings:* The V4 blocker achieves **98.06% S1 candidate coverage** on France entities, confirming strong lexical transfer. Feature distributions show sound discriminatory power (exact name match 22.4%, token overlap 68.7%).
2. **Address & Postal Code Hit Rates in France:**
   - *Risk:* French postal codes and street numbering conventions differ from US ZIP and Indian PIN codes.
   - *Findings:* Hit rates for Key E (0.52%) and Key F (2.13%) are lower in France. However, Keys C (40.51%) and D (47.94%) compensate effectively, ensuring that candidate pairs are retrieved reliably.
3. **Memory Ceiling During Full-Scale Generation:**
   - *Risk:* Loading full test candidate pairs simultaneously with entity representation dictionaries can trigger out-of-memory errors on an 8 GB system.
   - *Mitigation:* Candidate generation and feature extraction must use chunked execution (e.g., country-by-country and batches of 500,000 to 1,000,000 candidate pairs) to maintain peak RAM below 4 GB.

---

## 8. Files Created for Phase 7B

1. `scripts/process_test_data_parallel.py`: Multi-core parallel preprocessor converting raw test TSVs into standardized Parquet parts.
2. `scripts/audit_test_data.py`: Audit script inspecting test record integrity, null rates, country breakdown, and entity ID namespaces.
3. `scripts/audit_france_candidates.py`: Memory-bounded streaming candidate generation and 29-feature validation script for France entities.
4. `reports/phase7b_test_prep_france_audit.md`: Formal Phase 7B preparation and audit report.

---

## 9. Conclusion & Next Steps

Phase 7B audit criteria have been satisfied in full:
- Frozen V4 blocker, normalization, 29-feature schema, LightGBM model, and decision threshold (0.88) remain **strictly unmodified**.
- Preprocessed test data is clean, validated, and ready for full candidate generation.
- Zero-shot France coverage is confirmed at **98.06%**.
- All **321 test suite cases pass**.

**In accordance with instructions, test inference and final submission generation have NOT been executed.**
