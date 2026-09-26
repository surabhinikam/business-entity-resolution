# Data Analysis & Profiling Reports

This directory contains empirical profiling reports, statistical analyses, and blocking evaluations conducted for the Amazon ML Challenge 2026 — Business Entity Resolution project.

All analyses were executed strictly offline in WSL2 Ubuntu using local text processing and fast streaming readers (Polars, PyArrow). External APIs, network calls, and external databases were strictly prohibited and not used.

---

## 1. Directory Structure

```text
reports/data_analysis/
├── README.md                           # This guide: script documentation & report index
├── dataset_summary.md                  # Comprehensive raw dataset profiling (Source 1, 2, 3)
├── ground_truth_analysis.md            # Topology & cardinality of train_ground_truth.tsv
├── normalization_analysis.md           # Parquet partition discovery & normalization effects
├── match_similarity_analysis.md        # Similarity features: True matches vs Random & Hard negatives
├── blocking_analysis.md                # Empirical evaluation of candidate blocking strategies
├── data_analysis_summary.json          # Machine-readable JSON summary of raw & processed datasets
└── matches_and_blocking_summary.json   # Machine-readable JSON metrics for ground truth & blocking
```

---

## 2. Overview of Analysis Reports

### 1. `dataset_summary.md`
- **Data Analyzed**: Raw datasets (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`).
- **Core Findings**:
  - `Source 1`: 2,206,821 records (100% clean English/Latin reference businesses, 0% missing values).
  - `Source 2`: 5,034,616 records (noisy businesses; 3.356% missing addresses; 9.4% non-Latin Indic scripts).
  - `Source 3`: 5,285,603 records (noisy businesses; 3.328% missing addresses; 5.3% non-Latin Indic scripts).
  - `Country Distribution`: 60.0% US, 40.0% India in training data. Unseen test set includes France (~15%), confirming open-set country requirement.
  - Zero duplicate `entity_id` values within any source.

### 2. `ground_truth_analysis.md`
- **Data Analyzed**: `train_ground_truth.tsv` (2,206,821 Source 1 entities).
- **Core Findings**:
  - **7,638,365 total true match pairs** (3,693,619 to S2; 3,944,746 to S3).
  - **Singleton Rate**: 123,247 S1 entities (**5.585%**) have zero matches. Models must support predicting empty match sets.
  - **Match Cardinality**: Matched S1 entities have an average of **3.666 matches** (median: 4.0, max: 11).
  - **Dual-Source Coverage**: 80.48% of S1 entities match records in both Source 2 and Source 3 simultaneously.
  - **Strict 1-to-1 Topology from Candidates**: Every matched S2 or S3 entity links to exactly **one** S1 entity. There are zero many-to-one candidate collisions.

### 3. `normalization_analysis.md`
- **Data Analyzed**: 1,254 Parquet partition files across `data/processed/train/`.
- **Core Findings**:
  - Discovered and validated all 221 parts of `source1`, 504 parts of `source2`, and 529 parts of `source3`.
  - **Anomaly Resolved**: Identified 79 misplaced `train_source1_part_*.parquet` files in `data/processed/train/source2/`. Partition discovery filters them using strict filename stem matching.
  - **Multilingual Bridging**: Sources 2 and 3 contain over 750,000 non-Latin/mixed records across 9 Indic scripts (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi). Transliteration is essential because Source 1 is 100% Latin.
  - **Collision Compression**: Normalization reduces distinct name vocabulary by 15% - 20%, compressing syntactic variations into canonical keys.

### 4. `match_similarity_analysis.md`
- **Data Analyzed**: 50,000 true match pairs compared against 25,000 random non-matches and 25,000 hard negative non-matches.
- **Core Findings**:
  - **Country Filtering**: 100.0% of true matches share normalized country. Filtering cuts candidate space by ~48% with zero recall loss.
  - **Exact Equality Surge**: Exact name equality jumps from 10.74% (raw) to 27.29% (normalized) — a 2.5x increase.
  - **Token Overlap vs Jaccard**: True matches average **0.787** Overlap vs **0.646** Jaccard, demonstrating that Overlap is far more robust to legal suffix truncation.
  - **Address Numbers Differentiate Hard Negatives**: Hard negatives sharing a name token have only **0.44%** shared address numbers, whereas true matches have **72.49%**. Numeric tokens are decisive discriminators.

### 5. `blocking_analysis.md`
- **Data Analyzed**: Candidate generation strategies evaluated against the 22.77 trillion Cartesian product space.
- **Core Findings**:
  - `Country + Exact Normalized Name`: 27.29% recall, 17,126 candidates, 99.9993% reduction ratio.
  - `Country + First Name Token`: 67.59% recall, 1,282,004 candidates (subject to generic token explosion).
  - `Country + First Two Name Tokens`: 55.26% recall, 139,963 candidates, 99.9943% reduction ratio.
  - `Multi-Index Union (Exact Name OR First Two Tokens OR [First Token + Addr Num])`: Achieves **65.17% recall** with only **181,653 candidates** (5.6 candidates per true match, 99.9927% reduction ratio).

---

## 3. How to Reproduce the Analysis

All commands must be executed in **WSL2 Ubuntu** from the repository root:

```bash
# 1. Activate the Python virtual environment
source .venv/bin/activate

# 2. Run raw dataset and normalization analysis
python scripts/analyze_data.py --output-dir reports/data_analysis

# 3. Run ground truth, match pair similarity, and blocking evaluation
python scripts/analyze_matches.py --sample-size 50000 --output-dir reports/data_analysis

# 4. Run automated test suite
python -m unittest discover -s tests
# or
pytest tests/
```
