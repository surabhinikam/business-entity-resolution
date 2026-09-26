# Amazon ML Challenge 2026 — Business Entity Resolution

An offline, scalable, multilingual entity resolution system designed to resolve duplicate and matching business records across heterogeneous sources without external network access or geocoding APIs.

---

## 1. Problem Statement

Business entity resolution (record linkage) requires matching noisy records from secondary databases against a canonical reference database. 

In this challenge, we have three data sources:
- **Source 1 (`train_source1.tsv`)**: Deduplicated reference database containing canonical business entities.
- **Source 2 (`train_source2.tsv`)**: Noisy business records with potential typographical errors, abbreviations, legal suffix variations, and missing fields.
- **Source 3 (`train_source3.tsv`)**: Highly heterogeneous noisy records, including native non-Latin Indic scripts and missing fields.

### Record Schema
Each entity record contains:
- `entity_id`: Source-namespaced unique identifier (e.g., `S1-925783039`, `S2-166376419`, `S3-202863386`).
- `business_name`: Organization name (noisy text, multiple scripts, legal entity types).
- `business_address`: Physical address (street, unit, city, state, postal code, or missing).
- `country`: Country identifier.

### Ground Truth Linkage
The training ground truth (`train_ground_truth.tsv`) defines the query-to-candidate mapping:
$$\text{Source 1 entity} \longrightarrow \{\text{matching Source 2 and/or Source 3 entities}\}$$

An entity in Source 1 can link to:
- **Zero matches (Singletons)**: ~5.58% of reference entities have no match in S2 or S3.
- **One match**: ~5.40% of reference entities have exactly one match.
- **Multiple matches (1-to-Many)**: ~89.02% of reference entities match 2 or more records across S2 and S3 (average: 3.67 matches, maximum: 11 matches).

### Strict Competition Constraints
1. **Offline Only**: Strictly zero external network calls. No Google Maps, geocoding APIs, external registries, or internet lookups.
2. **Open-Set Geographic Handling**: The training split contains records from the **US** (~60%) and **India** (~40%). The unseen test split contains **France** (~15%), which is absent from training data. All geographical logic, normalization, and blocking must treat country as an open-set string rather than hard-coding US/India.
3. **Execution Environment**: All pipeline scripts, data processing, and analysis must run in **WSL2 Ubuntu**.

---

## 2. Repository Structure

```text
business-entity-resolution/
├── data/
│   ├── raw/
│   │   ├── train/
│   │   │   ├── train_source1.tsv           # Reference entities (2.21M rows, 200MB)
│   │   │   ├── train_source2.tsv           # Noisy entities (5.03M rows, 467MB)
│   │   │   ├── train_source3.tsv           # Noisy entities (5.29M rows, 480MB)
│   │   │   └── train_ground_truth.tsv      # S1 -> S2/S3 mapping (2.21M rows, 121MB)
│   │   └── test/
│   │       ├── test_source1.tsv            # Test reference entities (1.73M rows)
│   │       ├── test_source2.tsv            # Test candidates (4.89M rows)
│   │       └── test_source3.tsv            # Test candidates (5.08M rows)
│   └── processed/
│       ├── checkpoints/                    # Processing checkpoints
│       └── train/
│           ├── source1/                    # Partitioned Parquet (221 parts, 2.21M rows)
│           ├── source2/                    # Partitioned Parquet (504 valid parts, 5.03M rows)
│           └── source3/                    # Partitioned Parquet (529 parts, 5.29M rows)
│
├── reports/
│   └── data_analysis/
│       ├── README.md                       # Analysis documentation & reproduction guide
│       ├── dataset_summary.md              # Detailed raw dataset profiling
│       ├── ground_truth_analysis.md        # Linkage topology, cardinality & singleton metrics
│       ├── normalization_analysis.md       # Partition audit & vocabulary compression
│       ├── match_similarity_analysis.md    # True matches vs random & hard negative pairs
│       ├── blocking_analysis.md            # Empirical candidate blocking evaluation
│       ├── data_analysis_summary.json      # Structured raw/processed dataset metrics
│       └── matches_and_blocking_summary.json # Structured match & blocking metrics
│
├── src/
│   ├── analysis/                           # Reusable analysis & evaluation modules
│   │   ├── data_loader.py                  # Resilient raw/partitioned parquet loaders
│   │   ├── text_similarity.py              # Offline token, character & numeric similarity
│   │   └── blocking_evaluator.py           # Blocking recall, reduction ratio & indexer
│   ├── normalization/                      # Text, business, address & country normalizers
│   ├── transliteration/                    # Unicode script detector & Indic transliterator
│   ├── language_detection/                 # Lightweight offline language inference
│   ├── preprocessing/                      # Chunked TSV ingestion & Parquet part writer
│   └── validation/                         # Schema & partition validators
│
├── scripts/
│   ├── analyze_data.py                     # Entry point: Raw & processed dataset analysis
│   ├── analyze_matches.py                  # Entry point: Ground truth, matches & blocking
│   └── process_dataset.py                  # Streaming ingestion & normalization pipeline
│
├── tests/
│   ├── test_data_loading.py                # Partition discovery & ground truth tests
│   ├── test_analysis_utils.py              # Similarity metrics & feature extraction tests
│   ├── test_normalization_analysis.py      # Blocking evaluator & recall tests
│   └── ...                                 # Existing component test suites
│
├── requirements.txt
└── README.md
```

---

## 3. Data Structure: Raw vs. Processed

### Raw Data (Immutable TSVs)
- Stored under `data/raw/train/` and `data/raw/test/`.
- Tab-delimited (`\t`), UTF-8 encoded, 4 core fields: `entity_id`, `business_name`, `business_address`, `country`.
- Source 1 contains zero missing values. Sources 2 and 3 exhibit ~3.34% missing addresses (~344k records).

### Processed Data (Partitioned Parquet)
- Stored under `data/processed/train/{source1,source2,source3}/` in ~10,000-row chunks.
- Each partition implements the **16-column production canonical schema**:

| Field # | Canonical Column | Data Type | Description |
| :--- | :--- | :--- | :--- |
| 1 | `source` | `String` | Source identifier (`source1`, `source2`, `source3`) |
| 2 | `entity_id` | `String` | Unique entity primary key |
| 3 | `business_name` | `String` | Original raw business name |
| 4 | `business_name_script` | `String` | Detected Unicode script (`Latin`, `Devanagari`, `Telugu`, etc.) |
| 5 | `business_name_language` | `String` | Inferred language tag |
| 6 | `business_name_transliterated` | `String` | Phonetically transliterated name into Latin/IAST |
| 7 | `business_name_normalized` | `String` | Lowercased, diacritic-stripped, corporate-suffix canonicalized name |
| 8 | `business_name_tokens` | `List<String>` | Tokenized meaningful name segments |
| 9 | `business_address` | `String` | Original raw business address |
| 10 | `business_address_script` | `String` | Detected Unicode script of address |
| 11 | `business_address_language`| `String` | Inferred language tag of address |
| 12 | `business_address_transliterated` | `String`| Phonetically transliterated address |
| 13 | `business_address_normalized` | `String` | Lowercased, diacritic-stripped, punctuation-normalized address |
| 14 | `business_address_tokens` | `List<String>` | Tokenized address components (preserving numeric tokens) |
| 15 | `country` | `String` | Original raw country string |
| 16 | `country_normalized` | `String` | Canonical lowercase country name (`united states`, `india`, etc.) |

---

## 4. Normalization Stage (Existing Implementation)

The preprocessing and normalization modules (located in `src/normalization/` and `src/transliteration/`) provide critical data transformations:

1. **Multilingual Script Bridging**:
   - `Source 1` is 100% Latin-script records.
   - `Source 2` and `Source 3` contain over **750,000 native Indic-script records** across 9 regional scripts (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi) and ~53,000 mixed-script records.
   - The transliterator converts non-Latin scripts to Latin phonetic approximations, allowing cross-script matching.
2. **Vocabulary Compression**:
   - Standardizes corporate suffixes (`Private Limited` -> `pvt ltd`, `Inc.` -> `inc`, `LLC` -> `llc`).
   - Normalization reduces unique business names by **15.2% - 20.3%**, transforming noisy variations into identical canonical keys.
3. **Partition Discovery Anomaly Handled**:
   - During our partition audit, 79 misplaced `train_source1_part_*.parquet` files were discovered inside `data/processed/train/source2/`.
   - The data loader (`src/analysis/data_loader.py`) specifically applies strict source-stem matching (`_source2_`) to exclude these duplicate files and load exactly the 504 valid partitions (5,034,616 records).

---

## 5. Current Stage: Data, Ground Truth, and Blocking Analysis

The current phase focused exclusively on understanding the dataset, ground truth linkage topology, true match characteristics, negative pair distributions, and blocking strategies.

### Key Empirical Findings:

1. **Ground Truth Topology**:
   - Total True Pairs: **7,638,365** ($3.69\text{M}$ to S2; $3.94\text{M}$ to S3).
   - Reference Singletons: **123,247 entities (5.58%)** have zero matches.
   - Candidate Uniqueness: Every matched S2 or S3 record matches **exactly one** S1 entity (strictly 1-to-1 from candidates to reference).
   - Dual-Source Match Rate: **80.48%** of reference entities match both S2 and S3 records simultaneously.
2. **Feature Discriminability**:
   - **Country Match**: 100.0% of true matches share normalized country. Country filtering immediately eliminates 48% of the Cartesian space with zero recall loss.
   - **Exact Normalized Name Equality**: Normalization increases exact name equality from 10.74% (raw) to **27.29%** (+16.55% absolute gain).
   - **Name Token Overlap**: True matches have a mean Overlap of **0.787** (median: **1.0**) vs **0.040** for random non-matches.
   - **Shared Address Numbers**: 72.49% of true matches share at least one address numeric token (house/plot/pincode), compared to only **0.44%** of hard negatives sharing a name token. This makes numeric tokens the primary tie-breaker against hard negatives.
3. **Blocking Strategies Evaluation**:
   - Exact Normalized Name: 27.29% recall, 17,126 candidates, 99.9993% reduction ratio.
   - First Two Name Tokens: 55.26% recall, 139,963 candidates, 99.9943% reduction ratio.
   - First Name Token + First Address Number: 43.24% recall, 24,564 candidates, 99.9990% reduction ratio.
   - **Multi-Index Disjunctive Rule (Exact Name OR First Two Tokens OR [First Token + Addr Num])**:
     - **65.17% Recall** with only **181,653 candidates** (5.6 candidates per true match, **>99.99% Reduction Ratio**).

---

## 6. How to Run the Analysis

All commands must be run inside **WSL2 Ubuntu**:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run raw and processed dataset profiling
python scripts/analyze_data.py --output-dir reports/data_analysis

# 3. Run ground truth, similarity, and candidate blocking analysis
python scripts/analyze_matches.py --sample-size 50000 --output-dir reports/data_analysis

# 4. Execute test suite
pytest tests/
# or
python -m unittest discover -s tests
```

---

## 7. Generated Reports

The analysis generates five detailed markdown reports under `reports/data_analysis/`:

1. [`dataset_summary.md`](reports/data_analysis/dataset_summary.md): Dataset sizes, schemas, missingness, country breakdowns, name/address length distributions, top duplicates.
2. [`ground_truth_analysis.md`](reports/data_analysis/ground_truth_analysis.md): Linkage coverage, singleton rates, match count histograms, source overlap, and candidate-to-query topology.
3. [`normalization_analysis.md`](reports/data_analysis/normalization_analysis.md): Partition discovery verification, 16-column schema audit, script distribution, transformation modification rates, and vocabulary compression.
4. [`match_similarity_analysis.md`](reports/data_analysis/match_similarity_analysis.md): Empirical comparison of true matches vs random negatives and hard negatives across token overlap, character similarity, and numeric tokens.
5. [`blocking_analysis.md`](reports/data_analysis/blocking_analysis.md): Empirical evaluation of 7 single-key and 3 multi-index blocking rules against the 22.7 trillion candidate space.
6. [`data_analysis_summary.json`](reports/data_analysis/data_analysis_summary.json) and [`matches_and_blocking_summary.json`](reports/data_analysis/matches_and_blocking_summary.json): Complete machine-readable metrics.

---

## 8. What Has NOT Been Implemented Yet

To maintain rigorous stage separation and avoid premature optimization, the following components have **deliberately NOT been implemented yet**:

- **Candidate Generation / Blocking Index Construction**: Final production blocking inverted indexes and candidate retrieval pipelines.
- **Feature Engineering Pipeline**: Pairwise dense feature matrix creation for ML models.
- **Machine Learning Matcher**: GBDT (LightGBM/XGBoost/CatBoost) classifiers, neural ranking models, or cross-encoders.
- **Threshold Tuning**: Optimization of probability thresholds for F0.5 score.
- **Final Prediction Pipeline**: Formatting and generation of competition submission files.

---

## 9. Next Planned Stage: Candidate Generation & Blocking

Building directly on the empirical results of this analysis phase, the next stage will implement:

1. **Multi-Index Candidate Generator**:
   - Construct inverted index over training/test splits using the optimal multi-index blocking strategy:
     - Anchor 1: `(country_normalized, business_name_normalized)`
     - Anchor 2: `(country_normalized, first_two_name_tokens)`
     - Anchor 3: `(country_normalized, first_name_token + first_address_number)`
     - Fallback: Address-independent token blocking for the ~3.3% records with null addresses.
2. **Frequency Capping & Stopword Filtering**:
   - Suppress high-frequency non-informative tokens (`pvt`, `ltd`, `inc`, `kumar`, `shri`, `company`) to prevent block explosion.
   - Cap maximum block size (e.g. at 500 candidates per block) to maintain bounded candidate generation latency.
3. **Candidate Verification & F0.5 Optimization**:
   - Ensure the candidate generator achieves >85-90% true match recall before progressing to feature engineering and ranking.
