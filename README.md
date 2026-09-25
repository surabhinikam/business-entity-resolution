# Business Entity Resolution

A Python-based, offline entity resolution pipeline designed to process large-scale business records (TSV format) and resolve duplicate or matching entities without relying on external network calls or APIs.

## Project Structure

```text
business-entity-resolution/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── test/
│   └── processed/
│       ├── train/
│       └── test/
│
├── src/
│   ├── preprocessing/
│   ├── transliteration/
│   ├── language_detection/
│   ├── normalization/
│   └── validation/
│
├── scripts/
├── tests/
├── reports/
├── notebooks/
└── configs/
```

## Directory Overview

- **`data/raw/`**: Storage location for the original, unmodified datasets partitioned into `train/` and `test/`. These files are excluded from version control and must remain immutable.
- **`data/processed/`**: Staging directory for cleaned, normalized, or intermediate chunked outputs (`train/` and `test/`), also excluded from Git.
- **`src/`**: Core modular application source code.
  - **`src/preprocessing/`**: Modules handling chunked/streaming I/O, column filtering, and schema ingestion.
  - **`src/transliteration/`**: Offline transliteration engines to map multilingual/phonetic variations into unified representations.
  - **`src/language_detection/`**: Lightweight, offline language identification for routing records to language-specific normalization pipelines.
  - **`src/normalization/`**: Text cleaning, casing, punctuation removal, address/business abbreviation expansion, and canonicalization.
  - **`src/validation/`**: Data integrity checks, schema validation, and partition consistency checks.
- **`scripts/`**: Executable command-line scripts for end-to-end pipeline execution, chunked batch jobs, and evaluation.
- **`tests/`**: Unit and integration test suites for pipeline modules.
- **`reports/`**: Generated performance metrics, profiling logs, match rate audits, and summary reports.
- **`notebooks/`**: Exploratory data analysis (EDA) and experimental prototyping notebooks.
- **`configs/`**: Configuration files (e.g., YAML, JSON) managing hyperparameters, thresholds, chunk sizes, and pipeline options.

## Key Constraints & Guidelines

1. **Strict Offline Execution**: External web APIs and network requests are strictly prohibited. All dependencies and models must operate locally.
2. **Dataset Immutability**: Never overwrite, edit, or mutate original organizer data in `data/raw/`.
3. **Chunked / Streaming Reads**: Ingestion and transformations must process large TSV files in streams or chunks to manage memory footprint efficiently.
4. **Git Hygiene**: Raw and processed datasets are kept strictly out of version control via `.gitignore`.
