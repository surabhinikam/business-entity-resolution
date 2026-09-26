# Ground Truth Structure & Linkage Analysis Report

## 1. Executive Summary

This report provides a complete empirical analysis of the official training ground truth linkage (`train_ground_truth.tsv`):
- **Total Source 1 Entities**: 2,206,821
- **Total True Match Pairs**: 7,638,365
- Evaluates singleton rates, match cardinality distributions, source allocation, and linkage topology.

## 2. Match Coverage & Singleton Rate

| Metric | Count | Percentage of Source 1 |
| :--- | :--- | :--- |
| Total Reference Entities (Source 1) | 2,206,821 | 100.0% |
| Source 1 Entities with >= 1 Match | 2,083,574 | **94.415%** |
| Source 1 Singletons (0 Matches) | 123,247 | **5.585%** |

> [!IMPORTANT]
> **Singletons Account for 5.58%**: 123,247 Source 1 entities have **zero** matching records in Source 2 or Source 3. The candidate generation and matching system must be capable of predicting empty matches (`None` / empty string) rather than forcing every S1 entity to select a candidate.

## 3. Match Cardinality per Source 1 Entity

| Metric | Value |
| :--- | :--- |
| Minimum Matches per S1 | 0 |
| Maximum Matches for One S1 | 11 |
| Mean Matches per S1 (Overall) | 3.461 |
| Mean Matches per Matched S1 | **3.666** |
| Median Matches per Matched S1 | **4.0** |
| 90th Percentile Matches | 6.0 |
| 99th Percentile Matches | 8.0 |

### Match Count Distribution:

| Matches per S1 | S1 Entity Count | Percentage | Cumulative (%) |
| :--- | :--- | :--- | :--- |
| 0 matches | 123,247 | 5.585% | 5.58% |
| 1 matches | 119,157 | 5.399% | 10.98% |
| 2 matches | 375,212 | 17.002% | 27.99% |
| 3 matches | 530,841 | 24.055% | 52.04% |
| 4 matches | 484,115 | 21.937% | 73.98% |
| 5 matches | 321,957 | 14.589% | 88.57% |
| 6 matches | 164,868 | 7.471% | 96.04% |
| 7 matches | 63,968 | 2.899% | 98.94% |
| 8 matches | 18,680 | 0.846% | 99.78% |
| 9 matches | 4,205 | 0.191% | 99.97% |
| 10 matches | 534 | 0.024% | 100.00% |
| 11 matches | 37 | 0.002% | 100.00% |

## 4. Source Target Breakdown

| Match Relationship | Total Pairs | Percentage of Pairs |
| :--- | :--- | :--- |
| Total True Pairs | 7,638,365 | 100.0% |
| Source 1 -> Source 2 Pairs | 3,693,619 | 48.356% |
| Source 1 -> Source 3 Pairs | 3,944,746 | 51.644% |

### Source Overlap for Source 1 Entities:

| Overlap Type | S1 Entities | Percentage of All S1 |
| :--- | :--- | :--- |
| S1 matching BOTH Source 2 and Source 3 | 1,776,047 | **80.48%** |
| S1 matching ONLY Source 2 | 143,029 | 6.481% |
| S1 matching ONLY Source 3 | 164,498 | 7.454% |
| S1 Singletons (Neither Source) | 123,247 | 5.585% |

## 5. Candidate Target Entity Topology

| Candidate Dataset | Total Records | Unique Matched Entities | Unmatched Records in Dataset | Match Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| Source 2 | 5,034,616 | 3,693,619 | 1,340,997 | 73.36% |
| Source 3 | 5,285,603 | 3,944,746 | 1,340,857 | 74.63% |

> [!IMPORTANT]
> **Strict 1-to-1 Mapping from Candidate to Reference**: Every matched S2 entity matches exactly ONE S1 entity, and every matched S3 entity matches exactly ONE S1 entity. There are ZERO many-to-one collisions where multiple S1 entities claim the same noisy S2 or S3 record.

## 6. Strategic Implications for Later Pipeline Stages

1. **One-to-Many Architecture**: Because an S1 entity averages 3.67 matches and has up to 12 matches, entity resolution cannot use a 1-nearest-neighbor thresholding rule. It must use multi-match prediction / probability thresholding.
2. **Dual-Source Alignment**: 80.5% of S1 entities have matching records across both S2 and S3 simultaneously. Candidate generation must query both S2 and S3 indexes independently.
3. **Singleton Prediction**: With 123k singletons (5.6%), the model score threshold must cleanly separate true matches from non-matches to avoid hallucinating false links on singletons.
