# Candidate Blocking Strategies & Evaluation Report

## 1. Executive Summary

This report systematically evaluates candidate generation and blocking keys on empirical training ground truth.
- **Total Cartesian Pair Space**: $|S1| \times (|S2| + |S3|) \approx 2.277 \times 10^{13}$ pairs (22.7 trillion).
- **Objective**: Maximize **Blocking Recall** (fraction of ground-truth matches placed into at least one common block) while minimizing candidate set size (maximizing **Reduction Ratio**).

## 2. Blocking Strategies Empirical Evaluation Table

| Strategy / Key Formulation | Total Common Blocks | Candidate Pairs Generated | Ground Truth Recall (%) | Reduction Ratio (%) | Candidates / True Match |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Country + Exact Normalized Name** | 13,383 | 17,126 | **27.29%** | 99.999307% | 1.3 |
| **2. Country + Exact Transliterated Name** | 5,444 | 6,889 | **10.74%** | 99.999721% | 1.3 |
| **3. Country + First Name Token (len >= 3)** | 13,216 | 1,282,004 | **67.59%** | 99.948141% | 37.9 |
| **4. Country + First Two Name Tokens** | 24,101 | 139,963 | **55.26%** | 99.994338% | 5.1 |
| **5. Country + First Name Token + First Address Number** | 21,125 | 24,564 | **43.24%** | 99.999006% | 1.1 |
| **6. Country + Exact Normalized Address** | 4,209 | 4,243 | **8.43%** | 99.999828% | 1.0 |
| **7. Country + Address First Number + First Address Token** | 12,754 | 262,053 | **41.2%** | 99.9894% | 12.7 |
| **Multi-Index 1: Exact Name OR (First Name Token + Addr Num)** | 34,508 | 41,690 | **54.02%** | 99.998314% | 1.5 |
| **Multi-Index 2: Exact Name OR First Two Name Tokens OR (First Token + Addr Num)** | 58,609 | 181,653 | **65.17%** | 99.992652% | 5.6 |
| **Multi-Index 3: Exact Name OR Exact Address OR (First Token + Addr Num)** | 38,717 | 45,933 | **57.94%** | 99.998142% | 1.6 |

## 3. Deep Dive into Strategy Trade-offs

### Strategy 1: `Country + Exact Normalized Name`
- **Recall**: **27.29%**
- **Candidate Count**: Extremely compact (17,126 pairs across sample).
- **Strength**: Zero noise, high precision, very fast.
- **Weakness**: Misses true matches containing typos, abbreviations, or missing tokens.

### Strategy 3: `Country + First Name Token (len >= 3)`
- **Recall**: **67.59%**
- **Candidate Count**: Generates 1,282,004 candidates (~74.9x more than exact name).
- **Strength**: Captures names where suffixes or trailing tokens vary.
- **Weakness**: Frequent generic tokens (e.g. `kumar`, `shri`, `national`) produce massive blocks.

### Strategy 4: `Country + First Two Name Tokens`
- **Recall**: **55.26%**
- **Candidate Count**: 139,963 candidates (efficient balance).
- **Strength**: Substantially curbs block explosion while maintaining strong recall.
- **Weakness**: Fails if the initial token has a typo or transposed word order.

### Strategy 5: `Country + First Name Token + First Address Number`
- **Recall**: **43.24%**
- **Candidate Count**: Very small (24,564 candidates).
- **Strength**: Tremendous precision by coupling name and address numbers.
- **Weakness**: Cannot match records with missing addresses (3.3% of S2/S3).

### Multi-Index Disjunctive Blocking (The Optimal Path Forward)
- **Multi-Index 2 (Exact Name OR First Two Name Tokens OR [First Token + Addr Num])**:
  - **Achieves 65.17% Blocking Recall** while maintaining a **>99.99% Reduction Ratio** (181,653 candidates).
  - Combines high-precision anchors with structural address anchors and token prefix fallbacks.

## 4. Recommendations for Candidate Generation (Next Phase)

1. **Disjunctive Multi-Key Indexing**: Never rely on a single blocking key. A union of 2 to 3 complementary keys is required to achieve high recall (>65%-90%).
2. **Block Size Capping**: Generic terms (e.g. `pvt`, `inc`, `kumar`, `shri`, `company`) generate massive blocks (>10,000 candidates). Candidate generation must apply stopword filtering on prefix tokens and cap block sizes to prevent quadratic blowup.
3. **Address Fallback**: For records where address is null (~344k rows in S2/S3), the candidate generator must automatically fall back to multi-token name blocking.
