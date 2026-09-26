# True Match vs Non-Match Similarity Analysis Report

## 1. Executive Summary

This report analyzes the empirical distribution of similarity features across three distinct pair populations:
1. **True Match Pairs**: Valid ground-truth links (S1 <-> S2/S3).
2. **Random Non-Match Pairs**: Pairs randomly drawn from the Cartesian product representing background noise.
3. **Hard Negative Pairs**: Non-matching pairs that share the exact same country and at least one significant business name token.

## 2. Feature Comparison Table

| Feature | True Matches | Random Non-Matches | Hard Negatives | Discriminative Power |
| :--- | :--- | :--- | :--- | :--- |
| **Country Match (Norm)** | **100.0%** | 52.144% | 100.0% | **Critical Pre-Filter** |
| **Name Exact (Raw)** | **10.736%** | 0.0% | 0.0% | High Precision Anchor |
| **Name Exact (Norm)** | **27.292%** | 0.0% | 0.236% | High Precision Anchor |
| **Name Exact (Translit)** | **10.744%** | 0.0% | 0.0% | Bridges Cross-Script |
| **Name Token Jaccard (Mean / Med)** | **0.6461 / 0.6667** | 0.0226 / 0.0 | 0.2529 / 0.2 | **Extremely Strong** |
| **Name Token Overlap (Mean / Med)** | **0.7865 / 1.0** | 0.0404 / 0.0 | 0.457 / 0.5 | **Robust to Truncation** |
| **Name Char 3-gram (Mean / Med)** | **0.7119 / 0.75** | 0.0213 / 0.0 | 0.2477 / 0.2143 | Highly Discriminative |
| **Address Exact (Norm)** | **8.434%** | 0.0% | 0.0% | High Precision |
| **Address Token Jaccard (Mean / Med)** | **0.591 / 0.6154** | 0.008 / 0.0 | 0.0152 / 0.0 | **Decisive Differentiator** |
| **Address Token Overlap (Mean / Med)** | **0.7508 / 0.8** | 0.0 / 0.0 | 0.2529 | Decisive Differentiator |
| **Shared Address Numbers (%)** | **72.49%** | 0.3% | 0.436% | **Crucial for Disambiguation** |

## 3. Analysis of Key Findings

### 1. Country Is a 100% Deterministic Separator
- **100.0% of true match pairs share the same normalized country**.
- In the Cartesian product, cross-country pairs represent ~48% of combinations (US vs India). Filtering by country immediately cuts the candidate space in half with **zero loss in recall**.

### 2. Name Token Overlap vs Jaccard
- True matches have a mean Name Token Overlap of **0.7865** (median: **1.0**) compared to **0.6461** for Token Jaccard.
- This occurs because noisy records frequently add or omit legal designations (`Inc`, `Pvt Ltd`, `LLP`, `Trading Co`), reducing Jaccard while keeping Overlap near 1.0.
- Overlap coefficient is significantly more robust to suffix noise.

### 3. Address Numbers Break Ties Among Hard Negatives
- For Hard Negatives that intentionally share a common name token, their Name Token Jaccard is elevated (mean: **0.2529**).
- However, their **Shared Address Number rate is only 0.436%**, whereas True Matches exhibit **72.49% shared address numbers**.
- Street numbers, building numbers, and postal pincodes provide extraordinary discriminative power to reject hard negatives.

### 4. Normalization Effect on Equality Matches
- Raw exact name equality among true matches is **10.736%**.
- After normalization, exact name equality surges to **27.292%** (+16.56% absolute gain, a 2.5x increase).
- This confirms that normalization recovers nearly triple the raw equality true matches through casing, diacritic, and suffix cleanup alone.
