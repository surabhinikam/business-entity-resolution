# Phase 4C: Targeted Interaction Feature Ablation Report

## 1. Executive Summary

This report documents the rigorous **Phase 4C Interaction Feature Ablation** for the Amazon ML Challenge 2026 Business Entity Resolution system.

Building upon the **Phase 4 corrected baseline** (`phase4_dev_corrected.parquet`, 144,048 candidate pairs, 858 ground-truth positives), we investigated whether adding a small set of targeted interaction features mitigates the primary error modes identified in the baseline error analysis:
1. **False Positives**: Entities sharing identical/similar names when candidate addresses are missing or sparse.
2. **Synergy**: Strong joint evidence across both name and address modalities.
3. **Address Strengthening**: Street number token agreement bolstering the address character similarity signal.

### Controlled Experimental Framework
- **Dataset**: `phase4_dev_corrected.parquet` (115,272 train pairs with 686 positives, 28,776 val pairs with 172 positives).
- **Split**: Exact 80/20 entity-disjoint split by `source1_entity_id` (Seed=42; 0 S1 overlap; 0 pair overlap).
- **Model**: LightGBM (`n_estimators=150`, `learning_rate=0.05`, `num_leaves=31`, `seed=42`).
- **Ablation Set**:
  - `BASELINE`: 29 original engineered features.
  - `EXP_A`: 29 base + 2 missing-address interactions (`name_char_3gram_x_address_missing_cand`, `name_token_overlap_x_address_missing_cand`).
  - `EXP_B`: 29 base + 1 name $\times$ address interaction (`name_char_3gram_x_address_char_3gram`).
  - `EXP_C`: 29 base + 1 address-number $\times$ address-similarity interaction (`shared_address_num_x_address_char_3gram`).
  - `EXP_ALL`: 29 base + all 4 interaction features.

---

## 2. Primary Comparison Table

| Experiment | Features | F0.5@0.50 | F0.5@0.90 | PR-AUC | ROC-AUC | Best F0.5 | Best $\tau$ | Prec@0.50 | Rec@0.50 | Prec@0.90 | Rec@0.90 | Confusion @ 0.90 (TP/FP/TN/FN) | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **BASELINE** | 29 | 0.9112 | 0.9313 | 0.9608 | 0.9997 | **0.9423** | 0.93 | 0.9123 | 0.9070 | 0.9490 | 0.8663 | 149/8/28596/23 | 1.26s |
| **EXP_A** | 31 | 0.8937 | 0.9234 | 0.9613 | 0.9997 | **0.9343** | 0.93 | 0.8947 | 0.8895 | 0.9423 | 0.8547 | 147/9/28595/25 | 0.77s |
| **EXP_B** | 30 | 0.9002 | 0.9375 | 0.9755 | 0.9998 | **0.9440** | 0.95 | 0.8971 | 0.9128 | 0.9554 | 0.8721 | 150/7/28597/22 | 0.65s |
| **EXP_C** | 30 | 0.9096 | 0.9217 | 0.9734 | 0.9998 | **0.9392** | 0.97 | 0.9118 | 0.9012 | 0.9419 | 0.8488 | 146/9/28595/26 | 0.64s |
| **EXP_ALL** | 33 | 0.9028 | 0.9375 | 0.9611 | 0.9997 | **0.9391** | 0.82 | 0.9017 | 0.9070 | 0.9554 | 0.8721 | 150/7/28597/22 | 1.13s |

---

## 3. Deltas Relative to Corrected Baseline

| Experiment | Features | $\Delta$ F0.5@0.50 | $\Delta$ F0.5@0.90 | $\Delta$ Best F0.5 | $\Delta$ PR-AUC | $\Delta$ FP@0.90 | $\Delta$ FN@0.90 | $\Delta$ Pattern A FP@0.90 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | 29 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |
| **EXP_A** | 31 | -0.0175 | -0.0079 | -0.0080 | +0.0005 | +1 | +2 | 0 |
| **EXP_B** | 30 | -0.0110 | +0.0062 | +0.0017 | +0.0147 | -1 | -1 | 0 |
| **EXP_C** | 30 | -0.0016 | -0.0095 | -0.0031 | +0.0126 | +1 | +3 | 0 |
| **EXP_ALL** | 33 | -0.0084 | +0.0062 | -0.0033 | +0.0003 | -1 | -1 | 0 |

> [!NOTE]
> All metrics are evaluated on the identical 28,776 validation pairs with 172 true positive matches.
> Negative deltas for False Positives (FP) and False Negatives (FN) indicate improved error reduction.

---

## 4. Interaction Feature Importances (Gain Analysis)

#### EXP_A: 29 original + 2 missing-address interactions

| Interaction Feature | Gain Rank | Gain Score | Relative Share (%) |
| :--- | :---: | :---: | :---: |
| `name_char_3gram_x_address_missing_cand` | 6 / 31 | 2329.05 | 2.11% |
| `name_token_overlap_x_address_missing_cand` | 18 / 31 | 112.27 | 0.10% |

#### EXP_B: 29 original + 1 name x address interaction

| Interaction Feature | Gain Rank | Gain Score | Relative Share (%) |
| :--- | :---: | :---: | :---: |
| `name_char_3gram_x_address_char_3gram` | 1 / 30 | 81266.31 | 73.19% |

#### EXP_C: 29 original + 1 address-number x address-similarity interaction

| Interaction Feature | Gain Rank | Gain Score | Relative Share (%) |
| :--- | :---: | :---: | :---: |
| `shared_address_num_x_address_char_3gram` | 1 / 30 | 76645.11 | 69.43% |

#### EXP_ALL: 29 original + all 4 interaction features

| Interaction Feature | Gain Rank | Gain Score | Relative Share (%) |
| :--- | :---: | :---: | :---: |
| `name_char_3gram_x_address_missing_cand` | 8 / 33 | 2203.30 | 1.99% |
| `name_token_overlap_x_address_missing_cand` | 17 / 33 | 414.68 | 0.37% |
| `name_char_3gram_x_address_char_3gram` | 4 / 33 | 3619.36 | 3.27% |
| `shared_address_num_x_address_char_3gram` | 15 / 33 | 649.81 | 0.59% |


---

## 5. Targeted Error Pattern Analysis

We specifically tracked how each feature set impacts the two known failure modes on validation data:
- **Pattern A (Name Only False Positives)**: Pairs with `name_char_3gram_jaccard >= 0.80` and `address_missing_candidate == 1`.
- **Pattern B (Address Divergence False Negatives)**: True positives missed due to severe address divergence (`address_char_3gram_similarity < 0.20` or `name_char_3gram_jaccard < 0.50`).

| Experiment | Pat A FP @ 0.50 | Pat A FP @ 0.90 | Pat B FN @ 0.50 | Pat B FN @ 0.90 | Total FP @ 0.50 | Total FP @ 0.90 | Total FN @ 0.50 | Total FN @ 0.90 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | 0 | 0 | 7 | 9 | 15 | 8 | 16 | 23 |
| **EXP_A** | 0 | 0 | 7 | 8 | 18 | 9 | 19 | 25 |
| **EXP_B** | 0 | 0 | 8 | 9 | 18 | 7 | 15 | 22 |
| **EXP_C** | 0 | 0 | 9 | 10 | 15 | 9 | 17 | 26 |
| **EXP_ALL** | 0 | 0 | 7 | 9 | 17 | 7 | 16 | 22 |

---

## 6. Evaluation & Practical Significance

### Statistical & Practical Significance Assessment

1. **EXP_A (Missing Address Interactions — `name_char_3gram_x_address_missing_cand`, `name_token_overlap_x_address_missing_cand`)**:
   - **Performance Delta**: Delta F0.5@0.50 = -0.0175, Delta F0.5@0.90 = -0.0079, Delta Best F0.5 = -0.0080.
   - **Error Impact**: Total FP @ 0.90 increased from 8 to 9; total FN @ 0.90 increased from 23 to 25.
   - **Theoretical Explanation**: Tree-based algorithms (LightGBM) inherently perform axis-aligned orthogonal step splits on `address_missing_candidate` and name similarity features. Explicitly multiplying a continuous name similarity by a binary indicator does not introduce non-linear geometry that decision trees cannot already capture; instead, it introduces collinearity and dilutes split purity.
   - **Verdict**: **Clear Degradation — Reject.**

2. **EXP_B (Name x Address Synergy — `name_char_3gram_x_address_char_3gram`)**:
   - **Performance Delta**: Delta F0.5@0.90 = +0.0062 (0.9313 -> 0.9375), Delta Best F0.5 = +0.0017 (0.9423 -> 0.9440 at tau = 0.95), Delta PR-AUC = +0.0147 (0.9608 -> 0.9755).
   - **Error Impact**: At tau = 0.90, reduces False Positives by 1 (8 -> 7) and reduces False Negatives by 1 (23 -> 22). Precision increases from 0.9490 to 0.9554.
   - **Feature Importance**: Ranked **#1 by gain** with a commanding 73.19% relative share (gain = 81,266.31), surpassing individual address and name 3-grams.
   - **Practical Significance Assessment**:
     While the PR-AUC boost (+0.0147) and gain dominance confirm genuine positive interaction signal, the net change in Best F0.5 is **+0.0017**, which corresponds to a net shift of only **1 candidate pair** on the 28,776-pair validation set. By our pre-established criterion (Delta F0.5 >= 0.003), an improvement of +0.0017 must be explicitly classified as **statistically marginal / inconclusive**.
   - **Verdict**: **Marginally Positive / Inconclusive.**

3. **EXP_C (Address Number x Address Similarity — `shared_address_num_x_address_char_3gram`)**:
   - **Performance Delta**: Delta F0.5@0.90 = -0.0095 (0.9313 -> 0.9217), Delta Best F0.5 = -0.0031 (0.9423 -> 0.9392).
   - **Error Impact**: False Negatives @ 0.90 increased by 3 (23 -> 26), dropping Recall from 0.8663 to 0.8488.
   - **Theoretical Explanation**: While this feature captured 69.43% gain share, it penalizes valid entity matches that do not possess explicit street numbers in their addresses (e.g., mall locations, industrial estates, unnumbered avenues), skewing probability mass toward very high thresholds (tau = 0.97).
   - **Verdict**: **Degradation in F0.5 — Reject.**

4. **EXP_ALL (All 4 Interactions Combined)**:
   - **Performance Delta**: Delta F0.5@0.90 = +0.0062, Delta Best F0.5 = -0.0033 (0.9423 -> 0.9391), Delta PR-AUC = +0.0003.
   - **Error Impact**: Gains and split importance become fragmented across collinear terms (`name_char_3gram_x_address_char_3gram` drops from 73.19% in EXP_B to 3.27% in EXP_ALL). Best F0.5 degrades relative to both BASELINE and EXP_B.
   - **Verdict**: **Collinear Dilution — Reject.**

---

## 7. Recommendation for Phase 4D

Based strictly on the empirical evidence from this controlled ablation:

1. **Primary Recommendation**: **Carry the BASELINE 29 Features into Phase 4D**.
   - **Rationale**: The Principle of Parsimony strongly favors simpler, highly tested feature schemas unless an addition produces clear, unambiguous, and statistically meaningful improvements (Delta F0.5 >= 0.003).
   - The Best F0.5 delta for the best interaction (EXP_B) is only +0.0017 (1 pair shift on validation), which is strictly inconclusive.
   - Retaining the 29-feature schema keeps the production pipeline lightweight, completely aligned with the frozen Phase 3 schema, and free of redundant interaction terms.

2. **Alternate Candidate for Cross-Validation (Phase 4D)**:
   - Because EXP_B (`name_char_3gram_x_address_char_3gram`) produced a notable PR-AUC increase (+0.0147) and dominated feature gain, it can be retained as an optional secondary comparison in Phase 4D cross-validation to see whether its advantage persists across all K folds.
   - All other candidate interactions (EXP_A, EXP_C, EXP_ALL) are definitively eliminated.
