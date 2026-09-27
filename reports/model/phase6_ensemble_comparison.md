# Phase 6: Controlled Multi-Model Ensemble Experiment Report

**Date:** 2026-09-27  
**Dataset:** `data/processed/train/dev_features/phase4_dev_corrected.parquet` (144,048 candidate pairs, 858 GT positives)  
**Features:** Frozen 29 baseline features (`BASELINE_FEATURES`)  
**Evaluation Protocol:** 5-Fold Entity-Disjoint Cross-Validation (seed=42)  
**Execution Runtime:** 59.1s  

---

## Executive Summary

In Phase 6, we conducted a rigorous, leakage-safe ensemble experiment comparing **10 configurations**: the standalone LightGBM baseline, 5 probability blends, 3 rank-averaged ensembles, and 1 precision-oriented blend. All evaluations used the identical 5 entity-disjoint folds and out-of-fold predictions established in Phase 5.

### Key Findings:
1. **Standalone LightGBM Reference:** Achieves OOF PR-AUC = **0.9616**, F0.5 @ 0.90 = **0.9307** (P = 0.9767, R = 0.7832, 16 FPs, 186 FNs) with 5-fold mean F0.5 = **0.9307 ± 0.0090**.
2. **Best Performing Probability Blend (`blend_lgb0.75_xgb0.25`):** Achieves OOF PR-AUC = **0.9630**, F0.5 @ 0.90 = **0.9300** (P = 0.9780, R = 0.7774, 15 FPs, 191 FNs) with 5-fold mean F0.5 = **0.9298 ± 0.0114**.
3. **Empirical Verdict:** **Degradation:** Ensembling degraded F0.5 @ 0.90 by -0.0007.
4. **Rank Averaging Assessment:** Rank averaging normalizes candidate percentiles but shifts the effective threshold range because positives represent only 0.6% of pairs. At threshold 0.90, rank averaging retains too many candidates (poor precision); however, at its optimal operating threshold (~0.99), rank averaging achieves competitive PR-AUC.

---

## 1. Multi-Threshold Performance Comparison

The table below shows all 10 configurations across standard operational thresholds (0.80, 0.82, 0.85, 0.90, 0.95) and their optimal operating points on the complete 144,048-pair OOF dataset:

| Ensemble Configuration | Method | PR-AUC | ROC-AUC | F0.5 @ 0.80 | F0.5 @ 0.82 | F0.5 @ 0.85 | F0.5 @ 0.90 | P @ 0.90 | R @ 0.90 | FP/FN @ 0.90 | F0.5 @ 0.95 | Optimal F0.5 (tau) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **lightgbm_baseline** | baseline | 0.9616 | 0.9994 | 0.9304 | 0.9320 | 0.9302 | **0.9307** | 0.9767 | 0.7832 | 16 / 186 | 0.9259 | 0.9320 (0.82) |
| `blend_lgb0.75_xgb0.25` | probability_blend | 0.9630 | 0.9995 | 0.9307 | 0.9333 | 0.9309 | 0.9300 | 0.9780 | 0.7774 | 15 / 191 | 0.9216 | 0.9333 (0.82) |
| `blend_lgb0.50_xgb0.50` | probability_blend | 0.9633 | 0.9995 | 0.9310 | 0.9316 | 0.9332 | 0.9271 | 0.9791 | 0.7646 | 14 / 202 | 0.9192 | 0.9332 (0.85) |
| `blend_lgb0.75_cat0.25` | probability_blend | 0.9638 | 0.9996 | 0.9336 | 0.9346 | 0.9339 | 0.9282 | 0.9778 | 0.7716 | 15 / 196 | 0.9204 | 0.9353 (0.83) |
| `blend_lgb0.50_cat0.50` | probability_blend | 0.9645 | 0.9996 | 0.9350 | 0.9349 | 0.9339 | 0.9260 | 0.9790 | 0.7611 | 14 / 205 | 0.9110 | 0.9360 (0.81) |
| `blend_three_model_equal` | probability_blend | 0.9646 | 0.9996 | 0.9326 | 0.9356 | 0.9332 | 0.9266 | 0.9818 | 0.7564 | 12 / 209 | 0.9106 | 0.9356 (0.82) |
| `rank_lgb_xgb` | rank_average | 0.9634 | 0.9995 | 0.0393 | 0.0433 | 0.0511 | 0.0754 | 0.0613 | 0.9988 | 13134 / 1 | 0.1557 | 0.9310 (1.00) |
| `rank_lgb_cat` | rank_average | 0.9645 | 0.9995 | 0.0394 | 0.0435 | 0.0520 | 0.0765 | 0.0621 | 0.9988 | 12933 / 1 | 0.1599 | 0.9343 (0.99) |
| `rank_all_three` | rank_average | 0.9647 | 0.9995 | 0.0396 | 0.0440 | 0.0523 | 0.0757 | 0.0615 | 0.9988 | 13086 / 1 | 0.1636 | 0.9316 (0.99) |
| `blend_precision_oriented` | probability_blend | 0.9636 | 0.9995 | 0.9330 | 0.9330 | 0.9329 | 0.9296 | 0.9780 | 0.7762 | 15 / 192 | 0.9192 | 0.9340 (0.81) |

---

## 2. Fold-Level Stability Analysis (F0.5 @ 0.90)

To verify whether improvements are consistent across disjoint partitions, F0.5 @ 0.90 is evaluated separately on each of the 5 entity-disjoint validation folds:

| Ensemble Configuration | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean F0.5 | Std Dev | Stability Rating |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `lightgbm_baseline` | 0.9358 | 0.9444 | 0.9304 | 0.9237 | 0.9190 | **0.9307** | ±0.0090 | High |
| `blend_lgb0.75_xgb0.25` | 0.9409 | 0.9444 | 0.9286 | 0.9202 | 0.9151 | **0.9298** | ±0.0114 | High |
| `blend_lgb0.50_xgb0.50` | 0.9426 | 0.9393 | 0.9302 | 0.9166 | 0.9057 | **0.9269** | ±0.0139 | High |
| `blend_lgb0.75_cat0.25` | 0.9358 | 0.9427 | 0.9286 | 0.9184 | 0.9151 | **0.9281** | ±0.0104 | High |
| `blend_lgb0.50_cat0.50` | 0.9358 | 0.9393 | 0.9302 | 0.9147 | 0.9091 | **0.9258** | ±0.0119 | High |
| `blend_three_model_equal` | 0.9409 | 0.9357 | 0.9321 | 0.9179 | 0.9050 | **0.9263** | ±0.0131 | High |
| `rank_lgb_xgb` | 0.0857 | 0.0638 | 0.0741 | 0.0803 | 0.0768 | **0.0762** | ±0.0073 | High |
| `rank_lgb_cat` | 0.0827 | 0.0686 | 0.0766 | 0.0832 | 0.0735 | **0.0769** | ±0.0056 | High |
| `rank_all_three` | 0.0811 | 0.0677 | 0.0739 | 0.0817 | 0.0757 | **0.0760** | ±0.0051 | High |
| `blend_precision_oriented` | 0.9392 | 0.9444 | 0.9286 | 0.9202 | 0.9151 | **0.9295** | ±0.0111 | High |

---

## 3. Detailed Threshold Sweep (0.50 -> 0.99)

Fine threshold sweep on the pooled OOF predictions for the baseline (`lightgbm_baseline`) and top ensemble (`blend_lgb0.75_xgb0.25`):

| Threshold | LGB Prec | LGB Rec | LGB F0.5 | Ens Prec | Ens Rec | Ens F0.5 | Delta F0.5 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.50 | 0.9200 | 0.8718 | 0.9100 | 0.9239 | 0.8776 | 0.9143 | +0.0043 |
| 0.60 | 0.9365 | 0.8590 | 0.9199 | 0.9326 | 0.8543 | 0.9158 | -0.0041 |
| 0.70 | 0.9438 | 0.8415 | 0.9214 | 0.9474 | 0.8392 | 0.9236 | +0.0022 |
| 0.75 | 0.9508 | 0.8333 | 0.9247 | 0.9543 | 0.8275 | 0.9259 | +0.0012 |
| 0.80 | 0.9619 | 0.8228 | 0.9304 | 0.9642 | 0.8170 | 0.9307 | +0.0003 |
| 0.82 | 0.9656 | 0.8182 | 0.9320 | 0.9706 | 0.8089 | 0.9333 | +0.0013 |
| 0.85 | 0.9690 | 0.8019 | 0.9302 | 0.9716 | 0.7972 | 0.9309 | +0.0006 |
| 0.88 | 0.9755 | 0.7902 | 0.9318 | 0.9754 | 0.7855 | 0.9304 | -0.0014 |
| 0.90 | 0.9767 | 0.7832 | 0.9307 | 0.9780 | 0.7774 | 0.9300 | -0.0007 |
| 0.92 | 0.9765 | 0.7762 | 0.9286 | 0.9807 | 0.7692 | 0.9296 | +0.0010 |
| 0.94 | 0.9805 | 0.7599 | 0.9267 | 0.9832 | 0.7494 | 0.9254 | -0.0012 |
| 0.95 | 0.9818 | 0.7541 | 0.9259 | 0.9844 | 0.7343 | 0.9216 | -0.0043 |
| 0.96 | 0.9843 | 0.7319 | 0.9208 | 0.9855 | 0.7145 | 0.9160 | -0.0048 |
| 0.98 | 0.9880 | 0.6702 | 0.9024 | 0.9892 | 0.6375 | 0.8909 | -0.0115 |

---

## 4. Error Overlap & Error Mode Exchange Analysis (@ tau = 0.90)

Comparing top candidate ensembles against the LightGBM baseline reference at threshold 0.90:

| Ensemble Configuration | Base FPs | Base FNs | Corrected FPs | Corrected FNs | Introduced FPs | Introduced FNs | Net Error Change | Final FPs / FNs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `blend_lgb0.75_xgb0.25` | 16 | 186 | **1** | **0** | 0 | 5 | **-4** | 15 / 191 |
| `blend_lgb0.50_xgb0.50` | 16 | 186 | **3** | **0** | 1 | 16 | **-14** | 14 / 202 |
| `blend_lgb0.75_cat0.25` | 16 | 186 | **1** | **0** | 0 | 10 | **-9** | 15 / 196 |
| `blend_lgb0.50_cat0.50` | 16 | 186 | **2** | **1** | 0 | 20 | **-17** | 14 / 205 |
| `blend_three_model_equal` | 16 | 186 | **4** | **1** | 0 | 24 | **-19** | 12 / 209 |
| `rank_lgb_xgb` | 16 | 186 | **0** | **185** | 13118 | 0 | **-12933** | 13134 / 1 |
| `rank_lgb_cat` | 16 | 186 | **0** | **185** | 12917 | 0 | **-12732** | 12933 / 1 |
| `rank_all_three` | 16 | 186 | **0** | **185** | 13070 | 0 | **-12885** | 13086 / 1 |
| `blend_precision_oriented` | 16 | 186 | **1** | **0** | 0 | 6 | **-5** | 15 / 192 |

### Error Categories for `blend_lgb0.75_xgb0.25` vs LightGBM Reference

| Error Status | Category | Count | Interpretation |
|:---|:---|:---:|:---|
| `corrected_fp` | `ADDRESS_NUMBER_COLLISION` | 1 | Status in ensemble relative to LightGBM |
| `introduced_fn` | `OTHER_FALSE_NEGATIVE` | 4 | Status in ensemble relative to LightGBM |
| `introduced_fn` | `ABBREVIATION_OR_ACRONYM` | 1 | Status in ensemble relative to LightGBM |
| `shared_fn` | `OTHER_FALSE_NEGATIVE` | 70 | Status in ensemble relative to LightGBM |
| `shared_fn` | `MISSING_ADDRESS_EVIDENCE` | 36 | Status in ensemble relative to LightGBM |
| `shared_fn` | `ABBREVIATION_OR_ACRONYM` | 35 | Status in ensemble relative to LightGBM |
| `shared_fn` | `SEVERE_ADDRESS_DIVERGENCE` | 29 | Status in ensemble relative to LightGBM |
| `shared_fn` | `TRANSLITERATION_OR_SCRIPT_VARIANT` | 13 | Status in ensemble relative to LightGBM |
| `shared_fn` | `NAME_DIVERGENCE` | 3 | Status in ensemble relative to LightGBM |
| `shared_fp` | `OTHER_FALSE_POSITIVE` | 11 | Status in ensemble relative to LightGBM |
| `shared_fp` | `SHARED_LOCATION_DIFFERENT_BIZ` | 3 | Status in ensemble relative to LightGBM |
| `shared_fp` | `SIMILAR_NAME_BRANCH_MISMATCH` | 1 | Status in ensemble relative to LightGBM |

---

## 5. Subgroup Diagnostics: Source and Country (@ tau = 0.90)

Evaluating subgroup performance at tau = 0.90 for LightGBM vs `blend_lgb0.75_xgb0.25`:

| Subgroup | Model | Candidates | Positives | TP | FP | FN | Precision | Recall | F0.5 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `candidate_source == source2` | LightGBM | 66300 | 420 | 337 | 9 | 83 | 0.9740 | 0.8024 | 0.9340 |
| `candidate_source == source2` | **blend_lgb0.75_xgb0.25** | 66300 | 420 | 335 | 8 | 85 | 0.9767 | 0.7976 | 0.9347 |
| `candidate_source == source3` | LightGBM | 77748 | 438 | 335 | 7 | 103 | 0.9795 | 0.7648 | 0.9275 |
| `candidate_source == source3` | **blend_lgb0.75_xgb0.25** | 77748 | 438 | 332 | 7 | 106 | 0.9794 | 0.7580 | 0.9253 |
| `country == india` | LightGBM | 88509 | 300 | 225 | 8 | 75 | 0.9657 | 0.7500 | 0.9131 |
| `country == india` | **blend_lgb0.75_xgb0.25** | 88509 | 300 | 222 | 8 | 78 | 0.9652 | 0.7400 | 0.9098 |
| `country == united states` | LightGBM | 55539 | 558 | 447 | 8 | 111 | 0.9824 | 0.8011 | 0.9399 |
| `country == united states` | **blend_lgb0.75_xgb0.25** | 55539 | 558 | 445 | 7 | 113 | 0.9845 | 0.7975 | 0.9404 |

---

## 6. Complexity and Cost Analysis

| Architecture | Number of Models | Training Time Overhead | Inference Latency Multiplier | Memory Multiplier | Operational Complexity |
|:---|:---:|:---:|:---:|:---:|:---|
| **LightGBM (Standalone)** | 1 | 1.0x (16.2s) | 1.0x | 1.0x (6.5 MB) | Low — Single model serialization, single predict call |
| **2-Model Blend (LGB + XGB)** | 2 | ~1.6x (26.5s) | ~1.8x | ~1.6x (10.1 MB) | Moderate — Requires two distinct runtime engines (LightGBM + XGBoost) |
| **2-Model Blend (LGB + CB)** | 2 | ~2.2x (35.9s) | ~2.4x | ~1.6x (10.5 MB) | Moderate — LightGBM + CatBoost C++ bindings |
| **3-Model Blend (LGB + XGB + CB)** | 3 | ~2.8x (46.2s) | ~3.1x | ~2.2x (14.1 MB) | High — Three distinct model inference dependencies and weights maintenance |
| **Rank Averaging** | 2-3 | ~1.6x - 2.8x | High (>10x batch) | High | High — Requires global batch sorting/ranking across all candidate predictions |

---

## 7. Phase 6 Final Decisions & Answers to the 9 Core Questions

### 1. Does any ensemble beat standalone LightGBM?
**Answer:** **Degradation:** Ensembling degraded F0.5 @ 0.90 by -0.0007. Standalone LightGBM achieves F0.5 @ 0.90 of **0.9307** (PR-AUC = **0.9616**), while the best probability blend (`blend_lgb0.75_xgb0.25`) achieves F0.5 @ 0.90 of **0.9300** (PR-AUC = **0.9630**).

### 2. Is the improvement consistent across all five folds?
**Answer:** For `blend_lgb0.75_xgb0.25`, the 5-fold mean is **0.9298 ± 0.0114** compared to LightGBM's **0.9307 ± 0.0090**. Consistency across individual validation folds is evaluated above in Section 2.

### 3. Does it improve F0.5 @ 0.90?
**Answer:** F0.5 @ 0.90 delta is **-0.0007** (0.9307 -> 0.9300).

### 4. Does it improve the precision/recall tradeoff?
**Answer:** Precision changes from **0.9767** to **0.9780** (+0.0013) while recall changes from **0.7832** to **0.7774** (-0.0058).

### 5. Which error categories does it actually fix?
**Answer:** As detailed in Section 4, the ensemble corrects **1** false positives and **0** false negatives.

### 6. What additional errors does it introduce?
**Answer:** It introduces **0** new false positives and **5** new false negatives, leading to a net error change of **-4** examples.

### 7. Is the additional complexity justified?
**Answer:** NO / MARGINAL — maintaining a multi-model serving pipeline for negligible delta is not justified for production deployment.

### 8. Should standalone LightGBM remain the final model?
**Answer:** YES, standalone LightGBM should remain the official primary production model.

### 9. If an ensemble is promising, what exact configuration should be tested next?
**Answer:** If ensembling is pursued, `blend_lgb0.75_xgb0.25` represents the optimal candidate configuration.
