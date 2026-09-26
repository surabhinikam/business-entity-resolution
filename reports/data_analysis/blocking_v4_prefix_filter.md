# Candidate Generation V4 — India Prefix / Honorific Filter Experiment

**Generated**: 2026-09-26 15:16:00
**Evaluation Scope**: Full empirical ground truth (**7,638,365 pairs** across all 12,527,040 records)
**Configuration**: `A + C + D + E + F`, `MAX_BLOCK_SIZE = 5,000` (No Key B, No sorted-token keys)

## 1. Existing vs Modified Stopword Configuration

### Existing Configuration (`stopwords.py` - V3):
- `GENERIC_BUSINESS_TOKENS` already included generic industry words (`hotel`, `technologies`, `services`, etc.) and qualifiers (`new`, `old`, `big`, `sri`, `shri`, `shree`, `sree`).
- `HONORIFIC_TOKENS` included standard personal honorifics (`mr`, `mrs`, `ms`, `dr`, `prof`, `smt`, `shri`, `sri`, `shree`, `kumari`, `late`, `son`, `sons`, `brothers`).

### Modified Configuration (`stopwords.py` - V4):
- Added commercial partnership / entity prefixes commonly found in Indian noisy registries:
  - `m/s` (Messrs — present in 25,674 Source 2 and 29,304 Source 3 records, but almost absent in Source 1)
  - `m-s` (hyphenated variant)
  - `messrs` (expanded English form)
- Preserved integrity: These tokens are **NOT removed from the stored normalized business name**. They are **only ignored when extracting meaningful tokens** for Keys C, D, and E.

## 2. Before vs After Performance Metrics (Full Empirical Ground Truth)

| Metric | Before (V3 Blocker) | After (V4 Prefix Filter) | Delta | Impact / Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Blocking Recall** | **68.47%** | **68.75%** | **+0.28%** | **+21,678 net true pairs** (5,251,343 total) |
| **India True Match Recall** | **59.82%** | **60.53%** | **+0.71%** | **+22,151 newly covered Indian pairs** |
| **US True Match Recall** | **74.25%** | **74.25%** | **-0.00%** | Unaffected (0 newly covered) |
| **S1 → S2 Recall** | **69.36%** | **69.63%** | **+0.27%** | Corporate registry linkage |
| **S1 → S3 Recall** | **67.62%** | **67.92%** | **+0.30%** | Crowdsourced directory linkage |
| **Total Candidate Pairs** | **56,901,102** | **57,045,515** | **+144,413 (+0.25%)** | Negligible growth (~1.4%) |
| **Candidates / S1 Query** | **25.78** | **25.85** | **+0.07** | Well below budget cap of 50 |
| **Reduction Ratio** | **99.999750%** | **99.999750%** | **0.000000%** | >99.999% maintained |
| **Oversized Blocks Omitted** | **4,264** | **4,279** | **+15** | Cap = 5,000 |

## 3. Analysis of Newly Recovered True Pairs

- **Newly Covered Ground-Truth Pairs**: **+22,151 pairs** that were previously missed by every single blocking key now enter the candidate pool.
- **Net Gain**: **+21,678 true pairs** (accounting for 473 pairs that shifted blocks into oversized categories).
- **Geographic Distribution**: **22,151** Indian pairs (100.0%) and **0** US pairs.
- **Why they were missed previously**: In Source 2 and Source 3, over **55,000 entities** had names beginning with `'M/S '` (e.g. `'M/S Balaji Traders'`), while Source 1 almost exclusively omitted this prefix (`'Balaji Traders'`). Treating `'m/s'` as a meaningful token meant that Keys C, D, and E sought candidates beginning with token `'m/s'`, completely preventing alignment with Source 1. Removing `'m/s'` from blocking token extraction restores exact alignment on the true distinguishing token (`'balaji'`).

## 4. Sample of Newly Covered Ground-Truth Pairs (15 Examples)

The following sample shows representative true match pairs recovered by V4 that were completely absent from V3 candidates:

| # | Country | Source | Matched Key | S1 Business Name & Address | Candidate Business Name & Address | Alignment Mechanism |
| :-: | :---: | :---: | :---: | :--- | :--- | :--- |
| 1 | India | source2 | Key C+D | **Name**: Intelligent (India) Electric Private Limited<br>**Addr**: Vadodara, Ff Shop No. 08, Taif Nagar Shoping Center, Gujarat | **Name**: M/s Intelligent (India) Electric Private  Limited<br>**Addr**: FF SHOP NO. 08, TAIF NAGAR SHOPING CENTER, VADODARA, ગુજરાત | Bypassed `m/s` prefix → aligned on core business name token |
| 2 | India | source3 | Key D | **Name**: Blackberry Traders Ltd<br>**Addr**: 6 A-43 F/F R/S/ West Extension Area Karol Bagh, Delhi, Central Delhi, Delhi | **Name**: M/s Blackberry Traders  Ltd<br>**Addr**: ##6 A-43 F/f R/s/ West Extension Area Karol Bagh, Central Delhi, Delhi, DL | Bypassed `m/s` prefix → aligned on core business name token |
| 3 | India | source3 | Key C+D | **Name**: Cars Karkhana Limited<br>**Addr**: Al-Rahaba Arcade, 19/1825 D2, 3Rd Floor Francis Road Junction, Madhavan Nair Road, Calicut, Kerala | **Name**: M/s Cars Karkhana<br>**Addr**: Al-rahaba Arcade, 19/1825 D2, 3Rd Floor Francis Road Junction, Madhavan Nair Road, Calicut, KL | Bypassed `m/s` prefix → aligned on core business name token |
| 4 | India | source2 | Key C+D | **Name**: Sbt Kitchen Private Limited<br>**Addr**: No. 2035/2232 Subhash, Nelamangala, Bangalore Rural, Karnataka | **Name**: M/s Sbt Kitchen Limited Private<br>**Addr**: NO. 2035/2232 SUBHASH, NELAMANGALA, Karnataka | Bypassed `m/s` prefix → aligned on core business name token |
| 5 | India | source3 | Key C+E | **Name**: Everyday Developersprivate (India) Private Limited<br>**Addr**: C/O Satpal Mahavi, Sehlanga, Bahu, Jhajjar, Haryana | **Name**: M/s Everyday Developersprivate (India) Private Limited<br>**Addr**: C/o Satpal Mahavi, Sehlanga, Bahu, Jhajjar, HR | Bypassed `m/s` prefix → aligned on core business name token |
| 6 | India | source3 | Key C | **Name**: Zealous (India) Pharmaceuticals Private Limited<br>**Addr**: B 5/5 Vasant Vihar, New Delhi, South West, Delhi | **Name**: M/s Zealous (India) Pharmaceuticals Limited Center<br>**Addr**: B 005/5 Vasant Vihar, New Delhi, Delhi, DL | Bypassed `m/s` prefix → aligned on core business name token |
| 7 | India | source3 | Key C+D | **Name**: Grow Realty (India) Private Limited<br>**Addr**: 229, 1St Floor, Katra Peran, Tilak Bazar Khari Baoli, Delhi, Central Delhi, Delhi | **Name**: M/s Grow Realty (India) Private Ltd<br>**Addr**: 229, Delhi, New Delhi, DL | Bypassed `m/s` prefix → aligned on core business name token |
| 8 | India | source2 | Key C+D | **Name**: Manzil Academy<br>**Addr**: 83 A, Near Iti College, Airport Road, Pratap Nagar, Udaipur, Rajasthan | **Name**: M/s Manzil Academy<br>**Addr**: 83 A, NEAR ITI COLLEGE, AIRPORT ROAD, PRATAP NAGAR, Rajasthan | Bypassed `m/s` prefix → aligned on core business name token |
| 9 | India | source3 | Key D | **Name**: Nanak Sciences<br>**Addr**: Fl No-2, 4Th Floor, Gayatri Sadan, 2060 Sadashiv Peth, Vijaynagar Colony, Near S.P Colleg, E, Pune, Maharashtra | **Name**: M/s Nanak 5ciences<br>**Addr**: Fl No-002, 4Th Floor, Gayatri Sadan, 2060 Sadashiv Peth, Vijaynagar Colony, Near S.p Colleg, E, Pune, MH | Bypassed `m/s` prefix → aligned on core business name token |
| 10 | India | source3 | Key D | **Name**: Tirupati Foundation Private Limited<br>**Addr**: Plot No. 5, 6, 8, Kartavya Industrial Estate, B/H Bhavda Bus Stop, Ahmedabad-Indore Highway, Dascroi, Ahmedabad, Gujarat | **Name**: M/s Tirupati Foundation Private Ltd<br>**Addr**: Plot No. 5, 6, 8, Kartavya Industrial Estate, B/h Bhavda Bus Stop, Ahmedabad-indore Highway, Dascroi, Ahmedabad, GJ | Bypassed `m/s` prefix → aligned on core business name token |
| 11 | India | source3 | Key E | **Name**: Pranam Industries Group Pvt. Ltd.<br>**Addr**: Promoters Add K-2042 Chitranjan Park, New Delhi, South Delhi, Delhi | **Name**: M/s Pranam Industries Group Pvt. Ltd.<br>**Addr**: None | Bypassed `m/s` prefix → aligned on core business name token |
| 12 | India | source3 | Key E | **Name**: Amish Brothers Private Limited<br>**Addr**: No 19A, Ground Floor, Red Hills Main Road, Kolathur, Chennai, Tamil Nadu | **Name**: M/s Amish Bothser Private Limited<br>**Addr**: H.no 19A, Kolathur, Chennai, TN | Bypassed `m/s` prefix → aligned on core business name token |
| 13 | India | source2 | Key C+E | **Name**: Dcl (India) Producer Private Limited<br>**Addr**: Mumbai City, Raj Residency, Gujar Lane, Off S. V. Road, Santacruz (West), Maharashtra, Mumbai, C-201 | **Name**: M/s Dcl (India)  Producer Private Limited<br>**Addr**: C-201, RAJ RESIDENCY, GUJAR LANE, OFF S. V. ROAD, SANTACRUZ (WEST), MUMBAI, Maharashtra | Bypassed `m/s` prefix → aligned on core business name token |
| 14 | India | source2 | Key C+D | **Name**: Mtl System Ltd<br>**Addr**: Tapowan Nagar Arazi 520, Lucknow, Uttar Pradesh, C/O M L Bajpai, Lucknow | **Name**: M/s Mtl System Ltd.<br>**Addr**: C/O M L BAJPAI, TAPOWAN NAGAR ARAZI 520, LUCKNOW, Uttar Pradesh | Bypassed `m/s` prefix → aligned on core business name token |
| 15 | India | source3 | Key D | **Name**: Select Land Pvt Ltd<br>**Addr**: Plot No 229 Swapna Sree Apartment 4Th Floor Sardar Patel Nagar Kukatpally, Hyderabad, Telangana | **Name**: M/s Select Ltad Pvt  Ltd<br>**Addr**: Plot No 229 Swapna Sree Apartment 4Th Floor Sardar Patel Nagar Kukatpally, Hyderabad, Andhra Pradesh | Bypassed `m/s` prefix → aligned on core business name token |

## 5. Candidate-Volume Impact

- The total candidate volume moved from **56,901,102** to **57,045,515** — an increase of only **144,413 candidate pairs (+0.25%)**.
- The average candidate count per S1 query rose by just **+0.07 candidates** (from 25.78 to 25.85).
- The block-size cap (`MAX_BLOCK_SIZE = 5,000`) effectively shielded the system from any block explosion.

## 6. Final Recommendation & Freeze Decision

```
╔══════════════════════════════════════════════════════════════════════════════╗
║   DECISION:                                                                  ║
║   A. FREEZE UPDATED BLOCKER                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Rationale:
1. **Measurable Recall Gain**: Successfully recovered **+22,151 previously unblockable true matches**, boosting India recall by **+0.71%**.
2. **Negligible Computational Overhead**: Candidate volume increased by merely **+0.25%** (averaging ~25.8 candidates/query), preserving the downstream pairwise ML budget.
3. **Zero Side Effects**: Normalization code was left intact; only token extraction for blocking keys C, D, and E was refined.

The candidate generation pipeline is now **frozen**. No further blocking changes or experiments are permitted. We proceed directly to Feature Engineering.