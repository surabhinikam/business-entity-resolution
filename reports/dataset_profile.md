# Business Entity Resolution - Dataset Profiling Report

## 1. Executive Summary & Macro Metrics

This document provides a thorough profiling of the organizer datasets located under `../student_resource/dataset/`.
All profiling was performed using streaming reads to ensure zero memory exhaustion and zero in-place mutations.

### 1.1 Complete Dataset Overview Table

| Dataset | Partition | File Size | Row Count | Delimiter | Encoding | Columns | Missing Addr (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | Train | 200.34 MB | 2,206,821 | `\t` | UTF-8 | `entity_id, business_name, business_address, country` | 0.0% |
| `train_source2.tsv` | Train | 466.63 MB | 5,034,616 | `\t` | UTF-8 | `entity_id, business_name, business_address, country` | 3.356% |
| `train_source3.tsv` | Train | 480.37 MB | 5,285,603 | `\t` | UTF-8 | `entity_id, business_name, business_address, country` | 3.328% |
| `test_source1.tsv` | Test | 166.91 MB | 1,732,544 | `\t` | UTF-8 | `entity_id, business_name, business_address, country` | 0.0% |
| `test_source2.tsv` | Test | 485.86 MB | 4,887,273 | `\t` | UTF-8 | `entity_id, business_name, business_address, country` | 2.648% |
| `test_source3.tsv` | Test | 482.56 MB | 5,082,316 | `\t` | UTF-8 | `entity_id, business_name, business_address, country` | 2.678% |
| `train_ground_truth.tsv` | Train (GT) | 121.13 MB | 2,206,821 | `\t` | UTF-8 | `source1_entity_id, matched_entity_ids` | N/A |

### 1.2 Country Distribution Across Datasets

| Dataset | Country Breakdown |
| :--- | :--- |
| `train_source1.tsv` | US: 1,323,633 (60.0%), India: 883,188 (40.0%) |
| `train_source2.tsv` | India: 2,017,799 (40.1%), US: 3,016,817 (59.9%) |
| `train_source3.tsv` | US: 3,170,056 (60.0%), India: 2,115,547 (40.0%) |
| `test_source1.tsv` | US: 663,106 (38.3%), France: 259,452 (15.0%), India: 809,986 (46.8%) |
| `test_source2.tsv` | India: 2,312,565 (47.3%), France: 703,378 (14.4%), US: 1,871,330 (38.3%) |
| `test_source3.tsv` | India: 2,405,000 (47.3%), France: 731,615 (14.4%), US: 1,945,701 (38.3%) |

> [!CRITICAL]
> **Out-of-Distribution Country in Test Data**: The `France` category accounts for ~15% of records in the test split (`test_source1`: 259,452 rows; `test_source2`: ~710k rows; `test_source3`: ~735k rows), but **0% in the training split**. Any hard-coded geographic rules or models conditioned strictly on US/India will fail on France test records.

## 2. Individual Source Dataset Profiles

### 2.1 `train_source1.tsv`

- **Path**: `C:\Users\surab\OneDrive\Documents\HACKATHONS\Amazon ML\student_resource\dataset\train\train_source1.tsv`
- **File Size**: 200.34 MB (210,069,713 bytes)
- **Total Rows**: 2,206,821
- **Encoding & Delimiter**: UTF-8, delimiter `\t (tab)`
- **Core Fields Verified**:
  - `business_name`: Present
  - `business_address`: Present
  - `country`: Present

#### Schema, Inferred Types, and Missing Values

| Column | Inferred Type | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- |
| `entity_id` | `string` | 0 | 0.0% |
| `business_name` | `string` | 0 | 0.0% |
| `business_address` | `string` | 0 | 0.0% |
| `country` | `string` | 0 | 0.0% |

#### Script Distribution (Sampled)

- **`business_name` Scripts**:
  - Latin: 100.0%
- **`business_address` Scripts**:
  - Latin: 100.0%

#### Character & Script Examples Found

| Script | Example Business Name | Example Address |
| :--- | :--- | :--- |
| Latin | Orelee's Barbershop | 1795 Westchester Drive, High Point, NC |

#### Representative Sample Records (5 Samples)

| entity_id | business_name | business_address | country |
| :--- | :--- | :--- | :--- |
| `S1-925783039` | Orelee's Barbershop | 1795 Westchester Drive, High Point, NC | US |
| `S1-773889195` | Prime Money | 17560 Ellis Road, Tahlequah, OK | US |
| `S1-309349399` | Callicoat & Dailey Inc | Charlotte, NC, 833 Reliance Street | US |

---

### 2.2 `train_source2.tsv`

- **Path**: `C:\Users\surab\OneDrive\Documents\HACKATHONS\Amazon ML\student_resource\dataset\train\train_source2.tsv`
- **File Size**: 466.63 MB (489,301,488 bytes)
- **Total Rows**: 5,034,616
- **Encoding & Delimiter**: UTF-8, delimiter `\t (tab)`
- **Core Fields Verified**:
  - `business_name`: Present
  - `business_address`: Present
  - `country`: Present

#### Schema, Inferred Types, and Missing Values

| Column | Inferred Type | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- |
| `entity_id` | `string` | 0 | 0.0% |
| `business_name` | `string` | 0 | 0.0% |
| `business_address` | `string` | 168,967 | 3.356% |
| `country` | `string` | 0 | 0.0% |

#### Script Distribution (Sampled)

- **`business_name` Scripts**:
  - Latin: 90.91%
  - Devanagari: 5.37%
  - Telugu: 0.78%
  - Kannada: 0.74%
  - Tamil: 0.68%
  - Gujarati: 0.62%
  - Bengali: 0.61%
  - Malayalam: 0.38%
  - Oriya: 0.15%
  - Gurmukhi: 0.13%
- **`business_address` Scripts**:
  - Latin: 96.61%
  - Devanagari: 5.51%
  - [No Letters/Empty]: 3.39%
  - Kannada: 0.76%
  - Tamil: 0.69%
  - Telugu: 0.67%
  - Bengali: 0.64%
  - Gujarati: 0.63%
  - Malayalam: 0.31%
  - Gurmukhi: 0.14%

#### Character & Script Examples Found

| Script | Example Business Name | Example Address |
| :--- | :--- | :--- |
| Devanagari | राम मार्केटिंग प्राइवेट लिमिटेड | PLOT NO B-78/1, ADDITIONAL MIDC ANAND NAGAR, AMBERNATH EAST, THANE, महाराष्ट्र |
| Latin | -- Holloway Peak Inc Seafood | KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi |
| Tamil | குளோபல் பிசினஸ் பிரைவேட் லிமிடெட் | #44, KILATHOPE STREET, MADURAI MADURAI, MADURAI, தமிழ்நாடு |
| Gujarati | શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ | PLOT 008, CHAITANYA IND AREA, GANGA FORGING GATE, KOTDA SANGHANI, ગુજરાત |
| Bengali | গোল্ড প্রডিউসার স্টোর্স লিমিটেড | NO. 28 PSIXL-3RD FLOOR, NEWTOWN ROAD, UNIT NO. 305, PO RAJARHAT GOPALPUR, KOLKATA, KOLKATA, পশ্চিমবঙ্গ |
| Kannada | ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | NO ##14 , SHREE SHIDRAM NILAYA, OPP KEC GANDHINAGAR, SANMARGA NAGAR, HUBLI, DHARWAD, ಕರ್ನಾಟಕ |
| Telugu | పర్‌ఫెక్ట్ యునైటెడ్ మీడియా ప్రైవేట్ లిమిటెడ్ | ఆంధ్రప్రదేశ్, DOOR NO.11/1/471, KRUPANANDA NAGAR, ANANTAPUR |
| Malayalam | ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ് | 9-557-4, VETTICKAL HOUSE, PALAKKAD, കേരളം |

#### Representative Sample Records (5 Samples)

| entity_id | business_name | business_address | country |
| :--- | :--- | :--- | :--- |
| `S2-166376419` | राम मार्केटिंग प्राइवेट लिमिटेड | KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi | India |
| `S2-764573417` | -- Holloway Peak Inc Seafood | 105 ELM ST, MORGANTON, NC | US |
| `S2-639257739` | आदित्य प्रॉपर्टीज एलएलपी | G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh | India |
| `S2-187379771` | Guerra And Krueger Table, LLC |  | US |

---

### 2.3 `train_source3.tsv`

- **Path**: `C:\Users\surab\OneDrive\Documents\HACKATHONS\Amazon ML\student_resource\dataset\train\train_source3.tsv`
- **File Size**: 480.37 MB (503,705,637 bytes)
- **Total Rows**: 5,285,603
- **Encoding & Delimiter**: UTF-8, delimiter `\t (tab)`
- **Core Fields Verified**:
  - `business_name`: Present
  - `business_address`: Present
  - `country`: Present

#### Schema, Inferred Types, and Missing Values

| Column | Inferred Type | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- |
| `entity_id` | `string` | 0 | 0.0% |
| `business_name` | `string` | 0 | 0.0% |
| `business_address` | `string` | 175,916 | 3.328% |
| `country` | `string` | 0 | 0.0% |

#### Script Distribution (Sampled)

- **`business_name` Scripts**:
  - Latin: 95.37%
  - Devanagari: 2.99%
  - Telugu: 0.43%
  - Kannada: 0.42%
  - Tamil: 0.37%
  - Bengali: 0.35%
  - Gujarati: 0.34%
  - Malayalam: 0.22%
  - Gurmukhi: 0.08%
  - Oriya: 0.08%
- **`business_address` Scripts**:
  - Latin: 96.69%
  - Devanagari: 5.23%
  - [No Letters/Empty]: 3.31%
  - Kannada: 0.75%
  - Tamil: 0.66%
  - Telugu: 0.64%
  - Bengali: 0.6%
  - Gujarati: 0.58%
  - Malayalam: 0.3%
  - Gurmukhi: 0.14%

#### Character & Script Examples Found

| Script | Example Business Name | Example Address |
| :--- | :--- | :--- |
| Latin | wilfordhancock.com | Mack Rd, Haltom City, Texas |
| Kannada | ಬ್ಲ್ಯಾಕ್ ಇಂಜಿನಿಯರಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ |
| Tamil | அரிஹந்த் Foundation Private Limited | Basement Shop No. B-14/2, Chennai, தமிழ்நாடு |
| Devanagari | ब्लू टेक्नोलॉजीज | Plot No.: 3829/3250, Near Metro Pillar No: 234, Kanhaiya Nagar, Tri Nagar, Delhi, New Delhi, दिल्ली |
| Malayalam | സിൽവർ കൺസൾട്ടൻസി പ്രൈവറ്റ് ലിമിറ്റഡ് | 6/170, Saravana Poyka, Madathuvilakam Perukavu P.o., Malayinkeezhu, Thiruvananthapuram, Thiruvananthapuram, Trivandrum, കേരളം |
| Telugu | గుజరాత్ Logistics లిమిటెడ్ | Hyderabad, No 12-5-35/C/47- Ambedkar Nagar, తెలంగాణ, Secunderabad |
| Gujarati | આલ્ફા ફાઉન્ડેશન પ્રાઇવેટ લિમિટેડ | Core House Off C G Roadnr Parimal Garden Ellisbridge, Ahmedbad, Ahmedabad, ગુજરાત |
| Bengali | রেড টেক প্রাইভেট লিমিটেড | 867 Ground-floor G-1, Domjur, 296/1 G.t.road Belur, Bally, পশ্চিমবঙ্গ |

#### Representative Sample Records (5 Samples)

| entity_id | business_name | business_address | country |
| :--- | :--- | :--- | :--- |
| `S3-202863386` | wilfordhancock.com | Mack Rd, Haltom City, Texas | US |
| `S3-859268022` | International South Consultants Private Ltd |  | India |
| `S3-45067784` | அரிஹந்த் Foundation Private Limited | Saranampatti Roadvellakinar Post Coimbatore, Tamilnadu, Coimbatore, TN | India |
| `S3-32805948` | സിൽവർ കൺസൾട്ടൻസി പ്രൈവറ്റ് ലിമിറ്റഡ് | Ernakulam, KL, 18/479E/3 Peringattil Tower Kangarappady Kakkanad, Ernakulam | India |

---

### 2.4 `test_source1.tsv`

- **Path**: `C:\Users\surab\OneDrive\Documents\HACKATHONS\Amazon ML\student_resource\dataset\test\test_source1.tsv`
- **File Size**: 166.91 MB (175,022,086 bytes)
- **Total Rows**: 1,732,544
- **Encoding & Delimiter**: UTF-8, delimiter `\t (tab)`
- **Core Fields Verified**:
  - `business_name`: Present
  - `business_address`: Present
  - `country`: Present

#### Schema, Inferred Types, and Missing Values

| Column | Inferred Type | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- |
| `entity_id` | `string` | 0 | 0.0% |
| `business_name` | `string` | 0 | 0.0% |
| `business_address` | `string` | 0 | 0.0% |
| `country` | `string` | 0 | 0.0% |

#### Script Distribution (Sampled)

- **`business_name` Scripts**:
  - Latin: 100.0%
- **`business_address` Scripts**:
  - Latin: 100.0%

#### Character & Script Examples Found

| Script | Example Business Name | Example Address |
| :--- | :--- | :--- |
| Latin | Zephay Labs Inc | 2621 Cotten Road, Tyler, TX |

#### Representative Sample Records (5 Samples)

| entity_id | business_name | business_address | country |
| :--- | :--- | :--- | :--- |
| `S1-714132312` | Zephay Labs Inc | 2621 Cotten Road, Tyler, TX | US |
| `S1-106407869` | Vision Partners Corp | IA, Iowa City, 1064 Newton Rd, Unit 11 | US |
| `S1-913506265` | Thermal & Fils SASU | 20 Rue Parmentier, Dunkerque, Hauts-de-France | France |

---

### 2.5 `test_source2.tsv`

- **Path**: `C:\Users\surab\OneDrive\Documents\HACKATHONS\Amazon ML\student_resource\dataset\test\test_source2.tsv`
- **File Size**: 485.86 MB (509,456,422 bytes)
- **Total Rows**: 4,887,273
- **Encoding & Delimiter**: UTF-8, delimiter `\t (tab)`
- **Core Fields Verified**:
  - `business_name`: Present
  - `business_address`: Present
  - `country`: Present

#### Schema, Inferred Types, and Missing Values

| Column | Inferred Type | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- |
| `entity_id` | `string` | 0 | 0.0% |
| `business_name` | `string` | 0 | 0.0% |
| `business_address` | `string` | 129,408 | 2.648% |
| `country` | `string` | 0 | 0.0% |

#### Script Distribution (Sampled)

- **`business_name` Scripts**:
  - Latin: 89.26%
  - Devanagari: 6.29%
  - Telugu: 0.92%
  - Kannada: 0.9%
  - Tamil: 0.8%
  - Gujarati: 0.74%
  - Bengali: 0.72%
  - Malayalam: 0.45%
  - Oriya: 0.17%
  - Gurmukhi: 0.16%
- **`business_address` Scripts**:
  - Latin: 97.35%
  - Devanagari: 6.5%
  - [No Letters/Empty]: 2.65%
  - Kannada: 0.92%
  - Tamil: 0.84%
  - Telugu: 0.8%
  - Bengali: 0.78%
  - Gujarati: 0.75%
  - Malayalam: 0.37%
  - Gurmukhi: 0.16%

#### Character & Script Examples Found

| Script | Example Business Name | Example Address |
| :--- | :--- | :--- |
| Latin | Brahma Infosoft | COIMATORE COLONY, HUNSUR TQMYSORE DIST., Karnataka |
| Gujarati | ૐ Foundation પ્રાઇવેટ લિમિટેડ | CRYSTAL PLAZA, TOWER NO.1, 2ND FLOOR, 2-B GOTRI MAIN ROAD, VADODARA, ગુજરાત |
| Devanagari | आनंद फाउंडेशन प्राइवेट लिमिटेड | H.NO 4/A ASHTAVINAYAK, NAGAR, NAGPUR (URBAN), NAGPUR, महाराष्ट्र |
| Telugu | కృష్ణా ఇంపెక్స్ లిమిటెడ్ | 0098 GALAXY, FLOORS 22-24, PLOT NO. 1 SURVEY NO 83/1, SERI LINGAMPALLY, తెలంగాణ |
| Bengali | ইনোভেটিভ প্রোডাক্টস রেস্টুরেন্ট লিমিটেড | NO D/38 SONAMUKHI ROAD DASPARA, ASHUTI, LP-211/32 UG, THAKURPUKUR MAHESTOLA, পশ্চিমবঙ্গ |
| Tamil | யுனைடெட் சிஸ்டம்ஸ் கன்ஸ்ட்ரக்ஷன்ஸ் பிரைவேட் லிமிடெட் | G-134, VALLUVAR KOTTAM HIGH ROAD, NUNGAMBAKKAM, CHENNAI, தமிழ்நாடு |
| Malayalam | അൽ കൺസ്ട്രക്ഷൻസ് ഫുഡ്സ് പ്രൈവറ്റ് ലിമിറ്റഡ് | XVIII/C-37, FIRST FLOOR, U BROTHERS BUILDING, GURUVAYOOR ROAD, KUNNAMKULAM POST, THRISSUR, കേരളം |
| Kannada | ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | 3980/3988 HOSAKEREHALLI MAIN RD, BSK 2 STG BANGALORE, BANGALORE SOUTH, ಕರ್ನಾಟಕ |

#### Representative Sample Records (5 Samples)

| entity_id | business_name | business_address | country |
| :--- | :--- | :--- | :--- |
| `S2-192345572` | Brahma Infosoft | COIMATORE COLONY, HUNSUR TQMYSORE DIST., Karnataka | India |
| `S2-566025912` | Marina Ecole France Sarl | 63 R. DE DIEPPE, LILLE, Hauts-de-France | France |
| `S2-956479781` | కృష్ణా ఇంపెక్స్ లిమిటెడ్ | H.NO 27TH FLOOR, Telangana, HYDERABAD, SERI LINGAMPALLY, G SQUARE, NEAR WELLS FARGO, RAIDURGAM | India |
| `S2-875724478` | ૐ Foundation પ્રાઇવેટ લિમિટેડ | #203, NARAYAN KRUPA SQUARE, B/H NATRAJ CINEMA, NR. SAKAR V, ASHRAM R, OAD, MEMNAGAR, Gujarat | India |
| `S2-919241341` | Labh Krishak |  | India |

---

### 2.6 `test_source3.tsv`

- **Path**: `C:\Users\surab\OneDrive\Documents\HACKATHONS\Amazon ML\student_resource\dataset\test\test_source3.tsv`
- **File Size**: 482.56 MB (506,002,772 bytes)
- **Total Rows**: 5,082,316
- **Encoding & Delimiter**: UTF-8, delimiter `\t (tab)`
- **Core Fields Verified**:
  - `business_name`: Present
  - `business_address`: Present
  - `country`: Present

#### Schema, Inferred Types, and Missing Values

| Column | Inferred Type | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- |
| `entity_id` | `string` | 0 | 0.0% |
| `business_name` | `string` | 0 | 0.0% |
| `business_address` | `string` | 136,098 | 2.678% |
| `country` | `string` | 0 | 0.0% |

#### Script Distribution (Sampled)

- **`business_name` Scripts**:
  - Latin: 94.52%
  - Devanagari: 3.51%
  - Telugu: 0.53%
  - Kannada: 0.51%
  - Tamil: 0.45%
  - Bengali: 0.41%
  - Gujarati: 0.4%
  - Malayalam: 0.26%
  - Oriya: 0.1%
  - Gurmukhi: 0.09%
- **`business_address` Scripts**:
  - Latin: 97.32%
  - Devanagari: 6.26%
  - [No Letters/Empty]: 2.68%
  - Kannada: 0.88%
  - Tamil: 0.77%
  - Telugu: 0.77%
  - Gujarati: 0.73%
  - Bengali: 0.71%
  - Malayalam: 0.36%
  - Gurmukhi: 0.16%

#### Character & Script Examples Found

| Script | Example Business Name | Example Address |
| :--- | :--- | :--- |
| Devanagari | मॉडर्न फाइनेंस | H.no 910 A 3503, Mumbai, महाराष्ट्र |
| Latin | Shri Sai Infratech Co | No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR |
| Tamil | ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமிடெட் | No 25 Wvn Manor, Chennai, தமிழ்நாடு |
| Kannada | ಡಿಜಿಟಲ್ ಬಿಲ್ಡರ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | #803 20/1, Bangalore South, Begur, ಕರ್ನಾಟಕ |
| Bengali | ইনোভেটিভ এনার্জি | Kolkata, F-06, 28 No. Road Garden Reach, Kolkata, Howrah, পশ্চিমবঙ্গ |
| Gujarati | Tech Food પ્રાઇવેટ લિમિટેડ | No 2-P, Mahavir Nagar, Plot No. 10/A, Gandhidham, Kachchh, ગુજરાત |
| Telugu | గుడ్ ఎనర్జీ ప్రైవేట్ లిమిటెడ్ | P.no.46, Sy.no.133, Fno 502, Samanthapuri Colony, Sai Ram Avenue, Above Indian Bank, Near Na, Gole, Mohan Nagar, Hyderabad, తెలంగాణ |
| Malayalam | സായ് മാനേജ്മെന്റ് എൻജിനീയറിംഗ് | Room No.18/93a Muvattupuzha P O, Muvattupuzha, Ernakulam, കേരളം |

#### Representative Sample Records (5 Samples)

| entity_id | business_name | business_address | country |
| :--- | :--- | :--- | :--- |
| `S3-462677478` | मॉडर्न फाइनेंस | No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR | India |
| `S3-374810425` | Shri Sai Infratech Co | 3/115, East Delhi, DL | India |
| `S3-184785497` | ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமிடெட் | 1410 Sri Mahalakshmi Mandiar 59, Justice Rathinavel Pandian Road, Chennai, TN | India |
| `S3-643284918` | Swing Maison  SARL |  | France |
| `S3-198586129` | Fractales Amis Groupe S.A.S | 23 Rue Icmre, La Teste-de-buch, Gironde | France |

---

## 3. Ground Truth Analysis (`train_ground_truth.tsv`)

- **Schema**: `source1_entity_id, matched_entity_ids`
- **Total Rows**: 2,206,821
- **Total Matched Entity Pairings**: 7,638,365
  - Matches to Source 2: 3,693,619
  - Matches to Source 3: 3,944,746
- **Unmatched Source 1 Records (No Ground Truth Match)**: 123,247 (5.585%)
- **Average Matches per Source 1 Record**: 3.46 (Average for matched: 3.67)
- **Unique Matched Source 2 Entities**: 3,693,619 / 5,034,616
- **Unique Matched Source 3 Entities**: 3,944,746 / 5,285,603

### 3.1 Linkage Mechanism

Each record maps a single query business entity from source1 (S1-*) to zero, one, or multiple candidate entities in source2 (S2-*) and source3 (S3-*). Multiple matches are represented as comma-separated identifiers.

Distribution of Match Cardinalities (number of candidate entities linked to one source1 entity):

- **3 matches**: 530,841 source1 records (24.05%)
- **4 matches**: 484,115 source1 records (21.94%)
- **2 matches**: 375,212 source1 records (17.0%)
- **5 matches**: 321,957 source1 records (14.59%)
- **6 matches**: 164,868 source1 records (7.47%)
- **0 matches**: 123,247 source1 records (5.58%)

Sample Ground Truth Rows:

| source1_entity_id | Total Matches | S2 Matches | S3 Matches | Matched Entity IDs (Sample) |
| :--- | :--- | :--- | :--- | :--- |
| `S1-965667` | 5 | 2 | 3 | `S2-681193310,S2-743505751,S3-775321672,S3-11291185,S3-860443364` |
| `S1-55344266` | 4 | 2 | 2 | `S2-249013014,S2-197070651,S3-478195123,S3-384364074` |
| `S1-343815751` | 3 | 2 | 1 | `S2-790675320,S2-479876582,S3-878454467` |
| `S1-656753428` | 3 | 2 | 1 | `S2-153058913,S2-24659151,S3-679606215` |
| `S1-102811957` | 6 | 3 | 3 | `S2-478959098,S2-553508714,S2-625774905,S3-728090388,S3-928796641,S3-449308785` |

---

## 4. Key Findings & Uncertainties to Resolve

### 4.1 Key Findings
1. **Unified Schema**: Every source dataset in both `train` and `test` strictly adheres to `[entity_id, business_name, business_address, country]` delimited by tabs (`\t`) with UTF-8 encoding.
2. **Source 1 is Uniform Latin**: In both train and test, `source1` contains 100% Latin text (even for Indian businesses, names/addresses are in English or transliterated Latin). There are 0 missing addresses in `source1`.
3. **Source 2 & 3 are Multilingual / Multi-Script**: Contains native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Gurmukhi, Oriya) alongside English. Addressing requires robust, offline transliteration to bridge script barriers with Source 1.
4. **Missing Addresses in Source 2 and 3**: ~2.6% to 3.4% of entities in Source 2 and 3 have blank `business_address`. Matching logic must handle entity resolution on `business_name` + `country` alone when addresses are absent.
5. **Ground Truth Structure**: Ground truth is 1:N mapping `source1_entity_id` -> `[S2-*, S3-*]`. Approximately 5.58% of `source1` entities have 0 matches.

### 4.2 Critical Uncertainties for Preprocessing
1. **Out-of-Distribution `France` in Test**: `France` records exist exclusively in the test sets (~15% of records). Preprocessing and normalization rules must handle French address conventions (e.g., *Rue*, *Avenue*, *Boulevard*, *Cedex*, postal codes) and legal suffixes (*SAS*, *SARL*, *SASU*) without overfitting to US/India data.
2. **Transliteration Strategy (Offline Only)**: Because network calls and external APIs are strictly prohibited, we must evaluate offline transliteration libraries (e.g. `indic-transliteration` or `polyglot`/rule-based mapping) to map Indic scripts to Latin without installing heavy or disallowed network dependencies.
3. **Country Naming Consistency**: Do countries have divergent naming forms (e.g. `US` vs `USA` vs `United States`)? Current profiling shows exact values `US`, `India`, `France`, but casing and whitespace must be strictly standardized.
4. **Memory Footprint & Blocking**: The total test dataset across the three sources is ~11.7 million rows. Naive pairwise matching ($1.7M \times 10M$) is impossible ($10^{13}$ pairs). High-efficiency blocking (by country, phonetic tokens, or locality) and chunked processing are mandatory.
