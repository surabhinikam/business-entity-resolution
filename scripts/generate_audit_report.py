"""
Script to generate the comprehensive Transliterated Vocabulary Audit report.
"""

from __future__ import annotations

import collections
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pyarrow.parquet as pq

DATA_DIR = "data/processed/train"
REPORT_PATH = "reports/transliterated_vocabulary_audit.md"

# Known business words mapping hypothesis for categorization
BUSINESS_KEYWORDS = {
    # Legal / corporate structure
    "ltd": ("limited", "High: Standard canonical legal suffix", ["लिमिटेड", "ലിമിറ്റഡ്", "লিমিটেড"]),
    "pvt": ("private", "High: Standard canonical legal suffix", ["प्राइवेट", "પ્રાઇવેટ"]),
    "praivet": ("private", "High: Kannada/Telugu transliteration of 'Private' (ಪ್ರೈವೇಟ್ / ప్రైవేట్)", ["ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"]),
    "limidhedh": ("limited", "High: Tamil transliteration of 'Limited' (லிமிடெட்)", ["குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்"]),
    "bhiraivedh": ("private", "High: Tamil transliteration of 'Private' (பிரைவேட்)", ["குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்"]),
    "praibheta": ("private", "High: Bengali transliteration of 'Private' (প্রাইভেট)", ["ডায়নামিক ইন্ডাস্ট্রিজ প্রাইভেট লিমিটেড"]),
    "llp": ("llp", "High: Standard canonical legal suffix for Limited Liability Partnership", ["एलएलपी", "એલએલપી"]),
    "limirrad": ("limited", "High: Malayalam transliteration of 'Limited' (ലിമിറ്റഡ്)", ["ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്"]),
    "praivarr": ("private", "High: Malayalam transliteration of 'Private' (പ്രൈവറ്റ്)", ["ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്"]),
    "praibhet": ("private", "High: Odia transliteration of 'Private' (ପ୍ରାଇଭେଟ୍)", ["ଭିଜନ୍ ଟେକ୍ନୋଲୋଜିସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍"]),
    "limatida": ("limited", "High: Gurmukhi transliteration of 'Limited' (ਲਿਮਟਿਡ)", ["ਸਕਾਈ ਈਸਟ ਇਸਟੇਟ ਲਿਮਟਿਡ"]),
    "praiveta": ("private", "High: Devanagari/Gurmukhi transliteration of 'Private' (प्राइवेट / ਪ੍ਰਾਈਵੇਟ)", ["ਸੂਰਜ ਡਿਵੈਲਪਰਜ਼ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ"]),
    "elelbhi": ("llp", "High: Tamil transliteration of 'LLP' (எல்எல்பி)", ["பிரைம் ஐடி எல்எல்பி"]),

    # Core business domain terms
    "marketimga": ("marketing", "High: Devanagari/Indic transliteration of 'Marketing' (मार्केटिंग / માર્કેટિંગ)", ["राम मार्केटिंग प्राइवेट लिमिटेड"]),
    "praopartija": ("properties", "High: Devanagari transliteration of 'Properties' (प्रॉपर्टीज)", ["आदित्य प्रॉपर्टीज एलएलपी"]),
    "emtarapraijeja": ("enterprises", "High: Devanagari/Indic transliteration of 'Enterprises' (एंटरप्राइजेज)", ["नॉर्थ कंसल्टेंट्स एंटरप्राइजेज प्राइवेट लिमिटेड"]),
    "emtar": ("enterprise", "High: Truncated/pre-segment in South Indic scripts (ಎಂಟರ್‌ / ఎంటర్‌)", ["ಗ್ಯಾಲಕ್ಸಿ ಎಂಟರ್‌ಪ್ರೈಸಸ್"]),
    "tredimga": ("trading", "High: Devanagari/Gujarati transliteration of 'Trading' (ट्रेडिंग / ટ્રેડિંગ)", ["પ્રાઇમ બાલાજી ટ્રેડિંગ પ્રા. લિ."]),
    "tredimg": ("trading", "High: Kannada/Telugu transliteration of 'Trading' (ಟ್ರೇಡಿಂಗ್ / ట్రేడింగ్)", ["ವೈಟ್ ಟ್ರೇಡಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"]),
    "tredarsa": ("traders", "High: Devanagari transliteration of 'Traders' (ट्रेडर्स)", ["माय जय प्रोड्यूसर ट्रेडर्स लिमिटेड"]),
    "imdastrija": ("industries", "High: Devanagari transliteration of 'Industries' (इंडस्ट्रीज)", ["इंडियन इंडस्ट्रीज"]),
    "imdastris": ("industries", "High: Telugu/Kannada transliteration of 'Industries' (ఇండస్ట్రీస్)", ["మోడర్న్ ఇండస్ట్రీస్ ప్రైవేట్ లిమిటెడ్"]),
    "teknolaojija": ("technologies", "High: Devanagari transliteration of 'Technologies' (टेक्नोलॉजीज)", ["अल्फा अल टेक्नोलॉजीज प्राइवेट लिमिटेड"]),
    "teknolaoji": ("technology", "High: Devanagari transliteration of 'Technology' (टेक्नोलॉजी)", ["श्री टेक्नोलॉजी लिमिटेड"]),
    "teknalajis": ("technologies", "High: Kannada transliteration of 'Technologies' (ಟೆಕ್ನಾಲಜೀಸ್)", ["ಫ್ಯೂಚರ್ ಟೆಕ್ನಾಲಜೀಸ್"]),
    "teknalaji": ("technology", "High: Kannada transliteration of 'Technology' (ಟೆಕ್ನಾಲಜಿ)", ["ಅರಿಹಂತ್ ಕ್ರಿಯೇಟಿವ್ ಟೆಕ್ನಾಲಜಿ"]),
    "sarviseja": ("services", "High: Devanagari transliteration of 'Services' (सर्विसेज)", ["प्राइम भारत सर्विसेज"]),
    "saolyusamsa": ("solutions", "High: Devanagari transliteration of 'Solutions' (सॉल्यूशंस)", ["एस सॉल्यूशंस एलएलपी"]),
    "solyusans": ("solutions", "High: Telugu transliteration of 'Solutions' (సొల్యూషన్స్)", ["సన్ బెస్ట్ సొల్యూషన్స్ ప్రైవేట్ లిమిటెడ్"]),
    "laojistiksa": ("logistics", "High: Devanagari transliteration of 'Logistics' (लॉजिस्टिक्स)", ["मॉडर्न लॉजिस्टिक्स प्राइवेट लिमिटेड"]),
    "kamstraksamsa": ("constructions", "High: Devanagari transliteration of 'Constructions' (कंस्ट्रक्शंस)", ["सन कंस्ट्रक्शंस प्राइवेट लिमिटेड"]),
    "kamstraksana": ("construction", "High: Devanagari transliteration of 'Construction' (कंस्ट्रक्शन)", ["गुड गैलेक्सी कंस्ट्रक्शन"]),
    "straksans": ("constructions", "High: Kannada transliteration of '-structions' (ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್)", ["ವೈಟ್ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್"]),
    "imphra": ("infra", "High: Devanagari/Indic transliteration of 'Infra' (इंफ्रा / இன்ஃப்ரா)", ["मां इंफ्रा प्राइवेट लिमिटेड"]),
    "imphrastrakcara": ("infrastructure", "High: Devanagari transliteration of 'Infrastructure' (इंफ्रास्ट्रक्चर)", ["ब्राइट इंफ्रास्ट्रक्चर लिमिटेड"]),
    "phrastrakcar": ("infrastructure", "High: Kannada transliteration of '-frastructure' (ಇನ್‌ಫ್ರಾಸ್ಟ್ರಕ್ಚರ್)", ["ಸಾಯಿ ಇನ್‌ಫ್ರಾಸ್ಟ್ರಕ್ಚರ್"]),
    "imjiniyarimga": ("engineering", "High: Devanagari transliteration of 'Engineering' (इंजीनियरिंग)", ["भारत इंजीनियरिंग प्राइवेट लिमिटेड"]),
    "kamsaltemtsa": ("consultants", "High: Devanagari transliteration of 'Consultants' (कंसल्टेंट्स)", ["नॉर्थ कंसल्टेंट्स एंटरप्राइजेज"]),
    "kamsaltimga": ("consulting", "High: Devanagari transliteration of 'Consulting' (कंसल्टिंग)", ["ग्लोबल कंसल्टिंग प्राइवेट लिमिटेड"]),
    "kamsaltemsi": ("consultancy", "High: Devanagari transliteration of 'Consultancy' (कंसल्टेंसी)", ["स्काई कंसल्टेंसी प्राइवेट लिमिटेड"]),
    "kansaltensi": ("consultancy", "High: Kannada transliteration of 'Consultancy' (ಕನ್ಸಲ್ಟೆನ್ಸಿ)", ["ಜಯ್ ಕನ್ಸಲ್ಟೆನ್ಸಿ"]),
    "devalaparsa": ("developers", "High: Devanagari transliteration of 'Developers' (डेवलपर्स)", ["लक्ष्मी डेवलपर्स प्राइवेट लिमिटेड"]),
    "bildarsa": ("builders", "High: Devanagari transliteration of 'Builders' (बिल्डर्स)", ["सन बिल्डर्स प्राइवेट लिमिटेड"]),
    "investamemtsa": ("investments", "High: Devanagari transliteration of 'Investments' (इन्वेस्टमेंट्स)", ["ब्लू विजय इन्वेस्टमेंट्स"]),
    "investamemta": ("investment", "High: Devanagari transliteration of 'Investment' (इन्वेस्टमेंट)", ["इंडियन इन्वेस्टमेंट प्रा. लि."]),
    "invest": ("investments", "High: Telugu transliteration of 'Investments' (ఇన్వెస్ట్‌మెంట్స్)", ["గెలాక్సీ ఇన్వెస్ట్‌మెంట్స్"]),
    "bijanesa": ("business", "High: Devanagari transliteration of 'Business' (बिजनेस)", ["जैन बिजनेस प्राइवेट लिमिटेड"]),
    "phaumdesana": ("foundation", "High: Devanagari transliteration of 'Foundation' (फाउंडेशन)", ["जैन फाउंडेशन प्राइवेट लिमिटेड"]),
    "phaumdesan": ("foundation", "High: Kannada transliteration of 'Foundation' (ಫೌಂಡೇಶನ್)", ["ಎಸ್ಎಸ್ ಫೌಂಡೇಶನ್"]),
    "phainemsa": ("finance", "High: Devanagari transliteration of 'Finance' (फाइनेंस)", ["बेस्ट फाइनेंस टेक्नोलॉजीज"]),
    "phainans": ("finance", "High: Kannada transliteration of 'Finance' (ಫೈನಾನ್ಸ್)", ["ವಿಜಯ್ ಫೈನಾನ್ಸ್"]),
    "helthakeyara": ("healthcare", "High: Devanagari transliteration of 'Healthcare' (हेल्थकेयर)", ["जय नॉर्थ हेल्थकेयर"]),
    "keyara": ("care", "High: Devanagari transliteration of 'Care' (केयर)", ["ओम केयर प्राइवेट लिमिटेड"]),
    "ker": ("care", "High: Telugu transliteration of 'Care' (కేర్)", ["లోటస్ కేర్"]),
    "saophtaveyara": ("software", "High: Devanagari transliteration of 'Software' (सॉफ्टवेयर)", ["अरिहंत सॉफ्टवेयर"]),
    "sapht": ("software", "High: Telugu transliteration of 'Soft-' (సాఫ్ట్‌వేర్)", ["పయనీర్ సాఫ్ట్‌వేర్"]),
    "mainejamemta": ("management", "High: Devanagari transliteration of 'Management' (मैनेजमेंट)", ["इंडो मैनेजमेंट प्रा. लि."]),
    "memt": ("management", "High: Telugu transliteration of '-ment' (మేనేజ్‌మెంట్)", ["క్లాసిక్ మేనేజ్‌మెంట్"]),
    "imphoteka": ("infotech", "High: Devanagari transliteration of 'Infotech' (इंफोटेक)", ["सूर्य इंफोटेक लिमिटेड"]),
    "teka": ("tech", "High: Devanagari transliteration of 'Tech' (टेक)", ["रॉयल टेक प्रा. लि."]),
    "tek": ("tech", "High: Kannada/Telugu transliteration of 'Tech' (ಟೆಕ್ / టెక్)", ["ಆಲ್ ಟೆಕ್"]),
    "aiti": ("it", "High: Indic transliteration of 'IT' (आईटी / ಐಟಿ / ఐటీ)", ["सदर्न एसएस आईटी प्राइवेट लिमिटेड"]),
    "globala": ("global", "High: Devanagari transliteration of 'Global' (ग्लोबल)", ["ग्लोबल इन्वेस्टमेंट"]),
    "global": ("global", "High: Kannada/Telugu transliteration of 'Global' (ಗ್ಲೋಬಲ್)", ["ಡೈನಾಮಿಕ್ ವಿಜಯ್ ಗ್ಲೋಬಲ್"]),
    "imtaranesanala": ("international", "High: Devanagari transliteration of 'International' (इंटरनेशनल)", ["विजय इंटरनेशनल प्राइवेट लिमिटेड"]),
    "phudsa": ("foods", "High: Devanagari transliteration of 'Foods' (फूड्स)", ["सन सिस्टम्स फूड्स"]),
    "phuda": ("food", "High: Devanagari transliteration of 'Food' (फूड)", ["ईस्ट फूड प्राइवेट लिमिटेड"]),
    "phud": ("food", "High: Kannada transliteration of 'Food' (ಫುಡ್)", ["ಸೂಪರ್ ಬ್ಲ್ಯಾಕ್ ಫುಡ್"]),
    "phuds": ("foods", "High: Odia transliteration of 'Foods' (ଫୁଡ୍ସ୍)", ["ଶ୍ୟାମ ଫୁଡ୍ସ୍"]),
    "impeksa": ("impex", "High: Devanagari/Bengali transliteration of 'Impex' (इम्पेक्स / ইম্পেক্স)", ["হরি ইম্পেক্স"]),
    "eksaportsa": ("exports", "High: Devanagari/Gujarati transliteration of 'Exports' (एक्सपोर्ट्स)", ["નોર્થ એક્સપોર્ટ્સ"]),
    "prodyusara": ("producer", "High: Devanagari transliteration of 'Producer' (प्रोड्यूसर)", ["शिव प्रोड्यूसर"]),
    "prodaktsa": ("products", "High: Devanagari/Gujarati transliteration of 'Products' (प्रोडक्ट्स)", ["શક્તિ અર્બન પ્રોડક્ટ્સ"]),
    "prodakts": ("products", "High: Malayalam transliteration of 'Products' (പ്രൊഡക്ട്സ്)", ["വൺ പ്രൊഡക്ട്സ്"]),
    "projektsa": ("projects", "High: Devanagari transliteration of 'Projects' (प्रोजेक्ट्स)", ["एपेक्स प्रोजेक्ट्स"]),
    "prajekts": ("projects", "High: Telugu transliteration of 'Projects' (ప్రాజెక్ట్స్)", ["ఆల్ స్టార్ ప్రాజెక్ట్స్"]),
    "vemcarsa": ("ventures", "High: Devanagari transliteration of 'Ventures' (वेंचर्स)", ["होटल वेंचर्स"]),
    "pavara": ("power", "High: Devanagari transliteration of 'Power' (पावर)", ["बेस्ट नॉर्थ पावर"]),
    "sistamsa": ("systems", "High: Devanagari transliteration of 'Systems' (सिस्टम्स)", ["इंडो सिस्टम्स"]),
    "esteta": ("estate", "High: Devanagari transliteration of 'Estate' (एस्टेट)", ["बॉम्बे एस्टेट"]),
    "estet": ("estate", "High: Telugu transliteration of 'Estate' (ఎస్టేట్)", ["నార్త్ ఎస్టేట్"]),
    "midiya": ("media", "High: Indic transliteration of 'Media' (मीडिया / మీడియా)", ["పర్‌ఫెక్ట్ యునైటెడ్ మీడియా"]),
    "egro": ("agro", "High: Devanagari transliteration of 'Agro' (एग्रो)", ["बिग एग्रो लिमिटेड"]),
    "agro": ("agro", "High: Telugu transliteration of 'Agro' (ఆగ్రో)", ["శ్రీ ఆగ్రో"]),
    "enarji": ("energy", "High: Indic transliteration of 'Energy' (एनर्जी / એનર્જી)", ["सिटी आदित्य एनर्जी"]),
    "hotala": ("hotel", "High: Devanagari transliteration of 'Hotel' (होटल)", ["होटल वेंचर्स लिमिटेड"]),
    "haospitailiti": ("hospitality", "High: Devanagari transliteration of 'Hospitality' (हॉस्पिटैलिटी)", ["सेवन हॉस्पिटैलिटी"]),
    "haspitaliti": ("hospitality", "High: Kannada transliteration of 'Hospitality' (ಹಾಸ್ಪಿಟಾಲಿಟಿ)", ["ಗುಜರಾತ್ ಹಾಸ್ಪಿಟಾಲಿಟಿ"]),
    "dijitala": ("digital", "High: Devanagari transliteration of 'Digital' (डिजिटल)", ["बेस्ट डिजिटल इंफ्रा"]),
    "haiteka": ("hitech", "High: Bengali/Indic transliteration of 'Hitech' (হাইটেক)", ["হাইটেক ইনফ্রা"]),
    "krietiva": ("creative", "High: Gujarati transliteration of 'Creative' (ક્રિએટિવ)", ["ક્રિએટિવ ટેક્નોલોજી"]),
    "yunika": ("unique", "High: Gujarati transliteration of 'Unique' (યુનિક)", ["યુનિક ઇન્ડસ્ટ્રીઝ"]),
    "yunivarsala": ("universal", "High: Gujarati transliteration of 'Universal' (યુનિવર્સલ)", ["યુનિવર્સલ એન્ટરપ્રાઇઝિસ"]),
    "suprima": ("supreme", "High: Devanagari transliteration of 'Supreme' (सुप्रीम)", ["सुप्रीम फूड्स"]),
    "primiyara": ("premier", "High: Devanagari transliteration of 'Premier' (प्रीमियर)", ["प्रीमियर एक्सपोर्ट्स"]),
    "gaileksi": ("galaxy", "High: Devanagari transliteration of 'Galaxy' (गैलेक्सी)", ["गुड गैलेक्सी कंस्ट्रक्शन"]),
    "motarsa": ("motors", "High: Devanagari transliteration of 'Motors' (मोटर्स)", ["पायोनियर कंसल्टेंसी मोटर्स"]),
    "pharmesi": ("pharmacy", "High: Devanagari/Odia transliteration of 'Pharmacy' (फार्मेसी / ଫାର୍ମାସି)", ["रेड पावर फार्मेसी"]),
    "pharnicara": ("furniture", "High: Devanagari transliteration of 'Furniture' (फर्नीचर)", ["सुप्रीम फूड्स फर्नीचर"]),
    "provijana": ("provision", "High: Devanagari transliteration of 'Provision' (प्रोविजन)", ["कृष्णा कंसल्टिंग प्रोविजन"]),
}


def generate_report():
    print(f"Reading dataset from {DATA_DIR}...")
    dataset = pq.read_table(DATA_DIR)
    total_records = dataset.num_rows

    scripts = dataset["business_name_script"].to_pylist()
    tokens_col = dataset["business_name_tokens"].to_pylist()
    raw_names = dataset["business_name"].to_pylist()
    translit_names = dataset["business_name_transliterated"].to_pylist()
    norm_names = dataset["business_name_normalized"].to_pylist()

    token_freq = collections.Counter()
    token_examples = collections.defaultdict(list)
    script_counts = collections.Counter()
    suffix_counter = collections.Counter()
    suffix_examples = collections.defaultdict(list)
    non_latin_count = 0

    for idx, script in enumerate(scripts):
        if script and script not in ("Latin", "Unknown"):
            non_latin_count += 1
            script_counts[script] += 1
            toks = tokens_col[idx] or []
            r = raw_names[idx]
            t = translit_names[idx]
            n = norm_names[idx]

            # Detect suffix
            if toks:
                if len(toks) >= 2 and f"{toks[-2]} {toks[-1]}" in ("pvt ltd", "public limited"):
                    suf = f"{toks[-2]} {toks[-1]}"
                else:
                    suf = toks[-1]
                suffix_counter[suf] += 1
                if len(suffix_examples[suf]) < 3:
                    suffix_examples[suf].append((r, t, n))

            for tok in toks:
                token_freq[tok] += 1
                if len(token_examples[tok]) < 3:
                    token_examples[tok].append((r, t, n))

    top_200 = token_freq.most_common(200)

    # Format output markdown
    lines = []
    lines.append("# Transliterated Vocabulary Audit Report")
    lines.append(f"**Dataset**: 100,000 Processed Source 2 Records (`data/processed/train`)\n")
    lines.append(f"- **Total Records**: {total_records:,}")
    lines.append(f"- **Non-Latin Business Names**: {non_latin_count:,} ({non_latin_count / total_records * 100:.2f}%)")
    lines.append(f"- **Total Unique Transliterated Tokens**: {len(token_freq):,}\n")

    lines.append("## 1. Script Distribution in Non-Latin Business Names\n")
    lines.append("| Script | Record Count | Percentage |")
    lines.append("|---|---|---|")
    for s, c in script_counts.most_common():
        lines.append(f"| {s} | {c:,} | {c / non_latin_count * 100:.1f}% |")
    lines.append("")

    lines.append("## 2. Top Transliterated Legal Suffix Variants\n")
    lines.append("Legal and corporate suffixes undergo phonetic transliteration depending on the source Indic script:")
    lines.append("")
    lines.append("| Rank | Ending Token(s) | Frequency | % of Non-Latin | Canonical Target | Primary Script(s) & Sample Original Name |")
    lines.append("|---|---|---|---|---|---|")
    for rank, (suf, count) in enumerate(suffix_counter.most_common(15), 1):
        sample = suffix_examples[suf][0] if suffix_examples[suf] else ("", "", "")
        canonical = "pvt ltd" if "pvt" in suf or "praiv" in suf or "limidhedh" in suf or "limirrad" in suf or "limatida" in suf else ("ltd" if "ltd" in suf or "limi" in suf else ("llp" if "llp" in suf or "pi" in suf or "bhi" in suf else "Other"))
        lines.append(f"| {rank} | `{suf}` | {count:,} | {count / non_latin_count * 100:.2f}% | `{canonical}` | `{sample[0]}` |")
    lines.append("")

    lines.append("## 3. High-Frequency Transliterated English/Business Words\n")
    lines.append("The table below highlights high-frequency transliterated tokens that represent common English business, corporate, industry, and organizational terms written in Indic scripts:\n")
    lines.append("| Token | Frequency | Possible Canonical English Form | Confidence / Linguistic Reason | Sample Original Business Name(s) |")
    lines.append("|---|---|---|---|---|")

    # Match tokens from top 200 against business keywords
    matched_count = 0
    for tok, count in top_200:
        if tok in BUSINESS_KEYWORDS:
            canonical, reason, _ = BUSINESS_KEYWORDS[tok]
            examples = "; ".join([f"`{r}`" for r, _, _ in token_examples[tok][:2]])
            lines.append(f"| `{tok}` | {count:,} | **{canonical}** | {reason} | {examples} |")
            matched_count += 1

    lines.append(f"\n*Total English/business keyword candidates identified in top 200: {matched_count} tokens.*\n")

    lines.append("## 4. Complete Top 200 Transliterated Tokens by Frequency\n")
    lines.append("| Rank | Token | Frequency | Sample Original Business Names |")
    lines.append("|---|---|---|---|")
    for rank, (tok, count) in enumerate(top_200, 1):
        examples = "; ".join([f"`{r}`" for r, _, _ in token_examples[tok][:2]])
        lines.append(f"| {rank} | `{tok}` | {count:,} | {examples} |")
    lines.append("")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report generated successfully: {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
