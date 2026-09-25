"""Benchmark and validation script for offline script detection and transliteration."""

import os
import sys
import time

# Ensure UTF-8 output across Windows consoles
sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.transliteration.script_detector import detect_script, ScriptDetector
from src.transliteration.transliterator import IndicTransliterator, transliterate_text


BENCHMARK_EXAMPLES = [
    # 1. Devanagari
    ("Devanagari", "राम मार्केटिंग प्राइवेट लिमिटेड"),
    ("Devanagari (Address)", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi, महाराष्ट्र"),
    # 2. Bengali
    ("Bengali", "গোল্ড প্রডিউসার স্টোর্স লিমিটেড"),
    ("Bengali (Address)", "NO. 28 PSIXL-3RD FLOOR, NEWTOWN ROAD, KOLKATA, পশ্চিমবঙ্গ"),
    # 3. Gujarati
    ("Gujarati", "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ"),
    ("Gujarati (Address)", "PLOT 008, CHAITANYA IND AREA, KOTDA SANGHANI, ગુજરાત"),
    # 4. Gurmukhi
    ("Gurmukhi", "ਗੁਰੂ ਨਾਨਕ ਐਂਟਰਪ੍ਰਾਈਜਿਜ਼"),
    ("Gurmukhi (Address)", "ਦੁਕਾਨ ਨੰਬਰ 45, ਜੀ.ਟੀ. ਰੋਡ, ਲੁਧਿਆਣਾ, ਪੰਜਾਬ"),
    # 5. Kannada
    ("Kannada", "ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"),
    ("Kannada (Address)", "NO ##14, SHREE SHIDRAM NILAYA, HUBLI, DHARWAD, ಕರ್ನಾಟಕ"),
    # 6. Malayalam
    ("Malayalam", "ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്"),
    ("Malayalam (Address)", "9-557-4, VETTICKAL HOUSE, PALAKKAD, കേരളം"),
    # 7. Odia
    ("Odia", "ଶ୍ରୀ ରାମ ଟ୍ରେଡର୍ସ"),
    ("Odia (Address)", "ପ୍ଲଟ ନଂ ୧୨, ଚନ୍ଦ୍ରଶେଖରପୁର, ଭୁବନେଶ୍ୱର, ଓଡ଼ିଶା"),
    # 8. Tamil
    ("Tamil", "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்"),
    ("Tamil (Address)", "#44, KILATHOPE STREET, MADURAI, தமிழ்நாடு"),
    # 9. Telugu
    ("Telugu", "కృష్ణా ఇంపెక్స్ లిమిటెడ్"),
    ("Telugu (Address)", "DOOR NO.11/1/471, KRUPANANDA NAGAR, ANANTAPUR, ఆంధ్రప్రదేశ్"),
    # 10. Latin
    ("Latin", "Vision Partners Corp"),
    ("Latin (Address)", "1064 Newton Rd, Unit 11, Iowa City, IA, US"),
    # 11. Mixed-Script Text
    ("Mixed (Gujarati + Latin)", "Tech Food પ્રાઇવેટ લિમિટેડ"),
    ("Mixed (Tamil + Latin)", "அரிஹந்த் Foundation Private Limited"),
    ("Mixed (Devanagari + Latin)", "ब्लू Technologies Pvt Ltd, Plot No: 3829, New Delhi"),
    # 12. Edge Cases
    ("Unknown (Digits/Symbols)", "12345 / 67890 - #@!&"),
    ("Empty String", ""),
]


def run_benchmark():
    print("=" * 80)
    print("OFFLINE SCRIPT DETECTION & TRANSLITERATION BENCHMARK")
    print("=" * 80)

    detector = ScriptDetector()
    transliterator = IndicTransliterator()

    print("\n--- REPRESENTATIVE SAMPLE EVALUATION ---\n")
    for category, sample_input in BENCHMARK_EXAMPLES:
        detected = detector.detect(sample_input)
        transliterated = transliterator.transliterate(sample_input, script=detected)

        print(f"Category:            {category}")
        print(f"input:               {sample_input}")
        print(f"detected_script:     {detected}")
        print(f"transliterated_output: {transliterated}")
        print("-" * 60)

    # Throughput benchmark
    print("\n--- THROUGHPUT BENCHMARK (10,000 iterations over mixed batch) ---\n")
    batch = [text for _, text in BENCHMARK_EXAMPLES if text]
    num_iterations = 10000 // len(batch)
    total_evals = num_iterations * len(batch)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        for text in batch:
            sc = detector.detect(text)
            _ = transliterator.transliterate(text, script=sc)
    elapsed = time.perf_counter() - start_time

    rate = total_evals / elapsed
    print(f"Evaluated {total_evals:,} texts in {elapsed:.3f} seconds.")
    print(f"Throughput: {rate:,.0f} records/second.")
    print(f"Average latency: {(elapsed / total_evals) * 1000:.4f} ms/record.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
