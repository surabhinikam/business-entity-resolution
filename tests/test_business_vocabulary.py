"""
Unit tests for Indic-English business vocabulary normalization layer.
"""

import unittest
from src.normalization.business_vocabulary import (
    apply_legal_suffix,
    normalize_business_vocabulary,
    BUSINESS_TERMS,
)
from src.normalization.business_normalizer import (
    normalize_business_name,
    tokenize_business_name,
)


class TestBusinessVocabulary(unittest.TestCase):
    # -------------------------------------------------------------------------
    # 1. LEGAL SUFFIX PATTERNS & BOUNDARY BEHAVIOR
    # -------------------------------------------------------------------------
    def test_each_legal_suffix_pattern(self):
        cases = [
            # Tamil
            ("global business bhiraivedh limidhedh", "global business pvt ltd"),
            ("apex trading limidhedh", "apex trading ltd"),
            ("prime it elelbhi", "prime it llp"),
            # Malayalam
            ("shakti impex praivarr limirrad", "shakti impex pvt ltd"),
            ("life engineering limirrad", "life engineering ltd"),
            ("green solutions elelpi", "green solutions llp"),
            # Kannada / Telugu
            ("modern consultants praivet ltd", "modern consultants pvt ltd"),
            ("red je products elel pi", "red je products llp"),
            # Bengali
            ("dynamic industries praibheta ltd", "dynamic industries pvt ltd"),
            # Odia
            ("vision technologies praibhet ltd", "vision technologies pvt ltd"),
            ("red healthcare el pi", "red healthcare llp"),
            # Gurmukhi
            ("suraj developers praiveta limatida", "suraj developers pvt ltd"),
            ("sky east estate limatida", "sky east estate ltd"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(apply_legal_suffix(inp), expected)

    def test_legal_suffix_only_at_end_of_name(self):
        # Suffix patterns must NOT match if they occur in the middle of a business name
        self.assertEqual(
            apply_legal_suffix("bhiraivedh enterprise bank limited"),
            "bhiraivedh enterprise bank ltd",
        )
        self.assertEqual(
            apply_legal_suffix("praivet clinic healthcare ltd"),
            "praivet clinic healthcare ltd",
        )
        self.assertEqual(
            apply_legal_suffix("praibheta diagnostic center ltd"),
            "praibheta diagnostic center ltd",
        )

    def test_legal_looking_token_in_middle_must_not_be_changed(self):
        # Mid-string legal tokens are preserved and not prematurely canonicalized
        name = "praivet capital holdings"
        self.assertEqual(apply_legal_suffix(name), "praivet capital holdings")

        name2 = "elelbhi advisory services"
        self.assertEqual(apply_legal_suffix(name2), "elelbhi advisory services")

    # -------------------------------------------------------------------------
    # 2. BUSINESS TERM MAPPINGS & EXACT TOKEN MATCHING
    # -------------------------------------------------------------------------
    def test_representative_business_term_mappings(self):
        cases = [
            ("rama marketimga pvt ltd", "rama marketing pvt ltd"),
            ("aditya praopartija llp", "aditya properties llp"),
            ("balaji emtarapraijeja ltd", "balaji enterprises ltd"),
            ("prime tredimga pvt ltd", "prime trading pvt ltd"),
            ("white tredimg pvt ltd", "white trading pvt ltd"),
            ("indian imdastrija ltd", "indian industries ltd"),
            ("modern imdastris pvt ltd", "modern industries pvt ltd"),
            ("alpha teknolaojija pvt ltd", "alpha technologies pvt ltd"),
            ("future teknalajis general", "future technologies general"),
            ("royal teka pvt ltd", "royal tech pvt ltd"),
            ("bharat aiti services", "bharat it services"),
            ("vijay imtaranesanala pvt ltd", "vijay international pvt ltd"),
            ("sun sistamsa phudsa pvt ltd", "sun systems foods pvt ltd"),
            ("shakti urban prodaktsa pvt ltd", "shakti urban products pvt ltd"),
            ("laxmi devalaparsa pvt ltd", "laxmi developers pvt ltd"),
            ("bombay esteta pvt ltd", "bombay estate pvt ltd"),
            ("big egro ltd", "big agro ltd"),
            ("apex projektsa pvt ltd", "apex projects pvt ltd"),
            ("hotel vemcarsa ltd", "hotel ventures ltd"),
            ("best pavara hardware ltd", "best power hardware ltd"),
            ("blue vijay investamemtsa pvt ltd", "blue vijay investments pvt ltd"),
            ("jain bijanesa pvt ltd", "jain business pvt ltd"),
            ("jain phaumdesana pvt ltd", "jain foundation pvt ltd"),
            ("north kamsaltemtsa pvt ltd", "north consultants pvt ltd"),
            ("global kamsaltimga pvt ltd", "global consulting pvt ltd"),
            ("sky kamsaltemsi pvt ltd", "sky consultancy pvt ltd"),
            ("best phainemsa pvt ltd", "best finance pvt ltd"),
            ("om keyara pvt ltd", "om care pvt ltd"),
            ("maa imphra pvt ltd", "maa infra pvt ltd"),
            ("bright imphrastrakcara ltd", "bright infrastructure ltd"),
            ("arihant saophtaveyara pvt ltd", "arihant software pvt ltd"),
            ("indo mainejamemta pvt ltd", "indo management pvt ltd"),
            ("surya imphoteka ltd", "surya infotech ltd"),
            ("modern laojistiksa pvt ltd", "modern logistics pvt ltd"),
            ("seven haospitailiti pvt ltd", "seven hospitality pvt ltd"),
            ("seven hotala pvt ltd", "seven hotel pvt ltd"),
            ("best dijitala pvt ltd", "best digital pvt ltd"),
            ("hitech haiteka pvt ltd", "hitech hitech pvt ltd"),
            ("creative krietiva pvt ltd", "creative creative pvt ltd"),
            ("unique yunika pvt ltd", "unique unique pvt ltd"),
            ("supreme suprima pvt ltd", "supreme supreme pvt ltd"),
        ]
        for inp, expected in cases:
            with self.subTest(inp=inp):
                self.assertEqual(normalize_business_vocabulary(inp), expected)

    def test_exact_token_matching_no_substring_replacement(self):
        # Must NOT replace substrings in non-matching words
        # e.g., 'supermarketimga' should NOT become 'supermarketing'
        text = "supermarketimga superprodaktsa"
        self.assertEqual(normalize_business_vocabulary(text), text)

        # Token embedded in alphanumeric string should NOT be touched
        text2 = "tek123 aiti99"
        self.assertEqual(normalize_business_vocabulary(text2), text2)

    # -------------------------------------------------------------------------
    # 3. END-TO-END PIPELINE INTEGRATION
    # -------------------------------------------------------------------------
    def test_end_to_end_indic_name_normalization(self):
        # 1. Devanagari Marketing
        res1 = normalize_business_name(
            name="राम मार्केटिंग प्राइवेट लिमिटेड",
            transliterated="rāma mārkeṭiṃga prāiveṭa limiṭeḍa",
        )
        self.assertEqual(res1, "rama marketing pvt ltd")
        self.assertEqual(tokenize_business_name(res1), ["rama", "marketing", "pvt", "ltd"])

        # 2. Kannada Technologies + Private Limited
        res2 = normalize_business_name(
            name="ಫ್ಯೂಚರ್ ಟೆಕ್ನಾಲಜೀಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
            transliterated="phyūcar ṭeknālajīs praiveṭ limiṭeḍ",
        )
        self.assertEqual(res2, "phyucar technologies pvt ltd")

        # 3. Tamil Business + Private Limited
        res3 = normalize_business_name(
            name="குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்",
            transliterated="ghuḻobhal bhijhiṉas bhiraiveḍh limiḍhèḍh",
        )
        self.assertEqual(res3, "ghulobhal bhijhinas pvt ltd")

        # 4. Malayalam Software + Private Limited
        res4 = normalize_business_name(
            name="ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്",
            transliterated="śakti impeks praivaṟṟ limiṟṟaḍ",
        )
        self.assertEqual(res4, "sakti impex pvt ltd")

    def test_existing_latin_names_remain_unchanged(self):
        # Existing Latin company names must not be accidentally modified or distorted
        latin_cases = [
            ("Acme Corporation", "acme corp"),
            ("Johnson & Johnson, Inc.", "johnson & johnson inc"),
            ("A-One Solutions Pvt Ltd", "a-one solutions pvt ltd"),
            ("Global Services LLC", "global services llc"),
            ("Vodafone Public Limited", "vodafone plc"),
            ("Wipro Limited", "wipro ltd"),
        ]
        for original, expected in latin_cases:
            with self.subTest(original=original):
                self.assertEqual(normalize_business_name(original), expected)

    def test_null_and_empty_handling(self):
        self.assertIsNone(normalize_business_name(None))
        self.assertEqual(normalize_business_name(""), "")
        self.assertEqual(normalize_business_name("   "), "")
        self.assertEqual(tokenize_business_name(None), [])
        self.assertEqual(tokenize_business_name(""), [])


if __name__ == "__main__":
    unittest.main()
