"""
Production record preprocessing pipeline for Business Entity Resolution.

Transforms raw canonical records through:
1. Unicode/script detection
2. Language inference
3. Transliteration (for Indic / non-Latin scripts)
4. Text & legal suffix normalization
5. Address structural normalization
6. Tokenization
7. Country normalization

Preserves original fields and safely handles nulls without producing string 'nan' / 'None'.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from src.preprocessing.schema import CanonicalRecord
from src.transliteration.script_detector import ScriptDetector
from src.transliteration.transliterator import IndicTransliterator
from src.language_detection.detector import LanguageDetector
from src.normalization.business_normalizer import (
    normalize_business_name,
    tokenize_business_name,
)
from src.normalization.address_normalizer import (
    normalize_address,
    tokenize_address,
)
from src.normalization.country_normalizer import normalize_country


class PreprocessingPipeline:
    """Production record transformer."""

    def __init__(
        self,
        script_detector: Optional[ScriptDetector] = None,
        transliterator: Optional[IndicTransliterator] = None,
        language_detector: Optional[LanguageDetector] = None,
    ):
        self.script_detector = script_detector or ScriptDetector()
        self.transliterator = transliterator or IndicTransliterator()
        self.language_detector = language_detector or LanguageDetector()

    def process_record(self, record: CanonicalRecord) -> Dict[str, Any]:
        """
        Process a single CanonicalRecord into the 16-column production schema.
        Handles missing fields cleanly with None (never "nan" or "None" strings).
        """
        source = record.source
        entity_id = record.entity_id

        # 1. Business Name processing
        raw_name = record.business_name
        if raw_name is not None and raw_name.strip():
            raw_name = raw_name.strip()
            name_script = self.script_detector.detect(raw_name)
            name_lang = self.language_detector.detect(raw_name, script=name_script)

            if name_script not in ("Latin", "Unknown"):
                name_translit = self.transliterator.transliterate(raw_name, script=name_script)
            else:
                name_translit = raw_name

            name_norm = normalize_business_name(raw_name, transliterated=name_translit)
            name_tokens = tokenize_business_name(name_norm)
        else:
            raw_name = None
            name_script = None
            name_lang = None
            name_translit = None
            name_norm = None
            name_tokens = None

        # 2. Business Address processing
        raw_addr = record.business_address
        if raw_addr is not None and raw_addr.strip():
            raw_addr = raw_addr.strip()
            addr_script = self.script_detector.detect(raw_addr)
            addr_lang = self.language_detector.detect(raw_addr, script=addr_script)

            if addr_script not in ("Latin", "Unknown"):
                addr_translit = self.transliterator.transliterate(raw_addr, script=addr_script)
            else:
                addr_translit = raw_addr

            addr_norm = normalize_address(raw_addr, transliterated=addr_translit)
            addr_tokens = tokenize_address(addr_norm)
        else:
            raw_addr = None
            addr_script = None
            addr_lang = None
            addr_translit = None
            addr_norm = None
            addr_tokens = None

        # 3. Country processing
        raw_country = record.country
        if raw_country is not None and raw_country.strip():
            raw_country = raw_country.strip()
            country_norm = normalize_country(raw_country)
        else:
            raw_country = None
            country_norm = None

        return {
            "source": source,
            "entity_id": entity_id,
            "business_name": raw_name,
            "business_name_script": name_script,
            "business_name_language": name_lang,
            "business_name_transliterated": name_translit,
            "business_name_normalized": name_norm,
            "business_name_tokens": name_tokens,
            "business_address": raw_addr,
            "business_address_script": addr_script,
            "business_address_language": addr_lang,
            "business_address_transliterated": addr_translit,
            "business_address_normalized": addr_norm,
            "business_address_tokens": addr_tokens,
            "country": raw_country,
            "country_normalized": country_norm,
        }
