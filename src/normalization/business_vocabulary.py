"""
Indic-English Business Vocabulary Normalization Layer.

Provides high-confidence vocabulary normalization:
1. LEGAL_SUFFIX: Boundary-aware end-of-name corporate structure normalization.
2. BUSINESS_TERM: Exact token-level mapping of transliterated Indic commercial terms.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# =============================================================================
# 1. LEGAL SUFFIX RULES (Boundary-aware, end-of-name only)
# =============================================================================
# Must match ONLY at the end of the normalized string ($).
# Compound/specific variants are evaluated before single tokens.
LEGAL_SUFFIX_RULES: List[Tuple[re.Pattern, str]] = [
    # Compound Private Limited variants (Indic transliterated forms and English)
    (
        re.compile(
            r"\b(?:"
            r"private\s+limited|"
            r"pvt\s+ltd|"
            r"pvt\s+limited|"
            r"private\s+ltd|"
            r"pvt|"
            r"praiveta\s+limiteda|"
            r"pra\s+li|"
            r"bhiraivedh\s+(?:limidhedh|ltd|limited)|"
            r"praivarr\s+(?:limirrad|ltd|limited)|"
            r"praivet\s+(?:ltd|limited|limiteda)|"
            r"praibheta\s+(?:ltd|limited|limiteda)|"
            r"praibhet\s+(?:ltd|limited|limiteda)|"
            r"praiveta\s+(?:limatida|ltd|limited|limiteda)"
            r")\b\.?$",
            re.IGNORECASE,
        ),
        "pvt ltd",
    ),
    # Public limited / plc
    (
        re.compile(r"\b(?:public\s+limited|plc)\b\.?$", re.IGNORECASE),
        "plc",
    ),
    # Limited variants (English and Indic transliterated forms)
    (
        re.compile(
            r"\b(?:limited|ltd|limiteda|limidhedh|limirrad|limatida)\b\.?$",
            re.IGNORECASE,
        ),
        "ltd",
    ),
    # LLC
    (
        re.compile(r"\b(?:limited\s+liability\s+company|llc)\b\.?$", re.IGNORECASE),
        "llc",
    ),
    # LLP variants (English and Indic transliterated forms)
    (
        re.compile(
            r"\b(?:"
            r"limited\s+liability\s+partnership|"
            r"llp|"
            r"elaelapi|"
            r"elelbhi|"
            r"elelpi|"
            r"elel\s+pi|"
            r"el\s+pi"
            r")\b\.?$",
            re.IGNORECASE,
        ),
        "llp",
    ),
    # Inc / Incorporated
    (
        re.compile(r"\b(?:incorporated|inc)\b\.?$", re.IGNORECASE),
        "inc",
    ),
    # Corporation / Corp
    (
        re.compile(r"\b(?:corporation|corp)\b\.?$", re.IGNORECASE),
        "corp",
    ),
    # Company / Co
    (
        re.compile(r"\b(?:company|co)\b\.?$", re.IGNORECASE),
        "co",
    ),
]


def apply_legal_suffix(name: str) -> str:
    """
    Standardize the legal/corporate suffix at the END of a business name.
    Strictly boundary-aware: applies only if pattern occurs at the end of the name.
    Does NOT replace tokens in the beginning or middle of the name.
    """
    trimmed = re.sub(r"[,.\s]+$", "", name).strip()
    for pattern, canonical in LEGAL_SUFFIX_RULES:
        sub_name, count = pattern.subn(canonical, trimmed)
        if count > 0:
            return re.sub(r"\s+", " ", sub_name).strip()
    return trimmed


# =============================================================================
# 2. BUSINESS TERM MAPPINGS (Exact token matching only)
# =============================================================================
# High-confidence phonetic transliterations of English commercial/business words.
# Replaces exact individual tokens; never performs substring replacements.
BUSINESS_TERMS: Dict[str, str] = {
    # Core domain & industry terms
    "marketimga": "marketing",
    "praopartija": "properties",
    "emtarapraijeja": "enterprises",
    "emtar": "enterprise",
    "tredimga": "trading",
    "tredimg": "trading",
    "tredarsa": "traders",
    "imdastrija": "industries",
    "imdastris": "industries",
    "teknolaojija": "technologies",
    "teknolaoji": "technology",
    "teknalajis": "technologies",
    "teknalaji": "technology",
    "teka": "tech",
    "tek": "tech",
    "sarviseja": "services",
    "saolyusamsa": "solutions",
    "solyusans": "solutions",
    "laojistiksa": "logistics",
    "enarji": "energy",
    "aiti": "it",
    "globala": "global",
    "imtaranesanala": "international",
    "phudsa": "foods",
    "phuds": "foods",
    "phuda": "food",
    "phud": "food",
    "impeksa": "impex",
    "impeks": "impex",
    "imjiniyarimga": "engineering",
    "kamstraksamsa": "constructions",
    "kamstraksana": "construction",
    "straksans": "constructions",
    "prodaktsa": "products",
    "prodakts": "products",
    "devalaparsa": "developers",
    "esteta": "estate",
    "estet": "estate",
    "midiya": "media",
    "egro": "agro",
    "projektsa": "projects",
    "prajekts": "projects",
    "vemcarsa": "ventures",
    "eksaportsa": "exports",
    "prodyusara": "producer",
    "pavara": "power",
    "sistamsa": "systems",
    "bildarsa": "builders",
    "investamemtsa": "investments",
    "investamemta": "investment",
    "invest": "investments",
    "bijanesa": "business",
    "phaumdesana": "foundation",
    "phaumdesan": "foundation",
    "kamsaltemtsa": "consultants",
    "kamsaltimga": "consulting",
    "kamsaltemsi": "consultancy",
    "kansaltensi": "consultancy",
    "phainemsa": "finance",
    "phainans": "finance",
    "keyara": "care",
    "ker": "care",
    "imphra": "infra",
    "imphrastrakcara": "infrastructure",
    "phrastrakcar": "infrastructure",
    "saophtaveyara": "software",
    "sapht": "software",
    "mainejamemta": "management",
    "memt": "management",
    "imphoteka": "infotech",
    "helthakeyara": "healthcare",
    "haospitailiti": "hospitality",
    "haspitaliti": "hospitality",
    "hotala": "hotel",
    "dijitala": "digital",
    "haiteka": "hitech",
    "krietiva": "creative",
    "yunika": "unique",
    "yunivarsala": "universal",
    "suprima": "supreme",
    "primiyara": "premier",
    "gaileksi": "galaxy",
    "motarsa": "motors",
    "pharmesi": "pharmacy",
    "pharnicara": "furniture",
    "provijana": "provision",
}


def normalize_business_vocabulary(text: str) -> str:
    """
    Replace exact transliterated business tokens with their canonical English equivalents.
    Splits by whitespace, maps tokens present in BUSINESS_TERMS, and rejoins.
    Guarantees no substring replacement on non-matching or longer tokens.
    """
    if not text:
        return text

    words = text.split()
    normalized_words = [BUSINESS_TERMS.get(w, w) for w in words]
    return " ".join(normalized_words)
