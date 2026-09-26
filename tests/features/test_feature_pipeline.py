"""
Integration tests for Part 5: Feature Pipeline Integration.

Validates end-to-end integration of all four feature modules:
1. Business Name Features (Person 1, 12 features)
2. Business Address Features (Person 2, 10 features)
3. Cross-Field Features (Person 2, 1 feature)
4. Blocking Provenance Features (Person 2, 6 features)

Coverage:
- All feature groups appear in the final output (29 features + 3 identity columns = 32 columns)
- Schema, deterministic column ordering, and Polars dtypes
- Composite pair identity preservation: (source1_entity_id, candidate_entity_id, candidate_source)
- Empty and null input handling with preservation of missing-value semantics
- Execution on a realistic diverse sample of candidate pairs
- Exact consistency between pipeline output and individual feature module functions
- Contract validation for combine_person1_and_person2_features
"""

from __future__ import annotations

import math
import pytest
import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    NAME_FEATURE_NAMES,
    NAME_FEATURE_SCHEMA,
    ADDRESS_FEATURE_NAMES,
    CROSS_FEATURE_NAMES,
    BLOCKING_FEATURE_NAMES,
    PERSON2_FEATURE_NAMES,
    PERSON2_FEATURE_SCHEMA,
    ALL_FEATURE_NAMES,
    FULL_PIPELINE_COLUMNS,
    FULL_FEATURE_SCHEMA,
    FEATURE_SCHEMA,
    AddressRecordRepresentation,
    BlockingRecordRepresentation,
)
from src.features.record_representation import (
    NameRepresentation,
    build_name_representation,
    build_address_representation,
)
from src.features.name_features import compute_name_features
from src.features.address_features import compute_address_pair_features
from src.features.cross_features import compute_cross_features
from src.features.blocking_features import compute_blocking_features
from src.features.feature_pipeline import (
    FeaturePipeline,
    EntityResolutionFeaturePipeline,
    Person2FeaturePipeline,
    combine_person1_and_person2_features,
)


@pytest.fixture
def realistic_sample_records():
    """Provides a realistic, heterogeneous set of S1 and candidate entity records."""
    s1_records = {
        "S1-101": {
            "entity_id": "S1-101",
            "source": "source1",
            "business_name": "Starbucks Corporation",
            "business_name_normalized": "starbucks corp",
            "business_name_tokens": ["starbucks", "corp"],
            "business_name_transliterated": "starbucks corp",
            "business_address": "2401 Utah Ave S, Seattle, WA 98134",
            "business_address_normalized": "2401 utah ave s seattle wa 98134",
            "business_address_tokens": ["2401", "utah", "ave", "s", "seattle", "wa", "98134"],
            "country_normalized": "united states",
        },
        "S1-102": {
            "entity_id": "S1-102",
            "source": "source1",
            "business_name": "Tata Consultancy Services Ltd",
            "business_name_normalized": "tata consultancy services ltd",
            "business_name_tokens": ["tata", "consultancy", "services", "ltd"],
            "business_name_transliterated": "tata consultancy services ltd",
            "business_address": "Nirlon Knowledge Park, Goregaon East, Mumbai 400063",
            "business_address_normalized": "nirlon knowledge park goregaon east mumbai 400063",
            "business_address_tokens": ["nirlon", "knowledge", "park", "goregaon", "east", "mumbai", "400063"],
            "country_normalized": "india",
        },
        "S1-103": {
            "entity_id": "S1-103",
            "source": "source1",
            "business_name": "Acme Tools LLC",
            "business_name_normalized": "acme tools llc",
            "business_name_tokens": ["acme", "tools", "llc"],
            "business_name_transliterated": "acme tools llc",
            "business_address": None,
            "business_address_normalized": None,
            "business_address_tokens": None,
            "country_normalized": "united states",
        },
        "S1-104": {
            "entity_id": "S1-104",
            "source": "source1",
            "business_name": "Apollo Pharmacy",
            "business_name_normalized": "apollo pharmacy",
            "business_name_tokens": ["apollo", "pharmacy"],
            "business_name_transliterated": "apollo pharmacy",
            "business_address": "120 MG Road, Bangalore 560001",
            "business_address_normalized": "120 mg road bangalore 560001",
            "business_address_tokens": ["120", "mg", "road", "bangalore", "560001"],
            "country_normalized": "india",
        },
    }

    cand_records = {
        # Candidate 1: Exact match with S1-101
        "S2-201": {
            "entity_id": "S2-201",
            "source": "source2",
            "business_name": "Starbucks Coffee Corp",
            "business_name_normalized": "starbucks coffee corp",
            "business_name_tokens": ["starbucks", "coffee", "corp"],
            "business_name_transliterated": "starbucks coffee corp",
            "business_address": "2401 Utah Avenue South, Seattle, WA 98134",
            "business_address_normalized": "2401 utah avenue south seattle wa 98134",
            "business_address_tokens": ["2401", "utah", "avenue", "south", "seattle", "wa", "98134"],
            "country_normalized": "united states",
        },
        # Candidate 2: Near match with S1-102
        "S2-202": {
            "entity_id": "S2-202",
            "source": "source2",
            "business_name": "Tata Consultancy Services",
            "business_name_normalized": "tata consultancy services",
            "business_name_tokens": ["tata", "consultancy", "services"],
            "business_name_transliterated": "tata consultancy services",
            "business_address": "Goregaon East, Mumbai 400063",
            "business_address_normalized": "goregaon east mumbai 400063",
            "business_address_tokens": ["goregaon", "east", "mumbai", "400063"],
            "country_normalized": "india",
        },
        # Candidate 3: S1-103 pair with address missing on S1
        "S3-303": {
            "entity_id": "S3-303",
            "source": "source3",
            "business_name": "Acme Tools",
            "business_name_normalized": "acme tools",
            "business_name_tokens": ["acme", "tools"],
            "business_name_transliterated": "acme tools",
            "business_address": "500 Industrial Pkwy",
            "business_address_normalized": "500 industrial pkwy",
            "business_address_tokens": ["500", "industrial", "pkwy"],
            "country_normalized": "united states",
        },
        # Candidate 4: Unrelated pair for S1-104
        "S3-304": {
            "entity_id": "S3-304",
            "source": "source3",
            "business_name": "Burger King",
            "business_name_normalized": "burger king",
            "business_name_tokens": ["burger", "king"],
            "business_name_transliterated": "burger king",
            "business_address": "777 Market St",
            "business_address_normalized": "777 market st",
            "business_address_tokens": ["777", "market", "st"],
            "country_normalized": "united states",
        },
    }

    pairs_df = pl.DataFrame({
        "source1_entity_id": ["S1-101", "S1-102", "S1-103", "S1-104"],
        "candidate_entity_id": ["S2-201", "S2-202", "S3-303", "S3-304"],
        "candidate_source": ["source2", "source2", "source3", "source3"],
    })

    provenance = {
        ("S1-101", "S2-201"): {"C", "D"},
        ("S1-102", "S2-202"): {"A", "C"},
        ("S1-103", "S3-303"): {"E"},
        ("S1-104", "S3-304"): set(),
    }

    return s1_records, cand_records, pairs_df, provenance


class TestFeaturePipelineSchemaAndOrdering:
    """Tests schema conformance, column ordering, and feature group coverage."""

    def test_all_feature_groups_present_in_pipeline_output(self, realistic_sample_records):
        s1_records, cand_records, pairs_df, provenance = realistic_sample_records
        pipeline = FeaturePipeline()

        out_df = pipeline.generate_features_from_records(
            candidate_pairs_df=pairs_df,
            s1_records=s1_records,
            cand_records=cand_records,
            pair_provenance=provenance,
        )

        assert out_df.height == 4
        # Verify 3 identity columns + 29 features = 32 columns
        assert len(out_df.columns) == 32
        assert len(ALL_FEATURE_NAMES) == 29

        # 1. Name features group (12)
        for col in NAME_FEATURE_NAMES:
            assert col in out_df.columns, f"Missing name feature: {col}"

        # 2. Address features group (10)
        for col in ADDRESS_FEATURE_NAMES:
            assert col in out_df.columns, f"Missing address feature: {col}"

        # 3. Cross-field features group (1)
        for col in CROSS_FEATURE_NAMES:
            assert col in out_df.columns, f"Missing cross feature: {col}"

        # 4. Blocking features group (6)
        for col in BLOCKING_FEATURE_NAMES:
            assert col in out_df.columns, f"Missing blocking feature: {col}"

    def test_strict_deterministic_column_ordering(self, realistic_sample_records):
        s1_records, cand_records, pairs_df, provenance = realistic_sample_records
        pipeline = FeaturePipeline()

        out_df = pipeline.generate_features_from_records(
            candidate_pairs_df=pairs_df,
            s1_records=s1_records,
            cand_records=cand_records,
            pair_provenance=provenance,
        )

        assert out_df.columns == FULL_PIPELINE_COLUMNS
        assert out_df.columns[:3] == PAIR_ID_COLUMNS

    def test_schema_dtypes_exact_match(self, realistic_sample_records):
        s1_records, cand_records, pairs_df, provenance = realistic_sample_records
        pipeline = FeaturePipeline()

        out_df = pipeline.generate_features_from_records(
            candidate_pairs_df=pairs_df,
            s1_records=s1_records,
            cand_records=cand_records,
            pair_provenance=provenance,
        )

        for col, expected_dtype in FULL_FEATURE_SCHEMA.items():
            actual_dtype = out_df[col].dtype
            assert actual_dtype == expected_dtype, f"Dtype mismatch on {col}: expected {expected_dtype}, got {actual_dtype}"


class TestCompositeIdentityPreservation:
    """Tests preservation of (source1_entity_id, candidate_entity_id, candidate_source)."""

    def test_composite_identity_unaltered(self, realistic_sample_records):
        s1_records, cand_records, pairs_df, provenance = realistic_sample_records
        pipeline = FeaturePipeline()

        out_df = pipeline.generate_features_from_records(
            candidate_pairs_df=pairs_df,
            s1_records=s1_records,
            cand_records=cand_records,
            pair_provenance=provenance,
        )

        # Check values and row ordering match input candidate pairs exactly
        assert out_df["source1_entity_id"].to_list() == pairs_df["source1_entity_id"].to_list()
        assert out_df["candidate_entity_id"].to_list() == pairs_df["candidate_entity_id"].to_list()
        assert out_df["candidate_source"].to_list() == pairs_df["candidate_source"].to_list()

    def test_missing_identity_column_raises_error(self):
        pipeline = FeaturePipeline()
        bad_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            # candidate_source missing
        })
        with pytest.raises(ValueError, match="required composite identity column"):
            pipeline.validate_candidate_pairs(bad_df)


class TestNullAndEmptyHandling:
    """Tests empty DataFrames, missing entities, and null field semantics."""

    def test_empty_candidate_pairs_dataframe(self):
        pipeline = FeaturePipeline()
        empty_pairs = pl.DataFrame({
            "source1_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_entity_id": pl.Series([], dtype=pl.Utf8),
            "candidate_source": pl.Series([], dtype=pl.Utf8),
        })

        out_df = pipeline.generate_features(
            candidate_pairs_df=empty_pairs,
            s1_name_reps={},
            cand_name_reps={},
            s1_address_reps={},
            cand_address_reps={},
        )

        assert out_df.height == 0
        assert out_df.columns == FULL_PIPELINE_COLUMNS
        for col, dtype in FULL_FEATURE_SCHEMA.items():
            assert out_df[col].dtype == dtype

    def test_missing_address_semantics_preserved(self, realistic_sample_records):
        """Pair S1-103 vs S3-303 has missing address on S1."""
        s1_records, cand_records, pairs_df, provenance = realistic_sample_records
        pipeline = FeaturePipeline()

        out_df = pipeline.generate_features_from_records(
            candidate_pairs_df=pairs_df,
            s1_records=s1_records,
            cand_records=cand_records,
            pair_provenance=provenance,
        )

        row_103 = out_df.filter(pl.col("source1_entity_id") == "S1-103").to_dicts()[0]
        # Missing categorical flags must be -1
        assert row_103["address_exact"] == -1
        assert row_103["address_number_overlap"] == -1
        assert row_103["postal_match"] == -1
        assert row_103["address_length_difference"] == -1
        # Missing indicators
        assert row_103["address_missing_s1"] == 1
        assert row_103["address_missing_candidate"] == 0
        # Text similarities
        assert row_103["address_token_jaccard"] == 0.0

    def test_unrelated_pair_and_cross_country_semantics(self, realistic_sample_records):
        """Pair S1-104 vs S3-304: India vs US, completely different businesses."""
        s1_records, cand_records, pairs_df, provenance = realistic_sample_records
        pipeline = FeaturePipeline()

        out_df = pipeline.generate_features_from_records(
            candidate_pairs_df=pairs_df,
            s1_records=s1_records,
            cand_records=cand_records,
            pair_provenance=provenance,
        )

        row_104 = out_df.filter(pl.col("source1_entity_id") == "S1-104").to_dicts()[0]
        # Cross-field
        assert row_104["country_match"] == 0
        # Blocking
        assert row_104["matched_key_count"] == 0
        assert row_104["matched_key_A"] == 0
        # Names
        assert row_104["name_exact_norm"] == 0
        assert row_104["name_token_jaccard"] == 0.0


class TestConsistencyWithIndividualModules:
    """Verifies pipeline output matches individual feature module functions identically."""

    def test_pipeline_values_match_individual_modules(self, realistic_sample_records):
        s1_records, cand_records, pairs_df, provenance = realistic_sample_records

        # Build individual representations
        s1_name_reps = {eid: build_name_representation(r.get("business_name_normalized"), r.get("business_name_tokens")) for eid, r in s1_records.items()}
        cand_name_reps = {eid: build_name_representation(r.get("business_name_normalized"), r.get("business_name_tokens")) for eid, r in cand_records.items()}
        s1_addr_reps = {eid: build_address_representation(r) for eid, r in s1_records.items()}
        cand_addr_reps = {eid: build_address_representation(r) for eid, r in cand_records.items()}

        pipeline = FeaturePipeline()
        pipeline_df = pipeline.generate_features(
            candidate_pairs_df=pairs_df,
            s1_name_reps=s1_name_reps,
            cand_name_reps=cand_name_reps,
            s1_address_reps=s1_addr_reps,
            cand_address_reps=cand_addr_reps,
            s1_translit={eid: r.get("business_name_transliterated") for eid, r in s1_records.items()},
            cand_translit={eid: r.get("business_name_transliterated") for eid, r in cand_records.items()},
            pair_provenance=provenance,
        )

        for row in pipeline_df.iter_rows(named=True):
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]

            s1_rec = s1_records[s1_id]
            cand_rec = cand_records[cand_id]

            # 1. Name features module
            ind_name = compute_name_features(
                s1_name_reps[s1_id],
                cand_name_reps[cand_id],
                translit_a=s1_rec.get("business_name_transliterated"),
                translit_b=cand_rec.get("business_name_transliterated"),
            )
            for k, expected in ind_name.items():
                actual = row[k]
                if isinstance(expected, float):
                    assert math.isclose(actual, expected, abs_tol=1e-5), f"Name mismatch on {k} for {s1_id}-{cand_id}"
                else:
                    assert actual == expected, f"Name mismatch on {k} for {s1_id}-{cand_id}"

            # 2. Address features module
            ind_addr = compute_address_pair_features(s1_addr_reps[s1_id], cand_addr_reps[cand_id])
            for k, expected in ind_addr.items():
                actual = row[k]
                if isinstance(expected, float):
                    assert math.isclose(actual, expected, abs_tol=1e-5), f"Address mismatch on {k} for {s1_id}-{cand_id}"
                else:
                    assert actual == expected, f"Address mismatch on {k} for {s1_id}-{cand_id}"

            # 3. Cross-field module
            ind_cross = compute_cross_features(s1_rec.get("country_normalized"), cand_rec.get("country_normalized"))
            assert row["country_match"] == ind_cross["country_match"]

            # 4. Blocking provenance module
            matched_keys = provenance.get((s1_id, cand_id), set())
            ind_block = compute_blocking_features(matched_keys)
            for k, expected in ind_block.items():
                assert row[k] == expected, f"Blocking mismatch on {k} for {s1_id}-{cand_id}"


class TestCombinePerson1AndPerson2Features:
    """Tests combine_person1_and_person2_features contract."""

    def test_combine_contract(self):
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "candidate_source": ["source2"],
        })

        p1_df = pairs_df.with_columns([
            pl.lit(1).cast(pl.Int8).alias("name_exact_norm"),
            pl.lit(0.85).cast(pl.Float32).alias("name_token_jaccard"),
        ])

        p2_df = pairs_df.with_columns([
            pl.lit(1).cast(pl.Int8).alias("address_exact"),
            pl.lit(1).cast(pl.Int8).alias("country_match"),
            pl.lit(1).cast(pl.Int8).alias("matched_key_A"),
        ])

        combined = combine_person1_and_person2_features(p1_df, p2_df)
        assert combined.height == 1
        assert "name_exact_norm" in combined.columns
        assert "address_exact" in combined.columns
        assert "country_match" in combined.columns
        assert "matched_key_A" in combined.columns
