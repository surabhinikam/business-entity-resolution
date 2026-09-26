"""
Unit tests for Person 2 feature scaffolding and schema contracts.
"""

import pytest
import polars as pl

from src.features.feature_schema import (
    PAIR_ID_COLUMNS,
    ADDRESS_FEATURE_NAMES,
    CROSS_FEATURE_NAMES,
    BLOCKING_FEATURE_NAMES,
    PERSON2_FEATURE_NAMES,
    PERSON2_FEATURE_SCHEMA,
    FEATURE_METADATA,
    AddressRecordRepresentation,
    BlockingRecordRepresentation,
)
from src.features.address_features import (
    build_address_representation,
    compute_address_pair_features,
    extract_address_features_batch,
)
from src.features.cross_features import (
    compute_cross_features,
    extract_cross_features_batch,
)
from src.features.blocking_features import (
    compute_blocking_features,
    extract_blocking_features_from_provenance_dict,
    extract_blocking_features_from_keys,
)
from src.features.feature_pipeline import (
    Person2FeaturePipeline,
    combine_person1_and_person2_features,
)


class TestFeatureSchema:
    def test_composite_pair_identity(self):
        """Canonical composite identity must be strictly 3 columns."""
        assert PAIR_ID_COLUMNS == [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]

    def test_feature_counts(self):
        """Verify feature count breakdown."""
        assert len(ADDRESS_FEATURE_NAMES) == 10
        assert len(CROSS_FEATURE_NAMES) == 1
        assert len(BLOCKING_FEATURE_NAMES) == 6
        assert len(PERSON2_FEATURE_NAMES) == 17

    def test_schema_completeness(self):
        """Every feature must have an entry in PERSON2_FEATURE_SCHEMA and FEATURE_METADATA."""
        for feat in PERSON2_FEATURE_NAMES:
            assert feat in PERSON2_FEATURE_SCHEMA, f"Missing dtype for {feat}"
            assert feat in FEATURE_METADATA, f"Missing metadata for {feat}"

        for col in PAIR_ID_COLUMNS:
            assert col in PERSON2_FEATURE_SCHEMA


class TestAddressRepresentation:
    def test_build_address_representation_with_address(self):
        record = {
            "entity_id": "S1-12345",
            "source": "source1",
            "country_normalized": "united states",
            "business_address": "108 Norle Street, College Twp, PA",
            "business_address_normalized": "108 norle street college twp pa",
            "business_address_tokens": ["108", "norle", "street", "college", "twp", "pa"],
        }
        rep = build_address_representation(record)
        assert rep.entity_id == "S1-12345"
        assert rep.source == "source1"
        assert rep.country_normalized == "united states"
        assert rep.normalized_address == "108 norle street college twp pa"
        assert rep.address_tokens == ["108", "norle", "street", "college", "twp", "pa"]
        assert rep.primary_address_number == "108"
        assert not rep.is_address_missing

    def test_build_address_representation_missing_address(self):
        record = {
            "entity_id": "S2-99999",
            "source": "source2",
            "country_normalized": "india",
            "business_address": None,
            "business_address_normalized": None,
            "business_address_tokens": None,
        }
        rep = build_address_representation(record)
        assert rep.entity_id == "S2-99999"
        assert rep.primary_address_number is None
        assert rep.is_address_missing
        assert rep.address_tokens == []


class TestAddressFeatures:
    def test_missing_address_contract(self):
        s1 = AddressRecordRepresentation(
            entity_id="S1-1",
            source="source1",
            is_address_missing=False,
            normalized_address="108 norle street",
            address_tokens=["108", "norle", "street"],
        )
        cand_missing = AddressRecordRepresentation(
            entity_id="S2-2",
            source="source2",
            is_address_missing=True,
            normalized_address=None,
            address_tokens=[],
        )
        feat = compute_address_pair_features(s1, cand_missing)
        assert feat["address_exact"] == -1
        assert feat["address_token_jaccard"] == 0.0
        assert feat["address_missing_s1"] == 0
        assert feat["address_missing_candidate"] == 1
        assert feat["postal_match"] == -1

    def test_extract_address_features_batch(self):
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-2"],
            "candidate_source": ["source2"],
        })
        s1_rep = AddressRecordRepresentation(
            entity_id="S1-1", source="source1", normalized_address="108 main st"
        )
        cand_rep = AddressRecordRepresentation(
            entity_id="S2-2", source="source2", normalized_address="108 main st"
        )
        res = extract_address_features_batch(
            pairs_df, {"S1-1": s1_rep}, {"S2-2": cand_rep}
        )
        assert res.height == 1
        for col in ADDRESS_FEATURE_NAMES:
            assert col in res.columns


class TestCrossFeatures:
    def test_compute_cross_features(self):
        assert compute_cross_features("united states", "united states")["country_match"] == 1
        assert compute_cross_features("united states", "india")["country_match"] == 0
        assert compute_cross_features(None, "india")["country_match"] == -1
        assert compute_cross_features("united states", "")["country_match"] == -1

    def test_extract_cross_features_batch(self):
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-2"],
            "candidate_source": ["source2", "source3"],
        })
        s1_c = {"S1-1": "united states", "S1-2": "india"}
        c_c = {"S2-1": "united states", "S3-2": "united states"}
        res = extract_cross_features_batch(pairs_df, s1_c, c_c)
        assert res["country_match"].to_list() == [1, 0]


class TestBlockingFeatures:
    def test_compute_blocking_features(self):
        res = compute_blocking_features({"A", "C"})
        assert res["matched_key_A"] == 1
        assert res["matched_key_C"] == 1
        assert res["matched_key_D"] == 0
        assert res["matched_key_E"] == 0
        assert res["matched_key_F"] == 0
        assert res["matched_key_count"] == 2

    def test_extract_from_provenance_dict(self):
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-2"],
            "candidate_source": ["source2"],
        })
        prov = {("S1-1", "S2-2"): {"D", "F"}}
        res = extract_blocking_features_from_provenance_dict(pairs_df, prov)
        assert res["matched_key_D"][0] == 1
        assert res["matched_key_F"][0] == 1
        assert res["matched_key_count"][0] == 2

    def test_extract_from_keys(self):
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-2"],
            "candidate_source": ["source2"],
        })
        s1_keys = {"S1-1": BlockingRecordRepresentation(entity_id="S1-1", key_A="A||us||apple", key_C="C||us||ap inc")}
        cand_keys = {"S2-2": BlockingRecordRepresentation(entity_id="S2-2", key_A="A||us||apple", key_C="C||us||ap other")}
        res = extract_blocking_features_from_keys(pairs_df, s1_keys, cand_keys)
        assert res["matched_key_A"][0] == 1
        assert res["matched_key_C"][0] == 0
        assert res["matched_key_count"][0] == 1


class TestPerson2Pipeline:
    def test_full_pipeline_scaffolding(self):
        pairs_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-2"],
            "candidate_source": ["source2"],
        })
        s1_addr = {"S1-1": AddressRecordRepresentation(entity_id="S1-1", source="source1", country_normalized="india", normalized_address="mg road")}
        cand_addr = {"S2-2": AddressRecordRepresentation(entity_id="S2-2", source="source2", country_normalized="india", normalized_address="mg road")}
        prov = {("S1-1", "S2-2"): {"C", "D"}}

        pipeline = Person2FeaturePipeline()
        out_df = pipeline.generate_person2_features(
            candidate_pairs_df=pairs_df,
            s1_address_reps=s1_addr,
            cand_address_reps=cand_addr,
            pair_provenance=prov,
        )

        assert out_df.height == 1
        # Check all schema columns present
        for col, dtype in PERSON2_FEATURE_SCHEMA.items():
            assert col in out_df.columns
            assert out_df[col].dtype == dtype

    def test_combine_person1_and_person2(self):
        p1_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-2"],
            "candidate_source": ["source2"],
            "name_token_jaccard": [0.85],
        })
        p2_df = pl.DataFrame({
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-2"],
            "candidate_source": ["source2"],
            "address_token_jaccard": [0.72],
            "country_match": [1],
        })
        combined = combine_person1_and_person2_features(p1_df, p2_df)
        assert combined.height == 1
        assert "name_token_jaccard" in combined.columns
        assert "address_token_jaccard" in combined.columns
        assert "country_match" in combined.columns
