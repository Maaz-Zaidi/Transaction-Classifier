"""Tests for the ensemble orchestrator."""

import json

import pytest

from transaction_classifier.knowledge.merchant_kb import MerchantKnowledgeBase
from transaction_classifier.models.ensemble import Ensemble, ClassificationResult
from transaction_classifier.models.sgd_model import SGDModel
from transaction_classifier.rules.engine import RulesEngine


@pytest.fixture
def ensemble():
    rules_engine = RulesEngine()
    # train a minimal sgd model
    sgd = SGDModel()
    texts = [
        "TIM HORTONS", "STARBUCKS", "PIZZA",
        "UBER", "GAS STATION", "TAXI",
        "AMAZON", "STORE", "MALL",
        "NETFLIX", "CINEMA", "GAME",
        "PHARMACY", "DENTAL", "DOCTOR",
        "ROGERS", "HYDRO", "BELL",
        "MORTGAGE", "TRANSFER", "LOAN",
        "PAYROLL", "SALARY", "DEPOSIT",
        "CRA", "TAX", "GOVERNMENT",
        "CHARITY", "DONATION", "RED CROSS",
    ]
    labels = [
        "Food & Dining"] * 3 + [
        "Transportation"] * 3 + [
        "Shopping & Retail"] * 3 + [
        "Entertainment & Recreation"] * 3 + [
        "Healthcare & Medical"] * 3 + [
        "Utilities & Services"] * 3 + [
        "Financial Services"] * 3 + [
        "Income"] * 3 + [
        "Government & Legal"] * 3 + [
        "Charity & Donations"] * 3
    sgd.train(texts, labels)
    return Ensemble(rules_engine=rules_engine, sgd_model=sgd)


def test_classify_batch_returns_correct_length(ensemble):
    transactions = [
        "POS PURCHASE - 1234 TIM HORTONS OTTAWA ON",
        "SOME UNKNOWN MERCHANT",
        "INTERAC E-TRANSFER TO JOHN",
    ]
    results = ensemble.classify_batch(transactions)
    assert len(results) == 3


def test_known_merchant_uses_rules(ensemble):
    results = ensemble.classify_batch(["TIM HORTONS"])
    assert results[0].source == "rules"
    assert results[0].category == "Food & Dining"
    assert results[0].confidence == pytest.approx(0.98)
    assert results[0].flagged_for_review is False


def test_unknown_merchant_uses_sgd(ensemble):
    results = ensemble.classify_batch(["TOTALLY RANDOM UNKNOWN THING"])
    assert results[0].source == "sgd"


def test_classify_single(ensemble):
    result = ensemble.classify_single("NETFLIX.COM")
    assert isinstance(result, ClassificationResult)
    assert result.category is not None


def test_results_have_all_fields(ensemble):
    result = ensemble.classify_single("STARBUCKS COFFEE")
    assert result.transaction == "STARBUCKS COFFEE"
    assert result.cleaned  # not empty
    assert result.category
    assert result.confidence > 0
    assert result.source in (
        "rules",
        "knowledge_base",
        "zeroshot",
        "canine",
        "finetune",
        "finetune_metadata",
        "setfit",
        "fasttext",
        "sgd",
    )
    assert isinstance(result.flagged_for_review, bool)


def test_high_confidence_knowledge_base_hit_short_circuits_model(tmp_path):
    kb_path = tmp_path / "merchant_kb.json"
    with open(kb_path, "w") as f:
        json.dump(
            [
                {
                    "canonical_name": "REAL FRUIT BUBBLE TEA",
                    "aliases": ["REAL FRUIT BUBBLE TEA"],
                    "mapped_category": "Food & Dining",
                    "mapping_confidence": 0.98,
                    "metadata_text": "bubble tea shop and cafe",
                    "source": "curated_public",
                }
            ],
            f,
        )

    kb = MerchantKnowledgeBase(kb_path)
    ensemble = Ensemble(
        rules_engine=RulesEngine(),
        knowledge_base=kb,
        sgd_model=SGDModel(),
    )

    result = ensemble.classify_single("REAL FRUIT BUBBLE TEA OTTAWA ON")
    assert result.source == "knowledge_base"
    assert result.category == "Food & Dining"


def test_metadata_enrichment_feeds_primary_model(tmp_path):
    kb_path = tmp_path / "merchant_kb.json"
    with open(kb_path, "w") as f:
        json.dump(
            [
                {
                    "canonical_name": "UNIVERSITY OF OTTAWA",
                    "aliases": ["HTSP- UNIV OTTAWA"],
                    "mapped_category": None,
                    "mapping_confidence": 0.0,
                    "metadata_text": "public university with tuition and student fees",
                    "source": "curated_public",
                }
            ],
            f,
        )

    class RecordingModel:
        def __init__(self):
            self.last_texts = None

        def predict(self, texts):
            self.last_texts = texts
            return [("Government & Legal", 0.87) for _ in texts]

    model = RecordingModel()
    kb = MerchantKnowledgeBase(kb_path)
    ensemble = Ensemble(
        rules_engine=RulesEngine(),
        finetune_model=model,
        knowledge_base=kb,
        primary_model_name="finetune",
    )

    result = ensemble.classify_single("HTSP- UNIV OTTAWA")
    assert result.source == "finetune_metadata"
    assert result.category == "Government & Legal"
    assert "tuition" in model.last_texts[0]


def test_low_confidence_primary_uses_zeroshot(monkeypatch):
    class LowConfidenceModel:
        def predict(self, texts):
            return [("Shopping & Retail", 0.42) for _ in texts]

    class DummyZeroShot:
        is_loaded = True

        def predict(self, texts):
            return [("Food & Dining", 0.93) for _ in texts]

    monkeypatch.setattr("transaction_classifier.models.ensemble.settings.use_zeroshot", True)

    ensemble = Ensemble(
        rules_engine=RulesEngine(),
        finetune_model=LowConfidenceModel(),
        zeroshot_model=DummyZeroShot(),
        primary_model_name="finetune",
    )

    result = ensemble.classify_single("TOTALLY UNKNOWN CAFE")
    assert result.source == "zeroshot"
    assert result.category == "Food & Dining"
    assert result.flagged_for_review is False


def test_low_confidence_primary_stays_flagged_when_zeroshot_is_weak(monkeypatch):
    class LowConfidenceModel:
        def predict(self, texts):
            return [("Shopping & Retail", 0.42) for _ in texts]

    class WeakZeroShot:
        is_loaded = True

        def predict(self, texts):
            return [("Food & Dining", 0.25) for _ in texts]

    monkeypatch.setattr("transaction_classifier.models.ensemble.settings.use_zeroshot", True)

    ensemble = Ensemble(
        rules_engine=RulesEngine(),
        finetune_model=LowConfidenceModel(),
        zeroshot_model=WeakZeroShot(),
        primary_model_name="finetune",
    )

    result = ensemble.classify_single("TOTALLY UNKNOWN CAFE")
    assert result.source == "finetune"
    assert result.category == "Shopping & Retail"
    assert result.flagged_for_review is True
