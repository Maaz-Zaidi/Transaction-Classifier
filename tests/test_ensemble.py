"""tests for the ensemble flow."""

import json

import pytest

from transaction_classifier.knowledge.merchant_kb import MerchantKnowledgeBase
from transaction_classifier.models.ensemble import Ensemble, ClassificationResult, _resolve_descriptor_override
from transaction_classifier.models.sgd_model import SGDModel
from transaction_classifier.rules.engine import RulesEngine


class DummyDenseEmbedder:
    is_available = False


class DummyReranker:
    is_available = True

    def __init__(self, scores):
        self._scores = list(scores)

    def score(self, query, documents):
        return self._scores[: len(documents)]


@pytest.fixture
def ensemble():
    rules_engine = RulesEngine()
    # train a tiny sgd model for the test
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


def test_structural_pattern_uses_rules(ensemble):
    results = ensemble.classify_batch(["PARKING LOT DOWNTOWN"])
    assert results[0].source == "rules"
    assert results[0].category == "Transportation"
    assert results[0].confidence == pytest.approx(0.98)
    assert results[0].flagged_for_review is False


def test_unknown_merchant_uses_sgd(ensemble):
    results = ensemble.classify_batch(["TOTALLY RANDOM UNKNOWN THING"])
    assert results[0].source == "sgd"


def test_classify_single(ensemble):
    result = ensemble.classify_single("NSF FEE")
    assert isinstance(result, ClassificationResult)
    assert result.category is not None


def test_results_have_all_fields(ensemble):
    result = ensemble.classify_single("DENTAL CLINIC")
    assert result.transaction == "DENTAL CLINIC"
    assert result.cleaned  # should not be empty
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

    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([0.96]),
    )
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
    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([0.82]),
    )
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


def test_descriptor_override_changes_kb_category(tmp_path):
    """When a descriptor unambiguously maps to a different category, override the KB."""
    kb_path = tmp_path / "merchant_kb.json"
    with open(kb_path, "w") as f:
        json.dump(
            [
                {
                    "canonical_name": "AMAZON",
                    "aliases": ["AMAZON"],
                    "mapped_category": "Shopping & Retail",
                    "mapping_confidence": 0.95,
                    "metadata_text": "online retailer",
                    "source": "curated_public",
                }
            ],
            f,
        )

    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([0.96]),
    )
    ensemble = Ensemble(
        rules_engine=RulesEngine(),
        knowledge_base=kb,
        sgd_model=SGDModel(),
    )

    # "AMAZON DOWNLOADS" has descriptor "digital content downloads" -> Entertainment
    result = ensemble.classify_single("AMAZON DOWNLOADS")
    assert result.source == "knowledge_base"
    assert result.category == "Entertainment & Recreation"


def test_descriptor_no_override_when_consistent(tmp_path):
    """When the descriptor matches the KB category, no override happens."""
    kb_path = tmp_path / "merchant_kb.json"
    with open(kb_path, "w") as f:
        json.dump(
            [
                {
                    "canonical_name": "SHELL",
                    "aliases": ["SHELL"],
                    "mapped_category": "Transportation",
                    "mapping_confidence": 0.95,
                    "metadata_text": "gas station and fuel",
                    "source": "curated_public",
                }
            ],
            f,
        )

    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([0.96]),
    )
    ensemble = Ensemble(
        rules_engine=RulesEngine(),
        knowledge_base=kb,
        sgd_model=SGDModel(),
    )

    result = ensemble.classify_single("SHELL GAS")
    assert result.source == "knowledge_base"
    assert result.category == "Transportation"


def test_dense_lexical_quality_gate(tmp_path):
    """Dense/lexical matches below 0.80 should not enrich the ML model input."""
    kb_path = tmp_path / "merchant_kb.json"
    with open(kb_path, "w") as f:
        json.dump(
            [
                {
                    "canonical_name": "SOME MERCHANT",
                    "aliases": ["SOME MERCHANT"],
                    "mapped_category": None,
                    "mapping_confidence": 0.0,
                    "metadata_text": "random metadata that should not appear",
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
            return [("Shopping & Retail", 0.87) for _ in texts]

    model = RecordingModel()
    kb = MerchantKnowledgeBase(
        kb_path,
        dense_embedder=DummyDenseEmbedder(),
        reranker=DummyReranker([]),  # reranker returns nothing
    )
    # the only match will be dense_lexical with similarity around 0.7
    ensemble = Ensemble(
        rules_engine=RulesEngine(),
        finetune_model=model,
        knowledge_base=kb,
        primary_model_name="finetune",
    )

    # for this test, the KB won't match "TOTALLY DIFFERENT QUERY" at all via exact,
    # so the match will be None and the model gets raw text
    result = ensemble.classify_single("TOTALLY DIFFERENT QUERY")
    assert result.source == "finetune"
    # the model should NOT see "random metadata" since there's no match
    assert model.last_texts is not None
    assert "random metadata" not in model.last_texts[0]


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


@pytest.mark.parametrize(
    "descriptor, kb_cat, expected",
    [
        ("gas station, fuel", "Shopping & Retail", "Transportation"),
        ("digital content downloads", "Shopping & Retail", "Entertainment & Recreation"),
        ("pharmacy drugstore", "Shopping & Retail", "Healthcare & Medical"),
        ("cloud computing, technology services", "Shopping & Retail", "Utilities & Services"),
        ("grocery food retail", "Shopping & Retail", "Food & Dining"),
        ("gas station, fuel", "Transportation", None),  # consistent, no override
        ("wholesale warehouse shopping", "Shopping & Retail", None),  # not in overrides
    ],
)
def test_resolve_descriptor_override(descriptor, kb_cat, expected):
    assert _resolve_descriptor_override(descriptor, kb_cat) == expected
