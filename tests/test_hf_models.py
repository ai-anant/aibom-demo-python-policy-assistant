"""Tests for the local Hugging Face model wrappers (no weights downloaded)."""

from __future__ import annotations

from policy_assistant.agent import PolicyAgent
from policy_assistant.config import KB_DIR, Settings
from policy_assistant.hf_models import HuggingFaceModels
from policy_assistant.rag import KnowledgeBase


class FakeEmbedder:
    def encode(self, texts, **_kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeReranker:
    """Ranks whatever mentions 'remote work' first, whatever the query."""

    def predict(self, pairs, **_kwargs):
        return [1.0 if "Remote Work" in text else 0.1 for _query, text in pairs]


class FakeClassifier:
    def __init__(self, result):
        self.result = result

    def __call__(self, text, **_kwargs):
        return self.result


def _models(toxicity: float = 0.0) -> HuggingFaceModels:
    return HuggingFaceModels(
        Settings(provider="stub", use_hf_local=True),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        topic_classifier=FakeClassifier(
            {"labels": ["travel and expenses"], "scores": [0.93]}
        ),
        toxicity_classifier=FakeClassifier(
            [{"label": "neutral", "score": 1 - toxicity}, {"label": "toxic", "score": toxicity}]
        ),
    )


def test_embed_returns_plain_float_vectors() -> None:
    vectors = _models().embed(["a", "b"])
    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_rerank_returns_sorted_indexes() -> None:
    ranked = _models().rerank("hotel limit", ["Travel policy", "Remote Work policy"])
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1]


def test_topic_returns_top_label_and_score() -> None:
    label, score = _models().topic("Can I expense a hotel?")
    assert label == "travel and expenses"
    assert score == 0.93


def test_toxicity_ignores_neutral_label() -> None:
    assert _models(toxicity=0.0).toxicity("hello") == 0.0
    assert _models(toxicity=0.91).toxicity("hello") == 0.91


def test_knowledge_base_uses_cross_encoder_rerank() -> None:
    kb = KnowledgeBase(
        Settings(provider="stub", use_hf_local=True), KB_DIR, models=_models()
    )
    results = kb.retrieve("manager approval for remote arrangements", k=2)
    assert results[0].doc_id == "remote-work"


def test_agent_tags_answer_with_topic() -> None:
    agent = PolicyAgent(
        settings=Settings(provider="stub", use_hf_local=True), models=_models()
    )
    answer = agent.ask("What is the hotel limit in tier-1 cities?")
    assert answer.topic == "travel and expenses"
    assert answer.escalated is False


def test_agent_escalates_when_hf_classifier_flags_toxicity() -> None:
    agent = PolicyAgent(
        settings=Settings(provider="stub", use_hf_local=True),
        models=_models(toxicity=0.99),
    )
    answer = agent.ask("What is the hotel limit in tier-1 cities?")
    assert answer.escalated is True
    assert "human policy review" in answer.answer
