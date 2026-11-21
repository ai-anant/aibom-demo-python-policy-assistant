"""Local Hugging Face models used by the assistant.

Models are loaded lazily on first use and can be injected for tests, so the
service still starts (and the offline demo still runs) without downloading
weights or importing torch.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

from .config import HF_TOPIC_LABELS, Settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


class Reranker(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs: Any) -> Any: ...


class HuggingFaceModels:
    """Wrappers around the local Hugging Face models declared in config."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Any | None = None,
        reranker: Any | None = None,
        topic_classifier: Any | None = None,
        toxicity_classifier: Any | None = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._reranker = reranker
        self._topic_classifier = topic_classifier
        self._toxicity_classifier = toxicity_classifier

    # -- lazy loaders -------------------------------------------------------
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading HF embedding model %s", self.settings.hf_embedding_model)
            self._embedder = SentenceTransformer(
                self.settings.hf_embedding_model, device=self.settings.hf_device
            )
        return self._embedder

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading HF reranker %s", self.settings.hf_reranker_model)
            self._reranker = CrossEncoder(self.settings.hf_reranker_model)
        return self._reranker

    @property
    def topic_classifier(self) -> Any:
        if self._topic_classifier is None:
            from transformers import pipeline

            logger.info("loading HF topic model %s", self.settings.hf_topic_model)
            self._topic_classifier = pipeline(
                "zero-shot-classification",
                model=self.settings.hf_topic_model,
                device=-1,
            )
        return self._topic_classifier

    @property
    def toxicity_classifier(self) -> Any:
        if self._toxicity_classifier is None:
            from transformers import pipeline

            logger.info("loading HF toxicity model %s", self.settings.hf_toxicity_model)
            self._toxicity_classifier = pipeline(
                "text-classification",
                model=self.settings.hf_toxicity_model,
                top_k=None,
                device=-1,
            )
        return self._toxicity_classifier

    # -- operations ---------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.embedder.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(value) for value in vector] for vector in vectors]

    def rerank(
        self, query: str, texts: Sequence[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """Return ``(index, score)`` pairs ordered by cross-encoder relevance."""

        if not texts:
            return []
        scores = self.reranker.predict([(query, text) for text in texts])
        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[: (top_n or self.settings.rerank_top_n)]

    def topic(self, text: str, labels: Sequence[str] | None = None) -> tuple[str, float]:
        result = self.topic_classifier(
            text,
            candidate_labels=list(labels or HF_TOPIC_LABELS),
            multi_label=False,
        )
        return str(result["labels"][0]), float(result["scores"][0])

    def toxicity(self, text: str) -> float:
        scores = self.toxicity_classifier(text)
        if isinstance(scores, dict):
            scores = [scores]
        if scores and isinstance(scores[0], list):
            scores = scores[0]
        risky = [
            float(item.get("score", 0.0))
            for item in scores
            if str(item.get("label", "")).lower()
            not in {"non_toxic", "neutral", "safe", "not_toxic"}
        ]
        return max(risky) if risky else 0.0
