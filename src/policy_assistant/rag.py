"""Retrieval over the policy knowledge base.

Two retrieval backends are supported:

* ``memory``  - a dependency-free TF-IDF index, used for the offline demo.
* ``chroma``  - a persistent Chroma collection built from local or OpenAI
  embeddings (config: ``VECTOR_STORE=chroma``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from .config import Settings
from .hf_models import HuggingFaceModels

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "what",
    "when", "where", "which", "who", "will", "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of policy text."""

    doc_id: str
    title: str
    text: str
    score: float = 0.0


def load_documents(kb_dir: Path) -> list[Chunk]:
    """Read every Markdown file in ``kb_dir`` into a chunk."""

    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = path.stem.replace("-", " ").title()
        for line in raw.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        chunks.append(Chunk(doc_id=path.stem, title=title, text=raw.strip()))
    return chunks


class KeywordIndex:
    """Small in-memory TF-IDF index (no external services required)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._tokens = [tokenize(c.text) for c in chunks]
        self._doc_freq: Counter[str] = Counter()
        for tokens in self._tokens:
            self._doc_freq.update(set(tokens))
        self._total = max(len(chunks), 1)

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        return math.log((self._total + 1) / (df + 0.5)) + 1.0

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scored: list[Chunk] = []
        for chunk, tokens in zip(self._chunks, self._tokens):
            counts = Counter(tokens)
            length = max(sum(counts.values()), 1)
            score = sum(
                counts[t] / length * self._idf(t) for t in set(q_tokens) if t in counts
            )
            if score > 0:
                scored.append(
                    Chunk(
                        doc_id=chunk.doc_id,
                        title=chunk.title,
                        text=chunk.text,
                        score=round(score, 4),
                    )
                )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]


class KnowledgeBase:
    """Facade over the configured retrieval backend."""

    def __init__(
        self,
        settings: Settings,
        kb_dir: Path,
        models: HuggingFaceModels | None = None,
    ) -> None:
        self.settings = settings
        self.kb_dir = kb_dir
        self.models = models
        self._chunks = load_documents(kb_dir)
        self._index = KeywordIndex(self._chunks)
        self._collection = None

    @property
    def documents(self) -> list[Chunk]:
        return list(self._chunks)

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        limit = k or self.settings.top_k
        if self.settings.vector_store == "chroma":
            return self._retrieve_chroma(query, limit)

        if self.models is not None and self.settings.use_hf_local:
            candidates = self._index.search(query, max(limit * 3, limit))
            ranked = self.models.rerank(
                query, [chunk.text for chunk in candidates], limit
            )
            return [
                replace(candidates[index], score=round(score, 4))
                for index, score in ranked
            ]

        return self._index.search(query, limit)

    def _retrieve_chroma(self, query: str, k: int) -> list[Chunk]:
        """Query a Chroma collection built from the same source documents."""

        collection = self._get_chroma_collection()
        if self.models is not None:
            vector = self.models.embed([query])[0]
            result = collection.query(query_embeddings=[vector], n_results=k)
        else:
            result = collection.query(query_texts=[query], n_results=k)
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        out: list[Chunk] = []
        for text, meta in zip(documents, metadatas):
            out.append(
                Chunk(
                    doc_id=str(meta.get("doc_id", "unknown")),
                    title=str(meta.get("title", "Policy")),
                    text=text,
                )
            )
        return out

    def _get_chroma_collection(self):
        if self._collection is not None:
            return self._collection

        import chromadb

        client = chromadb.PersistentClient(path=".chroma")
        collection = client.get_or_create_collection(name=self.settings.collection)
        if collection.count() == 0:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_openai import OpenAIEmbeddings

            texts = [c.text for c in self._chunks]
            if self.models is not None:
                vectors = self.models.embed(texts)
            else:
                embeddings = (
                    OpenAIEmbeddings(model=self.settings.embedding_model)
                    if self.settings.provider == "openai"
                    else HuggingFaceEmbeddings(
                        model_name=self.settings.local_embedding_model
                    )
                )
                vectors = embeddings.embed_documents(texts)
            collection.add(
                ids=[c.doc_id for c in self._chunks],
                documents=texts,
                embeddings=vectors,
                metadatas=[{"doc_id": c.doc_id, "title": c.title} for c in self._chunks],
            )
        self._collection = collection
        return collection
