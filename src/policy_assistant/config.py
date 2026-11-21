"""Runtime configuration and the model catalog for the assistant.

Every model identifier the application can use is declared in this module so
that an AI bill of materials (AIBOM) scan has a single, resolvable place to
look. Values can be overridden through environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Model catalog -----------------------------------------------------------
# Chat / reasoning models
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_ESCALATION_MODEL = "gpt-4o"
ANTHROPIC_CHAT_MODEL = "claude-sonnet-4-5"

# Embedding models
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Local Hugging Face models ----------------------------------------------
# These run in-process through transformers / sentence-transformers.
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_TOPIC_MODEL = "facebook/bart-large-mnli"
HF_TOXICITY_MODEL = "unitary/toxic-bert"
HF_TOPIC_LABELS = (
    "travel and expenses",
    "remote work and equipment",
    "acceptable use and AI security",
    "benefits and leave",
)

# Hosted dataset used to bootstrap the offline evaluation set
EVAL_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_dir(relative: str, env_var: str) -> Path:
    """Locate a data directory for both source checkouts and installed wheels.

    Resolution order: explicit environment override, repository root (editable
    install), current working directory (container image), package directory.
    """

    override = os.getenv(env_var)
    candidates = [Path(override)] if override else []
    candidates += [
        REPO_ROOT / relative,
        Path.cwd() / relative,
        Path(__file__).resolve().parent / relative,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


PROMPT_DIR = _resolve_dir("prompts", "PROMPT_DIR")
KB_DIR = _resolve_dir("data/knowledge_base", "KB_DIR")


@dataclass(frozen=True)
class Settings:
    """Resolved settings for one process."""

    provider: str = "stub"
    chat_model: str = OPENAI_CHAT_MODEL
    escalation_model: str = OPENAI_ESCALATION_MODEL
    embedding_model: str = OPENAI_EMBEDDING_MODEL
    local_embedding_model: str = LOCAL_EMBEDDING_MODEL
    vector_store: str = "memory"
    collection: str = "policy-kb"
    temperature: float = 0.1
    max_tokens: int = 800
    top_k: int = 4
    redact_pii: bool = True
    # Local Hugging Face models
    use_hf_local: bool = False
    hf_device: str = "cpu"
    hf_embedding_model: str = HF_EMBEDDING_MODEL
    hf_reranker_model: str = HF_RERANKER_MODEL
    hf_topic_model: str = HF_TOPIC_MODEL
    hf_toxicity_model: str = HF_TOXICITY_MODEL
    rerank_top_n: int = 3
    toxicity_threshold: float = 0.75

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("AI_PROVIDER")
        if not provider:
            provider = "openai" if os.getenv("OPENAI_API_KEY") else "stub"

        return cls(
            provider=provider,
            chat_model=os.getenv("CHAT_MODEL", OPENAI_CHAT_MODEL),
            escalation_model=os.getenv("ESCALATION_MODEL", OPENAI_ESCALATION_MODEL),
            embedding_model=os.getenv("EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL),
            local_embedding_model=os.getenv(
                "LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL
            ),
            vector_store=os.getenv("VECTOR_STORE", "memory"),
            collection=os.getenv("KB_COLLECTION", "policy-kb"),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "800")),
            top_k=int(os.getenv("TOP_K", "4")),
            redact_pii=os.getenv("REDACT_PII", "true").lower() != "false",
            use_hf_local=os.getenv("HF_LOCAL_MODELS", "false").lower()
            in {"1", "true", "yes"},
            hf_device=os.getenv("HF_DEVICE", "cpu"),
            hf_embedding_model=os.getenv("HF_EMBEDDING_MODEL", HF_EMBEDDING_MODEL),
            hf_reranker_model=os.getenv("HF_RERANKER_MODEL", HF_RERANKER_MODEL),
            hf_topic_model=os.getenv("HF_TOPIC_MODEL", HF_TOPIC_MODEL),
            hf_toxicity_model=os.getenv("HF_TOXICITY_MODEL", HF_TOXICITY_MODEL),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "3")),
            toxicity_threshold=float(os.getenv("TOXICITY_THRESHOLD", "0.75")),
        )
