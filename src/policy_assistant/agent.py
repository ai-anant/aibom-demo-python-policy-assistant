"""The policy assistant agent: retrieve, answer, escalate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import guardrails
from .config import KB_DIR, Settings
from .hf_models import HuggingFaceModels
from .prompts import (
    SYSTEM_PROMPT_FILE,
    build_escalation_prompt,
    build_rag_prompt,
    load_prompt,
)
from .providers import ChatModel, build_chat_model
from .rag import Chunk, KnowledgeBase

logger = logging.getLogger(__name__)

# Tool schemas advertised to tool-calling models. They map onto the local
# functions in TOOL_IMPLEMENTATIONS below.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_kb",
            "description": "Search the HR and security policy knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_escalation",
            "description": "Open a ticket with the human policy review queue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["summary"],
            },
        },
    },
]


@dataclass
class Answer:
    answer: str
    citations: list[str] = field(default_factory=list)
    blocked: bool = False
    escalated: bool = False
    provider: str = ""
    model: str = ""
    topic: str = ""


class PolicyAgent:
    """Answers employee policy questions grounded in local documents."""

    def __init__(
        self,
        settings: Settings | None = None,
        chat_model: ChatModel | None = None,
        kb_dir: Path | None = None,
        models: HuggingFaceModels | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        if models is None and self.settings.use_hf_local:
            models = HuggingFaceModels(self.settings)
        self.models = models
        self.kb = KnowledgeBase(self.settings, kb_dir or KB_DIR, models=models)
        self.chat = chat_model or build_chat_model(self.settings)
        self.system_prompt = load_prompt(SYSTEM_PROMPT_FILE)
        self._escalations: list[dict[str, str]] = []

    # -- public API ---------------------------------------------------------
    def ask(self, question: str) -> Answer:
        screen = guardrails.screen_input(question)
        if not screen.allowed:
            return Answer(
                answer=f"I can't process that request: {screen.reason}.",
                blocked=True,
                provider=self.settings.provider,
                model=self.chat.name,
            )

        chunks = self.kb.retrieve(question)
        topic = self._classify_topic(question)
        prompt = build_rag_prompt(question, chunks)
        try:
            draft = self.chat.invoke(self.system_prompt, prompt)
        except Exception as exc:  # provider/network failure -> human review
            logger.warning("chat invocation failed: %s", exc)
            return self._escalate(question, "The assistant is temporarily unavailable.")

        cleaned = guardrails.screen_output(draft, self.settings.redact_pii)
        answer = cleaned.text.strip()

        if self._is_toxic(answer):
            logger.warning("HF toxicity classifier blocked a draft answer")
            return self._escalate(question, answer, chunks)

        if self._needs_escalation(answer):
            return self._escalate(question, answer, chunks)

        return Answer(
            answer=answer,
            citations=[f"{c.title} ({c.doc_id})" for c in chunks],
            provider=self.settings.provider,
            model=self.chat.name,
            topic=topic,
        )

    @property
    def escalations(self) -> list[dict[str, str]]:
        return list(self._escalations)

    # -- internals ----------------------------------------------------------
    def _classify_topic(self, question: str) -> str:
        if self.models is None:
            return ""
        try:
            label, _score = self.models.topic(question)
        except Exception as exc:  # model unavailable -> topic stays empty
            logger.warning("HF topic classifier failed: %s", exc)
            return ""
        return label

    def _is_toxic(self, answer: str) -> bool:
        if self.models is None:
            return False
        try:
            score = self.models.toxicity(answer)
        except Exception as exc:
            logger.warning("HF toxicity classifier failed: %s", exc)
            return False
        return score >= self.settings.toxicity_threshold

    @staticmethod
    def _needs_escalation(answer: str) -> bool:
        lowered = answer.lower()
        return any(
            phrase in lowered
            for phrase in ("i don't know", "cannot determine", "escalate", "not covered")
        )

    def _escalate(
        self, question: str, draft: str, chunks: list[Chunk] | None = None
    ) -> Answer:
        summary_prompt = build_escalation_prompt(question, draft)
        try:
            summary = self.chat.invoke(self.system_prompt, summary_prompt)
        except Exception:  # never lose the escalation because summarising failed
            summary = draft
        ticket = {
            "summary": summary[:1000],
            "question": question,
            "severity": "medium",
        }
        self._escalations.append(ticket)
        return Answer(
            answer=(
                "I've escalated this to the human policy review queue. "
                f"Reference: ESC-{len(self._escalations):04d}."
            ),
            citations=[f"{c.title} ({c.doc_id})" for c in (chunks or [])],
            escalated=True,
            provider=self.settings.provider,
            model=self.chat.name,
        )

    def search_policy_kb(self, query: str) -> list[str]:
        """Tool implementation exposed to tool-calling models."""
        return [chunk.text for chunk in self.kb.retrieve(query)]

    def create_escalation(self, summary: str, severity: str = "medium") -> str:
        """Tool implementation exposed to tool-calling models."""
        self._escalations.append({"summary": summary, "severity": severity})
        return f"ESC-{len(self._escalations):04d}"
