"""Chat model providers.

The application talks to every model through the small ``ChatModel`` protocol
below. Hosted providers are constructed lazily so the service can boot (and
the test-suite can run) without API keys or heavyweight ML dependencies.
"""

from __future__ import annotations

from typing import Protocol

from .config import Settings


class ChatModel(Protocol):
    """Minimal surface the assistant needs from an LLM."""

    name: str

    def invoke(self, system: str, user: str) -> str:  # pragma: no cover - protocol
        ...


class StubChatModel:
    """Deterministic offline model used for demos and tests.

    It answers by selecting the sentences from the retrieved context that
    overlap with the question, which exercises the full request path without
    network access.
    """

    name = "stub-extractive"

    def invoke(self, system: str, user: str) -> str:
        question = _extract_question(user)
        context = _extract_context(user)
        if not context:
            return "not covered by policy"

        sentences = _split_sentences(context)
        query_terms = set(_terms(question))
        scored = [
            (len(query_terms & set(_terms(sentence))), index)
            for index, sentence in enumerate(sentences)
        ]
        ranked = [(score, index) for score, index in scored if score > 0]
        if not ranked:
            return "not covered by policy"

        ranked.sort(reverse=True)
        picked = sorted(ranked[:2], key=lambda item: item[1])
        return " ".join(sentences[index] for _, index in picked)

    def summarize(self, text: str) -> str:
        return text.strip().split("\n")[0][:400]


def _terms(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9][a-z0-9\-']+", text.lower())


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [part.strip() for part in parts if len(part.strip()) > 20]


def _extract_question(user_prompt: str) -> str:
    marker = "Question:"
    if marker not in user_prompt:
        return user_prompt
    return user_prompt.split(marker, 1)[1].splitlines()[0].strip()


class OpenAIChatModel:
    """Wrapper around ``langchain_openai.ChatOpenAI``."""

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import ChatOpenAI

        self.name = settings.chat_model
        self._client = ChatOpenAI(
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    def invoke(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._client.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return str(response.content)


class AnthropicChatModel:
    """Wrapper around ``langchain_anthropic.ChatAnthropic``."""

    def __init__(self, settings: Settings) -> None:
        from langchain_anthropic import ChatAnthropic

        self.name = settings.chat_model
        self._client = ChatAnthropic(
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    def invoke(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._client.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return str(response.content)


def build_chat_model(settings: Settings) -> ChatModel:
    """Instantiate the configured provider."""

    provider = settings.provider.lower()
    if provider == "openai":
        return OpenAIChatModel(settings)
    if provider == "anthropic":
        return AnthropicChatModel(settings)
    return StubChatModel()


def _extract_context(user_prompt: str) -> str:
    marker = "Context:"
    if marker not in user_prompt:
        return ""
    context = user_prompt.split(marker, 1)[1]
    context = context.split("Question:", 1)[0]
    return context.strip()
