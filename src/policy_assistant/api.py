"""FastAPI surface for the policy assistant."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import PolicyAgent
from .config import Settings

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    citations: list[str] = []
    blocked: bool = False
    escalated: bool = False
    model: str = ""
    topic: str = ""


def create_app(agent: PolicyAgent | None = None) -> FastAPI:
    settings = Settings.from_env()
    app = FastAPI(
        title="Policy Assistant",
        version="0.3.0",
        description="Sample RAG application used for AIBOM demonstrations.",
    )
    assistant = agent or PolicyAgent(settings=settings)
    app.state.agent = assistant

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "provider": settings.provider,
            "model": assistant.chat.name,
        }

    @app.get("/")
    def index() -> dict[str, object]:
        return {
            "service": "policy-assistant",
            "provider": settings.provider,
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "huggingface_models": {
                "embedding": settings.hf_embedding_model,
                "reranker": settings.hf_reranker_model,
                "topic": settings.hf_topic_model,
                "toxicity": settings.hf_toxicity_model,
            },
            "hf_local_enabled": settings.use_hf_local,
            "documents": len(assistant.kb.documents),
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        result = assistant.ask(payload.question)
        return AskResponse(
            answer=result.answer,
            citations=result.citations,
            blocked=result.blocked,
            escalated=result.escalated,
            model=result.model,
            topic=result.topic,
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "policy_assistant.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )


if __name__ == "__main__":
    main()
