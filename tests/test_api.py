from fastapi.testclient import TestClient

from policy_assistant.agent import PolicyAgent
from policy_assistant.api import create_app
from policy_assistant.config import Settings


def _client() -> TestClient:
    agent = PolicyAgent(settings=Settings(provider="stub"))
    return TestClient(create_app(agent))


def test_healthz() -> None:
    response = _client().get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_lists_models() -> None:
    body = _client().get("/").json()
    assert body["chat_model"] == "gpt-4o-mini"
    assert body["embedding_model"] == "text-embedding-3-small"
    assert body["documents"] == 3


def test_ask_endpoint() -> None:
    response = _client().post("/ask", json={"question": "When are expense reports due?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["blocked"] is False
    assert "30 days" in payload["answer"]


def test_ask_rejects_short_question() -> None:
    response = _client().post("/ask", json={"question": "hi"})
    assert response.status_code == 422
