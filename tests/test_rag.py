from policy_assistant.config import KB_DIR, Settings
from policy_assistant.rag import KnowledgeBase, tokenize


def test_tokenize_removes_stopwords() -> None:
    tokens = tokenize("What is the hotel limit in tier-1 cities?")
    assert "hotel" in tokens
    assert "the" not in tokens


def test_retrieval_prefers_matching_document() -> None:
    kb = KnowledgeBase(Settings(provider="stub"), KB_DIR)
    results = kb.retrieve("hotel limit tier-1 cities", k=2)
    assert results
    assert results[0].doc_id == "travel-and-expense"


def test_knowledge_base_loads_all_documents() -> None:
    kb = KnowledgeBase(Settings(provider="stub"), KB_DIR)
    assert len(kb.documents) == 3
    assert {doc.doc_id for doc in kb.documents} == {
        "travel-and-expense",
        "remote-work",
        "acceptable-use",
    }
