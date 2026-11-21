"""Prompt loading and assembly.

The prompt text lives in ``prompts/*.txt`` so it can be reviewed, versioned and
scanned independently of the Python code.
"""

from __future__ import annotations

from pathlib import Path

from .config import PROMPT_DIR
from .rag import Chunk

SYSTEM_PROMPT_FILE = "system_prompt.txt"
RAG_PROMPT_FILE = "rag_answer_prompt.txt"
ESCALATION_PROMPT_FILE = "escalation_summary_prompt.txt"


def load_prompt(name: str, prompt_dir: Path | None = None) -> str:
    directory = prompt_dir or PROMPT_DIR
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def format_context(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no matching policy sections)"
    return "\n\n".join(
        f"[{i + 1}] {chunk.title} ({chunk.doc_id})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )


def build_rag_prompt(question: str, chunks: list[Chunk]) -> str:
    template = load_prompt(RAG_PROMPT_FILE)
    return template.format(context=format_context(chunks), question=question)


def build_escalation_prompt(question: str, draft_answer: str) -> str:
    template = load_prompt(ESCALATION_PROMPT_FILE)
    return template.format(question=question, draft_answer=draft_answer)
