# Policy Assistant — Python AI sample for AIBOM demos

A small but realistic Retrieval-Augmented Generation (RAG) service that answers
employee policy questions from local documents. It is built as a **demo target
for AI Bill of Materials (AIBOM) generation**: it declares real model
identifiers, prompt templates, an embedding model, a vector store, a guardrail
layer, evaluation data and an agent/tool surface, so a scanner has plenty to
find.

- **Language / runtime:** Python 3.11+, FastAPI, LangChain
- **Models declared:** `gpt-4o-mini`, `gpt-4o`, `claude-sonnet-4-5`,
  `text-embedding-3-small`, `sentence-transformers/all-MiniLM-L6-v2`
- **Data:** local Markdown policy corpus + a hosted Hugging Face evaluation
  dataset (`bitext/Bitext-customer-support-llm-chatbot-training-data…`)
- **Runs offline:** the default `stub` provider needs no API key, so the demo
  works on a laptop or in CI.

## What the application does

```
employee question
   │
   ├── guardrails.screen_input        prompt-injection + acceptable-use screen
   ├── rag.KnowledgeBase.retrieve     TF-IDF (offline) or Chroma + embeddings
   ├── prompts.build_rag_prompt       prompts/*.txt templates
   ├── providers.<ChatModel>.invoke   OpenAI / Anthropic / stub
   └── guardrails.screen_output       PII redaction, then answer + citations
```

If the models cannot answer, the request is escalated to a human review queue
and summarised with the escalation prompt.

## Quick start (no API key)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the tests
pytest

# Start the API with the deterministic offline model
AI_PROVIDER=stub uvicorn policy_assistant.api:app --port 8080
```

```bash
curl -s localhost:8080/healthz
curl -s -X POST localhost:8080/ask \
  -H 'content-type: application/json' \
  -d '{"question":"What is the hotel limit in tier-1 cities?"}'
```

## Quick start (hosted models)

```bash
cp .env.example .env            # add OPENAI_API_KEY
export $(grep -v '^#' .env | xargs)
pip install -e .
uvicorn policy_assistant.api:app --port 8080
```

Switch providers with `AI_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. Select
Chroma retrieval with `VECTOR_STORE=chroma` (builds `.chroma/` on first use).

## Layout

```
src/policy_assistant/
  config.py      model catalog + settings (single place to resolve model IDs)
  providers.py   ChatModel protocol, OpenAI/Anthropic wrappers, offline stub
  rag.py         TF-IDF index, Chroma backend, document loader
  prompts.py     prompt loading/assembly
  guardrails.py  PII redaction, injection screening, OpenAI moderation call
  agent.py       orchestration, citations, escalation, tool implementations
  api.py         FastAPI routes
prompts/         system, RAG and escalation prompt templates (plain text)
data/knowledge_base/  Markdown policy corpus
data/eval/       evaluation questions (JSONL)
scripts/         evaluation-set builder using the hosted dataset
examples/        LangChain tool-calling agent variant
tests/           offline test-suite (no network, no keys)
docs/AIBOM_DEMO.md    how to produce an AIBOM from this repo
```

## Generating an AIBOM

See [docs/AIBOM_DEMO.md](docs/AIBOM_DEMO.md). In short:

```bash
uv tool install --python 3.13 "cisco-aibom[agentic,llm-openai]"
cisco-aibom analyze . --output-format json --output-file aibom-report.json \
  --llm-model gpt-4o-mini
```

Expected findings include models (`gpt-4o-mini`, `gpt-4o`, Claude, embedding
models), the Chroma vector store, the policy corpus and evaluation dataset,
three prompt templates, the moderation guardrail, the agent's tools
(`search_policy_kb`, `create_escalation`), and the declared AI dependencies
(langchain, openai, anthropic, transformers, chromadb, guardrails-ai, ragas,
langfuse).

## Notes on the demo design

* Every model name lives in `config.py` or in a prompt file, not scattered
  through the codebase — the AIBOM then resolves them consistently.
* Heavy dependencies (`torch`, `chromadb`, `transformers`) are imported lazily,
  so tests and the offline demo stay fast. They remain declared dependencies,
  which is what a dependency scanner reports.
* No secrets are committed; `.env` is ignored and `.env.example` holds
  placeholders only.

## License

MIT
