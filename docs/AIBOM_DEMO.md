# Generating an AIBOM from this repository

This repository is a target for AI Bill of Materials tooling. It intentionally
contains a spread of AI assets: hosted and local models, an embedding model, a
vector store, prompt templates, a dataset, a guardrail and agent tools.

## 1. Install a scanner

Cisco AI Defense AIBOM is used here because it scans source code and resolves
model identifiers deterministically:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --python 3.13 "cisco-aibom[agentic,llm-openai]"
cisco-aibom --help
```

`analyze` always requires an LLM for its third-stage enrichment pass. Point it
at a model you have credentials for:

```bash
export OPENAI_API_KEY=sk-...
```

## 2. Scan

```bash
cd aibom-demo-python-policy-assistant
cisco-aibom analyze . \
  --output-format json \
  --output-file aibom-report.json \
  --llm-model gpt-4o-mini
```

Useful variants:

```bash
# Human-readable inventory
cisco-aibom analyze . -o plaintext

# CycloneDX ML-BOM style output for downstream tooling
cisco-aibom analyze . -o cyclonedx -O aibom.cdx.json --llm-model gpt-4o-mini

# HTML dashboard for a screen-share demo
cisco-aibom analyze . -o html -O aibom-report.html --llm-model gpt-4o-mini
```

Scans of a git working tree are attributed to the repository URL and commit SHA,
which is what makes the report usable as a supply-chain artifact.

## 3. What to expect in the report

| Asset type | Where it comes from |
|---|---|
| `model` | `config.py`, `providers.py`, `guardrails.py`, `examples/langchain_agent.py` |
| `model` (Hugging Face) | `hf_models.py` — `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`, `bart-large-mnli`, `toxic-bert` |
| `embedding` | `rag.py` (`text-embedding-3-small`, `all-MiniLM-L6-v2`) |
| `vector_store` | `rag.py` Chroma collection |
| `prompt` | `prompts/system_prompt.txt`, `rag_answer_prompt.txt`, `escalation_summary_prompt.txt` (surfaced by the agentic pass, which inspects prompt files) |
| `dataset` | `config.py` + `scripts/build_eval_set.py`, `data/eval/questions.jsonl` |
| `guardrail` | `guardrails.py` (PII redaction, injection screen, moderation model) |
| `agent` | `agent.py` orchestration and `examples/langchain_agent.py` |
| `tool` | `search_policy_kb`, `create_escalation` schemas in `agent.py` |
| `dependency` | `pyproject.toml` AI packages (langchain, openai, chromadb, …) |

## 4. Verified deterministic output

Running only the deterministic stage (Cisco AIBOM 1.10.1, no LLM) on this
repository reports **33 components**, including:

* models `gpt-4o-mini`, `gpt-4o`, `claude-sonnet-4-5`, `omni-moderation-latest`
* Hugging Face models `sentence-transformers/all-MiniLM-L6-v2`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, `facebook/bart-large-mnli`,
  `unitary/toxic-bert`
* embeddings `text-embedding-3-small`, `sentence-transformers/all-MiniLM-L6-v2`
* dataset assets for `data/eval/questions.jsonl`
* dependency entries for `langchain*`, `openai`, `chromadb`, `datasets`,
  `sentence-transformers`
* secret references for `OPENAI_API_KEY`

The agentic pass adds prompt assets, removes false positives (for example a
non-model environment variable picked up by the env resolver) and confirms
agent/tool relationships.

## 5. Demo narrative

1. Show the architecture and the model catalog in `config.py`.
2. Run the scan and open the JSON/HTML output.
3. Walk through one model entry: name, provider, where it was found, and
   whether it is a hosted or local asset.
4. Show the prompt inventory — reviewers can see exactly what instructions the
   application sends.
5. Point out the dataset and guardrail entries, i.e. the assets that make an
   AIBOM different from a plain SBOM.
6. Commit `aibom-report.json` (or its CycloneDX form) as a build artifact to
   show drift between commits. The file is git-ignored by default so you can
   decide what to publish.

## 5. Adding more signal

* add an MCP client in `mcp_client.py` that talks to an internal knowledge
  server (scanner reports `mcp_client` / `mcp_server`);
* add a second vector store (Pinecone or Qdrant) behind the same interface;
* fine-tune a local model and register the artifact, which surfaces
  `model_artifact` and `training_run` entries.
