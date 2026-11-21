.PHONY: install dev test run aibom clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

run:
	AI_PROVIDER=$${AI_PROVIDER:-stub} python -m uvicorn policy_assistant.api:app --reload --port 8080

aibom:
	cisco-aibom analyze . --output-format json --output-file aibom-report.json \
		--llm-model $${AIBOM_LLM_MODEL:-gpt-4o-mini}

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info .chroma aibom-report.json
