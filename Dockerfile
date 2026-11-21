FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AI_PROVIDER=stub

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY data ./data

RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["uvicorn", "policy_assistant.api:app", "--host", "0.0.0.0", "--port", "8080"]
