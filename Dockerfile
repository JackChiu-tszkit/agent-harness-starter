FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_HARNESS_MODEL_PROVIDER=mock \
    AGENT_HARNESS_DATABASE_PATH=/data/harness.db \
    AGENT_HARNESS_OTEL_EXPORTER=console \
    AGENT_HARNESS_HOST=0.0.0.0 \
    AGENT_HARNESS_PORT=8000

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY examples ./examples
RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "agent_harness.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
