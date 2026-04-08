FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir \
    openenv-core==0.2.2 \
    fastapi==0.135.3 \
    uvicorn==0.44.0 \
    pydantic==2.12.5 \
    openai==2.30.0 \
    httpx==0.28.1

ENV PYTHONPATH="/app:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
