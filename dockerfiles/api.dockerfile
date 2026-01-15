FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY uv.lock uv.lock

RUN uv sync --frozen --no-install-project

COPY src src/
COPY models models/
COPY configs configs/

RUN uv sync --frozen

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "src.cds_repository.api:app", "--host", "0.0.0.0", "--port", "8000"]
