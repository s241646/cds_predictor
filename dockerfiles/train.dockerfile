FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY uv.lock uv.lock

RUN uv sync --frozen --no-install-project

COPY src src/
COPY data data/
COPY models models/
COPY configs configs/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/cds_repository/train.py"]
