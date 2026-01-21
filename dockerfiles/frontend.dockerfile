FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY uv.lock uv.lock

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

EXPOSE 8080

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/cds_repository/app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
