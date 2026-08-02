ARG PYTHON_VERSION=3.11

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS build

ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev ${EXTRAS:+--extra ${EXTRAS}}

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev ${EXTRAS:+--extra ${EXTRAS}}


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# The venv carries absolute paths from the build stage.
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app config.yml config.docker.yml config.docker.semantic.yml ./
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app data/ ./data/

RUN mkdir -p /app/logs /app/data/index "${HF_HOME}" \
    && chown -R app:app /app/logs /app/data /home/app/.cache

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/', timeout=4)"]

ENTRYPOINT ["rag-app", "-c", "config.docker.yml"]
