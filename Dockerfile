# syntax=docker/dockerfile:1
#
# Two stages, two different base images, each chosen for one job:
#
#   build   ghcr.io/astral-sh/uv:python3.11-bookworm-slim
#           uv is already on the image, so the lockfile installs with no
#           bootstrap step and `uv sync --frozen` reproduces uv.lock exactly.
#   runtime python:3.11-slim-bookworm
#           only needs an interpreter of the same minor version. uv, the build
#           caches, and the toolchain stay behind in the build stage.
#
# The heavy embedding stack is opt-in. The default build installs no torch and
# runs BM25 retrieval; pass --build-arg EXTRAS=bge-flag for local BGE-M3.

ARG PYTHON_VERSION=3.11

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS build

ARG EXTRAS=""
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve in their own layer, keyed only on the manifests, so
# editing application code does not reinstall them.
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
    HF_HOME=/home/app/.cache/huggingface \
    UI_HOST=0.0.0.0 \
    UI_OPEN_BROWSER=false

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# The venv carries absolute paths from the build stage, so it must land on the
# same path here.
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app config.yml pyproject.toml README.md ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app data/ ./data/

# Writable at runtime: the log file, the index the app derives on first run, and
# the HuggingFace cache. The cache directory has to exist here even though it is
# usually a volume - Docker seeds a fresh named volume from the image directory,
# and if the path is missing it creates it root-owned, which the app user then
# cannot write to.
RUN mkdir -p /app/logs /app/data/index "${HF_HOME}" \
    && chown -R app:app /app/logs /app/data /home/app/.cache

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/', timeout=4)"]

# No arguments launches the Gradio UI; pass a question to use the CLI instead.
ENTRYPOINT ["rag-app"]
