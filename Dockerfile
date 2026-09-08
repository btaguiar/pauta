# syntax=docker/dockerfile:1

# Build em dois estágios: o primeiro resolve dependências, o segundo carrega só
# o que roda. A instalação sai do uv.lock comitado, então a imagem é a mesma
# resolução que passou no CI, não uma resolução nova feita na hora do build.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Só o manifesto primeiro. A camada de dependências sobrevive a toda mudança de
# código, que é o que faz o rebuild ser rápido no uso diário.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Nada aqui precisa de root, e esta imagem fica exposta numa porta.
RUN useradd --create-home --uid 10001 pauta

WORKDIR /app

COPY --from=builder --chown=pauta:pauta /app/.venv /app/.venv
# O `src/` acompanha o venv porque a instalação é editável: o pacote resolve
# `samples/` a partir da raiz do projeto, e o retriever depende disso.
COPY --chown=pauta:pauta src/ ./src/
COPY --chown=pauta:pauta samples/ ./samples/

USER pauta

EXPOSE 8000

# O compose espera este healthcheck para liberar quem depende da app.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "pauta.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
