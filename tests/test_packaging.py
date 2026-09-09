"""Coerência entre Dockerfile, compose e `.env.example`.

Nada aqui sobe container. O que estes testes pegam é a divergência silenciosa:
alguém troca a senha do Postgres num arquivo e esquece do outro, e a app só
falha na máquina de quem clonou.

O `CLAUDE.md` já exige que `.env.example` e `docker-compose.yml` mudem no mesmo
commit. Isto é essa regra virando teste.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
ENV_EXAMPLE = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

#: Credencial de desenvolvimento local, a mesma nos três lugares.
USER = PASSWORD = DATABASE = "pauta"


def test_the_compose_declares_the_credential_the_url_uses() -> None:
    assert f"POSTGRES_USER: {USER}" in COMPOSE
    assert f"POSTGRES_PASSWORD: {PASSWORD}" in COMPOSE
    assert f"POSTGRES_DB: {DATABASE}" in COMPOSE


def test_the_services_reach_the_database_by_service_name() -> None:
    """Dentro da rede do compose o banco não atende por localhost."""
    inside = f"postgresql://{USER}:{PASSWORD}@postgres:5432/{DATABASE}"
    assert COMPOSE.count(f"DATABASE_URL: {inside}") == 2, "api e index precisam da mesma URL"


def test_the_env_example_points_at_the_developer_machine() -> None:
    """Fora do compose, quem roda `python -m pauta` fala com a porta publicada."""
    outside = f"DATABASE_URL=postgresql://{USER}:{PASSWORD}@localhost:5432/{DATABASE}"
    assert outside in ENV_EXAMPLE


def test_the_published_port_matches_what_the_url_expects() -> None:
    assert '"5432:5432"' in COMPOSE


def test_the_indexing_service_stays_out_of_the_default_up() -> None:
    """`docker compose up` não deve gastar chamada de embedding sem alguém pedir."""
    index_block = COMPOSE.split("  index:")[1]
    assert 'profiles: ["tools"]' in index_block
    assert '"--index-corpus"' in index_block


def test_the_api_waits_for_a_healthy_database() -> None:
    api_block = COMPOSE.split("  api:")[1].split("  index:")[0]
    assert "condition: service_healthy" in api_block


def test_the_image_installs_from_the_committed_lock() -> None:
    """`--frozen` faz a imagem usar a resolução que passou no CI, não uma nova."""
    assert "uv.lock" in DOCKERFILE
    assert "uv sync --frozen --no-dev" in DOCKERFILE


def test_the_image_carries_the_source_next_to_the_corpus() -> None:
    """A instalação é editável, e o retriever acha `samples/` a partir da raiz."""
    assert "COPY --chown=pauta:pauta src/ ./src/" in DOCKERFILE
    assert "COPY --chown=pauta:pauta samples/ ./samples/" in DOCKERFILE
    assert "WORKDIR /app" in DOCKERFILE


def test_the_image_does_not_run_as_root() -> None:
    assert "USER pauta" in DOCKERFILE
    assert DOCKERFILE.index("USER pauta") < DOCKERFILE.index("CMD [")


def test_the_command_points_at_a_module_that_exists() -> None:
    match = re.search(r'CMD \["uvicorn", "([^"]+)"', DOCKERFILE)
    assert match is not None
    module, _, attribute = match.group(1).partition(":")
    imported = __import__(module, fromlist=[attribute])
    assert hasattr(imported, attribute), f"{module} não expõe {attribute}"


def test_the_healthcheck_asks_an_endpoint_the_api_serves() -> None:
    """Healthcheck apontando para rota inexistente deixa o container sempre doente."""
    assert "/health" in DOCKERFILE
    assert "/health" in _routes()


def _routes() -> set[str]:
    import fastapi

    from pauta.api.main import register_routes

    app = fastapi.FastAPI()
    register_routes(app)
    return {route.path for route in app.routes if hasattr(route, "path")}


def test_the_api_serves_every_endpoint_the_contract_names() -> None:
    """Contrato da seção 8.6 do dossiê, menos o stream, que é a fase seguinte."""
    assert _routes() >= {
        "/health",
        "/runs",
        "/runs/{run_id}",
        "/runs/{run_id}/resume",
        "/runs/{run_id}/continue",
    }


def test_the_image_carries_the_demo_page() -> None:
    """A API serve a demo em /demo; sem o COPY, a rota responde 404 no container."""
    assert "COPY --chown=pauta:pauta demo/ ./demo/" in DOCKERFILE


def test_the_demo_page_exists_where_the_api_looks_for_it() -> None:
    from pauta.api.main import DEMO_PAGE

    assert DEMO_PAGE.is_file(), f"a API procura a demo em {DEMO_PAGE}"
    assert DEMO_PAGE == REPO_ROOT / "demo" / "index.html"


def test_the_demo_never_puts_server_text_into_html() -> None:
    """Briefing e descoberta vêm de um LLM. Texto de LLM entra no DOM como texto."""
    page = (REPO_ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    written = re.findall(r"\.(innerHTML|outerHTML|insertAdjacentHTML)\s*[=(]", page)
    assert written == [], f"a demo escreve HTML cru: {written}"
    assert "textContent" in page


def test_the_demo_needs_no_build_and_no_cdn() -> None:
    page = (REPO_ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    assert "<script src=" not in page
    assert "https://" not in page.split("<script>")[1]
