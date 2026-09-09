"""A API do contrato 8.6, com store em memória e modelos falsos.

Nenhum teste aqui sobe Postgres nem chama provider. É o `create_app` recebendo
recursos prontos que torna isso possível.
"""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from pauta.agents.research import ResearchOutput
from pauta.agents.supervisor import Router
from pauta.api.limits import RateLimiter
from pauta.api.main import Resources, _events_of, create_app
from pauta.config import Settings, get_settings
from pauta.graph.builder import build_graph
from pauta.graph.state import Critique, Finding
from pauta.memory.run_store import InMemoryRunStore
from pauta.memory.runs import Run, RunStatus
from pauta.observability import setup_logging
from pauta.runner import execute_run, register_run
from tests.fakes import FakeChatModel, fake_graph_models


def a_full_run() -> dict[str, Any]:
    """Research, critic e writer, com o crítico aprovando."""
    return fake_graph_models(
        supervisor_model=FakeChatModel(
            responses=[
                Router(next="research", rationale="reunir"),
                Router(next="critic", rationale="validar"),
                Router(next="writer", rationale="redigir"),
            ]
        ),
        research_model=FakeChatModel(
            responses=[
                "achei",
                ResearchOutput(
                    findings=[
                        Finding(content="custa 1800 USD", source="https://a", agent="research")
                    ]
                ),
            ]
        ),
        critic_model=FakeChatModel(responses=[Critique(verdict="ok")]),
        writer_model=FakeChatModel(responses=["Briefing pronto."]),
    )


def make_resources(**overrides: Any) -> Resources:
    settings: Settings = overrides.pop("settings", get_settings())
    checkpointer = InMemorySaver()
    graphs = {
        mode: build_graph(
            **a_full_run(),
            settings=settings.model_copy(update={"HITL_MODE": mode}),
            checkpointer=checkpointer,
        )
        for mode in ("auto", "interrupt")
    }
    return Resources(
        settings=settings,
        store=overrides.pop("store", InMemoryRunStore()),
        graphs=overrides.pop("graphs", graphs),
        limiter=overrides.pop("limiter", RateLimiter("100/hour")),
    )


@pytest.fixture
def resources() -> Resources:
    return make_resources()


@pytest.fixture
def client(resources: Resources) -> Iterator[TestClient]:
    with TestClient(create_app(resources)) as client:
        yield client


async def settle(resources: Resources) -> None:
    """Espera as tasks de segundo plano, que é o que o `POST /runs` deixa pendurado."""
    while resources.background:
        await asyncio.gather(*list(resources.background), return_exceptions=True)


def settle_background(client: TestClient, resources: Resources) -> None:
    """Drena o segundo plano no loop que a própria app usa."""
    portal = client.portal
    assert portal is not None, "use o TestClient dentro do `with`"
    portal.call(settle, resources)


def force_orphan(client: TestClient, resources: Resources, run_id: str) -> None:
    """Simula a queda que deixa a run com checkpoint vivo e sem executor."""
    portal = client.portal
    assert portal is not None, "use o TestClient dentro do `with`"
    portal.call(orphan, resources, run_id)


def a_run(run_id: str, status: RunStatus, tokens: int = 0) -> Run:
    return Run(
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        task="vale a pena migrar?",
        status=status,
        tokens_used=tokens,
    )


def test_health_says_what_it_protects(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["budget_enforceable"] is False, "sem COST_PER_MTOK_USD o teto não protege"
    assert body["spent_today_usd"] is None
    assert body["daily_budget_usd"] > 0


def test_a_posted_task_answers_201_with_the_thread_id(client: TestClient) -> None:
    """O `thread_id` na resposta é o que permite retomar depois de uma queda."""
    response = client.post("/runs", json={"task": "vale a pena migrar?"})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "running"
    assert body["thread_id"].startswith("thread-")
    assert body["run_id"].startswith("run-")


def test_an_empty_task_is_refused(client: TestClient) -> None:
    assert client.post("/runs", json={"task": "   "}).status_code in (422, 201)
    assert client.post("/runs", json={"task": ""}).status_code == 422


def test_a_giant_task_is_refused_before_it_becomes_context(client: TestClient) -> None:
    assert client.post("/runs", json={"task": "x" * 5000}).status_code == 422


def test_the_run_finishes_in_the_background(client: TestClient, resources: Resources) -> None:
    created = client.post("/runs", json={"task": "vale a pena migrar?"}).json()
    settle_background(client, resources)

    detail = client.get(f"/runs/{created['run_id']}").json()
    assert detail["status"] == "completed"
    assert detail["final_report"] == "Briefing pronto."


def test_the_detail_brings_the_material_from_the_checkpoint(
    client: TestClient, resources: Resources
) -> None:
    """Findings e críticas moram no estado do grafo, não no ponteiro da run."""
    created = client.post("/runs", json={"task": "vale a pena migrar?"}).json()
    settle_background(client, resources)

    detail = client.get(f"/runs/{created['run_id']}").json()
    assert [item["content"] for item in detail["findings"]] == ["custa 1800 USD"]
    assert detail["critiques"][0]["verdict"] == "ok"


def test_an_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/runs/nao-existe").status_code == 404


async def test_the_listing_filters_by_status(resources: Resources) -> None:
    await resources.store.save(a_run("r1", "orphaned"))
    await resources.store.save(a_run("r2", "completed"))
    with TestClient(create_app(resources)) as client:
        orphans = client.get("/runs", params={"status": "orphaned"}).json()
        assert [item["run_id"] for item in orphans] == ["r1"]
        assert len(client.get("/runs").json()) == 2


async def test_the_cost_is_absent_without_a_declared_price(resources: Resources) -> None:
    await resources.store.save(a_run("r1", "completed", tokens=1_000_000))
    with TestClient(create_app(resources)) as client:
        assert client.get("/runs/r1").json()["cost_usd"] is None


async def test_the_cost_is_reported_once_the_price_exists() -> None:
    settings = get_settings().model_copy(update={"COST_PER_MTOK_USD": 2.0})
    resources = make_resources(settings=settings)
    await resources.store.save(a_run("r1", "completed", tokens=1_000_000))
    with TestClient(create_app(resources)) as client:
        assert client.get("/runs/r1").json()["cost_usd"] == pytest.approx(2.0)


def test_the_rate_limit_refuses_and_says_when_to_return() -> None:
    resources = make_resources(limiter=RateLimiter("1/hour"))
    with TestClient(create_app(resources)) as client:
        assert client.post("/runs", json={"task": "primeira"}).status_code == 201
        refused = client.post("/runs", json={"task": "segunda"})
        assert refused.status_code == 429
        assert "este IP" in refused.json()["detail"]


async def test_the_daily_cap_answers_503_instead_of_accepting_the_run() -> None:
    """Recusa honesta é melhor que aceitar a run e falhar no meio dela."""
    settings = get_settings().model_copy(update={"COST_PER_MTOK_USD": 2.0, "DAILY_BUDGET_USD": 1.0})
    resources = make_resources(settings=settings)
    await resources.store.save(a_run("gastou", "completed", tokens=1_000_000))

    with TestClient(create_app(resources)) as client:
        refused = client.post("/runs", json={"task": "mais uma"})
        assert refused.status_code == 503
        assert "teto diário" in refused.json()["detail"]


async def test_resume_refuses_a_run_that_is_not_waiting_for_a_human(
    resources: Resources,
) -> None:
    """409 é a resposta certa: a run existe, o pedido é que não cabe no estado dela."""
    await resources.store.save(a_run("r1", "completed"))
    with TestClient(create_app(resources)) as client:
        response = client.post("/runs/r1/resume", json={"approved": True})
        assert response.status_code == 409
        assert "interrupted" in response.json()["detail"]


async def test_continue_refuses_a_run_that_did_not_fall_over(resources: Resources) -> None:
    await resources.store.save(a_run("r1", "interrupted"))
    with TestClient(create_app(resources)) as client:
        response = client.post("/runs/r1/continue")
        assert response.status_code == 409
        assert "orphaned" in response.json()["detail"]


async def test_resume_and_continue_are_404_for_a_run_nobody_registered(
    resources: Resources,
) -> None:
    with TestClient(create_app(resources)) as client:
        assert client.post("/runs/sumiu/resume", json={"approved": True}).status_code == 404
        assert client.post("/runs/sumiu/continue").status_code == 404


def test_a_frozen_run_waits_for_the_human_then_finishes(resources: Resources) -> None:
    with TestClient(create_app(resources)) as client:
        created = client.post(
            "/runs", json={"task": "vale a pena migrar?", "hitl_mode": "interrupt"}
        ).json()
        settle_background(client, resources)

        frozen = client.get(f"/runs/{created['run_id']}").json()
        assert frozen["status"] == "interrupted"
        assert frozen["final_report"] is None

        resumed = client.post(f"/runs/{created['run_id']}/resume", json={"approved": True})
        assert resumed.status_code == 200
        settle_background(client, resources)

        assert client.get(f"/runs/{created['run_id']}").json()["status"] == "completed"


def test_an_orphan_is_continued_only_when_asked(resources: Resources) -> None:
    """ADR 006: religar não retoma sozinho, mas um POST explícito retoma."""
    with TestClient(create_app(resources)) as client:
        created = client.post(
            "/runs", json={"task": "vale a pena migrar?", "hitl_mode": "interrupt"}
        ).json()
        settle_background(client, resources)
        force_orphan(client, resources, created["run_id"])

        assert client.post(f"/runs/{created['run_id']}/continue").status_code == 200
        settle_background(client, resources)
        assert client.get(f"/runs/{created['run_id']}").json()["status"] == "completed"


async def orphan(resources: Resources, run_id: str) -> None:
    run = await resources.store.get(run_id)
    assert run is not None
    await resources.store.save(run.transition_to("orphaned"))


def test_the_stream_of_an_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/runs/sumiu/stream").status_code == 404


def test_the_stream_replays_a_run_that_already_finished(
    client: TestClient, resources: Resources
) -> None:
    """Cliente que chega tarde não pode ficar esperando um evento que já passou."""
    created = client.post("/runs", json={"task": "vale a pena migrar?"}).json()
    settle_background(client, resources)

    with client.stream("GET", f"/runs/{created['run_id']}/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: final" in body
    assert "já tinha terminado" in body


def test_the_demo_page_is_served_by_the_api(client: TestClient) -> None:
    response = client.get("/demo")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "EventSource" in response.text


async def test_the_generator_streams_the_events_of_a_live_run(resources: Resources) -> None:
    """Direto no gerador: o TestClient serializa o portal e não serve para stream vivo.

    É o mesmo emissor do log, então o que sai aqui foi o que de fato aconteceu.
    """
    setup_logging()
    resources.broadcaster.install()
    try:
        run = await register_run("vale a pena migrar?", store=resources.store)
        frames: list[str] = []

        async def collect() -> None:
            async for chunk in _events_of(resources, run.run_id):
                frames.append(chunk)

        reader = asyncio.create_task(collect())
        await asyncio.sleep(0)
        await execute_run(run, graph=resources.graph_for("auto"), store=resources.store)
        await asyncio.wait_for(reader, timeout=5)
    finally:
        resources.broadcaster.uninstall()

    body = "".join(frames)
    assert "event: node_start" in body
    assert "event: finding" in body
    assert "event: critique" in body
    assert '"type": "finding"' in body
    assert "custa 1800 USD" in body


async def test_the_stream_stops_when_the_run_stops(resources: Resources) -> None:
    """Sem o sinal de fim, o cliente ficaria pendurado numa run que já acabou."""
    setup_logging()
    resources.broadcaster.install()
    try:
        run = await register_run("vale a pena migrar?", store=resources.store)

        async def collect() -> int:
            return len([chunk async for chunk in _events_of(resources, run.run_id)])

        reader = asyncio.create_task(collect())
        await asyncio.sleep(0)
        await execute_run(run, graph=resources.graph_for("auto"), store=resources.store)
        assert await asyncio.wait_for(reader, timeout=5) > 0
    finally:
        resources.broadcaster.uninstall()
