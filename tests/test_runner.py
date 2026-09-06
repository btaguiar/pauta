"""O runner: costura grafo, checkpoint e ponteiro, e grava o desfecho sempre."""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from pauta.agents.research import ResearchOutput
from pauta.agents.supervisor import Router
from pauta.config import Settings, get_settings
from pauta.graph.builder import build_graph
from pauta.graph.state import Critique, Finding
from pauta.memory.run_store import InMemoryRunStore
from pauta.memory.runs import Run
from pauta.runner import (
    RunAlreadyFinished,
    RunNotFound,
    new_ids,
    resume_run,
    start_run,
)
from tests.fakes import FakeChatModel, fake_graph_models


@pytest.fixture
def store() -> InMemoryRunStore:
    return InMemoryRunStore()


def hitl_settings(mode: str) -> Settings:
    return get_settings().model_copy(update={"HITL_MODE": mode})


def a_full_run() -> dict[str, Any]:
    """Research, critic e writer, na ordem, com o crítico aprovando."""
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
                    findings=[Finding(content="a", source="https://a", agent="research")]
                ),
            ]
        ),
        critic_model=FakeChatModel(responses=[Critique(verdict="ok")]),
        writer_model=FakeChatModel(responses=["Briefing pronto."]),
    )


def a_graph(mode: str = "auto") -> Any:
    return build_graph(**a_full_run(), settings=hitl_settings(mode), checkpointer=InMemorySaver())


def test_the_two_ids_are_different_and_paired() -> None:
    """`run_id` identifica a tentativa, `thread_id` identifica a linha de checkpoint."""
    run_id, thread_id = new_ids()
    assert run_id != thread_id
    assert run_id.startswith("run-")
    assert thread_id.startswith("thread-")
    assert new_ids() != new_ids()


async def test_a_finished_run_is_recorded_as_completed(store: InMemoryRunStore) -> None:
    outcome = await start_run("vale a pena migrar?", graph=a_graph(), store=store)
    assert outcome.report == "Briefing pronto."
    assert outcome.waiting_for_human is False
    assert outcome.run.status == "completed"

    saved = await store.by_thread(outcome.run.thread_id)
    assert saved is not None
    assert saved.status == "completed"
    assert saved.final_report == "Briefing pronto."
    assert saved.task == "vale a pena migrar?"


async def test_the_pointer_exists_before_the_graph_runs(store: InMemoryRunStore) -> None:
    """Gravar só no fim perderia toda run que morresse no meio."""
    await start_run("t", graph=a_graph(), store=store, thread_id="fixo")
    saved = await store.by_thread("fixo")
    assert saved is not None
    assert saved.created_at <= saved.updated_at


async def test_an_interrupted_run_is_recorded_as_interrupted(store: InMemoryRunStore) -> None:
    outcome = await start_run("t", graph=a_graph("interrupt"), store=store)
    assert outcome.waiting_for_human is True
    assert outcome.report is None
    assert outcome.run.status == "interrupted"
    assert outcome.run.waiting_for_human


async def test_a_failure_is_recorded_before_it_propagates(store: InMemoryRunStore) -> None:
    """Run que morre sem rastro no ponteiro é o modo de falha que o runner evita."""
    graph = build_graph(
        **fake_graph_models(
            supervisor_model=FakeChatModel(responses=[Router(next="research", rationale="reunir")]),
            research_model=FakeChatModel(responses=[RuntimeError("provider caiu")]),
        ),
        settings=get_settings().model_copy(update={"NODE_RETRIES": 0}),
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(RuntimeError, match="provider caiu"):
        await start_run("t", graph=graph, store=store, thread_id="quebrada")

    saved = await store.by_thread("quebrada")
    assert saved is not None
    assert saved.status == "failed"
    assert saved.error is not None


async def test_resume_finishes_a_frozen_run(store: InMemoryRunStore) -> None:
    graph = a_graph("interrupt")
    started = await start_run("t", graph=graph, store=store)
    resumed = await resume_run(started.run.thread_id, graph=graph, store=store)

    assert resumed.report == "Briefing pronto."
    assert resumed.run.status == "completed"
    assert resumed.run.thread_id == started.run.thread_id
    assert resumed.run.run_id == started.run.run_id


async def test_resume_carries_the_human_feedback(store: InMemoryRunStore) -> None:
    writer = FakeChatModel(responses=["Briefing revisado."])
    graph = build_graph(
        **a_full_run() | {"writer_model": writer},
        settings=hitl_settings("interrupt"),
        checkpointer=InMemorySaver(),
    )
    started = await start_run("t", graph=graph, store=store)
    resumed = await resume_run(
        started.run.thread_id, graph=graph, store=store, feedback="foque no custo de saída"
    )

    assert resumed.report == "Briefing revisado."
    assert "foque no custo de saída" in str(writer.calls[0][1].content)


async def test_an_orphan_can_be_resumed(store: InMemoryRunStore) -> None:
    """`orphaned` é retomável por pedido explícito, que é o que a ADR 006 exige."""
    graph = a_graph("interrupt")
    started = await start_run("t", graph=graph, store=store)
    await store.save(started.run.transition_to("orphaned"))

    resumed = await resume_run(started.run.thread_id, graph=graph, store=store)
    assert resumed.run.status == "completed"


async def test_resuming_an_unknown_thread_says_so(store: InMemoryRunStore) -> None:
    with pytest.raises(RunNotFound):
        await resume_run("nunca-existiu", graph=a_graph(), store=store)


async def test_a_finished_run_does_not_resume(store: InMemoryRunStore) -> None:
    await store.save(
        Run(run_id="r1", thread_id="t1", task="t", status="running").transition_to("completed")
    )
    with pytest.raises(RunAlreadyFinished):
        await resume_run("t1", graph=a_graph(), store=store)


async def test_a_run_killed_mid_flight_still_resumes(store: InMemoryRunStore) -> None:
    """A morte suja deixa o ponteiro em `running`, e daí ele precisa voltar."""
    graph = a_graph("interrupt")
    started = await start_run("t", graph=graph, store=store)
    await store.save(started.run.model_copy(update={"status": "running"}))

    resumed = await resume_run(started.run.thread_id, graph=graph, store=store)
    assert resumed.run.status == "completed"
