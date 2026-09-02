"""Interrupt de human-in-the-loop: o grafo congela antes do writer e retoma."""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from pauta.agents.research import ResearchOutput
from pauta.agents.supervisor import Router
from pauta.config import Settings, get_settings
from pauta.graph.builder import build_graph
from pauta.graph.state import Critique, Finding, new_state
from tests.fakes import FakeChatModel, fake_graph_models


def hitl_settings(mode: str) -> Settings:
    return get_settings().model_copy(update={"HITL_MODE": mode})


def approved_run() -> dict[str, object]:
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
        writer_model=FakeChatModel(responses=["Briefing aprovado."]),
    )


async def test_auto_mode_runs_to_the_end_without_stopping() -> None:
    graph = build_graph(
        **approved_run(), settings=hitl_settings("auto"), checkpointer=InMemorySaver()
    )
    config: RunnableConfig = {"configurable": {"thread_id": "auto-1"}}
    final = await graph.ainvoke(new_state(task="t", run_id="r1"), config=config)
    assert final["final_report"] == "Briefing aprovado."


async def test_interrupt_mode_freezes_before_the_writer() -> None:
    graph = build_graph(
        **approved_run(), settings=hitl_settings("interrupt"), checkpointer=InMemorySaver()
    )
    config: RunnableConfig = {"configurable": {"thread_id": "hitl-1"}}
    await graph.ainvoke(new_state(task="t", run_id="r1"), config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("writer",)
    assert snapshot.values.get("final_report") is None
    assert snapshot.values["critiques"][0].verdict == "ok"


async def test_resume_finishes_the_frozen_run() -> None:
    graph = build_graph(
        **approved_run(), settings=hitl_settings("interrupt"), checkpointer=InMemorySaver()
    )
    config: RunnableConfig = {"configurable": {"thread_id": "hitl-2"}}
    await graph.ainvoke(new_state(task="t", run_id="r1"), config=config)

    final = await graph.ainvoke(None, config=config)
    assert final["final_report"] == "Briefing aprovado."
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()


async def test_human_feedback_reaches_the_writer() -> None:
    writer = FakeChatModel(responses=["Briefing revisado."])
    graph = build_graph(
        **approved_run() | {"writer_model": writer},
        settings=hitl_settings("interrupt"),
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "hitl-3"}}
    await graph.ainvoke(new_state(task="t", run_id="r1"), config=config)

    final = await graph.ainvoke(
        Command(update={"hitl_feedback": "foque no custo de saída"}), config=config
    )
    assert final["final_report"] == "Briefing revisado."
    prompt = str(writer.calls[0][1].content)
    assert "foque no custo de saída" in prompt


async def test_the_frozen_state_survives_by_thread_id() -> None:
    """Congelado, o estado fica no checkpointer e é lido por outra chamada."""
    checkpointer = InMemorySaver()
    graph = build_graph(
        **approved_run(), settings=hitl_settings("interrupt"), checkpointer=checkpointer
    )
    config: RunnableConfig = {"configurable": {"thread_id": "hitl-4"}}
    await graph.ainvoke(new_state(task="minha tarefa", run_id="r1"), config=config)

    other = build_graph(
        **approved_run(), settings=hitl_settings("interrupt"), checkpointer=checkpointer
    )
    snapshot = await other.aget_state(config)
    assert snapshot.values["task"] == "minha tarefa"
    assert snapshot.next == ("writer",)


@pytest.mark.parametrize("mode", ["auto", "interrupt"])
async def test_the_mode_comes_from_settings(mode: str) -> None:
    graph = build_graph(**approved_run(), settings=hitl_settings(mode))
    expected = ("writer",) if mode == "interrupt" else ()
    assert tuple(graph.interrupt_before_nodes or ()) == expected
