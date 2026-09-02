"""Durabilidade (a): drain cooperativo.

O caso (b), morte suja com `kill -9`, vive em `tests/test_durability_kill.py` e
exige Postgres: com `InMemorySaver` o checkpoint morre junto com o processo, e o
teste não provaria nada.
"""

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl

from pauta.agents.research import ResearchOutput
from pauta.agents.supervisor import Router
from pauta.config import Settings, get_settings
from pauta.graph.builder import build_graph
from pauta.graph.state import Critique, Finding, new_state
from tests.fakes import FakeChatModel, fake_graph_models


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def a_full_run() -> dict[str, Any]:
    """Uma run que vai de research a writer, opcionalmente pedindo drain no meio."""
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
        writer_model=FakeChatModel(responses=["Briefing."]),
    )


async def test_drain_before_the_first_superstep_keeps_the_checkpoint_clean(
    settings: Settings,
) -> None:
    control = RunControl()
    control.request_drain("desligando")
    graph = build_graph(**a_full_run(), settings=settings, checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "drain-early"}}

    with pytest.raises(GraphDrained):
        await graph.ainvoke(new_state(task="t", run_id="r1"), config=config, control=control)

    snapshot = await graph.aget_state(config)
    assert snapshot.values.get("final_report") is None
    assert snapshot.values.get("iteration") == 0


async def test_drain_mid_run_finishes_the_current_superstep(settings: Settings) -> None:
    """Pedido no meio, o superstep corrente termina e o trabalho dele fica gravado."""
    control = RunControl()

    @tool
    def busca_que_pede_parada(query: str) -> str:
        """Busca que dispara o drain enquanto o nó research ainda trabalha."""
        control.request_drain("sinal no meio da busca")
        return f"resultado para {query}"

    from langchain_core.messages import AIMessage

    research = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "busca_que_pede_parada", "args": {"query": "x"}, "id": "c1"}],
            ),
            "achei",
            ResearchOutput(
                findings=[Finding(content="achado", source="https://a", agent="research")]
            ),
        ]
    )
    tools: list[BaseTool] = [busca_que_pede_parada]
    graph = build_graph(
        **a_full_run() | {"research_model": research},
        research_tools=tools,
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "drain-mid"}}

    with pytest.raises(GraphDrained):
        await graph.ainvoke(new_state(task="t", run_id="r1"), config=config, control=control)

    snapshot = await graph.aget_state(config)
    assert control.drain_requested is True
    assert [f.content for f in snapshot.values["findings"]] == ["achado"]
    assert snapshot.values.get("final_report") is None


async def test_a_drained_run_resumes_by_thread_id(settings: Settings) -> None:
    """O que o DoD pede: parou limpo, retoma pelo thread_id e termina."""
    control = RunControl()
    control.request_drain("desligando")
    checkpointer = InMemorySaver()
    graph = build_graph(**a_full_run(), settings=settings, checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": "drain-resume"}}

    with pytest.raises(GraphDrained):
        await graph.ainvoke(new_state(task="t", run_id="r1"), config=config, control=control)

    resumed = build_graph(**a_full_run(), settings=settings, checkpointer=checkpointer)
    final = await resumed.ainvoke(None, config=config)

    assert final["final_report"] == "Briefing."
    assert final["critiques"][0].verdict == "ok"
    assert final["iteration"] == 3


async def test_a_fresh_control_does_not_carry_a_stale_drain() -> None:
    """`RunControl` é por run: reusar um drenado deixaria a próxima run parada."""
    used = RunControl()
    used.request_drain("run anterior")
    assert used.drain_requested is True
    assert RunControl().drain_requested is False
