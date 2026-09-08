"""Timeout e retry por nó, via add_node. Sem wrapper manual."""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeTimeoutError

from pauta.agents.research import ResearchOutput
from pauta.agents.supervisor import Router
from pauta.config import Settings, get_settings
from pauta.graph.builder import build_graph
from pauta.graph.state import Critique, Finding, new_state
from tests.fakes import FakeChatModel, fake_graph_models


@tool
async def busca_lenta(query: str) -> str:
    """Tool que demora mais que o timeout do nó."""
    await asyncio.sleep(1.0)
    return f"resultado para {query}"


def impatient_settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update={"NODE_TIMEOUT_RESEARCH_S": 0.05, **overrides})


async def test_slow_tool_trips_the_node_timeout() -> None:
    settings = impatient_settings(NODE_RETRIES=0)
    research = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "busca_lenta", "args": {"query": "x"}, "id": "call-1"}],
            ),
            "nunca chega aqui",
            ResearchOutput(findings=[]),
        ]
    )
    graph = build_graph(
        **fake_graph_models(
            supervisor_model=FakeChatModel(responses=[Router(next="research", rationale="buscar")]),
            research_model=research,
            writer_model=FakeChatModel(responses=["nunca chega aqui"]),
        ),
        research_tools=[busca_lenta],
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "timeout-1"}}
    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke(new_state(task="t", run_id="r1"), config=config)


async def test_the_timeout_comes_from_settings() -> None:
    """O número vem de NODE_TIMEOUT_S, não de constante solta no builder."""
    settings = impatient_settings(NODE_TIMEOUT_RESEARCH_S=0.25)
    graph = build_graph(
        **fake_graph_models(
            supervisor_model=FakeChatModel(responses=[]),
            research_model=FakeChatModel(responses=[]),
            writer_model=FakeChatModel(responses=[]),
        ),
        settings=settings,
    )
    policy = graph.nodes["research"].timeout
    assert policy is not None
    assert policy.run_timeout == 0.25


async def test_tools_are_optional() -> None:
    tools: list[BaseTool] = []
    graph = build_graph(
        **fake_graph_models(
            supervisor_model=FakeChatModel(
                responses=[
                    Router(next="research", rationale="reunir material"),
                    Router(next="writer", rationale="material reunido"),
                ]
            ),
            research_model=FakeChatModel(
                responses=[
                    "sem tool",
                    ResearchOutput(
                        findings=[Finding(content="a", source="https://a", agent="research")]
                    ),
                ]
            ),
            writer_model=FakeChatModel(responses=["Briefing."]),
        ),
        research_tools=tools,
        settings=get_settings(),
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "sem-tools"}}
    final = await graph.ainvoke(new_state(task="t", run_id="r1"), config=config)
    assert final["final_report"] == "Briefing."


async def test_fruitless_research_gives_up_instead_of_burning_the_budget() -> None:
    """Foi assim que uma run real gastou 58 mil tokens sem produzir briefing.

    O research volta vazio sempre. Com o teto, a run chega ao writer, que redige
    dizendo o que não encontrou, em vez de repetir pesquisa até o orçamento acabar.
    """
    resolved = get_settings().model_copy(
        update={"MAX_EMPTY_RESEARCH": 2, "MAX_SUPERVISOR_STEPS": 20}
    )
    empty = ResearchOutput(findings=[])
    graph = build_graph(
        **fake_graph_models(
            supervisor_model=FakeChatModel(
                responses=[Router(next="research", rationale="tentar de novo")] * 12
            ),
            research_model=FakeChatModel(responses=["nada achei", empty] * 6),
            critic_model=FakeChatModel(responses=[Critique(verdict="refinar")] * 4),
            writer_model=FakeChatModel(responses=["Não encontrei material sobre isso."]),
        ),
        settings=resolved,
        checkpointer=InMemorySaver(),
    )
    final = await graph.ainvoke(
        new_state(task="algo que o corpus não responde", run_id="r1"),
        config={"configurable": {"thread_id": "sem-resposta"}},
    )

    assert final["final_report"] == "Não encontrei material sobre isso."
    assert final["empty_research"] == 2, "parou no teto, não seguiu tentando"
    assert final["tokens_used"] < resolved.BUDGET_TOKENS_PER_RUN
