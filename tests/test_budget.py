"""Guardrail de tokens por run e término garantido do loop de crítica."""

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from pauta.agents.research import ResearchOutput
from pauta.agents.supervisor import Router
from pauta.config import Settings, get_settings
from pauta.graph.budget import is_exhausted, remaining, would_exhaust
from pauta.graph.builder import build_graph
from pauta.graph.state import AgentState, Critique, Finding, new_state
from tests.fakes import FakeChatModel, fake_graph_models


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@tool
def busca_fake(query: str) -> str:
    """Busca determinística para teste."""
    return f"resultado para {query}"


def state_with(**overrides: object) -> AgentState:
    state = new_state(task="t", run_id="r1")
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_remaining_never_goes_negative(settings: Settings) -> None:
    state = state_with(tokens_used=settings.BUDGET_TOKENS_PER_RUN * 2)
    assert remaining(state, settings) == 0
    assert is_exhausted(state, settings) is True


def test_remaining_counts_down(settings: Settings) -> None:
    state = state_with(tokens_used=10_000)
    assert remaining(state, settings) == settings.BUDGET_TOKENS_PER_RUN - 10_000


def test_a_node_knows_it_spent_the_rest(settings: Settings) -> None:
    """O furo que o DoD da semana 1 expôs: o nó gasta tudo antes de devolver."""
    state = state_with(tokens_used=settings.BUDGET_TOKENS_PER_RUN - 1_000)
    assert would_exhaust(state, settings, spent_in_node=999) is False
    assert would_exhaust(state, settings, spent_in_node=1_000) is True


async def test_the_tool_loop_stops_when_the_budget_ends(settings: Settings) -> None:
    """Research para no meio em vez de descobrir o estouro depois de entregar."""
    tight = settings.model_copy(update={"BUDGET_TOKENS_PER_RUN": 100, "MAX_TOOL_ROUNDS": 5})
    from langchain_core.messages import AIMessage

    from pauta.agents.research import make_research_node

    calls = [
        AIMessage(
            content="",
            tool_calls=[{"name": "busca_fake", "args": {"query": "x"}, "id": f"c{i}"}],
            usage_metadata={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
        )
        for i in range(5)
    ]
    model = FakeChatModel(responses=[*calls, ResearchOutput(findings=[])])
    tools: list[BaseTool] = [busca_fake]
    node = make_research_node(model, tools, tight)
    await node(state=state_with())
    assert model.call_count < 6


async def test_the_critic_loop_always_terminates(settings: Settings) -> None:
    """Crítico que nunca aprova não faz a run rodar para sempre."""
    supervisor = FakeChatModel(
        responses=[
            Router(next="research", rationale="reunir"),
            *[Router(next="critic", rationale="mais uma volta") for _ in range(8)],
        ]
    )
    research = FakeChatModel(
        responses=[
            "achei",
            ResearchOutput(findings=[Finding(content="a", source="https://a", agent="research")]),
        ]
    )
    critic = FakeChatModel(
        responses=[Critique(verdict="refinar", gaps=["ainda falta"]) for _ in range(8)]
    )
    graph = build_graph(
        **fake_graph_models(
            supervisor_model=supervisor,
            research_model=research,
            critic_model=critic,
            writer_model=FakeChatModel(responses=["Briefing com ressalvas."]),
        ),
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "loop-1"}}
    final = await graph.ainvoke(new_state(task="t", run_id="r1"), config=config)

    assert final["final_report"] == "Briefing com ressalvas."
    assert final["critic_loops"] <= settings.MAX_CRITIC_LOOPS


async def test_a_blown_budget_still_produces_a_report(settings: Settings) -> None:
    graph = build_graph(
        **fake_graph_models(writer_model=FakeChatModel(responses=["Briefing do que houve."])),
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    state = state_with(tokens_used=settings.BUDGET_TOKENS_PER_RUN)
    config: RunnableConfig = {"configurable": {"thread_id": "budget-1"}}
    final = await graph.ainvoke(state, config=config)
    assert final["final_report"] == "Briefing do que houve."


async def test_the_step_limit_also_ends_the_run(settings: Settings) -> None:
    graph = build_graph(
        **fake_graph_models(writer_model=FakeChatModel(responses=["Briefing no limite."])),
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    state = state_with(iteration=settings.MAX_SUPERVISOR_STEPS)
    config: RunnableConfig = {"configurable": {"thread_id": "steps-1"}}
    final = await graph.ainvoke(state, config=config)
    assert final["final_report"] == "Briefing no limite."
