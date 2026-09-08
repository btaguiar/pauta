import io
import json
import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from pauta.agents.research import ResearchOutput, make_research_node, unanswered
from pauta.config import Settings, get_settings
from pauta.graph.state import AgentState, Critique, Finding, new_state
from pauta.observability import setup_logging
from tests.fakes import FakeChatModel


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def captured() -> io.StringIO:
    stream = io.StringIO()
    setup_logging(level=logging.INFO, stream=stream)
    return stream


def events(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


@tool
def busca_fake(query: str) -> str:
    """Busca determinística para teste."""
    return f"resultado para {query}"


@tool
def busca_quebrada(query: str) -> str:
    """Tool que sempre falha, para exercitar o caminho de erro."""
    raise RuntimeError("provedor fora do ar")


def state_with(**overrides: object) -> AgentState:
    state = new_state(task="compare custo de GPU e API", run_id="r1")
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def tool_call_message(name: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": "preço de GPU"}, "id": "call-1"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


async def test_collects_findings_with_sources(settings: Settings) -> None:
    model = FakeChatModel(
        responses=[
            "sem tool, já sei responder",
            ResearchOutput(
                findings=[
                    Finding(content="A100 custa X por hora", source="https://a", agent="research")
                ]
            ),
        ]
    )
    node = make_research_node(model, [], settings)
    result = await node(state_with())
    assert len(result["findings"]) == 1
    assert result["findings"][0].source == "https://a"
    assert result["findings"][0].agent == "research"


async def test_runs_the_tool_the_model_asked_for(settings: Settings, captured: io.StringIO) -> None:
    model = FakeChatModel(
        responses=[
            tool_call_message("busca_fake"),
            "agora sei",
            ResearchOutput(
                findings=[Finding(content="achado", source="https://a", agent="research")]
            ),
        ]
    )
    tools: list[BaseTool] = [busca_fake]
    node = make_research_node(model, tools, settings)
    await node(state_with())
    calls = [event for event in events(captured) if event["event"] == "tool_call"]
    assert calls[0]["tool"] == "busca_fake"


async def test_tool_failure_becomes_an_event_not_a_crash(
    settings: Settings, captured: io.StringIO
) -> None:
    model = FakeChatModel(
        responses=[
            tool_call_message("busca_quebrada"),
            "não consegui buscar",
            ResearchOutput(findings=[], notes="a busca falhou"),
        ]
    )
    tools: list[BaseTool] = [busca_quebrada]
    node = make_research_node(model, tools, settings)
    result = await node(state_with())
    assert result["findings"] == []
    errors = [event for event in events(captured) if event["event"] == "error"]
    assert any(event.get("tool") == "busca_quebrada" for event in errors)


async def test_tool_rounds_are_capped(settings: Settings) -> None:
    """Sem teto, o agente pede tool para sempre."""
    responses: list[object] = [tool_call_message("busca_fake") for _ in range(10)]
    responses.append(ResearchOutput(findings=[]))
    model = FakeChatModel(responses=responses)
    tools: list[BaseTool] = [busca_fake]
    node = make_research_node(model, tools, settings)
    await node(state_with())
    assert model.call_count == settings.MAX_TOOL_ROUNDS + 1


async def test_findings_without_source_are_dropped(settings: Settings) -> None:
    """Afirmação sem fonte não vira Finding."""
    model = FakeChatModel(
        responses=[
            "pronto",
            ResearchOutput(
                findings=[
                    Finding(content="afirmação solta", source="   ", agent="research"),
                    Finding(content="com fonte", source="https://a", agent="research"),
                ]
            ),
        ]
    )
    node = make_research_node(model, [], settings)
    result = await node(state_with())
    assert [f.content for f in result["findings"]] == ["com fonte"]


async def test_counts_tokens_across_every_call(settings: Settings) -> None:
    model = FakeChatModel(
        responses=["pronto", ResearchOutput(findings=[])],
        input_tokens=40,
        output_tokens=10,
    )
    node = make_research_node(model, [], settings)
    assert (await node(state_with()))["tokens_used"] == 100


async def test_the_critic_gaps_reach_the_researcher(settings: Settings) -> None:
    model = FakeChatModel(responses=["pronto", ResearchOutput(findings=[])])
    node = make_research_node(model, [], settings)
    await node(state_with(critiques=[Critique(verdict="refinar", gaps=["falta o preço da GPU"])]))
    first_prompt = str(model.calls[0][1].content)
    assert "falta o preço da GPU" in first_prompt


def a_tool_call(name: str, call_id: str = "call_1", tokens: int = 200) -> AIMessage:
    """Uma AIMessage pedindo tool, com uso declarado.

    O fake devolve `AIMessage` como veio, sem inventar `usage_metadata`, então o
    contador do nó só enxerga o que a própria mensagem carrega.
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": "x"}, "id": call_id, "type": "tool_call"}],
        usage_metadata={
            "input_tokens": tokens // 2,
            "output_tokens": tokens - tokens // 2,
            "total_tokens": tokens,
        },
    )


def dangling_tool_calls(messages: list[Any]) -> list[str]:
    """Ids de tool call que ficaram sem `ToolMessage`. O provider recusa a conversa assim."""
    asked = [
        str(call["id"])
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["id"]
    ]
    answered = {message.tool_call_id for message in messages if isinstance(message, ToolMessage)}
    return [call_id for call_id in asked if call_id not in answered]


def test_the_helper_closes_every_call_it_was_given() -> None:
    closed = unanswered(a_tool_call("busca_fake", "call_abc"))
    assert [message.tool_call_id for message in closed] == ["call_abc"]
    assert closed[0].status == "error"
    assert "orçamento" in closed[0].content


async def test_a_budget_stop_leaves_no_tool_call_unanswered(settings: Settings) -> None:
    """Foi assim que uma run real morreu com 400 depois de gastar 58 mil tokens."""
    broke = settings.model_copy(update={"BUDGET_TOKENS_PER_RUN": 1})
    model = FakeChatModel(
        responses=[a_tool_call("busca_fake"), ResearchOutput(findings=[])],
        input_tokens=100,
        output_tokens=100,
    )
    node = make_research_node(model, [busca_fake], broke)
    await node(new_state(task="t", run_id="r1"))

    extraction = model.calls[-1]
    assert dangling_tool_calls(extraction) == [], "o extractor recebeu conversa inválida"
