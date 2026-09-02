import io
import json
import logging

import pytest

from pauta.agents.critic import make_critic_node, render_material
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


def state_with(**overrides: object) -> AgentState:
    state = new_state(task="vale a pena migrar?", run_id="r1")
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def with_material(**overrides: object) -> AgentState:
    return state_with(
        findings=[Finding(content="custo mensal é 300", source="https://a", agent="research")],
        **overrides,
    )


async def test_records_the_verdict_and_counts_the_loop(settings: Settings) -> None:
    model = FakeChatModel(responses=[Critique(verdict="ok")])
    node = make_critic_node(model, settings)
    result = await node(state=with_material())
    assert result["critiques"][0].verdict == "ok"
    assert result["critic_loops"] == 1


async def test_gaps_and_delegation_survive(settings: Settings) -> None:
    verdict = Critique(
        verdict="refinar",
        gaps=["falta o preço por token"],
        delegate_to=["research"],
    )
    node = make_critic_node(FakeChatModel(responses=[verdict]), settings)
    result = await node(state=with_material())
    assert result["critiques"][0].gaps == ["falta o preço por token"]
    assert result["critiques"][0].delegate_to == ["research"]


async def test_a_critic_that_cannot_answer_does_not_approve(
    settings: Settings, captured: io.StringIO
) -> None:
    """Crítico mudo não aprova por omissão."""
    node = make_critic_node(FakeChatModel(responses=[ValueError("json quebrado")]), settings)
    result = await node(state=with_material())
    assert result["critiques"][0].verdict == "refinar"
    assert any(event["event"] == "error" for event in events(captured))


async def test_approving_empty_material_is_overridden(settings: Settings) -> None:
    """Aprovar o vazio é o modo de falha que a ADR 002 existe para pegar."""
    node = make_critic_node(FakeChatModel(responses=[Critique(verdict="ok")]), settings)
    result = await node(state=state_with())
    assert result["critiques"][0].verdict == "refinar"
    assert result["critiques"][0].delegate_to == ["research"]


async def test_emits_a_critique_event_with_the_loop_count(
    settings: Settings, captured: io.StringIO
) -> None:
    node = make_critic_node(FakeChatModel(responses=[Critique(verdict="refinar")]), settings)
    await node(state=with_material(critic_loops=1))
    critique = next(event for event in events(captured) if event["event"] == "critique")
    assert critique["loop"] == 2
    assert critique["max_loops"] == settings.MAX_CRITIC_LOOPS


async def test_counts_its_own_tokens(settings: Settings) -> None:
    model = FakeChatModel(responses=[Critique(verdict="ok")], input_tokens=200, output_tokens=50)
    node = make_critic_node(model, settings)
    assert (await node(state=with_material()))["tokens_used"] == 250


def test_the_critic_sees_its_previous_gaps() -> None:
    state = with_material(critiques=[Critique(verdict="refinar", gaps=["faltou a fonte"])])
    rendered = render_material(state)
    assert "faltou a fonte" in rendered
    assert "já avaliou este material 1 vez" in rendered


def test_material_is_shown_with_agent_and_source() -> None:
    rendered = render_material(with_material())
    assert "[research]" in rendered
    assert "fonte: https://a" in rendered
