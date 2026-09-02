import io
import json
import logging

import pytest

from pauta.agents.supervisor import Router, make_supervisor_node, render_state
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


async def test_routes_to_what_the_model_decided(settings: Settings) -> None:
    model = FakeChatModel(responses=[Router(next="research", rationale="falta fonte")])
    node = make_supervisor_node(model, settings)
    result = await node(state_with())
    assert result["next_agent"] == "research"
    assert result["iteration"] == 1


async def test_counts_the_tokens_the_call_spent(settings: Settings) -> None:
    model = FakeChatModel(
        responses=[Router(next="research", rationale="falta fonte")],
        input_tokens=90,
        output_tokens=10,
    )
    node = make_supervisor_node(model, settings)
    assert (await node(state_with()))["tokens_used"] == 100


async def test_falls_back_to_the_deterministic_rule_after_two_failures(
    settings: Settings, captured: io.StringIO
) -> None:
    """O grafo continua andando quando o LLM que decide o caminho falha."""
    model = FakeChatModel(responses=[ValueError("json quebrado"), ValueError("json quebrado")])
    node = make_supervisor_node(model, settings)
    result = await node(state_with())
    assert result["next_agent"] == "research"
    assert model.call_count == 2
    errors = [event for event in events(captured) if event["event"] == "error"]
    assert len(errors) == 2


async def test_does_not_call_the_model_when_the_budget_is_gone(settings: Settings) -> None:
    model = FakeChatModel(responses=[])
    node = make_supervisor_node(model, settings)
    state = state_with(tokens_used=settings.BUDGET_TOKENS_PER_RUN)
    assert (await node(state))["next_agent"] == "writer"
    assert model.call_count == 0


async def test_does_not_call_the_model_when_the_steps_are_gone(settings: Settings) -> None:
    model = FakeChatModel(responses=[])
    node = make_supervisor_node(model, settings)
    state = state_with(iteration=settings.MAX_SUPERVISOR_STEPS)
    assert (await node(state))["next_agent"] == "writer"
    assert model.call_count == 0


async def test_model_cannot_skip_the_critic(settings: Settings) -> None:
    model = FakeChatModel(responses=[Router(next="writer", rationale="acho que ja da")])
    node = make_supervisor_node(model, settings)
    state = state_with(findings=[Finding(content="a", source="s", agent="research")])
    assert (await node(state))["next_agent"] == "critic"


async def test_model_cannot_reopen_the_critic_loop(settings: Settings) -> None:
    model = FakeChatModel(responses=[Router(next="critic", rationale="mais uma volta")])
    node = make_supervisor_node(model, settings)
    state = state_with(
        findings=[Finding(content="a", source="s", agent="research")],
        critiques=[Critique(verdict="refinar"), Critique(verdict="refinar")],
        critic_loops=settings.MAX_CRITIC_LOOPS,
    )
    assert (await node(state))["next_agent"] == "writer"


async def test_rationale_reaches_the_event_stream(
    settings: Settings, captured: io.StringIO
) -> None:
    model = FakeChatModel(responses=[Router(next="research", rationale="falta preço de GPU")])
    node = make_supervisor_node(model, settings)
    await node(state_with())
    ends = [event for event in events(captured) if event["event"] == "node_end"]
    assert ends[-1]["rationale"] == "falta preço de GPU"


async def test_the_model_sees_numbers_not_prose(settings: Settings) -> None:
    state = state_with(
        findings=[Finding(content="a", source="s", agent="research")],
        critiques=[Critique(verdict="refinar", gaps=["falta fonte do preço"])],
        iteration=3,
        tokens_used=12_000,
    )
    rendered = render_state(state, settings)
    assert "Descobertas: 1" in rendered
    assert "último veredito: refinar" in rendered
    assert "falta fonte do preço" in rendered
    assert f"Iterações usadas: 3/{settings.MAX_SUPERVISOR_STEPS}" in rendered
    assert f"Tokens usados: 12000/{settings.BUDGET_TOKENS_PER_RUN}" in rendered
