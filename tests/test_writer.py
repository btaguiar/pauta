import io
import json
import logging

import pytest

from pauta.agents.writer import UNVALIDATED_NOTICE, make_writer_node, render_material
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


async def test_writes_the_report_and_ends_the_run(settings: Settings) -> None:
    model = FakeChatModel(responses=["Briefing: migrar não compensa hoje."])
    node = make_writer_node(model, settings)
    result = await node(state_with(critiques=[Critique(verdict="ok")]))
    assert result["final_report"].startswith("Briefing")
    assert result["next_agent"] == "END"


async def test_asks_for_caveats_when_the_critic_did_not_approve(settings: Settings) -> None:
    model = FakeChatModel(responses=["Briefing com ressalvas."])
    node = make_writer_node(model, settings)
    await node(state_with(critiques=[Critique(verdict="refinar", gaps=["falta fonte"])]))
    system_prompt = str(model.calls[0][0].content)
    assert UNVALIDATED_NOTICE in system_prompt


async def test_does_not_ask_for_caveats_when_approved(settings: Settings) -> None:
    model = FakeChatModel(responses=["Briefing limpo."])
    node = make_writer_node(model, settings)
    await node(state_with(critiques=[Critique(verdict="ok")]))
    system_prompt = str(model.calls[0][0].content)
    assert UNVALIDATED_NOTICE not in system_prompt


async def test_material_carries_every_source() -> None:
    state = state_with(
        findings=[
            Finding(content="preço caiu 10%", source="https://a", agent="research"),
            Finding(content="custo mensal é 300", source="cálculo", agent="analyst"),
        ]
    )
    rendered = render_material(state)
    assert "fonte: https://a" in rendered
    assert "fonte: cálculo" in rendered


async def test_empty_material_is_stated_not_hidden() -> None:
    assert "nenhuma descoberta foi registrada" in render_material(state_with())


async def test_human_feedback_reaches_the_writer() -> None:
    rendered = render_material(state_with(hitl_feedback="foque no custo de saída"))
    assert "foque no custo de saída" in rendered


async def test_emits_final_and_usage(settings: Settings, captured: io.StringIO) -> None:
    model = FakeChatModel(responses=["Briefing."], input_tokens=30, output_tokens=20)
    node = make_writer_node(model, settings)
    await node(state_with(tokens_used=1000, critiques=[Critique(verdict="ok")]))
    kinds = [event["event"] for event in events(captured)]
    assert "final" in kinds
    assert "usage" in kinds
    usage = next(event for event in events(captured) if event["event"] == "usage")
    assert usage["tokens_used"] == 1050
