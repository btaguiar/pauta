import io
import json
import logging

import pytest

from pauta.agents.analyst import AnalystOutput, make_analyst_node
from pauta.config import Settings, get_settings
from pauta.graph.state import AgentState, Finding, new_state
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
    state = new_state(task="quanto custa rodar por mês", run_id="r1")
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


async def test_collects_calculated_findings(settings: Settings) -> None:
    model = FakeChatModel(
        responses=[
            "conta feita na calculadora",
            AnalystOutput(
                findings=[
                    Finding(
                        content="custo mensal = 4090 USD",
                        source="cálculo: 0.55*744",
                        agent="analyst",
                    )
                ]
            ),
        ]
    )
    node = make_analyst_node(model, [], settings)
    result = await node(state_with(findings=[]))
    assert [f.content for f in result["findings"]] == ["custo mensal = 4090 USD"]
    assert result["findings"][0].agent == "analyst"


async def test_a_broken_extraction_becomes_an_event_not_a_crash(
    settings: Settings, captured: io.StringIO
) -> None:
    """Medido no gateway em 2026-09-26: JSON fora do schema no extrator derrubava
    a run. Parse quebrado registra zero resultados e a run continua."""
    model = FakeChatModel(responses=["pronto", ValueError("json fora do schema")])
    node = make_analyst_node(model, [], settings)
    result = await node(state_with(findings=[]))
    assert result["findings"] == []
    errors = [event for event in events(captured) if event["event"] == "error"]
    assert any("extração estruturada falhou" in str(event.get("error", "")) for event in errors)
