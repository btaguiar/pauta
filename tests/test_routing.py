import pytest

from pauta.config import Settings, get_settings
from pauta.graph.routing import enforce_rules, fallback_route, forced_route
from pauta.graph.state import AgentState, Critique, Finding, NextStep, new_state


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def state_with(**overrides: object) -> AgentState:
    state = new_state(task="vale a pena migrar?", run_id="r1")
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_fallback_asks_for_research_when_there_is_nothing(settings: Settings) -> None:
    assert fallback_route(state_with()) == "research"


def test_fallback_asks_for_critic_after_findings(settings: Settings) -> None:
    state = state_with(findings=[Finding(content="a", source="s", agent="research")])
    assert fallback_route(state) == "critic"


def test_fallback_ends_at_writer(settings: Settings) -> None:
    state = state_with(
        findings=[Finding(content="a", source="s", agent="research")],
        critiques=[Critique(verdict="ok")],
    )
    assert fallback_route(state) == "writer"


def test_budget_overrun_forces_the_writer(settings: Settings) -> None:
    state = state_with(tokens_used=settings.BUDGET_TOKENS_PER_RUN)
    assert forced_route(state, settings) == "writer"


def test_step_limit_forces_the_writer(settings: Settings) -> None:
    state = state_with(iteration=settings.MAX_SUPERVISOR_STEPS)
    assert forced_route(state, settings) == "writer"


def test_finished_run_goes_to_end(settings: Settings) -> None:
    assert forced_route(state_with(final_report="pronto"), settings) == "END"


def test_no_limit_reached_leaves_the_decision_to_the_model(settings: Settings) -> None:
    assert forced_route(state_with(), settings) is None


def test_writer_never_runs_before_the_critic(settings: Settings) -> None:
    """Regra 1 do prompt, aplicada em código e não na confiança."""
    state = state_with(findings=[Finding(content="a", source="s", agent="research")])
    assert enforce_rules(state, "writer", settings) == "critic"


def test_writer_does_not_write_about_nothing(settings: Settings) -> None:
    """Sem nenhuma descoberta, o pedido de writer volta para research."""
    assert enforce_rules(state_with(), "writer", settings) == "research"


def test_empty_writer_rule_yields_to_the_hard_limits(settings: Settings) -> None:
    """Orçamento estourado decide antes, e aí o writer redige com o que houver."""
    state = state_with(tokens_used=settings.BUDGET_TOKENS_PER_RUN)
    assert forced_route(state, settings) == "writer"


def test_without_a_research_node_the_empty_writer_rule_is_off(settings: Settings) -> None:
    available: frozenset[NextStep] = frozenset({"writer", "END"})
    assert enforce_rules(state_with(), "writer", settings, available) == "writer"


def test_critic_is_not_called_again_after_the_loop_limit(settings: Settings) -> None:
    state = state_with(critic_loops=settings.MAX_CRITIC_LOOPS)
    assert enforce_rules(state, "critic", settings) == "writer"


def test_end_without_report_becomes_writer(settings: Settings) -> None:
    assert enforce_rules(state_with(), "END", settings) == "writer"


def test_valid_route_passes_through(settings: Settings) -> None:
    assert enforce_rules(state_with(), "research", settings) == "research"


def capped(settings: Settings, loops: int) -> Settings:
    return settings.model_copy(update={"MAX_CRITIC_LOOPS": loops})


def test_a_zero_loop_limit_turns_the_critic_off(settings: Settings) -> None:
    """O `ge=0` da configuração promete que zero desliga o crítico."""
    state = state_with(findings=[Finding(content="a", source="s", agent="research")])
    assert enforce_rules(state, "writer", capped(settings, 0)) == "writer"


def test_the_first_loop_still_runs_when_the_limit_is_one(settings: Settings) -> None:
    state = state_with(findings=[Finding(content="a", source="s", agent="research")])
    assert enforce_rules(state, "writer", capped(settings, 1)) == "critic"


def test_the_second_loop_is_refused_when_the_limit_is_one(settings: Settings) -> None:
    state = state_with(
        findings=[Finding(content="a", source="s", agent="research")],
        critiques=[Critique(verdict="refinar")],
        critic_loops=1,
    )
    assert enforce_rules(state, "critic", capped(settings, 1)) == "writer"


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_the_exhausted_critic_is_never_proposed_again(settings: Settings, limit: int) -> None:
    """Esgotado o limite, nenhuma regra devolve o crítico, venha de onde vier a proposta."""
    resolved = capped(settings, limit)
    state = state_with(
        findings=[Finding(content="a", source="s", agent="research")],
        critic_loops=limit,
    )
    assert enforce_rules(state, "writer", resolved) != "critic"
    assert enforce_rules(state, "critic", resolved) != "critic"
