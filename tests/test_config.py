import pytest
from pydantic import ValidationError

from pauta.config import Settings, get_settings


def test_defaults_come_from_section_8_3() -> None:
    settings = get_settings()
    assert settings.MAX_SUPERVISOR_STEPS == 8
    assert settings.MAX_CRITIC_LOOPS == 2
    assert settings.BUDGET_TOKENS_PER_RUN == 60_000
    assert settings.RETRIEVER_TOP_K == 5
    assert settings.HITL_MODE == "auto"
    assert settings.LANGSMITH_TRACING is False


def test_role_models_have_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Um default de modelo no código é um modelo hardcoded por outro nome."""
    monkeypatch.delenv("MODEL_ROUTER", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "MODEL_ROUTER" in str(excinfo.value)


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CRITIC_LOOPS", "5")
    monkeypatch.setenv("HITL_MODE", "interrupt")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.MAX_CRITIC_LOOPS == 5
    assert settings.HITL_MODE == "interrupt"


def test_rejects_impossible_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_SUPERVISOR_STEPS", "0")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_hitl_mode_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HITL_MODE", "talvez")
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_database_url_matches_compose_default() -> None:
    """Se este valor mudar, o docker-compose.yml muda no mesmo commit."""
    assert get_settings().DATABASE_URL == "postgresql://pauta:pauta@localhost:5432/pauta"
