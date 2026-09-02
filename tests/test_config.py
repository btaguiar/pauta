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


def test_blank_optional_values_mean_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """O .env.example deixa opcionais em branco; branco não pode quebrar o parse."""
    monkeypatch.setenv("COST_PER_MTOK_USD", "")
    monkeypatch.setenv("JUDGE_MODEL", "   ")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.COST_PER_MTOK_USD is None
    assert settings.JUDGE_MODEL is None
    assert settings.TAVILY_API_KEY is None


def test_a_filled_optional_still_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COST_PER_MTOK_USD", "1.75")
    get_settings.cache_clear()
    assert Settings(_env_file=None).COST_PER_MTOK_USD == 1.75  # type: ignore[call-arg]


def test_the_shipped_env_example_parses() -> None:
    """O espelho vazio precisa carregar quando os obrigatórios vêm do ambiente."""
    settings = Settings(_env_file=".env.example")  # type: ignore[call-arg]
    assert settings.COST_PER_MTOK_USD is None
    assert settings.DATABASE_URL.startswith("postgresql://")
