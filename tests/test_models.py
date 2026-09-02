import pytest

from pauta.models import ROLE_ENV, get_model, model_name_for, reset_model_cache


def test_every_role_maps_to_an_environment_variable() -> None:
    assert set(ROLE_ENV) == {"supervisor", "research", "analyst", "critic", "writer"}
    assert ROLE_ENV["supervisor"] == "MODEL_ROUTER"
    assert ROLE_ENV["critic"] == "MODEL_CRITIC"
    assert ROLE_ENV["research"] == ROLE_ENV["analyst"] == ROLE_ENV["writer"] == "MODEL_WORKER"


def test_reads_the_name_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_ROUTER", "provider:modelo-de-teste")
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    assert model_name_for("supervisor") == "provider:modelo-de-teste"


def test_empty_value_fails_with_a_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_CRITIC", "   ")
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    with pytest.raises(ValueError, match="MODEL_CRITIC"):
        model_name_for("critic")


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="papel desconhecido"):
        model_name_for("redator")  # type: ignore[arg-type]


def test_no_model_name_appears_in_source() -> None:
    """Nenhum modelo hardcoded: models.py só conhece nomes de variável."""
    from pathlib import Path

    import pauta.models as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for marker in ("gpt-", "claude-", "gemini-", "llama", "o1-", "o3-"):
        assert marker not in source.lower(), f"nome de modelo no código: {marker}"


def test_get_model_is_cached_per_role(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_init(model: str, **kwargs: object) -> object:
        calls.append(model)
        return object()

    monkeypatch.setattr("pauta.models.init_chat_model", fake_init)
    reset_model_cache()
    get_model("research")
    get_model("research")
    assert calls == ["fake-worker"]
