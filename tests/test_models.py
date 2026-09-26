import pytest

from pauta.config import CHAT_GATEWAY_URL
from pauta.models import (
    ROLE_ENV,
    MissingGatewayKey,
    get_model,
    model_name_for,
    reset_model_cache,
)


def test_every_role_maps_to_an_environment_variable() -> None:
    assert set(ROLE_ENV) == {"supervisor", "research", "analyst", "critic", "writer", "judge"}
    assert ROLE_ENV["supervisor"] == "MODEL_ROUTER"
    assert ROLE_ENV["critic"] == "MODEL_CRITIC"
    assert ROLE_ENV["judge"] == "JUDGE_MODEL"
    assert ROLE_ENV["research"] == ROLE_ENV["analyst"] == ROLE_ENV["writer"] == "MODEL_WORKER"


def test_the_judge_is_a_role_so_it_does_not_build_its_own_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regra 1 do repositório: todo LLM sai de `get_model`, o juiz do eval também."""
    monkeypatch.setenv("JUDGE_MODEL", "outra-casa/modelo-juiz")
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    assert model_name_for("judge") == "outra-casa/modelo-juiz"


def test_an_unset_judge_model_fails_with_a_clear_message() -> None:
    """`JUDGE_MODEL` é opcional no Settings, então aqui pode chegar `None`."""
    with pytest.raises(ValueError, match="JUDGE_MODEL"):
        model_name_for("judge")


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
    from pauta.models import GatewayChatModel

    monkeypatch.setenv("CHAT_API_KEY", "chave-de-teste")
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    first = get_model("research")
    assert first is get_model("research")
    assert first is not get_model("critic")
    assert isinstance(first, GatewayChatModel)


def test_the_gateway_is_wired_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider declarado e base_url do gateway, nunca inferidos do id."""
    from pauta.models import GatewayChatModel

    monkeypatch.setenv("CHAT_API_KEY", "chave-de-teste")
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    model = get_model("critic")
    assert isinstance(model, GatewayChatModel)
    assert model.model_name == "fake/critic"
    assert model.temperature == 0
    assert model.openai_api_base == CHAT_GATEWAY_URL


def test_no_key_fails_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    with pytest.raises(MissingGatewayKey, match="CHAT_API_KEY"):
        get_model("writer")


def test_structured_method_comes_from_the_measured_table() -> None:
    """Sonda de 2026-09-26 contra o gateway: glm-5.3 e qwen3.6-flash honram
    function_calling; qwen3.8-max recusa tool_choice=required e só aceita
    json_schema. Id fora da tabela cai no default do LangChain."""
    from pauta.models import structured_method_for

    assert structured_method_for("glm-5.3") == "function_calling"
    assert structured_method_for("qwen3.6-flash") == "function_calling"
    assert structured_method_for("qwen3.8-max") == "json_schema"
    assert structured_method_for("modelo-nunca-medido") == "function_calling"


def test_get_model_carries_the_measured_structured_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_API_KEY", "chave-de-teste")
    monkeypatch.setenv("MODEL_CRITIC", "qwen3.8-max")
    monkeypatch.setenv("MODEL_ROUTER", "glm-5.3")
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    assert get_model("critic").structured_method == "json_schema"
    assert get_model("supervisor").structured_method == "function_calling"


def test_structured_output_uses_the_measured_method_unless_overridden() -> None:
    """Sem method explícito, o método medido entra; com method explícito, ele vence."""
    from langchain_core.runnables import RunnableBinding, RunnableSequence
    from pydantic import BaseModel, SecretStr

    from pauta.models import GatewayChatModel

    class Schema(BaseModel):
        next: str

    measured = GatewayChatModel(model="m", api_key=SecretStr("k"), structured_method="json_schema")
    chain = measured.with_structured_output(Schema)
    assert isinstance(chain, RunnableSequence)
    first = chain.first
    assert isinstance(first, RunnableBinding)
    assert first.kwargs["response_format"] is not None

    explicit = GatewayChatModel(model="m", api_key=SecretStr("k"), structured_method="json_schema")
    bound_chain = explicit.with_structured_output(Schema, method="function_calling")
    assert isinstance(bound_chain, RunnableSequence)
    bound = bound_chain.first
    assert isinstance(bound, RunnableBinding)
    assert bound.kwargs.get("tool_choice") is not None
