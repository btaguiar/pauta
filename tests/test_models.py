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
    with pytest.raises(MissingGatewayKey, match="CHAT_API_KEY") as excinfo:
        get_model("writer")
    assert "API_KEY_ALIBABA" in str(excinfo.value)


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


def test_google_auth_needs_no_chat_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com `CHAT_AUTH=google` não há chave: o token sai das credenciais padrão."""
    import pauta.models as module

    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_AUTH", "google")
    monkeypatch.setenv("CHAT_BASE_URL", "https://vertex.exemplo/openapi")
    real_auth = module.GoogleBearerAuth
    monkeypatch.setattr(module, "GoogleBearerAuth", lambda: real_auth(_FakeCredentials()))
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    model = get_model("writer")
    assert model.openai_api_base == "https://vertex.exemplo/openapi"
    assert model.http_client is not None and model.http_async_client is not None


class _FakeCredentials:
    def __init__(self) -> None:
        self.valid = False
        self.token: str | None = None
        self.refreshes = 0

    def refresh(self, _request: object) -> None:
        self.refreshes += 1
        self.token = f"token-{self.refreshes}"
        self.valid = True


def test_google_bearer_refreshes_only_when_the_token_expires() -> None:
    import httpx

    from pauta.models import GoogleBearerAuth

    credentials = _FakeCredentials()
    auth = GoogleBearerAuth(credentials)

    def bearer() -> str:
        request = httpx.Request("POST", "https://vertex.exemplo/chat/completions")
        return next(auth.auth_flow(request)).headers["Authorization"]

    assert bearer() == "Bearer token-1"
    assert bearer() == "Bearer token-1", "token válido não é renovado"
    credentials.valid = False
    assert bearer() == "Bearer token-2"
    assert credentials.refreshes == 2


def test_tool_call_extra_content_survives_the_round_trip() -> None:
    """O Gemini recusa o turno seguinte (400) se o `thought_signature` de cada
    tool call não voltar. O `ChatOpenAI` o descarta; o `GatewayChatModel` guarda
    por id e reinjeta no payload (ADR 014)."""
    from langchain_core.messages import HumanMessage, ToolMessage
    from pydantic import SecretStr

    from pauta.models import TOOL_CALL_EXTRA_KEY, GatewayChatModel

    model = GatewayChatModel(model="m", api_key=SecretStr("k"))
    signature = {"google": {"thought_signature": "assinatura-opaca"}}
    response = {
        "id": "r1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": '{"expr": "17*23"}'},
                            "extra_content": signature,
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    ai = model._create_chat_result(response).generations[0].message
    assert ai.additional_kwargs[TOOL_CALL_EXTRA_KEY] == {"call-1": signature}

    history = [HumanMessage("quanto é 17*23?"), ai, ToolMessage("391", tool_call_id="call-1")]
    payload = model._get_request_payload(history)
    sent = [m for m in payload["messages"] if m.get("tool_calls")]
    assert sent[0]["tool_calls"][0]["extra_content"] == signature
    assert TOOL_CALL_EXTRA_KEY not in sent[0], "a chave interna não vai para a API"


def test_a_response_without_extra_content_is_untouched() -> None:
    from pydantic import SecretStr

    from pauta.models import TOOL_CALL_EXTRA_KEY, GatewayChatModel

    model = GatewayChatModel(model="m", api_key=SecretStr("k"))
    response = {
        "id": "r2",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "oi"}}
        ],
    }
    ai = model._create_chat_result(response).generations[0].message
    assert TOOL_CALL_EXTRA_KEY not in ai.additional_kwargs


def test_each_call_has_a_timeout_and_the_supervisor_a_shorter_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem teto por chamada, uma chamada presa no provider espera até o nó morrer."""
    monkeypatch.setenv("CHAT_API_KEY", "chave-fake")
    monkeypatch.setenv("CHAT_AUTH", "api_key")
    monkeypatch.setenv("MODEL_REQUEST_TIMEOUT_S", "150")
    monkeypatch.setenv("MODEL_REQUEST_TIMEOUT_SUPERVISOR_S", "30")
    monkeypatch.setenv("MODEL_MAX_RETRIES", "2")
    from pauta.config import get_settings

    get_settings.cache_clear()
    reset_model_cache()
    supervisor = get_model("supervisor")
    writer = get_model("writer")
    assert supervisor.request_timeout == 30
    assert writer.request_timeout == 150
    assert supervisor.max_retries == 2 and writer.max_retries == 2
