"""Modelo por papel (ADR 007). Nenhum agente instancia cliente."""

import threading
from collections.abc import Generator
from functools import lru_cache
from typing import Any, Literal, cast

import httpx
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import get_settings

#: Onde o `thought_signature` de cada tool call fica guardado na mensagem.
#: O Gemini devolve a assinatura em `tool_calls[].extra_content` e recusa o
#: turno seguinte (400, "missing a thought_signature") se ela não voltar. O
#: `ChatOpenAI` descarta o campo, então o modelo o guarda aqui, por id da call,
#: e o reinjeta no payload (ADR 014).
TOOL_CALL_EXTRA_KEY = "tool_call_extra_content"

#: Escopo do token OAuth do Vertex.
GOOGLE_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

Role = Literal["supervisor", "research", "analyst", "critic", "writer", "judge"]

#: Qual variável de ambiente atende cada papel. Roteamento e crítica degradam
#: muito em modelo pequeno; pesquisa e redação não exigem raciocínio profundo.
#: O juiz do eval é papel como os outros: ele também não instancia cliente por
#: conta própria, e a metodologia pede que ele seja de outro provider.
ROLE_ENV: dict[Role, str] = {
    "supervisor": "MODEL_ROUTER",
    "critic": "MODEL_CRITIC",
    "research": "MODEL_WORKER",
    "analyst": "MODEL_WORKER",
    "writer": "MODEL_WORKER",
    "judge": "JUDGE_MODEL",
}

#: Como cada modelo medido aceita saída estruturada no gateway de chat. Sonda
#: de 2026-09-26: qwen3.8-max recusa `tool_choice=required` e só honra
#: `response_format` com `json_schema`; glm-5.3 e qwen3.6-flash honram
#: `function_calling`. Isto é capacidade medida do id, não escolha de modelo:
#: a escolha continua vindo do ambiente. Id fora da tabela usa
#: `function_calling`, que é o método que o gateway aceita em mais modelos.
#: Sonda de 2026-11-14 no Vertex (endpoint compatível com OpenAI, região
#: global): os três modelos do Gemini 4 honram os dois métodos, então ficam fora
#: da tabela e caem no default `function_calling` (ADR 014).
STRUCTURED_METHOD_BY_MODEL: dict[str, str] = {
    "glm-5.3": "function_calling",
    "qwen3.6-flash": "function_calling",
    "qwen3.8-max": "json_schema",
}


def structured_method_for(model: str) -> str:
    """O método de saída estruturada medido para o id, ou o mais compatível."""
    return STRUCTURED_METHOD_BY_MODEL.get(model, "function_calling")


def model_name_for(role: Role) -> str:
    """Nome do modelo configurado para o papel, direto do ambiente."""
    if role not in ROLE_ENV:
        raise ValueError(f"papel desconhecido: {role!r}; esperados {sorted(ROLE_ENV)}")
    # `JUDGE_MODEL` é opcional no `Settings`, então aqui pode chegar `None`.
    name = cast(str | None, getattr(get_settings(), ROLE_ENV[role]))
    if not name or not name.strip():
        raise ValueError(f"{ROLE_ENV[role]} está vazio; defina no .env antes de rodar o grafo")
    return name


class MissingGatewayKey(RuntimeError):
    """`CHAT_API_KEY` não está no ambiente."""


class GoogleBearerAuth(httpx.Auth):
    """Bearer com o token das credenciais padrão do Google, renovado ao vencer.

    O `api_key` do cliente da OpenAI é fixo, e o token do Vertex vence em uma
    hora. Este `httpx.Auth` sobrescreve o cabeçalho em cada requisição, com a
    renovação protegida por lock porque o grafo chama modelos em paralelo.
    """

    def __init__(self, credentials: Any = None) -> None:
        if credentials is None:
            import google.auth

            credentials, _ = google.auth.default(scopes=list(GOOGLE_SCOPES))
        self._credentials = credentials
        self._lock = threading.Lock()

    def _token(self) -> str:
        with self._lock:
            if not self._credentials.valid:
                from google.auth.transport.requests import Request

                self._credentials.refresh(Request())
            return str(self._credentials.token)

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._token()}"
        yield request


class GatewayChatModel(ChatOpenAI):
    """`ChatOpenAI` para o gateway, com o método estruturado medido por id.

    O `with_structured_output` de `ChatOpenAI` assume que o destino honra a API
    da OpenAI inteira, e o gateway não honra: um mesmo plano serve modelos que
    só aceitam `json_schema` e modelos que só aceitam `function_calling`. Aqui
    o default de `method` é o medido para o próprio id; quem chama pode passar
    `method=` explícito e vencer a tabela.
    """

    structured_method: str = "function_calling"

    def with_structured_output(
        self,
        schema: Any = None,
        *,
        method: str | None = None,
        include_raw: bool = False,
        strict: bool | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        return super().with_structured_output(
            schema,
            method=method or self.structured_method,  # type: ignore[arg-type]
            include_raw=include_raw,
            strict=strict,
            tools=tools,
            **kwargs,
        )

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Guarda o `extra_content` de cada tool call, que o `ChatOpenAI` descarta."""
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, raw.get("choices") or [], strict=False):
            calls = (choice.get("message") or {}).get("tool_calls") or []
            extra = {c["id"]: c["extra_content"] for c in calls if c.get("extra_content")}
            if extra:
                generation.message.additional_kwargs[TOOL_CALL_EXTRA_KEY] = extra
        return result

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Devolve o `extra_content` guardado às tool calls do histórico."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        extra: dict[str, Any] = {}
        for message in self._convert_input(input_).to_messages():
            if isinstance(message, AIMessage):
                extra.update(message.additional_kwargs.get(TOOL_CALL_EXTRA_KEY) or {})
        if extra:
            for message_dict in payload.get("messages", []):
                for call in message_dict.get("tool_calls") or []:
                    if call.get("id") in extra:
                        call["extra_content"] = extra[call["id"]]
        return payload


@lru_cache
def get_model(role: Role) -> GatewayChatModel:
    """Instância única por papel, com `temperature=0` para o eval ser reprodutível.

    O provider é declarado, nunca inferido: os ids deste gateway são nus
    (`qwen3.8-flash`, `glm-5.3`, `deepseek-v4-pro`), sem a casa de origem no
    nome, então não há prefixo de onde o LangChain deduzir coisa alguma.
    """
    settings = get_settings()
    name = model_name_for(role)
    limits: dict[str, Any] = {
        "timeout": request_timeout_for(role),
        "max_retries": settings.MODEL_MAX_RETRIES,
    }
    if settings.CHAT_AUTH == "google":
        # O `api_key` é só um marcador: o `GoogleBearerAuth` reescreve o
        # cabeçalho em cada requisição, nos dois clientes.
        auth = GoogleBearerAuth()
        return GatewayChatModel(
            model=name,
            temperature=0,
            base_url=settings.CHAT_BASE_URL,
            api_key=SecretStr("google-adc"),
            http_client=httpx.Client(auth=auth),
            http_async_client=httpx.AsyncClient(auth=auth),
            structured_method=structured_method_for(name),
            **limits,
        )
    if not settings.CHAT_API_KEY:
        raise MissingGatewayKey(
            "CHAT_API_KEY (ou API_KEY_ALIBABA) não definida; nenhum modelo pode ser criado"
        )
    return GatewayChatModel(
        model=name,
        temperature=0,
        base_url=settings.CHAT_BASE_URL,
        api_key=SecretStr(settings.CHAT_API_KEY),
        structured_method=structured_method_for(name),
        **limits,
    )


def request_timeout_for(role: Role) -> float:
    """Teto de uma chamada ao modelo do papel; o supervisor tem o seu."""
    settings = get_settings()
    if role == "supervisor":
        return settings.MODEL_REQUEST_TIMEOUT_SUPERVISOR_S
    return settings.MODEL_REQUEST_TIMEOUT_S


def reset_model_cache() -> None:
    """Descarta as instâncias. Usado quando o ambiente muda entre execuções do eval."""
    get_model.cache_clear()
