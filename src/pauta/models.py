"""Modelo por papel (ADR 007). Nenhum agente instancia cliente."""

from functools import lru_cache
from typing import Any, Literal, cast

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import get_settings

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


@lru_cache
def get_model(role: Role) -> GatewayChatModel:
    """Instância única por papel, com `temperature=0` para o eval ser reprodutível.

    O provider é declarado, nunca inferido: os ids deste gateway são nus
    (`qwen3.8-flash`, `glm-5.3`, `deepseek-v4-pro`), sem a casa de origem no
    nome, então não há prefixo de onde o LangChain deduzir coisa alguma.
    """
    settings = get_settings()
    if not settings.CHAT_API_KEY:
        raise MissingGatewayKey("CHAT_API_KEY não definida; nenhum modelo pode ser criado")
    name = model_name_for(role)
    return GatewayChatModel(
        model=name,
        temperature=0,
        base_url=settings.CHAT_BASE_URL,
        api_key=SecretStr(settings.CHAT_API_KEY),
        structured_method=structured_method_for(name),
    )


def reset_model_cache() -> None:
    """Descarta as instâncias. Usado quando o ambiente muda entre execuções do eval."""
    get_model.cache_clear()
