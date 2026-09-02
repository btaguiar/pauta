"""Modelo por papel (ADR 007). Nenhum agente instancia cliente."""

from functools import lru_cache
from typing import Literal, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .config import get_settings

Role = Literal["supervisor", "research", "analyst", "critic", "writer"]

#: Qual variável de ambiente atende cada papel. Roteamento e crítica degradam
#: muito em modelo pequeno; pesquisa e redação não exigem raciocínio profundo.
ROLE_ENV: dict[Role, str] = {
    "supervisor": "MODEL_ROUTER",
    "critic": "MODEL_CRITIC",
    "research": "MODEL_WORKER",
    "analyst": "MODEL_WORKER",
    "writer": "MODEL_WORKER",
}


def model_name_for(role: Role) -> str:
    """Nome do modelo configurado para o papel, direto do ambiente."""
    if role not in ROLE_ENV:
        raise ValueError(f"papel desconhecido: {role!r}; esperados {sorted(ROLE_ENV)}")
    name = cast(str, getattr(get_settings(), ROLE_ENV[role]))
    if not name.strip():
        raise ValueError(f"{ROLE_ENV[role]} está vazio; defina no .env antes de rodar o grafo")
    return name


@lru_cache
def get_model(role: Role) -> BaseChatModel:
    """Instância única por papel, com `temperature=0` para o eval ser reprodutível."""
    return init_chat_model(model_name_for(role), temperature=0)


def reset_model_cache() -> None:
    """Descarta as instâncias. Usado quando o ambiente muda entre execuções do eval."""
    get_model.cache_clear()
