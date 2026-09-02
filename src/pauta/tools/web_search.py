"""Busca web via Tavily. Wrapper fino: trocar de provedor é um arquivo."""

from functools import lru_cache
from typing import cast

from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch

from ..config import get_settings

#: Quantos resultados a busca devolve por chamada. Mais que isso vira ruído no
#: contexto e custa token sem melhorar a resposta.
WEB_SEARCH_MAX_RESULTS = 5


class MissingTavilyKey(RuntimeError):
    """`TAVILY_API_KEY` não está no ambiente."""


@lru_cache
def get_web_search_tool() -> BaseTool:
    """Instância única da tool de busca."""
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        raise MissingTavilyKey("TAVILY_API_KEY não definida; a busca web não pode ser usada")
    tool = TavilySearch(
        max_results=WEB_SEARCH_MAX_RESULTS,
        tavily_api_key=settings.TAVILY_API_KEY,
    )
    return cast(BaseTool, tool)


def reset_web_search_cache() -> None:
    get_web_search_tool.cache_clear()
