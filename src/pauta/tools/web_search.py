"""Busca web. Um provedor por vez, escolhido por configuração.

Wrapper fino: trocar de provedor é este arquivo. Os dois provedores saem com o
mesmo nome de tool, `web_search`, para `tools_called` ser comparável entre
rodadas. Artefato antigo traz `tavily_search`; a quebra de nome fica registrada
no `EVALUATION.md`.

A falha da busca não é tratada aqui. `run_tools` em `agents/_common.py` captura
a exceção, emite o evento `error` e devolve `ToolMessage` de erro. Tratar dentro
da tool e devolver lista vazia esconderia a falha.
"""

from functools import lru_cache
from typing import Any, cast

from ddgs import DDGS
from langchain_core.tools import BaseTool, tool
from langchain_tavily import TavilySearch

from ..config import get_settings

#: Quantos resultados a busca devolve por chamada. Mais que isso vira ruído no
#: contexto e custa token sem melhorar a resposta. Não varia por ambiente, por
#: isso é constante de módulo e não campo de `Settings`.
WEB_SEARCH_MAX_RESULTS = 5

#: Nome estável da tool nos dois provedores.
WEB_SEARCH_TOOL_NAME = "web_search"


class MissingTavilyKey(RuntimeError):
    """`TAVILY_API_KEY` não está no ambiente."""


def _format_results(results: list[dict[str, Any]]) -> str:
    """Formata os resultados do `ddgs` para o agente conseguir citar a fonte.

    As chaves do `ddgs` são `title`, `href` e `body`. Um resultado por bloco,
    porque a regra 1 do prompt do research cobra url ou caminho em toda
    afirmação.
    """
    if not results:
        return "nenhum resultado"
    blocks = [
        f"title: {item.get('title', '')}\n"
        f"href: {item.get('href', '')}\n"
        f"body: {item.get('body', '')}"
        for item in results
    ]
    return "\n\n".join(blocks)


def _ddgs_tool() -> BaseTool:
    """Busca sem chave e sem custo, sobre o `ddgs`."""

    @tool(WEB_SEARCH_TOOL_NAME)
    def web_search(query: str) -> str:
        """Busca na web e devolve títulos, links e trechos.

        Args:
            query: o que procurar.
        """
        settings = get_settings()
        # ddgs aceita timeout em segundos inteiros; o campo é float de propósito,
        # coerente com os NODE_TIMEOUT_*.
        results = DDGS(timeout=int(settings.WEB_SEARCH_TIMEOUT_S)).text(
            query,
            region=settings.SEARCH_REGION,
            max_results=WEB_SEARCH_MAX_RESULTS,
            backend=settings.SEARCH_BACKEND,
        )
        return _format_results(list(results))

    return web_search


def _tavily_tool() -> BaseTool:
    """Busca sobre a Tavily, já com o nome unificado.

    `TavilySearch` aceita `name=`, então não há embrulho: é o mesmo objeto de
    antes, com o nome que os dois provedores compartilham.
    """
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        raise MissingTavilyKey("TAVILY_API_KEY não definida; SEARCH_PROVIDER=tavily exige a chave")
    return cast(
        BaseTool,
        TavilySearch(
            name=WEB_SEARCH_TOOL_NAME,
            max_results=WEB_SEARCH_MAX_RESULTS,
            tavily_api_key=settings.TAVILY_API_KEY,
        ),
    )


@lru_cache
def get_web_search_tool() -> BaseTool:
    """Instância única da tool de busca, no provedor configurado.

    `SEARCH_PROVIDER` é `Literal`, então o pydantic já recusou valor inválido no
    carregamento das settings. Não há branch de erro aqui: seria código morto.
    """
    if get_settings().SEARCH_PROVIDER == "ddgs":
        return _ddgs_tool()
    return _tavily_tool()


def reset_web_search_cache() -> None:
    """Descarta a instância cacheada. Quem troca configuração tem que chamar."""
    get_web_search_tool.cache_clear()
