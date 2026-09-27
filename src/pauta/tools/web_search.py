"""Busca web. Um provedor por vez, escolhido por configuração.

Wrapper fino: trocar de provedor é este arquivo. Os dois provedores saem com o
mesmo nome de tool, `web_search`, o mesmo argumento `query` e o mesmo formato
de saída, para `tools_called` ser comparável entre rodadas. Artefato antigo traz
`tavily_search`; a quebra de nome fica registrada no `EVALUATION.md`.

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
    """Formata os resultados para o agente conseguir citar a fonte.

    As chaves são `title`, `href` e `body`, o formato comum que os dois
    provedores entregam. Um resultado por bloco, porque a regra 1 do prompt do
    research cobra url ou caminho em toda afirmação. Campo nulo ou ausente vira
    vazio, para o agente não ler `None` como conteúdo.
    """
    if not results:
        return "nenhum resultado"
    blocks = [
        f"title: {item.get('title') or ''}\n"
        f"href: {item.get('href') or ''}\n"
        f"body: {item.get('body') or ''}"
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
        # coerente com os NODE_TIMEOUT_*. O truncamento rende 0 abaixo de 1s, e 0
        # expira na hora: por isso o inteiro passado tem piso em 1.
        results = DDGS(timeout=max(1, int(settings.WEB_SEARCH_TIMEOUT_S))).text(
            query,
            region=settings.SEARCH_REGION,
            max_results=WEB_SEARCH_MAX_RESULTS,
            backend=settings.SEARCH_BACKEND,
        )
        return _format_results(list(results))

    return web_search


def _tavily_tool() -> BaseTool:
    """Busca sobre a Tavily, já com o nome e o formato unificados.

    O `TavilySearch` cru expõe os filtros do provider e devolve `title`, `url`
    e `content`. O embrulho aceita só `query` e renomeia as chaves para o
    formato dos dois provedores.
    """
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        raise MissingTavilyKey("TAVILY_API_KEY não definida; SEARCH_PROVIDER=tavily exige a chave")
    inner = TavilySearch(
        name=WEB_SEARCH_TOOL_NAME,
        max_results=WEB_SEARCH_MAX_RESULTS,
        tavily_api_key=settings.TAVILY_API_KEY,
    )

    @tool(WEB_SEARCH_TOOL_NAME)
    def web_search(query: str) -> str:
        """Busca na web e devolve títulos, links e trechos.

        Args:
            query: o que procurar.
        """
        payload = cast(dict[str, Any], inner.invoke({"query": query}))
        results = [
            {"title": item.get("title"), "href": item.get("url"), "body": item.get("content")}
            for item in payload["results"]
        ]
        return _format_results(results)

    return web_search


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
