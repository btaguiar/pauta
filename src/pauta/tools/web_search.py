"""Busca web. Um provedor por vez, escolhido por configuração.

Wrapper fino: trocar de provedor é este arquivo. Os dois provedores saem com o
mesmo nome de tool, `web_search`, o mesmo argumento `query` e o mesmo formato
de saída, para `tools_called` ser comparável entre rodadas. Artefato antigo traz
`tavily_search`; a quebra de nome fica registrada no `EVALUATION.md`.

A falha da busca não é tratada aqui. `run_tools` em `agents/_common.py` captura
a exceção, emite o evento `error` e devolve `ToolMessage` de erro. Tratar dentro
da tool e devolver lista vazia esconderia a falha.

O teto de tempo, sim, é daqui. Medido em 2026-09-28: com `timeout=10` no `DDGS`,
uma chamada levou 31,2s, e 20 de 25 runs do piloto da matriz morreram no timeout
de 240s do nó, com a busca consumindo o tempo. O `timeout` do pacote vale por
requisição HTTP, não pela chamada. O teto que vale é o `asyncio.wait_for` abaixo.
"""

import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from ddgs import DDGS
from langchain_core.tools import BaseTool, StructuredTool
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


class WebSearchError(RuntimeError):
    """A busca falhou e o provider devolveu o motivo no corpo da resposta."""


class WebSearchTimeout(WebSearchError):
    """A busca passou de `WEB_SEARCH_TIMEOUT_S`."""


def _bounded_tool(search: Callable[[str], str]) -> BaseTool:
    """Monta a tool com teto de tempo que vale de fato no caminho assíncrono.

    O grafo chama a tool por `ainvoke`, e é nesse caminho que o teto existe: a
    busca roda numa thread e `wait_for` para de esperar quando o prazo acaba.
    A thread não é morta, porque Python não mata thread; ela termina sozinha e o
    resultado é descartado. O caminho síncrono fica sem teto e só serve a teste.
    """

    async def bounded(query: str) -> str:
        limit = get_settings().WEB_SEARCH_TIMEOUT_S
        try:
            return await asyncio.wait_for(asyncio.to_thread(search, query), timeout=limit)
        except TimeoutError as exc:
            raise WebSearchTimeout(f"a busca passou de {limit:g}s e foi abandonada") from exc

    return StructuredTool.from_function(
        func=search,
        coroutine=bounded,
        name=WEB_SEARCH_TOOL_NAME,
        description="Busca na web e devolve títulos, links e trechos.",
    )


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

    return _bounded_tool(web_search)


def _tavily_results(payload: object) -> list[dict[str, Any]]:
    """Extrai os resultados reconhecendo as duas falhas que o provider devolve como dado.

    `TavilySearch` não levanta exceção na maioria das falhas: manda
    `{"error": ...}` em outage e uma string quando não há resultado. Os dois
    formatos viram `WebSearchError` com a mensagem do próprio provider. O resto
    também vira `WebSearchError`, dizendo o que chegou, em vez de estourar
    `KeyError` ou `TypeError` e apagar a causa.
    """
    if isinstance(payload, str):
        raise WebSearchError(payload)
    if not isinstance(payload, dict):
        raise WebSearchError(
            f"resposta da Tavily fora do formato: {type(payload).__name__}: {payload!r}"
        )
    if "error" in payload:
        raise WebSearchError(str(payload["error"]))
    results = payload.get("results")
    if not isinstance(results, list):
        raise WebSearchError(f"resposta da Tavily sem results nem error: {payload!r}")
    return results


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

    def web_search(query: str) -> str:
        """Busca na web e devolve títulos, links e trechos.

        Args:
            query: o que procurar.
        """
        results = [
            {"title": item.get("title"), "href": item.get("url"), "body": item.get("content")}
            for item in _tavily_results(inner.invoke({"query": query}))
        ]
        return _format_results(results)

    return _bounded_tool(web_search)


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
