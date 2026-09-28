"""A fábrica de busca. Nenhum teste aqui toca rede."""

import time
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.tools import BaseTool

from pauta.config import get_settings
from pauta.tools import web_search as ws


class FakeDDGS:
    """Substitui o DDGS real. Não chama rede, não depende de chave."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def text(self, query: str, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append((query, kwargs))
        return [
            {"title": "Um título", "href": "https://exemplo.dev/a", "body": "um trecho"},
            {"title": "Outro", "href": "https://exemplo.dev/b", "body": "outro trecho"},
        ]


@pytest.fixture
def fake_ddgs(monkeypatch: pytest.MonkeyPatch) -> FakeDDGS:
    instance = FakeDDGS()

    def factory(**kwargs: Any) -> FakeDDGS:
        instance.kwargs = kwargs
        return instance

    monkeypatch.setattr(ws, "DDGS", factory)
    return instance


class FakeTavilySearch:
    """Substitui o `TavilySearch` real. Devolve o shape documentado do provider."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return {
            "results": [
                {
                    "title": "Um título",
                    "url": "https://exemplo.dev/a",
                    "content": "um trecho",
                    "raw_content": None,
                },
                {
                    "title": "Outro",
                    "url": "https://exemplo.dev/b",
                    "content": "outro trecho",
                    "raw_content": None,
                },
            ],
            "query": payload["query"],
        }


@pytest.fixture
def fake_tavily(monkeypatch: pytest.MonkeyPatch) -> FakeTavilySearch:
    instance = FakeTavilySearch()

    def factory(**kwargs: Any) -> FakeTavilySearch:
        instance.kwargs = kwargs
        return instance

    monkeypatch.setattr(ws, "TavilySearch", factory)
    return instance


def test_ddgs_tool_has_the_stable_name(
    fake_ddgs: FakeDDGS, use_search_provider: Callable[[str], None]
) -> None:
    use_search_provider("ddgs")
    tool = ws.get_web_search_tool()
    assert isinstance(tool, BaseTool)
    assert tool.name == "web_search"


def test_ddgs_tool_passes_region_backend_and_timeout(
    fake_ddgs: FakeDDGS, use_search_provider: Callable[[str], None]
) -> None:
    use_search_provider("ddgs")
    tool = ws.get_web_search_tool()
    tool.invoke({"query": "chunking em português"})
    assert fake_ddgs.kwargs == {"timeout": int(get_settings().WEB_SEARCH_TIMEOUT_S)}
    query, kwargs = fake_ddgs.calls[0]
    assert query == "chunking em português"
    assert kwargs["region"] == "br-pt"
    assert kwargs["backend"] == "auto"
    assert kwargs["max_results"] == ws.WEB_SEARCH_MAX_RESULTS


def test_ddgs_tool_formats_title_href_body(
    fake_ddgs: FakeDDGS, use_search_provider: Callable[[str], None]
) -> None:
    use_search_provider("ddgs")
    tool = ws.get_web_search_tool()
    out = tool.invoke({"query": "x"})
    assert "title: Um título" in out
    assert "href: https://exemplo.dev/a" in out
    assert "body: um trecho" in out
    assert "title: Outro" in out


def test_ddgs_tool_returns_a_readable_marker_when_empty(
    fake_ddgs: FakeDDGS,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    use_search_provider("ddgs")
    monkeypatch.setattr(fake_ddgs, "text", lambda q, **k: [])
    tool = ws.get_web_search_tool()
    assert tool.invoke({"query": "x"}) == "nenhum resultado"


def test_tavily_tool_is_named_web_search_too(
    monkeypatch: pytest.MonkeyPatch, use_search_provider: Callable[[str], None]
) -> None:
    """O nome é estável nos dois provedores, para tools_called ser comparável.

    Artefato antigo traz tavily_search. Artefato novo traz web_search. A quebra
    de nome fica registrada no EVALUATION.md.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    tool = ws.get_web_search_tool()
    assert tool.name == "web_search"


def test_tavily_without_key_fails_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, use_search_provider: Callable[[str], None]
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "")
    use_search_provider("tavily")
    with pytest.raises(ws.MissingTavilyKey, match="TAVILY_API_KEY"):
        ws.get_web_search_tool()


def test_reset_makes_the_factory_build_again(use_search_provider: Callable[[str], None]) -> None:
    use_search_provider("ddgs")
    first = ws.get_web_search_tool()
    second = ws.get_web_search_tool()
    assert first is second
    ws.reset_web_search_cache()
    third = ws.get_web_search_tool()
    assert third is not first


def test_tavily_tool_exposes_only_query(
    monkeypatch: pytest.MonkeyPatch, use_search_provider: Callable[[str], None]
) -> None:
    """Os filtros do provider não vazam para o modelo: a tool é só `query`."""
    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    tool = ws.get_web_search_tool()
    assert list(tool.args) == ["query"]


def test_tavily_tool_maps_url_content_into_href_body(
    fake_tavily: FakeTavilySearch,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """O `url` do provider vira `href` e o `content` vira `body`, no bloco de sempre."""
    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    tool = ws.get_web_search_tool()
    out = tool.invoke({"query": "x"})
    assert fake_tavily.calls == [{"query": "x"}]
    assert out == (
        "title: Um título\n"
        "href: https://exemplo.dev/a\n"
        "body: um trecho\n"
        "\n"
        "title: Outro\n"
        "href: https://exemplo.dev/b\n"
        "body: outro trecho"
    )


def test_both_providers_render_the_same_blocks(
    fake_ddgs: FakeDDGS,
    fake_tavily: FakeTavilySearch,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """Saída idêntica para entrada equivalente: o agente cita fonte igual nos dois."""
    use_search_provider("ddgs")
    ddgs_out = ws.get_web_search_tool().invoke({"query": "x"})

    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    tavily_out = ws.get_web_search_tool().invoke({"query": "x"})

    assert ddgs_out == tavily_out
    assert "href: https://exemplo.dev/a" in tavily_out


def test_search_exceptions_propagate_out_of_the_tool(
    fake_ddgs: FakeDDGS,
    fake_tavily: FakeTavilySearch,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """A exceção sobe para o `run_tools`, que vira `ToolMessage` de erro.

    Tratar aqui e devolver `nenhum resultado` esconderia a falha.
    """

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("busca fora do ar")

    monkeypatch.setattr(fake_ddgs, "text", explode)
    monkeypatch.setattr(fake_tavily, "invoke", explode)

    use_search_provider("ddgs")
    with pytest.raises(RuntimeError, match="fora do ar"):
        ws.get_web_search_tool().invoke({"query": "x"})

    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    with pytest.raises(RuntimeError, match="fora do ar"):
        ws.get_web_search_tool().invoke({"query": "x"})


def test_tavily_provider_error_dict_raises_web_search_error(
    fake_tavily: FakeTavilySearch,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """Outage chega como `{"error": ...}`, não como exceção, e tem que sobreviver.

    Sem a tradução o acesso a `payload["results"]` estoura `KeyError` e a causa
    real, a mensagem do provider, desaparece antes do agente.
    """
    monkeypatch.setattr(
        fake_tavily, "invoke", lambda payload: {"error": ConnectionError("tavily fora do ar")}
    )
    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    tool = ws.get_web_search_tool()
    with pytest.raises(ws.WebSearchError, match="tavily fora do ar"):
        tool.invoke({"query": "x"})


def test_tavily_empty_result_string_raises_web_search_error(
    fake_tavily: FakeTavilySearch,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """Resultado vazio vira string no `BaseTool`, que é falha do provider.

    Não pode virar `nenhum resultado`: o provider trata vazio como erro, e
    responder como sucesso esconderia a anomalia. Também não pode virar
    `TypeError` de `string indices`.
    """
    empty = "No search results found for 'q'. Suggestions: ..."
    monkeypatch.setattr(fake_tavily, "invoke", lambda payload: empty)
    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    tool = ws.get_web_search_tool()
    with pytest.raises(ws.WebSearchError, match="No search results found"):
        tool.invoke({"query": "q"})


def test_tavily_unexpected_shape_raises_web_search_error(
    fake_tavily: FakeTavilySearch,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """Formato que não é dict nem string vira erro do domínio, dizendo o que chegou.

    Sem isso o caso estoura `TypeError` ou `KeyError` e o agente lê o nome da
    exceção no lugar do conteúdo que o provider mandou.
    """
    monkeypatch.setattr(fake_tavily, "invoke", lambda payload: 42)
    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    tool = ws.get_web_search_tool()
    with pytest.raises(ws.WebSearchError, match="42"):
        tool.invoke({"query": "x"})


def test_fractional_timeout_does_not_become_zero(
    fake_ddgs: FakeDDGS,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """0.5s não pode virar timeout=0, que expira na hora."""
    monkeypatch.setenv("WEB_SEARCH_TIMEOUT_S", "0.5")
    use_search_provider("ddgs")
    ws.get_web_search_tool().invoke({"query": "x"})
    assert fake_ddgs.kwargs == {"timeout": 1}


def test_none_fields_render_as_empty(
    fake_ddgs: FakeDDGS,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """Campo presente e nulo vira vazio: o agente não pode ler `None`."""
    monkeypatch.setattr(
        fake_ddgs, "text", lambda q, **k: [{"title": None, "href": None, "body": None}]
    )
    use_search_provider("ddgs")
    tool = ws.get_web_search_tool()
    assert tool.invoke({"query": "x"}) == "title: \nhref: \nbody: "


def slow_search(query: str, **kwargs: Any) -> list[dict[str, str]]:
    """Busca que demora mais que o teto dos testes abaixo."""
    time.sleep(0.5)
    return []


async def test_async_path_answers_like_the_sync_one(
    fake_ddgs: FakeDDGS, use_search_provider: Callable[[str], None]
) -> None:
    """O grafo chama por `ainvoke`; o teto não pode mudar o que a busca devolve."""
    use_search_provider("ddgs")
    out = await ws.get_web_search_tool().ainvoke({"query": "x"})
    assert "href: https://exemplo.dev/a" in out


async def test_a_slow_search_is_abandoned_at_the_deadline(
    fake_ddgs: FakeDDGS,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """Medido: com timeout=10 no DDGS uma chamada levou 31,2s. O teto é da tool."""
    monkeypatch.setattr(fake_ddgs, "text", slow_search)
    monkeypatch.setenv("WEB_SEARCH_TIMEOUT_S", "0.05")
    use_search_provider("ddgs")
    started = time.perf_counter()
    with pytest.raises(ws.WebSearchTimeout, match=r"0\.05s"):
        await ws.get_web_search_tool().ainvoke({"query": "x"})
    assert time.perf_counter() - started < 0.4


async def test_a_timed_out_search_becomes_an_error_message(
    fake_ddgs: FakeDDGS,
    monkeypatch: pytest.MonkeyPatch,
    use_search_provider: Callable[[str], None],
) -> None:
    """No grafo, o estouro vira `ToolMessage` de erro e o nó segue com o resto."""
    from langchain_core.messages import AIMessage

    from pauta.agents._common import run_tools

    monkeypatch.setattr(fake_ddgs, "text", slow_search)
    monkeypatch.setenv("WEB_SEARCH_TIMEOUT_S", "0.05")
    use_search_provider("ddgs")
    call = AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "c1"}],
    )
    [message] = await run_tools([ws.get_web_search_tool()], call, node="research", run_id="r1")
    assert message.status == "error"
    assert "passou de" in str(message.content)
