"""A fábrica de busca. Nenhum teste aqui toca rede."""

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
