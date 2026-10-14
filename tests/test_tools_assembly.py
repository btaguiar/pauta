"""Quais tools o grafo recebe, dado o que o ambiente oferece."""

from collections.abc import Callable, Sequence

import pytest
from langchain_core.tools import BaseTool

from pauta.tools import analyst_tools, research_tools
from pauta.tools.web_search import MissingTavilyKey


def search_tool_names(tools: Sequence[BaseTool]) -> set[str]:
    return {t.name for t in tools}


def test_ddgs_needs_no_key(
    monkeypatch: pytest.MonkeyPatch, use_search_provider: Callable[[str], None]
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "")
    use_search_provider("ddgs")
    names = search_tool_names(research_tools())
    assert "web_search" in names


def test_tavily_with_key_enters(
    monkeypatch: pytest.MonkeyPatch, use_search_provider: Callable[[str], None]
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "chave-fake")
    use_search_provider("tavily")
    names = search_tool_names(research_tools())
    assert "web_search" in names


def test_tavily_without_key_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch, use_search_provider: Callable[[str], None]
) -> None:
    """Silenciar aqui deixaria o research sem busca e sem dizer por quê."""
    monkeypatch.setenv("TAVILY_API_KEY", "")
    use_search_provider("tavily")
    with pytest.raises(MissingTavilyKey, match="TAVILY_API_KEY"):
        research_tools()


def test_analyst_never_gets_the_search_tool(
    monkeypatch: pytest.MonkeyPatch, use_search_provider: Callable[[str], None]
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "")
    use_search_provider("ddgs")
    names = search_tool_names(analyst_tools())
    assert "web_search" not in names
