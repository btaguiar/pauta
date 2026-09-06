"""Quais tools existem, dado o que o ambiente oferece.

O agente recebe as tools prontas. Quem monta o conjunto é o ponto de entrada, e
monta igual em todos eles: sem chave da Tavily não há busca web, sem corpus
indexável não há retriever, calculadora sempre.
"""

from collections.abc import Sequence
from pathlib import Path

from langchain_core.tools import BaseTool

from ..config import Settings, get_settings
from .calculator import get_calculator_tool
from .retriever import get_retriever_tool, load_documents, samples_dir
from .web_search import get_web_search_tool


def corpus_is_empty(directory: Path | None = None) -> bool:
    """Vazio é não ter documento indexável, não apenas não ter arquivo."""
    documents, _ = load_documents(directory or samples_dir())
    return not documents


def research_tools(settings: Settings | None = None) -> Sequence[BaseTool]:
    """Busca web quando há chave, corpus local quando há documento."""
    resolved = settings or get_settings()
    tools: list[BaseTool] = []
    if resolved.TAVILY_API_KEY:
        tools.append(get_web_search_tool())
    if not corpus_is_empty():
        tools.append(get_retriever_tool())
    return tools


def analyst_tools() -> Sequence[BaseTool]:
    """A calculadora nunca falta: conta feita de cabeça é o que o analyst evita."""
    tools: list[BaseTool] = [get_calculator_tool()]
    if not corpus_is_empty():
        tools.append(get_retriever_tool())
    return tools
