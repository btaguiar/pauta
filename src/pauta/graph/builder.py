"""Monta o grafo. Não define prompt e não instancia cliente por conta própria."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from ..agents.analyst import make_analyst_node
from ..agents.critic import make_critic_node
from ..agents.research import make_research_node
from ..agents.supervisor import make_supervisor_node
from ..agents.writer import make_writer_node
from ..config import Settings, get_settings
from ..models import get_model
from .routing import ALL_AGENTS
from .state import AgentState, NextStep

#: Nomes de exceção que indicam limite de taxa em qualquer provider.
RATE_LIMIT_MARKERS = ("ratelimit", "toomanyrequests", "overloaded")

#: O nó onde o grafo congela para o humano decidir, quando HITL_MODE=interrupt.
HITL_NODE = "writer"


def is_retryable(exc: Exception) -> bool:
    """Retry só para erro de rede e limite de taxa, nunca para validação de schema.

    O `default_retry_on` do LangGraph retenta 5xx mas não retenta 429, e este
    projeto precisa retentar 429.
    """
    if isinstance(exc, ValueError | TypeError | KeyError | AttributeError):
        return False
    name = type(exc).__name__.lower()
    if any(marker in name for marker in RATE_LIMIT_MARKERS):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    return isinstance(exc, ConnectionError | TimeoutError | OSError)


def retry_policy(settings: Settings) -> RetryPolicy:
    """`NODE_RETRIES` são as retentativas; `max_attempts` conta a primeira tentativa."""
    return RetryPolicy(max_attempts=settings.NODE_RETRIES + 1, retry_on=is_retryable)


def route_from_supervisor(state: AgentState) -> str:
    """Lê a decisão que o supervisor já gravou no estado. Nenhuma chamada de LLM aqui."""
    return state.get("next_agent", "writer")


def build_graph(
    *,
    supervisor_model: BaseChatModel | None = None,
    research_model: BaseChatModel | None = None,
    analyst_model: BaseChatModel | None = None,
    critic_model: BaseChatModel | None = None,
    writer_model: BaseChatModel | None = None,
    research_tools: Sequence[BaseTool] = (),
    analyst_tools: Sequence[BaseTool] = (),
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """Monta e compila os cinco nós.

    Os modelos podem ser injetados, o que é como o teste usa o `FakeChatModel`.
    Sem injeção, cada papel vem de `get_model`. Com `HITL_MODE=interrupt`, o grafo
    congela antes do writer e espera um humano retomar pelo `thread_id`.
    """
    resolved = settings or get_settings()
    supervisor = make_supervisor_node(
        supervisor_model or get_model("supervisor"), resolved, ALL_AGENTS
    )
    research = make_research_node(research_model or get_model("research"), research_tools, resolved)
    analyst = make_analyst_node(analyst_model or get_model("analyst"), analyst_tools, resolved)
    critic = make_critic_node(critic_model or get_model("critic"), resolved)
    writer = make_writer_node(writer_model or get_model("writer"), resolved)

    policy = retry_policy(resolved)
    graph: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)
    graph.add_node(
        "supervisor",
        supervisor,
        timeout=resolved.NODE_TIMEOUT_SUPERVISOR_S,
        retry_policy=policy,
    )
    graph.add_node(
        "research", research, timeout=resolved.NODE_TIMEOUT_RESEARCH_S, retry_policy=policy
    )
    graph.add_node("analyst", analyst, timeout=resolved.NODE_TIMEOUT_ANALYST_S, retry_policy=policy)
    graph.add_node("critic", critic, timeout=resolved.NODE_TIMEOUT_CRITIC_S, retry_policy=policy)
    graph.add_node("writer", writer, timeout=resolved.NODE_TIMEOUT_WRITER_S, retry_policy=policy)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research": "research",
            "analyst": "analyst",
            "critic": "critic",
            "writer": "writer",
            "END": END,
        },
    )
    for worker in ("research", "analyst", "critic"):
        graph.add_edge(worker, "supervisor")
    graph.add_edge("writer", END)

    interrupt_before: list[str] = [HITL_NODE] if resolved.HITL_MODE == "interrupt" else []
    return graph.compile(
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        interrupt_before=interrupt_before or None,
    )


def available_agents() -> frozenset[NextStep]:
    """Os cinco nós existem a partir da semana 2."""
    return ALL_AGENTS
