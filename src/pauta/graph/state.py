"""Estado do grafo. É a única parte que o type checker consegue proteger."""

import operator
from collections.abc import Awaitable
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

AgentName = Literal["research", "analyst", "critic", "writer"]
NextStep = Literal["research", "analyst", "critic", "writer", "END"]
Verdict = Literal["ok", "refinar"]


class Finding(BaseModel):
    """Uma descoberta com origem declarada. Afirmação sem fonte não vira Finding."""

    content: str
    source: str = Field(description='url, caminho de arquivo ou "cálculo"')
    agent: Literal["research", "analyst"]


class Critique(BaseModel):
    """Veredito do crítico sobre o material reunido até aqui."""

    verdict: Verdict
    gaps: list[str] = Field(default_factory=list)
    delegate_to: list[Literal["research", "analyst"]] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """Estado compartilhado do grafo.

    Os contadores usam soma como reducer: cada nó devolve o quanto gastou, em vez
    de ler o total e escrever de volta. Isso evita perder contagem quando dois
    nós escrevem no mesmo superstep.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    findings: Annotated[list[Finding], operator.add]
    critiques: Annotated[list[Critique], operator.add]
    iteration: Annotated[int, operator.add]
    critic_loops: Annotated[int, operator.add]
    tokens_used: Annotated[int, operator.add]
    next_agent: NextStep
    final_report: str | None
    hitl_feedback: str | None
    run_id: str


class GraphNode(Protocol):
    """Assinatura de um nó do grafo.

    O parâmetro precisa se chamar `state`: é assim que o protocolo interno do
    LangGraph aceita a função em `add_node`. O nó é assíncrono porque o
    `timeout` por nó do LangGraph 1.2 só vale para nós async: execução síncrona
    não pode ser cancelada com segurança dentro do processo.
    """

    def __call__(self, state: AgentState) -> Awaitable[dict[str, Any]]: ...


def new_state(task: str, run_id: str) -> AgentState:
    """Estado inicial de uma run. Todo contador começa em zero, explicitamente."""
    return AgentState(
        messages=[],
        task=task,
        findings=[],
        critiques=[],
        iteration=0,
        critic_loops=0,
        tokens_used=0,
        next_agent="research",
        final_report=None,
        hitl_feedback=None,
        run_id=run_id,
    )
