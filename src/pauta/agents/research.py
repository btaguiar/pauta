"""Research: reúne material com fonte declarada. Afirmação sem fonte não vira Finding."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..graph.budget import would_exhaust
from ..graph.state import AgentState, Finding, GraphNode
from ..observability import emit, node_span
from ._common import run_tools, tokens_from, unanswered

RESEARCH_PROMPT = """Você é o pesquisador de uma equipe de análise. Reúna o material
que responde à tarefa, usando as ferramentas disponíveis.

REGRAS:
1. Toda afirmação vem com a fonte de onde saiu: url ou caminho de arquivo.
2. O que você não encontrou, você diz que não encontrou. Não preencha lacuna com
   suposição.
3. Não redija o relatório. Só reúna o material."""

EXTRACTION_PROMPT = """Liste as descobertas do material acima, uma por item, cada uma
com a fonte exata. Se nada foi encontrado, devolva a lista vazia."""


class ResearchOutput(BaseModel):
    """Saída estruturada do pesquisador."""

    findings: list[Finding] = Field(default_factory=list)
    notes: str = Field(default="", description="o que não foi encontrado, se for o caso")


def make_research_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    settings: Settings | None = None,
) -> GraphNode:
    """Constrói o nó de pesquisa sobre um modelo e um conjunto de tools já prontos."""
    resolved = settings or get_settings()
    with_tools = model.bind_tools(list(tools)) if tools else model
    extractor = model.with_structured_output(ResearchOutput, include_raw=True)

    async def research(state: AgentState) -> dict[str, Any]:
        run_id = state.get("run_id", "desconhecida")
        iteration = state.get("iteration", 0)
        with node_span("research", run_id=run_id, iteration=iteration) as span:
            gaps = [gap for critique in state.get("critiques", []) for gap in critique.gaps]
            history: list[BaseMessage] = [
                SystemMessage(RESEARCH_PROMPT),
                HumanMessage(
                    f"Tarefa: {state['task']}\n"
                    f"Lacunas apontadas pelo crítico: {gaps if gaps else 'nenhuma'}"
                ),
            ]
            tokens = 0

            for _ in range(resolved.MAX_TOOL_ROUNDS):
                reply = await with_tools.ainvoke(history)
                tokens += tokens_from(reply)
                history.append(reply)
                if not isinstance(reply, AIMessage) or not reply.tool_calls:
                    break
                if would_exhaust(state, resolved, tokens):
                    emit(
                        "error",
                        node="research",
                        run_id=run_id,
                        error="orçamento da run esgotado no meio do nó; parando com o parcial",
                        tokens_used=tokens,
                    )
                    history.extend(unanswered(reply))
                    break
                history.extend(await run_tools(tools, reply, node="research", run_id=run_id))

            history.append(HumanMessage(EXTRACTION_PROMPT))
            result = await extractor.ainvoke(history)
            if isinstance(result, dict):
                tokens += tokens_from(result.get("raw"))
                parsed = result.get("parsed")
            else:
                parsed = result

            findings: list[Finding] = []
            if isinstance(parsed, ResearchOutput):
                findings = [
                    Finding(content=f.content, source=f.source, agent="research")
                    for f in parsed.findings
                    if f.content.strip() and f.source.strip()
                ]
            else:
                emit(
                    "error",
                    node="research",
                    run_id=run_id,
                    error="extração estruturada falhou; nenhuma descoberta registrada",
                )

            for finding in findings:
                emit(
                    "finding",
                    node="research",
                    run_id=run_id,
                    source=finding.source,
                    content=finding.content,
                )

            span.tokens_used = tokens
            span.extra["findings"] = len(findings)
            return {
                "findings": findings,
                "tokens_used": tokens,
                "empty_research": 0 if findings else 1,
            }

    return research
