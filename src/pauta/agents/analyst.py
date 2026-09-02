"""Analyst: processa e calcula sobre o material reunido. Conta vai na calculadora."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..graph.budget import would_exhaust
from ..graph.state import AgentState, Finding, GraphNode
from ..observability import emit, node_span

ANALYST_PROMPT = """Você é o analista de uma equipe de análise. Processe o material já
reunido: calcule, compare, converta unidade, cheque consistência.

REGRAS:
1. Toda conta passa pela calculadora. Não calcule de cabeça.
2. Cada resultado seu vira uma descoberta cuja fonte é "cálculo" seguido da conta
   que você usou, para alguém poder conferir.
3. Se o dado necessário para a conta não está no material, diga o que falta em vez
   de arbitrar um valor.
4. Não redija o relatório."""

EXTRACTION_PROMPT = """Liste o que você apurou, um item por resultado, com a conta na
fonte. Se não deu para calcular nada, devolva a lista vazia."""


class AnalystOutput(BaseModel):
    """Saída estruturada do analista."""

    findings: list[Finding] = Field(default_factory=list)
    notes: str = Field(default="", description="o dado que faltou, se for o caso")


def render_material(state: AgentState) -> str:
    findings = state.get("findings", [])
    if findings:
        lines = [f"- [{f.agent}] {f.content} (fonte: {f.source})" for f in findings]
    else:
        lines = ["- nenhuma descoberta foi registrada"]
    gaps = [gap for critique in state.get("critiques", []) for gap in critique.gaps]
    return "\n".join(
        [
            f"Tarefa: {state['task']}",
            "",
            "Material reunido:",
            *lines,
            "",
            f"Lacunas apontadas pelo crítico: {gaps if gaps else 'nenhuma'}",
        ]
    )


def _tokens_from(message: Any) -> int:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0))
    return 0


async def _run_tools(
    tools: Sequence[BaseTool],
    message: AIMessage,
    *,
    run_id: str,
) -> list[ToolMessage]:
    by_name = {tool.name: tool for tool in tools}
    results: list[ToolMessage] = []
    for call in message.tool_calls:
        emit("tool_call", node="analyst", run_id=run_id, tool=call["name"], args=call["args"])
        tool = by_name.get(call["name"])
        if tool is None:
            results.append(
                ToolMessage(
                    content=f"tool desconhecida: {call['name']}",
                    tool_call_id=call["id"] or "",
                    status="error",
                )
            )
            continue
        try:
            results.append(await tool.ainvoke(call))
        except Exception as exc:
            emit(
                "error",
                node="analyst",
                run_id=run_id,
                tool=call["name"],
                error_type=type(exc).__name__,
                error=str(exc),
            )
            results.append(
                ToolMessage(
                    content=f"a tool {call['name']} falhou: {exc}",
                    tool_call_id=call["id"] or "",
                    status="error",
                )
            )
    return results


def make_analyst_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    settings: Settings | None = None,
) -> GraphNode:
    """Constrói o nó analista sobre um modelo e um conjunto de tools já prontos."""
    resolved = settings or get_settings()
    with_tools = model.bind_tools(list(tools)) if tools else model
    extractor = model.with_structured_output(AnalystOutput, include_raw=True)

    async def analyst(state: AgentState) -> dict[str, Any]:
        run_id = state.get("run_id", "desconhecida")
        iteration = state.get("iteration", 0)
        with node_span("analyst", run_id=run_id, thread_id=run_id, iteration=iteration) as span:
            history: list[BaseMessage] = [
                SystemMessage(ANALYST_PROMPT),
                HumanMessage(render_material(state)),
            ]
            tokens = 0

            for _ in range(resolved.MAX_TOOL_ROUNDS):
                reply = await with_tools.ainvoke(history)
                tokens += _tokens_from(reply)
                history.append(reply)
                if not isinstance(reply, AIMessage) or not reply.tool_calls:
                    break
                if would_exhaust(state, resolved, tokens):
                    emit(
                        "error",
                        node="analyst",
                        run_id=run_id,
                        error="orçamento da run esgotado no meio do nó; parando com o parcial",
                        tokens_used=tokens,
                    )
                    break
                history.extend(await _run_tools(tools, reply, run_id=run_id))

            history.append(HumanMessage(EXTRACTION_PROMPT))
            result = await extractor.ainvoke(history)
            if isinstance(result, dict):
                tokens += _tokens_from(result.get("raw"))
                parsed = result.get("parsed")
            else:
                parsed = result

            findings: list[Finding] = []
            if isinstance(parsed, AnalystOutput):
                findings = [
                    Finding(content=f.content, source=f.source, agent="analyst")
                    for f in parsed.findings
                    if f.content.strip() and f.source.strip()
                ]
            else:
                emit(
                    "error",
                    node="analyst",
                    run_id=run_id,
                    error="extração estruturada falhou; nenhum resultado registrado",
                )

            for finding in findings:
                emit(
                    "finding",
                    node="analyst",
                    run_id=run_id,
                    source=finding.source,
                    content=finding.content,
                )

            span.tokens_used = tokens
            span.extra["findings"] = len(findings)
            return {"findings": findings, "tokens_used": tokens}

    return analyst
