"""Writer: redige o briefing. Relatório honesto sobre o que não validou."""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings, get_settings
from ..graph.state import AgentState
from ..observability import emit, node_span

WRITER_PROMPT = """Você é o redator de uma equipe de análise. Escreva um briefing curto
que responda à tarefa, usando apenas o material reunido.

REGRAS:
1. Toda afirmação carrega a fonte de onde veio.
2. O que não foi validado aparece escrito, em uma secção de ressalvas ao final.
   "Não foi possível validar X" é uma frase aceitável e esperada.
3. Se a tarefa não é respondível com o material disponível, diga isso e explique
   o que faltou. Não invente número, não estime sem dizer que é estimativa.
4. Frases curtas. Sem travessão."""

#: Quando o crítico não aprovou, o redator recebe esta instrução a mais.
UNVALIDATED_NOTICE = """O crítico não aprovou o material. Registre no briefing, de forma
explícita, o que ficou por validar e por quê."""


def render_material(state: AgentState) -> str:
    """Material disponível para a redação, com fonte em cada linha."""
    findings = state.get("findings", [])
    if findings:
        lines = [f"- {f.content} (fonte: {f.source}, por {f.agent})" for f in findings]
    else:
        lines = ["- nenhuma descoberta foi registrada"]

    critiques = state.get("critiques", [])
    gaps = [gap for critique in critiques for gap in critique.gaps]
    blocks = [
        f"Tarefa: {state['task']}",
        "",
        "Material reunido:",
        *lines,
        "",
        f"Lacunas apontadas pelo crítico: {gaps if gaps else 'nenhuma'}",
    ]
    feedback = state.get("hitl_feedback")
    if feedback:
        blocks += ["", f"Feedback do revisor humano: {feedback}"]
    return "\n".join(blocks)


def _tokens_from(message: Any) -> int:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0))
    return 0


def make_writer_node(
    model: BaseChatModel,
    settings: Settings | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    """Constrói o nó de redação sobre um modelo já instanciado."""
    resolved = settings or get_settings()

    def writer(state: AgentState) -> dict[str, Any]:
        run_id = state.get("run_id", "desconhecida")
        iteration = state.get("iteration", 0)
        with node_span("writer", run_id=run_id, thread_id=run_id, iteration=iteration) as span:
            critiques = state.get("critiques", [])
            approved = bool(critiques) and critiques[-1].verdict == "ok"
            prompt = WRITER_PROMPT if approved else f"{WRITER_PROMPT}\n\n{UNVALIDATED_NOTICE}"
            reply = model.invoke([SystemMessage(prompt), HumanMessage(render_material(state))])
            tokens = _tokens_from(reply)
            report = reply.text if isinstance(reply.text, str) else str(reply.content)

            span.tokens_used = tokens
            span.extra["approved_by_critic"] = approved
            emit(
                "final",
                node="writer",
                run_id=run_id,
                approved_by_critic=approved,
                characters=len(report),
            )
            emit(
                "usage",
                run_id=run_id,
                tokens_used=state.get("tokens_used", 0) + tokens,
                budget=resolved.BUDGET_TOKENS_PER_RUN,
            )
            return {"final_report": report, "tokens_used": tokens, "next_agent": "END"}

    return writer
