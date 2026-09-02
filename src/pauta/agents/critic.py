"""Critic: valida o material e aponta lacunas. Não reescreve resposta (ADR 002)."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings, get_settings
from ..graph.state import AgentState, Critique, GraphNode
from ..observability import emit, node_span

CRITIC_PROMPT = """Você é o crítico. Avalie se as descobertas sustentam uma resposta à
tarefa. Verifique: a fonte existe e é citada? o cálculo confere?
há contradição entre descobertas? o que falta para a tarefa ficar
respondida?

Para cada lacuna, diga qual agente resolve: research quando falta dado ou fonte,
analyst quando o dado existe e falta processar ou calcular.

Não reescreva a resposta. Só aponte o que falta."""


def render_material(state: AgentState) -> str:
    findings = state.get("findings", [])
    if findings:
        lines = [f"- [{f.agent}] {f.content} (fonte: {f.source})" for f in findings]
    else:
        lines = ["- nenhuma descoberta foi registrada"]
    previous = state.get("critiques", [])
    blocks = [
        f"Tarefa: {state['task']}",
        "",
        "Material reunido:",
        *lines,
    ]
    if previous:
        blocks += [
            "",
            f"Você já avaliou este material {len(previous)} vez(es).",
            f"Lacunas que apontou da última vez: {previous[-1].gaps or 'nenhuma'}",
        ]
    return "\n".join(blocks)


def _tokens_from(message: Any) -> int:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0))
    return 0


def make_critic_node(
    model: BaseChatModel,
    settings: Settings | None = None,
) -> GraphNode:
    """Constrói o nó crítico sobre um modelo já instanciado."""
    resolved = settings or get_settings()
    judge = model.with_structured_output(Critique, include_raw=True)

    async def critic(state: AgentState) -> dict[str, Any]:
        run_id = state.get("run_id", "desconhecida")
        iteration = state.get("iteration", 0)
        with node_span("critic", run_id=run_id, thread_id=run_id, iteration=iteration) as span:
            messages = [SystemMessage(CRITIC_PROMPT), HumanMessage(render_material(state))]
            tokens = 0
            verdict: Critique | None = None
            try:
                result = await judge.ainvoke(messages)
                if isinstance(result, dict):
                    tokens = _tokens_from(result.get("raw"))
                    parsed = result.get("parsed")
                    if isinstance(parsed, Critique):
                        verdict = parsed
            except Exception as exc:
                emit(
                    "error",
                    node="critic",
                    run_id=run_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

            if verdict is None:
                # Crítico que não responde não aprova por omissão. Sem veredito,
                # o material segue como não validado e a ressalva vai no relatório.
                verdict = Critique(
                    verdict="refinar",
                    gaps=["o crítico não conseguiu avaliar o material"],
                    delegate_to=[],
                )
                emit("error", node="critic", run_id=run_id, error="veredito não estruturado")

            if not state.get("findings") and verdict.verdict == "ok":
                # Aprovar o vazio é o modo de falha que a ADR 002 existe para pegar.
                verdict = Critique(
                    verdict="refinar",
                    gaps=["nenhuma descoberta foi registrada"],
                    delegate_to=["research"],
                )

            emit(
                "critique",
                node="critic",
                run_id=run_id,
                verdict=verdict.verdict,
                gaps=verdict.gaps,
                delegate_to=verdict.delegate_to,
                loop=state.get("critic_loops", 0) + 1,
                max_loops=resolved.MAX_CRITIC_LOOPS,
            )
            span.tokens_used = tokens
            span.extra["verdict"] = verdict.verdict
            return {"critiques": [verdict], "critic_loops": 1, "tokens_used": tokens}

    return critic
