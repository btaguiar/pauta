"""Supervisor: escolhe o próximo agente e justifica (ADR 001).

O agente define comportamento e não instancia cliente. O modelo chega pronto.
"""

from collections.abc import Collection
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..graph.routing import ALL_AGENTS, enforce_rules, fallback_route, forced_route
from ..graph.state import AgentState, GraphNode, NextStep
from ..observability import emit, node_span

SUPERVISOR_PROMPT = """Você é o supervisor de uma equipe de análise. Diante do estado atual
(tarefa original, descobertas, críticas, iterações restantes, orçamento
restante), escolha exatamente um próximo passo:
research (falta dado ou fonte), analyst (dado existe, falta processar
ou calcular), critic (há material suficiente, falta validar),
writer (validado ou limite atingido, falta redigir), END (pronto).

REGRAS:
1. Nunca envie para o writer sem o critic ter rodado.
2. Se o critic recusou duas vezes, writer com as ressalvas.
3. Se as iterações ou o orçamento acabarem, writer com o que houver.
4. Justifique sua escolha em uma frase."""

#: Tentativas de parse estruturado antes de cair na regra determinística.
STRUCTURED_PARSE_ATTEMPTS = 2


class Router(BaseModel):
    """Uma chamada, uma decisão, uma justificativa."""

    next: NextStep
    rationale: str = Field(description="uma frase explicando a escolha")


def render_state(state: AgentState, settings: Settings) -> str:
    """O que o supervisor vê. Números, não prosa."""
    critiques = state.get("critiques", [])
    last_verdict = critiques[-1].verdict if critiques else "nenhuma"
    gaps = critiques[-1].gaps if critiques else []
    return "\n".join(
        [
            f"Tarefa: {state['task']}",
            f"Descobertas: {len(state.get('findings', []))}",
            f"Críticas: {len(critiques)} (último veredito: {last_verdict})",
            f"Lacunas apontadas: {gaps if gaps else 'nenhuma'}",
            f"Iterações usadas: {state.get('iteration', 0)}/{settings.MAX_SUPERVISOR_STEPS}",
            f"Refações usadas: {state.get('critic_loops', 0)}/{settings.MAX_CRITIC_LOOPS}",
            f"Tokens usados: {state.get('tokens_used', 0)}/{settings.BUDGET_TOKENS_PER_RUN}",
            f"Feedback humano: {state.get('hitl_feedback') or 'nenhum'}",
        ]
    )


def _tokens_from(raw: Any) -> int:
    usage = getattr(raw, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0))
    return 0


def make_supervisor_node(
    model: BaseChatModel,
    settings: Settings | None = None,
    available: Collection[NextStep] = ALL_AGENTS,
) -> GraphNode:
    """Constrói o nó supervisor sobre um modelo já instanciado.

    `available` diz quais nós existem no grafo montado. O supervisor não manda
    trabalho para um nó que não foi registrado.
    """
    resolved = settings or get_settings()
    router = model.with_structured_output(Router, include_raw=True)

    async def supervisor(state: AgentState) -> dict[str, Any]:
        run_id = state.get("run_id", "desconhecida")
        iteration = state.get("iteration", 0)
        with node_span("supervisor", run_id=run_id, thread_id=run_id, iteration=iteration) as span:
            imposed = forced_route(state, resolved)
            if imposed is not None:
                emit(
                    "node_end",
                    node="supervisor",
                    run_id=run_id,
                    next_agent=imposed,
                    rationale="limite atingido, rota imposta sem consultar o modelo",
                )
                return {"next_agent": imposed, "iteration": 1}

            messages = [
                SystemMessage(SUPERVISOR_PROMPT),
                HumanMessage(render_state(state, resolved)),
            ]
            tokens = 0
            decision: Router | None = None
            for attempt in range(1, STRUCTURED_PARSE_ATTEMPTS + 1):
                try:
                    result = await router.ainvoke(messages)
                    if not isinstance(result, dict):
                        raise TypeError(
                            f"include_raw=True devia devolver dict, veio {type(result).__name__}"
                        )
                    tokens += _tokens_from(result.get("raw"))
                    parsed = result.get("parsed")
                    if isinstance(parsed, Router):
                        decision = parsed
                        break
                    raise ValueError(f"router devolveu {type(parsed).__name__}, esperava Router")
                except Exception as exc:  # a falha vira evento, não trava a run
                    emit(
                        "error",
                        node="supervisor",
                        run_id=run_id,
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

            if decision is None:
                route = fallback_route(state)
                rationale = "parse estruturado falhou; rota determinística aplicada"
            else:
                route = decision.next
                rationale = decision.rationale

            route = enforce_rules(state, route, resolved, available)
            span.tokens_used = tokens
            span.extra["next_agent"] = route
            span.extra["rationale"] = rationale
            return {"next_agent": route, "iteration": 1, "tokens_used": tokens}

    return supervisor
