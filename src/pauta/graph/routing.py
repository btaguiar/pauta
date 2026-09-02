"""Rota determinística. O grafo continua andando quando o LLM que decide falha."""

from collections.abc import Collection

from ..config import Settings
from .state import AgentState, NextStep

#: Todos os nós previstos para a v1. O grafo da semana 1 tem menos, e o
#: roteamento respeita o que existe em vez de mandar trabalho para um nó ausente.
ALL_AGENTS: frozenset[NextStep] = frozenset({"research", "analyst", "critic", "writer", "END"})


def fallback_route(state: AgentState) -> NextStep:
    """Regra dura, sem LLM, para quando o parse estruturado do router falha."""
    if not state.get("findings"):
        return "research"
    if not state.get("critiques"):
        return "critic"
    return "writer"


def budget_exhausted(state: AgentState, settings: Settings) -> bool:
    return state.get("tokens_used", 0) >= settings.BUDGET_TOKENS_PER_RUN


def steps_exhausted(state: AgentState, settings: Settings) -> bool:
    return state.get("iteration", 0) >= settings.MAX_SUPERVISOR_STEPS


def critic_loops_exhausted(state: AgentState, settings: Settings) -> bool:
    return state.get("critic_loops", 0) >= settings.MAX_CRITIC_LOOPS


def forced_route(state: AgentState, settings: Settings) -> NextStep | None:
    """Rota imposta pelos limites, antes de perguntar ao LLM.

    Estourou orçamento ou iterações, o writer redige com o que houver. Já existe
    relatório, a run acabou. Devolve `None` quando ainda cabe uma decisão de LLM.
    """
    if state.get("final_report"):
        return "END"
    if budget_exhausted(state, settings) or steps_exhausted(state, settings):
        return "writer"
    return None


def enforce_rules(
    state: AgentState,
    proposed: NextStep,
    settings: Settings,
    available: Collection[NextStep] = ALL_AGENTS,
) -> NextStep:
    """Aplica as regras do prompt do supervisor em código, não na confiança.

    Regra 1: nunca writer sem o critic ter rodado.
    Regra 2: o critic recusou até o limite, writer com as ressalvas.
    Regra 3: rota para nó que não existe no grafo montado vira writer.
    Regra 4: writer sem nenhuma descoberta volta para research. Redigir sobre
    nada gasta uma run inteira para produzir um texto que só informa a própria
    falta de dados. Quem passa por cima disto é o `forced_route`, que já
    decidiu antes quando o orçamento ou as iterações acabaram.
    """
    if proposed not in available:
        proposed = "writer"
    if proposed == "writer" and not state.get("findings") and "research" in available:
        return "research"
    if "critic" in available and proposed == "writer" and not state.get("critiques"):
        return "critic"
    if proposed == "critic" and critic_loops_exhausted(state, settings):
        return "writer"
    if proposed == "END" and not state.get("final_report"):
        return "writer"
    return proposed
