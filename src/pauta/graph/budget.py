"""Guardrail de tokens por run (camada 1 da ADR 004).

O DoD da semana 1 mostrou o furo: checar o contador só entre nós deixa uma run
passar de 60k, porque um único nó de pesquisa gasta dezenas de milhares de uma
vez. A checagem agora também acontece dentro do nó, entre rodadas de tool.
"""

from ..config import Settings
from .state import AgentState


class BudgetExhausted(RuntimeError):
    """O orçamento de tokens da run acabou."""


def remaining(state: AgentState, settings: Settings) -> int:
    """Quantos tokens ainda cabem nesta run. Nunca negativo."""
    return max(0, settings.BUDGET_TOKENS_PER_RUN - state.get("tokens_used", 0))


def is_exhausted(state: AgentState, settings: Settings) -> bool:
    return remaining(state, settings) <= 0


def would_exhaust(state: AgentState, settings: Settings, spent_in_node: int) -> bool:
    """O nó já gastou o que faltava, contando o que consumiu antes de devolver.

    Serve para o nó parar no meio do próprio trabalho, em vez de descobrir o
    estouro depois de entregar. Quem chama decide o que fazer com o parcial.
    """
    return spent_in_node >= remaining(state, settings)
