"""Guardrail de tokens por run (camada 1 da ADR 004).

O DoD da semana 1 mostrou o furo: checar o contador só entre nós deixa uma run
passar de 60k, porque um único nó de pesquisa gasta dezenas de milhares de uma
vez. A checagem agora também acontece dentro do nó, entre rodadas de tool.

Medido em 2026-09-28 com busca ddgs, o furo seguinte: 74.905 tokens contra 60k.
O writer rodava depois do teto sem reserva, e o research parava olhando só o
que já tinha gastado, sem contar a extração que ainda vinha. Os nós de trabalho
agora param antes da reserva do writer, e projetam a próxima chamada.
"""

from ..config import Settings
from .state import AgentState


class BudgetExhausted(RuntimeError):
    """O orçamento de tokens da run acabou."""


def work_budget(settings: Settings) -> int:
    """O que os nós de trabalho podem gastar: o teto menos a reserva do writer."""
    return settings.BUDGET_TOKENS_PER_RUN - settings.WRITER_RESERVE_TOKENS


def remaining(state: AgentState, settings: Settings) -> int:
    """Quantos tokens de trabalho ainda cabem nesta run. Nunca negativo."""
    return max(0, work_budget(settings) - state.get("tokens_used", 0))


def is_exhausted(state: AgentState, settings: Settings) -> bool:
    return remaining(state, settings) <= 0


def would_exhaust(
    state: AgentState,
    settings: Settings,
    spent_in_node: int,
    next_call: int = 0,
) -> bool:
    """O nó não cabe mais no que falta, contando a chamada que ainda vai fazer.

    Serve para o nó parar no meio do próprio trabalho, em vez de descobrir o
    estouro depois de entregar. `next_call` é a projeção da próxima chamada: o
    histórico só cresce dentro do nó, então a próxima custa pelo menos o mesmo
    que a última. Quem chama decide o que fazer com o parcial.
    """
    return spent_in_node + next_call >= remaining(state, settings)
