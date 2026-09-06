"""Métricas determinísticas do golden set.

Cada linha de `tasks.jsonl` carrega rótulos escritos à mão: o que a resposta
precisa conter, se a tarefa é uma armadilha que exige recusa, se ela exige conta,
se exige pesquisa, e em quantos passos devia caber. Este módulo é quem os lê.

Nada aqui chama modelo. O que não dá para decidir por regra fica para o juiz de
`judge.py`, e é o juiz que exige calibração antes de qualquer número dele valer.
"""

from collections.abc import Iterable, Mapping, Sequence
from statistics import mean, stdev
from typing import Any
from unicodedata import combining, normalize

#: Marcas de incerteza aceitas. Não são palpite: saem do prompt do writer, que
#: manda escrever "Não foi possível validar X" e abrir uma seção de ressalvas.
#: Mudou o prompt, muda esta lista, e o `EVALUATION.md` diz que a métrica é uma
#: aproximação por marcador, não compreensão de texto.
UNCERTAINTY_MARKERS: tuple[str, ...] = (
    "nao foi possivel",
    "nao e possivel",
    "nao encontrei",
    "nao ha como",
    "nao da para",
    "nao sei",
    "nao respondivel",
    "nao validad",
    "ressalva",
    "incerteza",
    "incerto",
    "estimativa",
)

#: A tool que prova que a conta não foi feita de cabeça.
CALCULATOR_TOOL = "calculator"

#: Chaves de score por tarefa, na ordem em que entram no relatório.
SCORE_KEYS: tuple[str, ...] = (
    "cobertura",
    "incerteza_sinalizada",
    "calculadora_usada",
    "pesquisa_feita",
    "dentro_do_teto",
)


def flatten(text: str) -> str:
    """Caixa e acento fora. `GPU`, `gpu` e `gpú` são a mesma exigência."""
    decomposed = normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not combining(char))


def coverage(report: str, required: Sequence[str]) -> float | None:
    """Fração das substrings exigidas que aparecem. `None` quando nada é exigido.

    `None` e `0.0` são coisas diferentes: a tarefa sem `must_contain` não tem o
    que cobrir, e entrar na média como zero puniria o que ninguém pediu.
    """
    if not required:
        return None
    haystack = flatten(report)
    return sum(1 for item in required if flatten(item) in haystack) / len(required)


def flags_uncertainty(report: str) -> bool:
    """O briefing admite não saber. É o sucesso das tarefas-armadilha."""
    flat = flatten(report)
    return any(marker in flat for marker in UNCERTAINTY_MARKERS)


def score_task(
    task: dict[str, Any],
    *,
    report: str,
    iterations: int,
    findings: int,
    tools_called: Iterable[str],
) -> dict[str, float | bool | None]:
    """Os rótulos de uma tarefa contra o que a run produziu.

    Cada chave vem `None` quando a tarefa não faz aquela exigência. Quem agrega
    ignora os `None` e conta quantas tarefas sustentaram cada média.
    """
    called = set(tools_called)
    max_steps = task.get("max_steps")
    return {
        "cobertura": coverage(report, task.get("must_contain") or ()),
        "incerteza_sinalizada": (
            flags_uncertainty(report) if task.get("should_flag_uncertainty") else None
        ),
        "calculadora_usada": (CALCULATOR_TOOL in called if task.get("needs_calculus") else None),
        "pesquisa_feita": findings > 0 if task.get("needs_research") else None,
        "dentro_do_teto": iterations <= max_steps if isinstance(max_steps, int) else None,
    }


def aggregate(scores: Sequence[dict[str, float | bool | None]]) -> dict[str, float | int]:
    """Média por métrica, com o `_n` que diz quantas tarefas a sustentaram.

    Uma média sem denominador esconde que ela veio de duas tarefas. O `_n` anda
    junto do número em todo lugar, inclusive no `EVALUATION.md`.
    """
    summary: dict[str, float | int] = {}
    for key in SCORE_KEYS:
        values = [float(value) for score in scores if (value := score.get(key)) is not None]
        summary[key] = round(mean(values), 4) if values else 0.0
        summary[f"{key}_n"] = len(values)
    return summary


def spread(values: Sequence[float]) -> tuple[float, float]:
    """Média e desvio amostral. Uma amostra só tem desvio zero, não indefinido."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return mean(values), stdev(values)


Score = dict[str, float | bool | None]


def collapse_repeats(repeats: Sequence[Score]) -> Score:
    """As N execuções da mesma tarefa viram um score só, pela média.

    Sem isto, uma tarefa rodada três vezes pesaria três vezes na média geral, e
    o número diria mais sobre quantas repetições houve que sobre o sistema.
    """
    collapsed: Score = {}
    for key in SCORE_KEYS:
        values = [float(value) for repeat in repeats if (value := repeat.get(key)) is not None]
        collapsed[key] = mean(values) if values else None
    return collapsed


def repeat_deviation(grouped: Mapping[str, Sequence[Score]]) -> dict[str, float]:
    """Quanto cada rótulo se mexe entre repetições da mesma tarefa.

    Definição, porque média de desvio não é óbvia: para cada tarefa calcula-se o
    desvio das suas repetições, e o número reportado é a média desses desvios
    entre as tarefas. Lê-se como "quanto o score de uma tarefa costuma variar de
    uma execução para outra". Tarefa com menos de duas repetições não entra.

    Sem este número não há como separar melhora de ruído entre dois commits.
    """
    deviations: dict[str, float] = {}
    for key in SCORE_KEYS:
        per_task: list[float] = []
        for repeats in grouped.values():
            values = [float(value) for repeat in repeats if (value := repeat.get(key)) is not None]
            if len(values) >= 2:
                per_task.append(stdev(values))
        deviations[f"{key}_desvio"] = round(mean(per_task), 4) if per_task else 0.0
        deviations[f"{key}_desvio_n"] = len(per_task)
    return deviations
