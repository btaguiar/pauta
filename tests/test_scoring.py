"""Métricas determinísticas: os rótulos do golden set finalmente sendo lidos."""

from typing import Any

import pytest

from scoring import (
    Score,
    aggregate,
    collapse_repeats,
    coverage,
    flags_uncertainty,
    flatten,
    repeat_deviation,
    score_task,
    spread,
)


def a_task(**overrides: Any) -> dict[str, Any]:
    task: dict[str, Any] = {"id": "t01", "task": "compare custo"}
    task.update(overrides)
    return task


def scored(task: dict[str, Any], **overrides: Any) -> dict[str, float | bool | None]:
    call: dict[str, Any] = {
        "report": "",
        "iterations": 1,
        "findings": 0,
        "tools_called": (),
    }
    call.update(overrides)
    return score_task(task, **call)


def test_accent_and_case_do_not_change_the_requirement() -> None:
    assert flatten("Cálculo POR TOKEN") == "calculo por token"


def test_coverage_counts_the_substrings_that_showed_up() -> None:
    assert coverage("usa GPU dedicada e cobra por token", ["GPU", "por token"]) == 1.0
    assert coverage("usa GPU dedicada", ["GPU", "por token"]) == 0.5
    assert coverage("nada disso", ["GPU", "por token"]) == 0.0


def test_a_task_that_requires_nothing_is_not_scored_as_zero() -> None:
    """`None` e `0.0` são coisas diferentes, e a média não pode confundi-las."""
    assert coverage("qualquer texto", []) is None


def test_coverage_ignores_accents_in_both_directions() -> None:
    assert coverage("o calculo confere", ["cálculo"]) == 1.0
    assert coverage("o cálculo confere", ["calculo"]) == 1.0


@pytest.mark.parametrize(
    "report",
    [
        "Não foi possível validar o preço.",
        "Não encontrei fonte para isso.",
        "Ressalvas: o dado de 2030 não é conhecível.",
        "O número é uma estimativa.",
    ],
)
def test_an_honest_refusal_is_recognised(report: str) -> None:
    assert flags_uncertainty(report) is True


def test_a_confident_answer_is_not_mistaken_for_a_refusal() -> None:
    assert flags_uncertainty("A saca custará 1200 reais em 2030.") is False


def test_the_trap_task_is_scored_only_when_it_is_a_trap() -> None:
    trap = a_task(should_flag_uncertainty=True)
    assert scored(trap, report="Não foi possível prever.")["incerteza_sinalizada"] is True
    assert scored(trap, report="Custará 1200 reais.")["incerteza_sinalizada"] is False
    assert scored(a_task())["incerteza_sinalizada"] is None


def test_the_calculator_has_to_have_actually_run() -> None:
    """Finding do analyst não prova conta feita na tool. A chamada prova."""
    needs = a_task(needs_calculus=True)
    assert scored(needs, tools_called=["calculator"])["calculadora_usada"] is True
    assert scored(needs, tools_called=["retriever"])["calculadora_usada"] is False
    assert scored(a_task(), tools_called=[])["calculadora_usada"] is None


def test_research_is_scored_by_findings() -> None:
    needs = a_task(needs_research=True)
    assert scored(needs, findings=3)["pesquisa_feita"] is True
    assert scored(needs, findings=0)["pesquisa_feita"] is False
    assert scored(a_task())["pesquisa_feita"] is None


def test_the_step_cap_is_checked_against_what_the_run_used() -> None:
    capped = a_task(max_steps=4)
    assert scored(capped, iterations=4)["dentro_do_teto"] is True
    assert scored(capped, iterations=5)["dentro_do_teto"] is False
    assert scored(a_task())["dentro_do_teto"] is None


def test_every_metric_carries_its_denominator() -> None:
    """Média sem `_n` esconde que ela veio de duas tarefas."""
    scores = [
        scored(a_task(must_contain=["GPU"]), report="usa GPU"),
        scored(a_task(must_contain=["GPU"]), report="nada"),
        scored(a_task()),
    ]
    summary = aggregate(scores)
    assert summary["cobertura"] == 0.5
    assert summary["cobertura_n"] == 2
    assert summary["incerteza_sinalizada_n"] == 0


def test_a_metric_nobody_exercised_reports_zero_over_zero() -> None:
    summary = aggregate([scored(a_task())])
    assert summary["calculadora_usada"] == 0.0
    assert summary["calculadora_usada_n"] == 0


def test_aggregating_nothing_does_not_explode() -> None:
    summary = aggregate([])
    assert summary["cobertura_n"] == 0


def test_one_sample_has_no_spread_instead_of_an_error() -> None:
    assert spread([]) == (0.0, 0.0)
    assert spread([2.0]) == (2.0, 0.0)


def test_spread_reports_mean_and_sample_deviation() -> None:
    average, deviation = spread([1.0, 2.0, 3.0])
    assert average == 2.0
    assert deviation == pytest.approx(1.0)


def test_repeats_collapse_into_one_score_per_task() -> None:
    """Uma tarefa rodada três vezes não pode pesar o triplo na média geral."""
    collapsed = collapse_repeats(
        [
            {"cobertura": 1.0, "calculadora_usada": True},
            {"cobertura": 0.0, "calculadora_usada": False},
        ]
    )
    assert collapsed["cobertura"] == 0.5
    assert collapsed["calculadora_usada"] == 0.5


def test_collapsing_keeps_none_for_what_the_task_never_required() -> None:
    collapsed = collapse_repeats([{"cobertura": 1.0}, {"cobertura": 1.0}])
    assert collapsed["pesquisa_feita"] is None


def test_the_deviation_is_the_mean_of_the_per_task_deviations() -> None:
    """Definição escolhida: quanto o score de uma tarefa varia entre execuções."""
    grouped: dict[str, list[Score]] = {
        "t01": [{"cobertura": 1.0}, {"cobertura": 0.0}],
        "t02": [{"cobertura": 1.0}, {"cobertura": 1.0}],
    }
    deviations = repeat_deviation(grouped)
    assert deviations["cobertura_desvio"] == pytest.approx(0.7071 / 2, abs=1e-3)
    assert deviations["cobertura_desvio_n"] == 2


def test_a_task_run_once_contributes_no_deviation() -> None:
    single: dict[str, list[Score]] = {"t01": [{"cobertura": 1.0}]}
    deviations = repeat_deviation(single)
    assert deviations["cobertura_desvio"] == 0.0
    assert deviations["cobertura_desvio_n"] == 0
