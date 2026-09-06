"""O runner do eval: seleção de tarefas e relatório. Nenhuma chamada de LLM aqui."""

import json
from pathlib import Path
from typing import Any

import pytest

import run_eval
from pauta.observability import emit, setup_logging

REPO_ROOT = Path(__file__).resolve().parent.parent


def task(task_id: str, *, requires_corpus: bool = False) -> dict[str, Any]:
    return {"id": task_id, "task": f"tarefa {task_id}", "requires_corpus": requires_corpus}


def test_golden_set_file_is_readable() -> None:
    tasks = run_eval.load_tasks(REPO_ROOT / "eval" / "tasks.jsonl")
    assert len(tasks) >= 10
    assert all("id" in item and "task" in item for item in tasks)


def test_corpus_tasks_are_skipped_when_there_is_no_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem corpus, a tarefa é pulada e contada. Não falha, não roda contra vazio."""
    monkeypatch.setattr(run_eval, "corpus_is_empty", lambda: True)
    selected, skipped = run_eval.select_tasks(
        [task("t01"), task("t03", requires_corpus=True), task("t05")], None
    )
    assert [item["id"] for item in selected] == ["t01", "t05"]
    assert skipped == 1


def test_corpus_tasks_run_once_the_corpus_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "corpus_is_empty", lambda: False)
    selected, skipped = run_eval.select_tasks([task("t03", requires_corpus=True)], None)
    assert [item["id"] for item in selected] == ["t03"]
    assert skipped == 0


def test_limit_caps_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "corpus_is_empty", lambda: False)
    selected, _ = run_eval.select_tasks([task(f"t{i}") for i in range(10)], 5)
    assert len(selected) == 5


def test_empty_samples_directory_counts_as_no_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "SAMPLES_DIR", REPO_ROOT / "nao-existe")
    assert run_eval.corpus_is_empty() is True


def test_cost_is_not_invented_without_a_declared_price() -> None:
    result = run_eval.TaskResult(
        task_id="t01",
        task="t",
        report="r",
        findings=1,
        iterations=2,
        tokens_used=50_000,
        latency_s=1.0,
    )
    assert result.cost_usd(None) is None
    assert result.cost_usd(2.0) == pytest.approx(0.1)


def test_report_states_how_many_were_skipped() -> None:
    result = run_eval.TaskResult(
        task_id="t01",
        task="compare custo",
        report="Briefing.",
        findings=2,
        iterations=3,
        tokens_used=1_000,
        latency_s=0.5,
    )
    rendered = run_eval.render_report([result], skipped=4, price=None)
    assert "puladas por falta de corpus: 4" in rendered
    assert "não calculado" in rendered
    assert "Briefing." in rendered


def test_report_shows_failures_instead_of_hiding_them() -> None:
    failed = run_eval.TaskResult(
        task_id="t02",
        task="preço do café em 2030",
        report="",
        findings=0,
        iterations=0,
        tokens_used=0,
        latency_s=0.1,
        error="RuntimeError: provider fora do ar",
    )
    rendered = run_eval.render_report([failed], skipped=0, price=1.0)
    assert "FALHOU" in rendered
    assert "falhas: 1" in rendered


def test_every_task_in_the_golden_set_has_an_id() -> None:
    raw = (REPO_ROOT / "eval" / "tasks.jsonl").read_text(encoding="utf-8")
    ids = [json.loads(line)["id"] for line in raw.splitlines() if line.strip()]
    assert len(ids) == len(set(ids))


def test_the_collector_picks_up_the_tools_the_graph_reported() -> None:
    """A prova de que a calculadora rodou vem do stream de eventos, não do estado."""
    setup_logging()
    with run_eval.collecting_events() as events:
        emit("tool_call", node="analyst", run_id="r1", tool="calculator", args={})
        emit("finding", node="analyst", run_id="r1", source="cálculo")
        emit("tool_call", node="research", run_id="r1", tool="retriever", args={})
    assert events.tools_called() == ["calculator", "retriever"]


def test_the_collector_stops_listening_after_the_task() -> None:
    setup_logging()
    with run_eval.collecting_events() as events:
        pass
    emit("tool_call", node="analyst", run_id="r1", tool="calculator", args={})
    assert events.tools_called() == []


def a_scored_result(**overrides: Any) -> run_eval.TaskResult:
    fields: dict[str, Any] = {
        "task_id": "t01",
        "task": "compare custo",
        "report": "Briefing com GPU.",
        "findings": 2,
        "iterations": 3,
        "tokens_used": 1_000,
        "latency_s": 0.5,
    }
    fields.update(overrides)
    return run_eval.TaskResult(**fields)


def test_the_report_shows_the_labels_each_task_was_scored_on() -> None:
    result = a_scored_result(
        scores={"cobertura": 1.0, "calculadora_usada": False, "pesquisa_feita": None}
    )
    rendered = run_eval.render_report([result], skipped=0, price=None)
    assert "cobertura=1.00" in rendered
    assert "calculadora_usada=NAO" in rendered
    assert "pesquisa_feita" not in rendered.split("rótulos:")[1].split("\n")[0]


def test_the_summary_carries_a_denominator_for_every_label() -> None:
    rendered = run_eval.render_report(
        [a_scored_result(scores={"cobertura": 0.5})], skipped=0, price=None
    )
    assert "cobertura: 0.5000 (n=1)" in rendered
    assert "calculadora_usada: sem tarefa que exija (n=0)" in rendered
