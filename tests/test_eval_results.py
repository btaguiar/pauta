"""O artefato por rodada: o que torna duas execuções comparáveis no tempo."""

import json
from pathlib import Path
from typing import Any

import pytest

import run_eval
from pauta.config import Settings, get_settings

SECRET = "sk-um-segredo-que-nunca-pode-vazar"


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(
        update={
            "OPENROUTER_API_KEY": SECRET,
            "TAVILY_API_KEY": SECRET,
            "LANGSMITH_API_KEY": SECRET,
            "DATABASE_URL": f"postgresql://user:{SECRET}@localhost:5432/pauta",
        }
    )


@pytest.fixture
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path / "results")
    return tmp_path / "results"


def a_result(task_id: str = "t01", **overrides: Any) -> run_eval.TaskResult:
    fields: dict[str, Any] = {
        "task_id": task_id,
        "task": "compare custo",
        "report": "Briefing com GPU.",
        "findings": 2,
        "iterations": 3,
        "tokens_used": 1_000,
        "latency_s": 0.5,
        "scores": {"cobertura": 1.0},
    }
    fields.update(overrides)
    return run_eval.TaskResult(**fields)


def test_the_config_block_names_the_models_that_produced_the_number(
    settings: Settings,
) -> None:
    block = run_eval.config_block(settings)
    assert block["model_router"] == settings.MODEL_ROUTER
    assert block["max_critic_loops"] == settings.MAX_CRITIC_LOOPS
    assert block["temperature"] == 0


def test_no_secret_reaches_the_artifact(settings: Settings, results_dir: Path) -> None:
    """A lista do bloco é explícita justamente para isto. Nunca um despejo do Settings."""
    run_eval.write_results(
        [a_result()], skipped=0, settings=settings, tasks_file="tasks.jsonl", limit=None
    )
    for path in results_dir.iterdir():
        assert SECRET not in path.read_text(encoding="utf-8")


def test_a_round_writes_the_three_files(settings: Settings, results_dir: Path) -> None:
    written = run_eval.write_results(
        [a_result()], skipped=0, settings=settings, tasks_file="tasks.jsonl", limit=None
    )
    prefixes = sorted(path.name.split("_")[0] for path in written)
    assert prefixes == ["bruto", "eval", "metricas"]
    assert all(path.exists() for path in written)


def test_the_filename_carries_the_commit_so_rounds_can_be_told_apart(
    settings: Settings, results_dir: Path
) -> None:
    written = run_eval.write_results(
        [a_result()], skipped=0, settings=settings, tasks_file="tasks.jsonl", limit=None
    )
    commit = run_eval.commit_hash()
    assert all(commit in path.name for path in written)
    assert commit != ""


def test_the_metrics_file_carries_numbers_without_the_briefings(
    settings: Settings, results_dir: Path
) -> None:
    """É o arquivo que alimenta um gráfico de evolução sem carregar texto junto."""
    run_eval.write_results(
        [a_result()], skipped=0, settings=settings, tasks_file="tasks.jsonl", limit=None
    )
    metrics_file = next(path for path in results_dir.iterdir() if path.name.startswith("metricas"))
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "metricas" in payload
    assert "itens" not in payload
    assert payload["config"]["model_worker"] == settings.MODEL_WORKER


def test_the_limit_is_recorded_because_it_changes_what_comparable_means(
    settings: Settings, results_dir: Path
) -> None:
    run_eval.write_results(
        [a_result()], skipped=0, settings=settings, tasks_file="tasks.jsonl", limit=5
    )
    payload = json.loads(
        next(p for p in results_dir.iterdir() if p.name.startswith("eval_")).read_text("utf-8")
    )
    assert payload["limit"] == 5


def test_failures_and_skips_are_counted_not_hidden() -> None:
    metrics = run_eval.metrics_of(
        [a_result(), a_result("t02", error="RuntimeError: caiu")], skipped=3, price=None
    )
    assert metrics["total_tarefas"] == 2
    assert metrics["falhas"] == 1
    assert metrics["puladas_sem_corpus"] == 3


def test_a_failed_task_does_not_drag_the_latency_average_to_zero() -> None:
    metrics = run_eval.metrics_of(
        [a_result(latency_s=2.0), a_result("t02", error="caiu", latency_s=0.0)],
        skipped=0,
        price=None,
    )
    assert metrics["latencia_media_s"] == 2.0
    assert metrics["latencia_n"] == 1


def test_cost_stays_absent_without_a_declared_price() -> None:
    assert run_eval.metrics_of([a_result()], skipped=0, price=None)["custo_usd"] is None
    assert run_eval.metrics_of([a_result()], skipped=0, price=2.0)["custo_usd"] == 0.002


def test_the_percentile_needs_no_extra_dependency() -> None:
    assert run_eval.percentile([], 0.95) == 0.0
    assert run_eval.percentile([1.0], 0.95) == 1.0
    assert run_eval.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
