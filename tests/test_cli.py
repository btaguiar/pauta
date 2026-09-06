"""Linha de comando: o que o usuário digita e o que ele lê de volta."""

import argparse

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from pauta.__main__ import build_parser, main_async, open_resources, render
from pauta.config import Settings, get_settings
from pauta.memory.run_store import InMemoryRunStore
from pauta.memory.runs import Run
from pauta.runner import RunOutcome


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def an_outcome(*, waiting: bool, report: str | None = "Briefing pronto.") -> RunOutcome:
    run = Run(
        run_id="run-abc",
        thread_id="thread-abc",
        task="vale a pena migrar?",
        status="interrupted" if waiting else "completed",
        final_report=report,
        tokens_used=4321,
        iterations=3,
    )
    return RunOutcome(run=run, report=report, waiting_for_human=waiting)


def test_the_parser_accepts_no_task_because_two_modes_have_none() -> None:
    """`--resume` e `--list` rodam sem pergunta. Quem recusa o vazio é o `mode_of`."""
    assert build_parser().parse_args([]).task is None


def test_a_run_is_durable_unless_asked_otherwise() -> None:
    """O default precisa ser o Postgres: durabilidade por omissão, não por lembrança."""
    assert build_parser().parse_args(["uma pergunta"]).ephemeral is False
    assert build_parser().parse_args(["uma pergunta", "--ephemeral"]).ephemeral is True


def test_a_finished_run_shows_the_briefing_and_the_numbers() -> None:
    text = render(an_outcome(waiting=False))
    assert "Briefing pronto." in text
    assert "thread-abc" in text
    assert "4321" in text


def test_a_frozen_run_shows_how_to_resume_it() -> None:
    """O `thread_id` é inútil se o usuário precisar adivinhar o comando seguinte."""
    text = render(an_outcome(waiting=True, report=None))
    assert "--resume thread-abc" in text


def test_a_silent_writer_does_not_print_an_empty_briefing() -> None:
    assert "(o writer não produziu texto)" in render(an_outcome(waiting=False, report=None))


async def test_ephemeral_resources_are_both_in_memory(settings: Settings) -> None:
    """Estado durável com ponteiro volátil daria run recuperável e invisível."""
    async with open_resources(settings, ephemeral=True) as (saver, store):
        assert isinstance(saver, InMemorySaver)
        assert isinstance(store, InMemoryRunStore)


async def test_missing_model_configuration_exits_with_a_readable_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("MODEL_WORKER", raising=False)
    get_settings.cache_clear()
    args = argparse.Namespace(task="t", ephemeral=True)

    assert await main_async(args) == 2
    assert "MODEL_WORKER" in capsys.readouterr().err
