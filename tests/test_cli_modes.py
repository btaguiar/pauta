"""Os três modos do CLI e a recusa de combinações que não descrevem nenhum."""

import argparse
from datetime import UTC, datetime

import pytest

from pauta.__main__ import UsageError, build_parser, main_async, mode_of, render_runs
from pauta.memory.runs import Run, RunStatus


def parse(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def a_run(thread_id: str, status: RunStatus, task: str = "vale a pena migrar?") -> Run:
    return Run(
        run_id=f"run-{thread_id}",
        thread_id=thread_id,
        task=task,
        status=status,
        created_at=datetime(2026, 9, 6, 14, 30, tzinfo=UTC),
    )


def test_a_question_starts_a_run() -> None:
    assert mode_of(parse("vale a pena migrar?")) == "start"


def test_a_thread_id_resumes_a_run() -> None:
    assert mode_of(parse("--resume", "thread-abc")) == "resume"


def test_the_list_flag_lists() -> None:
    assert mode_of(parse("--list")) == "list"


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("uma pergunta", "--list"),
        ("uma pergunta", "--resume", "thread-abc"),
        ("--list", "--resume", "thread-abc"),
    ],
)
def test_an_ambiguous_or_empty_request_is_refused(argv: tuple[str, ...]) -> None:
    with pytest.raises(UsageError):
        mode_of(parse(*argv))


def test_feedback_without_resume_is_refused() -> None:
    """Feedback numa run nova não tem onde entrar: ninguém revisou nada ainda."""
    with pytest.raises(UsageError, match="--feedback"):
        mode_of(parse("uma pergunta", "--feedback", "foque no custo"))


def test_ephemeral_resume_is_refused_before_it_confuses_anyone() -> None:
    """Sem store durável não há o que retomar, e o erro precisa dizer isso."""
    with pytest.raises(UsageError, match="--ephemeral"):
        mode_of(parse("--resume", "thread-abc", "--ephemeral"))


def test_feedback_rides_along_with_resume() -> None:
    args = parse("--resume", "thread-abc", "--feedback", "foque no custo de saída")
    assert mode_of(args) == "resume"
    assert args.feedback == "foque no custo de saída"


def test_an_empty_register_says_so() -> None:
    assert render_runs([]) == "nenhuma run registrada\n"


def test_the_listing_shows_thread_status_and_task() -> None:
    text = render_runs([a_run("thread-abc", "interrupted")])
    assert "thread-abc" in text
    assert "interrupted" in text
    assert "2026-09-06 14:30" in text
    assert "vale a pena migrar?" in text


def test_a_long_task_is_cut_instead_of_wrapping() -> None:
    text = render_runs([a_run("thread-abc", "completed", task="palavra " * 40)])
    assert "…" in text
    assert len(text.splitlines()[2]) < 140


def test_the_listing_offers_a_thread_worth_resuming() -> None:
    runs = [a_run("thread-done", "completed"), a_run("thread-frozen", "interrupted")]
    assert "--resume thread-frozen" in render_runs(runs)


def test_the_listing_offers_nothing_when_everything_finished() -> None:
    assert "--resume" not in render_runs([a_run("thread-done", "completed")])


async def test_an_unusable_combination_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        task=None, resume=None, list_runs=False, feedback=None, ephemeral=False
    )
    assert await main_async(args) == 2
    assert "escolha exatamente um" in capsys.readouterr().err
