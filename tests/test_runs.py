"""Máquina de estados da run (ADR 006)."""

import pytest

from pauta.memory.runs import (
    ALLOWED,
    TERMINAL_STATUSES,
    IllegalTransition,
    Run,
    RunStatus,
    all_statuses,
    mark_orphaned,
)


def a_run(status: RunStatus = "running") -> Run:
    return Run(run_id="r1", thread_id="t1", task="vale a pena migrar?", status=status)


def test_a_run_starts_running() -> None:
    assert a_run().status == "running"
    assert a_run().is_terminal is False


def test_the_happy_path_of_the_contract() -> None:
    """running -> interrupted -> running -> completed."""
    run = a_run().transition_to("interrupted")
    assert run.waiting_for_human is True
    run = run.transition_to("running")
    run = run.transition_to("completed")
    assert run.is_terminal is True


def test_a_finished_run_never_moves_again() -> None:
    for status in TERMINAL_STATUSES:
        with pytest.raises(IllegalTransition):
            a_run(status).transition_to("running")


def test_an_illegal_transition_says_what_is_allowed() -> None:
    with pytest.raises(IllegalTransition, match="permitidos"):
        a_run("completed").transition_to("orphaned")


def test_only_an_orphan_is_recoverable() -> None:
    assert a_run("orphaned").recoverable is True
    for status in ("running", "interrupted", "completed", "failed"):
        assert a_run(status).recoverable is False


def test_startup_marks_orphans_and_resumes_nothing() -> None:
    """Religar o servidor não pode gastar token de ninguém."""
    runs = [a_run("running"), a_run("interrupted"), a_run("completed"), a_run("failed")]
    updated, count = mark_orphaned(runs)
    assert count == 2
    assert [run.status for run in updated] == ["orphaned", "orphaned", "completed", "failed"]


def test_marking_orphans_twice_changes_nothing() -> None:
    once, first = mark_orphaned([a_run("running")])
    twice, second = mark_orphaned(once)
    assert first == 1
    assert second == 0
    assert twice[0].status == "orphaned"


def test_an_orphan_goes_back_to_running_when_asked() -> None:
    assert a_run("orphaned").transition_to("running").status == "running"


def test_the_transition_table_covers_every_status() -> None:
    assert set(ALLOWED) == set(all_statuses())


def test_the_timestamp_moves_on_transition() -> None:
    run = a_run()
    moved = run.transition_to("completed")
    assert moved.updated_at >= run.updated_at
    assert moved.created_at == run.created_at
