"""Varredura de startup: marca as órfãs e não retoma nenhuma (ADR 006)."""

import pytest

from pauta.api.recovery import sweep
from pauta.memory.run_store import InMemoryRunStore
from pauta.memory.runs import Run, RunStatus


@pytest.fixture
def store() -> InMemoryRunStore:
    return InMemoryRunStore()


def a_run(run_id: str, status: RunStatus) -> Run:
    return Run(
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        task="vale a pena migrar?",
        status=status,
    )


async def test_a_clean_start_finds_nothing(store: InMemoryRunStore) -> None:
    report = await sweep(store)
    assert report.orphaned == 0
    assert report.clean is True
    assert report.threads == []


async def test_runs_caught_mid_flight_become_orphans(store: InMemoryRunStore) -> None:
    await store.save(a_run("r1", "running"))
    await store.save(a_run("r2", "interrupted"))
    report = await sweep(store)

    assert report.orphaned == 2
    assert sorted(report.threads) == ["thread-r1", "thread-r2"]
    assert len(await store.list_runs("orphaned")) == 2


async def test_finished_runs_are_left_alone(store: InMemoryRunStore) -> None:
    """Religar o servidor não mexe em run que já acabou."""
    await store.save(a_run("r1", "completed"))
    await store.save(a_run("r2", "failed"))
    report = await sweep(store)

    assert report.orphaned == 0
    assert len(await store.list_runs("orphaned")) == 0


async def test_the_sweep_never_puts_a_run_back_to_running(store: InMemoryRunStore) -> None:
    """O ponto inteiro da ADR 006: subir a app não pode gastar token de ninguém."""
    await store.save(a_run("r1", "running"))
    await store.save(a_run("r2", "interrupted"))
    await sweep(store)

    assert await store.list_runs("running") == []
    assert await store.list_runs("interrupted") == []


async def test_a_second_start_finds_nothing_new(store: InMemoryRunStore) -> None:
    await store.save(a_run("r1", "running"))
    assert (await sweep(store)).orphaned == 1
    assert (await sweep(store)).orphaned == 0


async def test_the_report_names_the_threads_worth_continuing(store: InMemoryRunStore) -> None:
    """Saber que existem órfãs e não saber quais não ajuda ninguém."""
    await store.save(a_run("r1", "running"))
    await store.save(a_run("r2", "completed"))
    assert (await sweep(store)).threads == ["thread-r1"]
