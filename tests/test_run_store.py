"""Registro de runs: o ponteiro que faz um checkpoint ser encontrável de novo."""

import os
import uuid

import pytest

from pauta.memory.run_store import InMemoryRunStore, mark_orphans, postgres_run_store
from pauta.memory.runs import Run, RunStatus

requires_postgres = pytest.mark.skipif(
    os.environ.get("PAUTA_TEST_POSTGRES") != "1",
    reason="precisa de Postgres; rode docker compose up e exporte PAUTA_TEST_POSTGRES=1",
)


def a_run(run_id: str = "r1", status: RunStatus = "running") -> Run:
    return Run(
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        task="vale a pena migrar?",
        status=status,
    )


@pytest.fixture
def store() -> InMemoryRunStore:
    return InMemoryRunStore()


async def test_a_saved_run_comes_back(store: InMemoryRunStore) -> None:
    await store.save(a_run())
    found = await store.get("r1")
    assert found is not None
    assert found.task == "vale a pena migrar?"


async def test_an_unknown_run_is_none(store: InMemoryRunStore) -> None:
    assert await store.get("nunca-existiu") is None


async def test_a_run_is_found_by_its_thread(store: InMemoryRunStore) -> None:
    """É o `thread_id` que o `--resume` recebe, então é por ele que se procura."""
    await store.save(a_run())
    found = await store.by_thread("thread-r1")
    assert found is not None
    assert found.run_id == "r1"


async def test_saving_the_same_run_twice_replaces_it(store: InMemoryRunStore) -> None:
    await store.save(a_run())
    await store.save(a_run().transition_to("completed"))
    assert len(await store.list_runs()) == 1
    found = await store.get("r1")
    assert found is not None
    assert found.status == "completed"


async def test_listing_filters_by_status(store: InMemoryRunStore) -> None:
    await store.save(a_run("r1", status="interrupted"))
    await store.save(a_run("r2", status="completed"))
    assert [run.run_id for run in await store.list_runs("interrupted")] == ["r1"]
    assert len(await store.list_runs()) == 2


async def test_marking_orphans_takes_the_pending_ones(store: InMemoryRunStore) -> None:
    await store.save(a_run("r1", status="running"))
    await store.save(a_run("r2", status="interrupted"))
    assert await mark_orphans(store) == 2
    assert len(await store.list_runs("orphaned")) == 2


async def test_marking_orphans_leaves_the_terminal_ones(store: InMemoryRunStore) -> None:
    """Religar o servidor não mexe em run que já acabou."""
    await store.save(a_run("r1", status="completed"))
    await store.save(a_run("r2", status="failed"))
    assert await mark_orphans(store) == 0
    assert len(await store.list_runs("orphaned")) == 0


async def test_marking_orphans_twice_is_a_no_op(store: InMemoryRunStore) -> None:
    await store.save(a_run("r1", status="running"))
    assert await mark_orphans(store) == 1
    assert await mark_orphans(store) == 0


@requires_postgres
async def test_the_postgres_store_outlives_the_connection() -> None:
    """Gravado numa conexão, lido em outra. É o que o `--resume` faz depois de um kill."""
    run_id = f"pg-{uuid.uuid4()}"
    run = Run(run_id=run_id, thread_id=f"thread-{run_id}", task="sobreviver ao processo")

    async with postgres_run_store() as store:
        await store.save(run)
        await store.save(run.transition_to("interrupted"))

    async with postgres_run_store(setup=False) as store:
        found = await store.by_thread(f"thread-{run_id}")
        assert found is not None
        assert found.run_id == run_id
        assert found.status == "interrupted"
        assert found.task == "sobreviver ao processo"
        assert [r.run_id for r in await store.list_runs() if r.run_id == run_id] == [run_id]
