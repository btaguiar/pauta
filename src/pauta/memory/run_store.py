"""Persistência do ponteiro da run (ADR 006).

O checkpointer guarda o estado do grafo por `thread_id`. Esta tabela guarda o
ponteiro para ele: que runs existem, em que estado estão, e qual `thread_id`
retomar. Sem ela um processo que caia esperando aprovação deixa um checkpoint
recuperável e invisível ao mesmo tempo.

`PostgresRunStore` em produção, `InMemoryRunStore` em teste, pelo mesmo motivo
que o checkpointer tem os dois: o teste não pode depender de banco para provar
regra de negócio.
"""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from ..config import Settings, get_settings
from ..observability import emit
from .runs import Run, RunStatus, mark_orphaned

#: Nome da tabela. Fica no mesmo banco do checkpointer, que já está no compose.
TABLE = "pauta_runs"

CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    run_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    final_report TEXT,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    iterations INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
)
"""

COLUMNS = (
    "run_id",
    "thread_id",
    "task",
    "status",
    "final_report",
    "tokens_used",
    "iterations",
    "error",
    "created_at",
    "updated_at",
)

UPSERT = f"""
INSERT INTO {TABLE} ({", ".join(COLUMNS)})
VALUES ({", ".join("%s" for _ in COLUMNS)})
ON CONFLICT (run_id) DO UPDATE SET
    status = EXCLUDED.status,
    final_report = EXCLUDED.final_report,
    tokens_used = EXCLUDED.tokens_used,
    iterations = EXCLUDED.iterations,
    error = EXCLUDED.error,
    updated_at = EXCLUDED.updated_at
"""


class RunStore(Protocol):
    """O mínimo que o ponto de entrada precisa para achar uma run de novo."""

    async def setup(self) -> None: ...

    async def save(self, run: Run) -> None: ...

    async def get(self, run_id: str) -> Run | None: ...

    async def by_thread(self, thread_id: str) -> Run | None: ...

    async def list_runs(self, status: RunStatus | None = None) -> list[Run]: ...


class InMemoryRunStore:
    """Store de teste. Morre com o processo, e é essa a intenção."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    async def setup(self) -> None:
        return None

    async def save(self, run: Run) -> None:
        self._runs[run.run_id] = run

    async def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    async def by_thread(self, thread_id: str) -> Run | None:
        for run in self._runs.values():
            if run.thread_id == thread_id:
                return run
        return None

    async def list_runs(self, status: RunStatus | None = None) -> list[Run]:
        runs = [run for run in self._runs.values() if status is None or run.status == status]
        return sorted(runs, key=lambda run: run.created_at, reverse=True)


class PostgresRunStore:
    """Store de produção, sobre a conexão que o chamador abriu e fecha."""

    def __init__(self, connection: AsyncConnection[Any]) -> None:
        self._connection = connection

    async def setup(self) -> None:
        """Cria a tabela se faltar. Idempotente e barato, roda no startup."""
        await self._connection.execute(CREATE_TABLE)

    async def save(self, run: Run) -> None:
        values = [getattr(run, column) for column in COLUMNS]
        await self._connection.execute(UPSERT, values)

    async def get(self, run_id: str) -> Run | None:
        return await self._one(f"SELECT * FROM {TABLE} WHERE run_id = %s", (run_id,))

    async def by_thread(self, thread_id: str) -> Run | None:
        return await self._one(f"SELECT * FROM {TABLE} WHERE thread_id = %s", (thread_id,))

    async def list_runs(self, status: RunStatus | None = None) -> list[Run]:
        query = f"SELECT * FROM {TABLE}"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = %s"
            params = (status,)
        query += " ORDER BY created_at DESC"
        async with self._connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)
            return [Run.model_validate(row) for row in await cursor.fetchall()]

    async def _one(self, query: str, params: Sequence[Any]) -> Run | None:
        async with self._connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)
            row = await cursor.fetchone()
        return Run.model_validate(row) if row is not None else None


@asynccontextmanager
async def postgres_run_store(
    settings: Settings | None = None,
    *,
    setup: bool = True,
) -> AsyncIterator[RunStore]:
    """Abre o store de Postgres e fecha a conexão ao sair."""
    resolved = settings or get_settings()
    async with await AsyncConnection.connect(resolved.DATABASE_URL, autocommit=True) as connection:
        store = PostgresRunStore(connection)
        if setup:
            await store.setup()
        emit("node_start", node="run_store", message="registro de runs pronto")
        yield store


async def mark_orphans(store: RunStore) -> int:
    """Marca como órfã toda run não terminal. Não retoma nenhuma (ADR 006).

    Roda no startup. Devolve quantas foram marcadas, que é o número que o log do
    startup e o `--list` mostram.
    """
    pending = [run for run in await store.list_runs() if run.status in ("running", "interrupted")]
    updated, count = mark_orphaned(pending)
    for run in updated:
        await store.save(run)
    if count:
        emit("error", node="run_store", message="runs sem executor marcadas", orphaned=count)
    return count
