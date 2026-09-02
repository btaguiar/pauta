"""Checkpointer do grafo (ADR 003).

`AsyncPostgresSaver` em produção, `InMemorySaver` em teste. O checkpoint por
`thread_id` é o que dá retomada de execução e o interrupt de human-in-the-loop
sem banco caseiro nenhum.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ..config import Settings, get_settings
from ..observability import emit


@asynccontextmanager
async def postgres_checkpointer(
    settings: Settings | None = None,
    *,
    setup: bool = True,
) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    """Abre o checkpointer de Postgres e fecha a conexão ao sair.

    `setup=True` cria as tabelas se faltarem. É idempotente e barato, então roda
    no startup da app em vez de virar um passo manual que alguém esquece.
    """
    resolved = settings or get_settings()
    async with AsyncPostgresSaver.from_conn_string(resolved.DATABASE_URL) as saver:
        if setup:
            await saver.setup()
        emit("node_start", node="checkpointer", message="checkpointer de Postgres pronto")
        yield saver


def memory_checkpointer() -> BaseCheckpointSaver[Any]:
    """Checkpointer de teste. Morre com o processo, e é essa a intenção."""
    return InMemorySaver()
