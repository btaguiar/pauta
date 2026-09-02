"""Checkpointer do grafo (ADR 003).

`AsyncPostgresSaver` em produção, `InMemorySaver` em teste. O checkpoint por
`thread_id` é o que dá retomada de execução e o interrupt de human-in-the-loop
sem banco caseiro nenhum.
"""

import asyncio
import selectors
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ..config import Settings, get_settings
from ..observability import emit


def loop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    """Fábrica de event loop compatível com o psycopg async.

    O `ProactorEventLoop`, padrão do Windows, não serve: o psycopg recusa rodar
    em modo assíncrono sobre ele. Em Linux e macOS o padrão já serve e esta
    função devolve `None`, que quer dizer "use o de sempre".

    Quem abre o loop passa isto ao `asyncio.Runner`. Uma biblioteca não troca a
    policy global do processo por conta própria.
    """
    if sys.platform != "win32":
        return None
    return lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())


def run_async(coro: Any) -> Any:
    """Roda uma corrotina no loop certo para esta plataforma."""
    with asyncio.Runner(loop_factory=loop_factory()) as runner:
        return runner.run(coro)


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
