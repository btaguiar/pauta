"""Ponto de entrada de linha de comando.

    python -m pauta "sua pergunta"
    python -m pauta "sua pergunta" --ephemeral

Por padrão a run é durável: o checkpoint e o ponteiro vão para o Postgres do
`docker-compose.yml`. Com `--ephemeral` os dois viram memória e nada sobrevive ao
processo, o que serve para experimentar sem sujar o banco.

Ligar log e trace é decisão do ponto de entrada, nunca da biblioteca.
"""

import argparse
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import ValidationError

from .config import Settings, get_settings
from .graph.builder import build_graph
from .memory.checkpointer import memory_checkpointer, postgres_checkpointer, run_async
from .memory.run_store import InMemoryRunStore, RunStore, postgres_run_store
from .observability import configure_tracing, emit, setup_logging
from .runner import RunOutcome, start_run
from .tools import analyst_tools, research_tools

MISSING_CONFIG_MESSAGE = """Falta configuração para rodar o pauta.

Copie .env.example para .env e preencha MODEL_WORKER, MODEL_ROUTER, MODEL_CRITIC
e EMBEDDING_MODEL. Nenhum destes tem valor padrão no código, de propósito.

Detalhe do validador:
{detail}
"""

Resources = tuple[BaseCheckpointSaver[Any], RunStore]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pauta",
        description="produz um briefing analítico a partir de uma pergunta",
    )
    parser.add_argument("task", help="a pergunta que vira briefing")
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="roda sem Postgres; nada sobrevive ao processo",
    )
    return parser


@asynccontextmanager
async def open_resources(
    settings: Settings,
    *,
    ephemeral: bool,
) -> AsyncIterator[Resources]:
    """Checkpointer e registro de runs, os dois duráveis ou os dois em memória.

    Misturar os dois deixaria o estado sobrevivendo sem ponteiro, ou o contrário.
    """
    if ephemeral:
        yield memory_checkpointer(), InMemoryRunStore()
        return
    async with (
        postgres_checkpointer(settings) as saver,
        postgres_run_store(settings) as store,
    ):
        yield saver, store


def render(outcome: RunOutcome) -> str:
    """O que o usuário lê no fim. O `thread_id` sempre aparece: é o que retoma."""
    run = outcome.run
    if outcome.waiting_for_human:
        return (
            "\nA run congelou para revisão humana.\n"
            f"thread_id: {run.thread_id}\n"
            f"retome com: python -m pauta --resume {run.thread_id}\n"
        )
    report = outcome.report or "(o writer não produziu texto)"
    return (
        f"{report}\n\n"
        f"thread_id: {run.thread_id} · tokens: {run.tokens_used} · "
        f"ciclos: {run.iterations}\n"
    )


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    try:
        settings = get_settings()
    except ValidationError as exc:
        sys.stderr.write(MISSING_CONFIG_MESSAGE.format(detail=exc))
        return 2
    configure_tracing()

    try:
        async with open_resources(settings, ephemeral=args.ephemeral) as (saver, store):
            graph = build_graph(
                research_tools=research_tools(settings),
                analyst_tools=analyst_tools(),
                settings=settings,
                checkpointer=saver,
            )
            outcome = await start_run(args.task, graph=graph, store=store)
    except Exception as exc:
        emit("error", node="cli", error_type=type(exc).__name__, error=str(exc))
        sys.stderr.write(f"a run falhou: {type(exc).__name__}: {exc}\n")
        return 1

    sys.stdout.write(render(outcome))
    return 0


def main() -> int:
    return int(run_async(main_async(build_parser().parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
