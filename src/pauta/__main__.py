"""Ponto de entrada de linha de comando.

    python -m pauta "sua pergunta"
    python -m pauta "sua pergunta" --ephemeral
    python -m pauta --list
    python -m pauta --resume THREAD_ID
    python -m pauta --resume THREAD_ID --feedback "foque no custo de saída"

Por padrão a run é durável: o checkpoint e o ponteiro vão para o Postgres do
`docker-compose.yml`. Com `--ephemeral` os dois viram memória e nada sobrevive ao
processo, o que serve para experimentar sem sujar o banco.

Ligar log e trace é decisão do ponto de entrada, nunca da biblioteca.
"""

import argparse
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import ValidationError

from .config import Settings, get_settings
from .graph.builder import build_graph
from .memory.checkpointer import memory_checkpointer, postgres_checkpointer, run_async
from .memory.run_store import InMemoryRunStore, RunStore, postgres_run_store
from .memory.runs import Run
from .observability import configure_tracing, emit, setup_logging
from .runner import RunAlreadyFinished, RunNotFound, RunOutcome, resume_run, start_run
from .tools import analyst_tools, research_tools

MISSING_CONFIG_MESSAGE = """Falta configuração para rodar o pauta.

Copie .env.example para .env e preencha MODEL_WORKER, MODEL_ROUTER, MODEL_CRITIC
e EMBEDDING_MODEL. Nenhum destes tem valor padrão no código, de propósito.

Detalhe do validador:
{detail}
"""

Resources = tuple[BaseCheckpointSaver[Any], RunStore]
Mode = Literal["start", "resume", "list"]

#: Quanto da tarefa cabe numa linha do `--list` antes de virar ruído.
TASK_PREVIEW_CHARS = 60

#: Estados que esperam alguém decidir. São os que o `--list` sugere retomar.
RESUMABLE = ("interrupted", "orphaned")


class UsageError(ValueError):
    """A combinação de argumentos não descreve nenhum modo."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pauta",
        description="produz um briefing analítico a partir de uma pergunta",
    )
    parser.add_argument("task", nargs="?", help="a pergunta que vira briefing")
    parser.add_argument(
        "--resume",
        metavar="THREAD_ID",
        help="retoma a run congelada, órfã ou interrompida desse thread",
    )
    parser.add_argument(
        "--feedback",
        help="nota do revisor, que entra no prompt da redação; só junto de --resume",
    )
    parser.add_argument(
        "--list",
        dest="list_runs",
        action="store_true",
        help="mostra as runs registradas, com o thread_id de cada uma",
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="roda sem Postgres; nada sobrevive ao processo",
    )
    return parser


def mode_of(args: argparse.Namespace) -> Mode:
    """Qual dos três modos o usuário pediu. Combinação ambígua é recusada aqui.

    Recusar cedo é melhor que obedecer meio pedido. Um `--resume` sobre store em
    memória, por exemplo, nunca acharia nada, e a mensagem falaria de run
    inexistente em vez do argumento errado.
    """
    asked = [bool(args.task), bool(args.resume), bool(args.list_runs)]
    if sum(asked) != 1:
        raise UsageError("escolha exatamente um: uma pergunta, --resume THREAD_ID ou --list")
    if args.feedback and not args.resume:
        raise UsageError("--feedback só faz sentido junto de --resume")
    if args.ephemeral and not args.task:
        raise UsageError("--ephemeral só vale para uma run nova; nada fica gravado para retomar")
    if args.task:
        return "start"
    return "resume" if args.resume else "list"


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


def preview(task: str) -> str:
    if len(task) <= TASK_PREVIEW_CHARS:
        return task
    return task[: TASK_PREVIEW_CHARS - 1] + "…"


def render_runs(runs: list[Run]) -> str:
    """A lista que faz a aprovação humana parar de depender da memória de alguém."""
    if not runs:
        return "nenhuma run registrada\n"
    lines = [f"{len(runs)} run(s) registradas, horário em UTC:", ""]
    lines += [
        f"  {run.thread_id}  {run.status:<12}  {run.created_at:%Y-%m-%d %H:%M}  {preview(run.task)}"
        for run in runs
    ]
    waiting = [run for run in runs if run.status in RESUMABLE]
    if waiting:
        lines += ["", f"retome uma delas com: python -m pauta --resume {waiting[0].thread_id}"]
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    try:
        settings = get_settings()
    except ValidationError as exc:
        sys.stderr.write(MISSING_CONFIG_MESSAGE.format(detail=exc))
        return 2
    configure_tracing()

    try:
        mode = mode_of(args)
    except UsageError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    try:
        async with open_resources(settings, ephemeral=args.ephemeral) as (saver, store):
            if mode == "list":
                sys.stdout.write(render_runs(await store.list_runs()))
                return 0
            graph = build_graph(
                research_tools=research_tools(settings),
                analyst_tools=analyst_tools(),
                settings=settings,
                checkpointer=saver,
            )
            if mode == "start":
                outcome = await start_run(args.task, graph=graph, store=store)
            else:
                outcome = await resume_run(
                    args.resume, graph=graph, store=store, feedback=args.feedback
                )
    except (RunNotFound, RunAlreadyFinished) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
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
