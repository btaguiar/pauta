"""Executa uma run do começo ao fim e mantém o ponteiro dela atualizado.

O grafo sabe rodar, o checkpointer sabe guardar estado, o store sabe guardar o
ponteiro. Este módulo costura os três, para o ponto de entrada cuidar só de
argumento de linha de comando.

O `thread_id` nasce aqui, uma vez por run, e é o que o `--resume` recebe depois.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from .graph.state import AgentState, new_state
from .memory.run_store import RunStore
from .memory.runs import Run
from .observability import emit

Graph = CompiledStateGraph[AgentState, Any, Any, Any]


class RunNotFound(LookupError):
    """Nenhuma run registrada com esse `thread_id`."""


class RunAlreadyFinished(ValueError):
    """A run chegou a um estado terminal e não retoma."""


@dataclass(frozen=True)
class RunOutcome:
    """O que a execução deixou. `waiting_for_human` é o interrupt do HITL."""

    run: Run
    report: str | None
    waiting_for_human: bool


def new_ids() -> tuple[str, str]:
    """Um par `run_id`, `thread_id` para uma run nova.

    São dois porque medem coisas diferentes: o `run_id` identifica a tentativa
    nos logs, o `thread_id` identifica a linha de checkpoint que sobrevive ao
    processo. Uma retomada reusa o `thread_id` e continua a mesma run.
    """
    token = uuid.uuid4().hex[:12]
    return f"run-{token}", f"thread-{token}"


async def start_run(
    task: str,
    *,
    graph: Graph,
    store: RunStore,
    run_id: str | None = None,
    thread_id: str | None = None,
) -> RunOutcome:
    """Registra uma run nova e a executa até o fim ou até o interrupt."""
    run = await register_run(task, store=store, run_id=run_id, thread_id=thread_id)
    return await execute_run(run, graph=graph, store=store)


async def register_run(
    task: str,
    *,
    store: RunStore,
    run_id: str | None = None,
    thread_id: str | None = None,
) -> Run:
    """Grava o ponteiro da run e devolve os ids, sem executar nada.

    Registrar e executar são separados porque a API precisa dos dois momentos:
    ela responde 201 com o `thread_id` na hora e executa em segundo plano. Se o
    processo cair entre um e outro, a run existe no registro e a varredura de
    startup a encontra, que é o contrário de sumir sem rastro.
    """
    generated_run_id, generated_thread_id = new_ids()
    run = Run(
        run_id=run_id or generated_run_id,
        thread_id=thread_id or generated_thread_id,
        task=task,
    )
    await store.save(run)
    emit("node_start", node="runner", run_id=run.run_id, thread_id=run.thread_id, task=task)
    return run


async def execute_run(run: Run, *, graph: Graph, store: RunStore) -> RunOutcome:
    """Executa uma run já registrada, até o fim ou até o interrupt."""
    return await _drive(run, new_state(task=run.task, run_id=run.run_id), graph=graph, store=store)


async def resume_run(
    thread_id: str,
    *,
    graph: Graph,
    store: RunStore,
    feedback: str | None = None,
) -> RunOutcome:
    """Retoma uma run congelada ou órfã. Nunca acontece sozinho (ADR 006)."""
    run = await store.by_thread(thread_id)
    if run is None:
        raise RunNotFound(f"nenhuma run registrada com thread_id {thread_id!r}")
    if run.is_terminal:
        raise RunAlreadyFinished(f"a run {run.run_id} já está {run.status!r} e não retoma")

    if run.status == "running":
        # Uma run gravada como `running` que alguém pede para retomar é uma run
        # cujo executor morreu: é o que a morte suja deixa no ponteiro. A máquina
        # de estados não vai de `running` para `running`, e passar por `orphaned`
        # é o caminho que a ADR 006 desenhou para exatamente este caso.
        run = run.transition_to("orphaned")
        await store.save(run)

    running = run.transition_to("running")
    await store.save(running)
    emit(
        "node_start",
        node="runner",
        run_id=running.run_id,
        thread_id=thread_id,
        resumed_from=run.status,
        with_feedback=feedback is not None,
    )
    payload: Any = Command(update={"hitl_feedback": feedback}) if feedback else None
    return await _drive(running, payload, graph=graph, store=store)


async def _spent_so_far(graph: Graph, config: RunnableConfig) -> tuple[int, int]:
    """Tokens e iterações que o checkpoint já registrou, para uma run que falhou.

    Uma run que morre gastou tokens de verdade, e gravar zero no ponteiro faz o
    teto diário subcontabilizar justamente as runs que queimaram token sem
    entregar nada.

    O número é um piso, não o total exato: o superstep que falhou não chega a
    ser gravado, então o que ele consumiu antes de estourar não aparece aqui.
    Piso medido é melhor que zero inventado.
    """
    try:
        snapshot = await graph.aget_state(config)
    except Exception as exc:
        emit("error", node="runner", error=f"checkpoint ilegível após falha: {exc}")
        return 0, 0
    values = snapshot.values or {}
    return int(values.get("tokens_used", 0)), int(values.get("iteration", 0))


async def _drive(run: Run, payload: Any, *, graph: Graph, store: RunStore) -> RunOutcome:
    """Roda o grafo e grava o desfecho, qualquer que ele seja.

    Falha vira run `failed` no registro antes de propagar. Uma run que morre sem
    deixar rastro no ponteiro é o modo de falha que este módulo existe para
    evitar.
    """
    config: RunnableConfig = {"configurable": {"thread_id": run.thread_id}}
    try:
        final = await graph.ainvoke(payload, config=config)
    except Exception as exc:
        tokens, iterations = await _spent_so_far(graph, config)
        failed = run.model_copy(
            update={
                "error": f"{type(exc).__name__}: {exc}",
                "tokens_used": tokens,
                "iterations": iterations,
            }
        )
        await store.save(failed.transition_to("failed"))
        emit(
            "error",
            node="runner",
            run_id=run.run_id,
            thread_id=run.thread_id,
            tokens_used=tokens,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise

    snapshot = await graph.aget_state(config)
    waiting = bool(snapshot.next)
    report = final.get("final_report")
    finished = run.model_copy(
        update={
            "final_report": report,
            "tokens_used": final.get("tokens_used", 0),
            "iterations": final.get("iteration", 0),
        }
    ).transition_to("interrupted" if waiting else "completed")
    await store.save(finished)
    emit(
        "node_end",
        node="runner",
        run_id=finished.run_id,
        thread_id=finished.thread_id,
        status=finished.status,
        tokens_used=finished.tokens_used,
        iteration=finished.iterations,
    )
    return RunOutcome(run=finished, report=report, waiting_for_human=waiting)
