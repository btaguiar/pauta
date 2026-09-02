"""Executa tarefas do golden set contra o grafo e imprime o que custou.

Versão da semana 1: uma repetição, sem juiz. As repetições, o desvio padrão e o
LLM-as-judge de outro provider entram na semana 3, junto com o EVALUATION.md.

Uso:
    uv run python eval/run_eval.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from pauta.config import get_settings
from pauta.graph.builder import build_graph
from pauta.graph.state import new_state
from pauta.memory.checkpointer import run_async
from pauta.observability import setup_logging
from pauta.tools.calculator import get_calculator_tool
from pauta.tools.web_search import get_web_search_tool

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO_ROOT / "eval" / "tasks.jsonl"
SAMPLES_DIR = REPO_ROOT / "samples"
TOKENS_PER_MILLION = 1_000_000


@dataclass
class TaskResult:
    """O que uma tarefa produziu, medido e não estimado."""

    task_id: str
    task: str
    report: str
    findings: int
    iterations: int
    tokens_used: int
    latency_s: float
    error: str | None = None

    def cost_usd(self, price_per_mtok: float | None) -> float | None:
        if price_per_mtok is None:
            return None
        return self.tokens_used / TOKENS_PER_MILLION * price_per_mtok


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corpus_is_empty() -> bool:
    """O retriever ainda não indexou nada quando `samples/` não tem arquivo."""
    if not SAMPLES_DIR.is_dir():
        return True
    return not any(item.is_file() for item in SAMPLES_DIR.rglob("*"))


def select_tasks(
    tasks: list[dict[str, Any]], limit: int | None
) -> tuple[list[dict[str, Any]], int]:
    """Devolve as tarefas executáveis e quantas foram puladas por falta de corpus."""
    skipped = 0
    runnable: list[dict[str, Any]] = []
    empty = corpus_is_empty()
    for task in tasks:
        if task.get("requires_corpus") and empty:
            skipped += 1
            continue
        runnable.append(task)
    if limit is not None:
        runnable = runnable[:limit]
    return runnable, skipped


async def run_task(task: dict[str, Any], *, index: int) -> TaskResult:
    settings = get_settings()
    research_tools = [get_web_search_tool()] if settings.TAVILY_API_KEY else []

    graph = build_graph(
        research_tools=research_tools,
        analyst_tools=[get_calculator_tool()],
        settings=settings,
    )
    run_id = f"eval-{task['id']}"
    config: RunnableConfig = {"configurable": {"thread_id": f"{run_id}-{index}"}}
    started = time.perf_counter()
    try:
        final = await graph.ainvoke(new_state(task=task["task"], run_id=run_id), config=config)
    except Exception as exc:
        return TaskResult(
            task_id=task["id"],
            task=task["task"],
            report="",
            findings=0,
            iterations=0,
            tokens_used=0,
            latency_s=round(time.perf_counter() - started, 2),
            error=f"{type(exc).__name__}: {exc}",
        )
    return TaskResult(
        task_id=task["id"],
        task=task["task"],
        report=final.get("final_report") or "",
        findings=len(final.get("findings", [])),
        iterations=final.get("iteration", 0),
        tokens_used=final.get("tokens_used", 0),
        latency_s=round(time.perf_counter() - started, 2),
    )


def render_report(results: list[TaskResult], skipped: int, price: float | None) -> str:
    lines: list[str] = []
    for result in results:
        lines.append("=" * 78)
        lines.append(f"[{result.task_id}] {result.task}")
        lines.append("-" * 78)
        if result.error:
            lines.append(f"FALHOU: {result.error}")
        else:
            lines.append(result.report or "(o writer não produziu texto)")
        cost = result.cost_usd(price)
        cost_text = (
            f"{cost:.4f} USD" if cost is not None else "não calculado (COST_PER_MTOK_USD vazio)"
        )
        lines.append("")
        lines.append(
            f"descobertas: {result.findings} · ciclos: {result.iterations} · "
            f"tokens: {result.tokens_used} · custo: {cost_text} · {result.latency_s}s"
        )

    total_tokens = sum(r.tokens_used for r in results)
    failures = [r for r in results if r.error]
    total_cost = total_tokens / TOKENS_PER_MILLION * price if price is not None else None
    lines.append("=" * 78)
    lines.append(
        f"tarefas executadas: {len(results)} · falhas: {len(failures)} · "
        f"puladas por falta de corpus: {skipped}"
    )
    lines.append(
        f"tokens somados: {total_tokens} · custo somado: "
        + (f"{total_cost:.4f} USD" if total_cost is not None else "não calculado")
    )
    return "\n".join(lines)


MISSING_CONFIG_MESSAGE = """Falta configuração para rodar o eval.

Copie .env.example para .env e preencha MODEL_WORKER, MODEL_ROUTER, MODEL_CRITIC
e EMBEDDING_MODEL. Nenhum destes tem valor padrão no código, de propósito.

Detalhe do validador:
{detail}
"""


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    try:
        get_settings()
    except ValidationError as exc:
        sys.stderr.write(MISSING_CONFIG_MESSAGE.format(detail=exc))
        return 2
    tasks = load_tasks(Path(args.tasks))
    selected, skipped = select_tasks(tasks, args.limit)
    if not selected:
        sys.stdout.write(f"nenhuma tarefa executável; {skipped} puladas por falta de corpus\n")
        return 1
    results = [await run_task(task, index=i) for i, task in enumerate(selected)]
    sys.stdout.write(render_report(results, skipped, get_settings().COST_PER_MTOK_USD) + "\n")
    return 1 if any(r.error for r in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="roda tarefas do golden set contra o grafo")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="caminho do jsonl")
    parser.add_argument("--limit", type=int, default=None, help="quantas tarefas rodar")
    return int(run_async(main_async(parser.parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
