"""Executa tarefas do golden set contra o grafo e imprime o que custou.

Versão da semana 1: uma repetição, sem juiz. As repetições, o desvio padrão e o
LLM-as-judge de outro provider entram na semana 3, junto com o EVALUATION.md.

Uso:
    uv run python eval/run_eval.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from pauta import tools
from pauta.config import Settings, get_settings
from pauta.graph.builder import build_graph
from pauta.graph.state import new_state
from pauta.memory.checkpointer import memory_checkpointer, run_async
from pauta.observability import PAYLOAD_KEY, get_logger, setup_logging
from scoring import SCORE_KEYS, aggregate, score_task

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = REPO_ROOT / "eval" / "tasks.jsonl"
SAMPLES_DIR = REPO_ROOT / "samples"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
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
    tools_called: list[str] = field(default_factory=list)
    scores: dict[str, float | bool | None] = field(default_factory=dict)

    def cost_usd(self, price_per_mtok: float | None) -> float | None:
        if price_per_mtok is None:
            return None
        return self.tokens_used / TOKENS_PER_MILLION * price_per_mtok


class EventCollector(logging.Handler):
    """Junta os eventos estruturados que uma tarefa emitiu.

    O grafo já publica `tool_call` para cada ferramenta que roda. Ler o próprio
    stream é mais fiel que inferir do estado final se a calculadora foi usada:
    um finding do analyst prova que ele escreveu algo, não que ele calculou.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        payload = getattr(record, PAYLOAD_KEY, None)
        if isinstance(payload, dict):
            self.events.append(payload)

    def tools_called(self) -> list[str]:
        return [
            str(event["tool"])
            for event in self.events
            if event.get("event") == "tool_call" and "tool" in event
        ]


@contextmanager
def collecting_events() -> Iterator[EventCollector]:
    """Escuta o logger durante uma tarefa e larga o handler ao sair."""
    handler = EventCollector()
    logger = get_logger()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corpus_is_empty() -> bool:
    """Vazio é não ter documento indexável, não apenas não ter arquivo."""
    return tools.corpus_is_empty(SAMPLES_DIR)


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
    graph = build_graph(
        research_tools=tools.research_tools(settings),
        analyst_tools=tools.analyst_tools(),
        settings=settings,
        # Em memória, e de propósito: o eval roda dezenas de tarefas descartáveis
        # e não deve encher o banco de checkpoint que a operação usa. A escolha
        # fica escrita aqui em vez de vir do default de `build_graph`.
        checkpointer=memory_checkpointer(),
    )
    run_id = f"eval-{task['id']}"
    config: RunnableConfig = {"configurable": {"thread_id": f"{run_id}-{index}"}}
    started = time.perf_counter()
    with collecting_events() as events:
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
                tools_called=events.tools_called(),
            )
        called = events.tools_called()

    report = final.get("final_report") or ""
    findings = len(final.get("findings", []))
    iterations = final.get("iteration", 0)
    return TaskResult(
        task_id=task["id"],
        task=task["task"],
        report=report,
        findings=findings,
        iterations=iterations,
        tokens_used=final.get("tokens_used", 0),
        latency_s=round(time.perf_counter() - started, 2),
        tools_called=called,
        scores=score_task(
            task,
            report=report,
            iterations=iterations,
            findings=findings,
            tools_called=called,
        ),
    )


def render_score(value: float | bool) -> str:
    """Booleano vira sim ou nao; fração vira número. Cada rótulo lido do jeito dele."""
    if isinstance(value, bool):
        return "sim" if value else "NAO"
    return f"{value:.2f}"


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
        scored = {key: value for key, value in result.scores.items() if value is not None}
        if scored:
            lines.append(
                "rótulos: "
                + " · ".join(f"{key}={render_score(value)}" for key, value in scored.items())
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

    summary = aggregate([result.scores for result in results if result.scores])
    lines.append("")
    lines.append("rótulos do golden set:")
    for key in SCORE_KEYS:
        denominator = summary[f"{key}_n"]
        value = f"{summary[key]:.4f}" if denominator else "sem tarefa que exija"
        lines.append(f"  {key}: {value} (n={denominator})")
    return "\n".join(lines)


def commit_hash() -> str:
    """SHA curto do HEAD. Sem ele dois resultados não se comparam no tempo."""
    try:
        finished = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "sem-git"
    return finished.stdout.strip() or "sem-git"


def config_block(settings: Settings) -> dict[str, Any]:
    """O que muda o resultado, e só isso.

    A lista é explícita, nunca um despejo do `Settings`: chave de API não entra
    em artefato que vai para o disco e possivelmente para o repositório. Sem
    este bloco, dois JSON de commits diferentes não são comparáveis, porque não
    dá para saber se a diferença veio do código ou de outro modelo.
    """
    return {
        "model_worker": settings.MODEL_WORKER,
        "model_router": settings.MODEL_ROUTER,
        "model_critic": settings.MODEL_CRITIC,
        "embedding_model": settings.EMBEDDING_MODEL,
        "judge_model": settings.JUDGE_MODEL,
        "temperature": 0,
        "max_supervisor_steps": settings.MAX_SUPERVISOR_STEPS,
        "max_critic_loops": settings.MAX_CRITIC_LOOPS,
        "budget_tokens_per_run": settings.BUDGET_TOKENS_PER_RUN,
        "max_tool_rounds": settings.MAX_TOOL_ROUNDS,
        "retriever_top_k": settings.RETRIEVER_TOP_K,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "hitl_mode": settings.HITL_MODE,
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Percentil por interpolação linear. Sem dependência nova para uma conta destas."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 3)


def metrics_of(
    results: Sequence[TaskResult], *, skipped: int, price: float | None
) -> dict[str, Any]:
    """As métricas da rodada, cada média acompanhada do seu `_n`."""
    finished = [result for result in results if not result.error]
    tokens = [result.tokens_used for result in finished]
    latencies = [result.latency_s for result in finished]
    total_tokens = sum(tokens)
    return {
        "total_tarefas": len(results),
        "falhas": sum(1 for result in results if result.error),
        "puladas_sem_corpus": skipped,
        **aggregate([result.scores for result in results if result.scores]),
        "tokens_total": total_tokens,
        "tokens_media": round(mean(tokens), 1) if tokens else 0.0,
        "tokens_media_n": len(tokens),
        "latencia_media_s": round(mean(latencies), 3) if latencies else 0.0,
        "latencia_p95_s": percentile(latencies, 0.95),
        "latencia_n": len(latencies),
        "custo_usd": (
            round(total_tokens / TOKENS_PER_MILLION * price, 4) if price is not None else None
        ),
    }


def write_results(
    results: Sequence[TaskResult],
    *,
    skipped: int,
    settings: Settings,
    tasks_file: str,
    limit: int | None,
) -> list[Path]:
    """Grava a rodada em três arquivos, como o `grifo` faz.

    `bruto` guarda o que cada tarefa produziu, `eval` junta cabeçalho, métricas e
    itens, `metricas` fica só com os números, que é o arquivo que alimenta um
    gráfico de evolução por commit sem carregar briefing nenhum.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commit = commit_hash()
    header = {
        "timestamp": stamp,
        "commit": commit,
        "tasks_file": Path(tasks_file).name,
        "limit": limit,
        "config": config_block(settings),
    }
    metrics = metrics_of(results, skipped=skipped, price=settings.COST_PER_MTOK_USD)
    items = [asdict(result) for result in results]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, payload in (
        (f"bruto_{stamp}_{commit}.json", {**header, "itens": items}),
        (f"eval_{stamp}_{commit}.json", {**header, "metricas": metrics, "itens": items}),
        (f"metricas_{stamp}_{commit}.json", {**header, "metricas": metrics}),
    ):
        path = RESULTS_DIR / name
        # allow_nan=False: NaN é extensão do Python e não é JSON válido. Melhor
        # falhar na escrita que gravar um arquivo que nenhum leitor abre.
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        written.append(path)
    return written


MISSING_CONFIG_MESSAGE = """Falta configuração para rodar o eval.

Copie .env.example para .env e preencha MODEL_WORKER, MODEL_ROUTER, MODEL_CRITIC
e EMBEDDING_MODEL. Nenhum destes tem valor padrão no código, de propósito.

Detalhe do validador:
{detail}
"""


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    try:
        settings = get_settings()
    except ValidationError as exc:
        sys.stderr.write(MISSING_CONFIG_MESSAGE.format(detail=exc))
        return 2
    tasks = load_tasks(Path(args.tasks))
    selected, skipped = select_tasks(tasks, args.limit)
    if not selected:
        sys.stdout.write(f"nenhuma tarefa executável; {skipped} puladas por falta de corpus\n")
        return 1
    results = [await run_task(task, index=i) for i, task in enumerate(selected)]
    sys.stdout.write(render_report(results, skipped, settings.COST_PER_MTOK_USD) + "\n")

    written = write_results(
        results,
        skipped=skipped,
        settings=settings,
        tasks_file=args.tasks,
        limit=args.limit,
    )
    sys.stdout.write("\n" + "\n".join(f"gravado: {path}" for path in written) + "\n")
    return 1 if any(r.error for r in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="roda tarefas do golden set contra o grafo")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="caminho do jsonl")
    parser.add_argument("--limit", type=int, default=None, help="quantas tarefas rodar")
    return int(run_async(main_async(parser.parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
