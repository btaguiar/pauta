"""Compara configurações de modelo sobre o mesmo golden set.

    uv run python eval/matrix.py --limit 5

A pergunta que este script existe para responder: vale pagar modelo melhor em
qual papel? O supervisor decide rota e o código corrige a decisão dele, então a
hipótese do projeto é que router barato não custa qualidade. O crítico é o
oposto: crítico fraco tende a carimbar `verdict: ok`, e aí o loop de crítica
inteiro vira enfeite. Isto aqui mede as duas coisas em vez de supor.

Cada configuração roda o mesmo conjunto de tarefas, com a mesma semente de
repetições, e o resultado sai lado a lado com o delta contra a linha de base.

O script não escolhe nada. Ele produz o número; a escolha é de quem lê, e vira
ADR no `ARCHITECTURE.md` mais um comentário no `.env.example`.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pauta.config import get_settings
from pauta.memory.checkpointer import run_async
from pauta.models import reset_model_cache
from pauta.observability import emit, setup_logging
from run_eval import (
    DEFAULT_REPEATS,
    DEFAULT_TASKS,
    RESULTS_DIR,
    commit_hash,
    config_block,
    load_tasks,
    metrics_of,
    run_task,
    select_tasks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = REPO_ROOT / "eval" / "matrix.json"

#: As variáveis que uma configuração da matriz pode trocar. Só papéis de chat: o
#: embedding muda o índice, e trocá-lo no meio da matriz compararia corpus
#: diferentes em vez de modelos diferentes.
TUNABLE = ("MODEL_ROUTER", "MODEL_CRITIC", "MODEL_WORKER")

#: As métricas que entram na tabela, na ordem em que decidem.
COMPARED = (
    "cobertura",
    "incerteza_sinalizada",
    "calculadora_usada",
    "dentro_do_teto",
    "fidelidade",
    "tokens_media",
    "latencia_media_s",
    "custo_usd",
)


class InvalidSpec(ValueError):
    """O arquivo da matriz não descreve configurações rodáveis."""


@dataclass(frozen=True)
class Configuration:
    """Uma linha da matriz: um nome e os modelos que ela usa."""

    name: str
    models: dict[str, str]
    notes: str = ""


@dataclass
class ConfigurationResult:
    """O que uma configuração produziu sobre o golden set."""

    name: str
    models: dict[str, str]
    notes: str
    metrics: dict[str, Any]


def load_spec(path: Path) -> list[Configuration]:
    """Lê e valida a matriz. Configuração com modelo em branco não roda.

    Falhar aqui é barato. Descobrir no meio da terceira configuração que a
    quarta está vazia custa o dinheiro das três primeiras.
    """
    if not path.is_file():
        raise InvalidSpec(f"não achei a matriz em {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("configuracoes") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise InvalidSpec(f"{path.name} precisa de uma lista 'configuracoes' não vazia")

    configurations: list[Configuration] = []
    seen: set[str] = set()
    for entry in entries:
        name = str(entry.get("nome", "")).strip()
        if not name:
            raise InvalidSpec("toda configuração precisa de um 'nome'")
        if name in seen:
            raise InvalidSpec(f"nome repetido na matriz: {name!r}")
        seen.add(name)
        models = {key: str(entry.get(key, "")).strip() for key in TUNABLE}
        missing = sorted(key for key, value in models.items() if not value)
        if missing:
            raise InvalidSpec(
                f"a configuração {name!r} está com {', '.join(missing)} em branco; "
                f"preencha {path.name} antes de rodar a matriz"
            )
        configurations.append(
            Configuration(name=name, models=models, notes=str(entry.get("nota", "")))
        )
    return configurations


def apply_configuration(configuration: Configuration) -> None:
    """Coloca a configuração no ambiente e descarta os caches.

    Mexer em `os.environ` é coisa de ponto de entrada, e este script é um. Sem
    limpar os dois caches, `get_settings` e `get_model` devolveriam a
    configuração anterior e a matriz inteira mediria a mesma coisa várias vezes.
    """
    for key, value in configuration.models.items():
        os.environ[key] = value
    get_settings.cache_clear()
    reset_model_cache()


async def run_configuration(
    configuration: Configuration,
    tasks: list[dict[str, Any]],
    *,
    skipped: int,
    repeats: int,
) -> ConfigurationResult:
    apply_configuration(configuration)
    emit("node_start", node="matrix", configuration=configuration.name, **configuration.models)
    results = [
        await run_task(task, index=index, repeat=repeat)
        for repeat in range(repeats)
        for index, task in enumerate(tasks)
    ]
    metrics = metrics_of(results, skipped=skipped, price=get_settings().COST_PER_MTOK_USD)
    emit("node_end", node="matrix", configuration=configuration.name, falhas=metrics["falhas"])
    return ConfigurationResult(
        name=configuration.name,
        models=configuration.models,
        notes=configuration.notes,
        metrics=metrics,
    )


def delta_against(baseline: ConfigurationResult, other: ConfigurationResult) -> dict[str, float]:
    """Diferença de cada métrica contra a linha de base, que é a primeira linha."""
    deltas: dict[str, float] = {}
    for key in COMPARED:
        first, second = baseline.metrics.get(key), other.metrics.get(key)
        if isinstance(first, int | float) and isinstance(second, int | float):
            deltas[key] = round(float(second) - float(first), 4)
    return deltas


def cell(value: Any) -> str:
    if value is None:
        return "n/d"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_matrix(results: list[ConfigurationResult]) -> str:
    """A tabela lado a lado, com o `n` de cada métrica logo abaixo."""
    if not results:
        return "nenhuma configuração rodou\n"
    width = max(len(result.name) for result in results) + 2
    header = "configuração".ljust(width) + " ".join(key.rjust(14) for key in COMPARED)
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            result.name.ljust(width)
            + " ".join(cell(result.metrics.get(key)).rjust(14) for key in COMPARED)
        )

    baseline = results[0]
    if len(results) > 1:
        lines += ["", f"delta contra a linha de base ({baseline.name}):"]
        for result in results[1:]:
            deltas = delta_against(baseline, result)
            lines.append(
                result.name.ljust(width)
                + " ".join(f"{deltas.get(key, 0.0):+.4f}".rjust(14) for key in COMPARED)
            )

    denominators = " · ".join(
        f"{key}_n={baseline.metrics.get(f'{key}_n', 0)}"
        for key in ("cobertura", "incerteza_sinalizada", "fidelidade")
    )
    lines += [
        "",
        f"denominadores da linha de base: {denominators}",
        "",
        "Este script não escolhe. Leve o número para uma ADR no ARCHITECTURE.md",
        "e para um comentário no .env.example dizendo o que o justificou.",
    ]
    return "\n".join(lines)


def write_matrix(results: list[ConfigurationResult], *, repeats: int, limit: int | None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commit = commit_hash()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"matriz_{stamp}_{commit}.json"
    baseline = results[0] if results else None
    payload = {
        "timestamp": stamp,
        "commit": commit,
        "repeticoes": repeats,
        "limit": limit,
        # O bloco comum sai da configuração corrente; o que varia por linha está
        # em `modelos`, dentro de cada configuração.
        "config": config_block(get_settings()),
        "configuracoes": [
            {
                "nome": result.name,
                "nota": result.notes,
                "modelos": result.models,
                "metricas": result.metrics,
                "delta_vs_base": delta_against(baseline, result) if baseline else {},
            }
            for result in results
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return path


MISSING_CONFIG_MESSAGE = """Falta configuração para rodar a matriz.

Copie .env.example para .env e preencha OPENROUTER_API_KEY e EMBEDDING_MODEL.
Os três modelos de papel vêm de eval/matrix.json, uma linha por configuração.

Detalhe: {detail}
"""


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    try:
        configurations = load_spec(Path(args.spec))
    except (InvalidSpec, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    # Uma configuração precisa estar aplicada antes do primeiro `get_settings`,
    # senão o validador reclama dos modelos de papel que a matriz é quem fornece.
    apply_configuration(configurations[0])
    try:
        get_settings()
    except ValidationError as exc:
        sys.stderr.write(MISSING_CONFIG_MESSAGE.format(detail=exc))
        return 2

    tasks, skipped = select_tasks(load_tasks(Path(args.tasks)), args.limit)
    if not tasks:
        sys.stdout.write(f"nenhuma tarefa executável; {skipped} puladas por falta de corpus\n")
        return 1

    results = [
        await run_configuration(configuration, tasks, skipped=skipped, repeats=args.repeats)
        for configuration in configurations
    ]
    sys.stdout.write(render_matrix(results) + "\n")
    path = write_matrix(results, repeats=args.repeats, limit=args.limit)
    sys.stdout.write(f"\ngravado: {path}\n")
    return 1 if any(result.metrics["falhas"] for result in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="roda o golden set em várias configurações de modelo e compara"
    )
    parser.add_argument("--spec", default=str(DEFAULT_SPEC), help="caminho do matrix.json")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="caminho do jsonl")
    parser.add_argument("--limit", type=int, default=None, help="quantas tarefas por configuração")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"execuções por tarefa (padrão {DEFAULT_REPEATS})",
    )
    return int(run_async(main_async(parser.parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
