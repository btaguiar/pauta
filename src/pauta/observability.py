"""Log estruturado em JSON e toggle de trace.

Um emissor, dois destinos: o mesmo evento vira linha de log agora e frame SSE
quando a API existir (semana 3). Nada de `print` em lugar nenhum.

Os tipos de evento são os da secção 8.6 do dossiê, mais `node_end`, que carrega
o `latency_ms` de cada nó e não tem equivalente naquela lista.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import get_settings

EventType = Literal[
    "node_start",
    "node_end",
    "tool_call",
    "finding",
    "critique",
    "token",
    "interrupt",
    "final",
    "usage",
    "error",
]

LOGGER_NAME = "pauta"
_PAYLOAD_KEY = "pauta"


class JsonFormatter(logging.Formatter):
    """Serializa cada registro como uma linha de JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, _PAYLOAD_KEY, None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO, stream: Any | None = None) -> logging.Logger:
    """Configura o logger do projeto. Idempotente: chamar duas vezes não duplica saída."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def emit(event: EventType, **fields: Any) -> None:
    """Emite um evento estruturado. `fields` vira chave de primeiro nível no JSON."""
    level = logging.ERROR if event == "error" else logging.INFO
    get_logger().log(level, event, extra={_PAYLOAD_KEY: {"event": event, **fields}})


@dataclass
class NodeSpan:
    """Contexto de um nó em execução. O corpo do nó preenche o que mediu."""

    node: str
    run_id: str
    thread_id: str
    iteration: int
    tokens_used: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_fields(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "iteration": self.iteration,
            "tokens_used": self.tokens_used,
            **self.extra,
        }


@contextmanager
def node_span(
    node: str,
    *,
    run_id: str,
    thread_id: str,
    iteration: int,
) -> Iterator[NodeSpan]:
    """Emite `node_start`, mede a latência e emite `node_end`.

    Se o nó levantar, emite `error` com a latência até a falha e propaga. Falha
    visível é melhor que run travada em silêncio.
    """
    span = NodeSpan(node=node, run_id=run_id, thread_id=thread_id, iteration=iteration)
    emit("node_start", **span.as_fields())
    started = time.perf_counter()
    try:
        yield span
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        emit(
            "error",
            **span.as_fields(),
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    emit("node_end", **span.as_fields(), latency_ms=latency_ms)


def configure_tracing() -> bool:
    """Liga o trace do LangSmith quando `LANGSMITH_TRACING` está ligado.

    Devolve se o trace ficou ligado. Desligado por padrão, e desligado no CI.
    """
    settings = get_settings()
    if not settings.LANGSMITH_TRACING:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    if not settings.LANGSMITH_API_KEY:
        emit("error", message="LANGSMITH_TRACING ligado sem LANGSMITH_API_KEY; trace desligado")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    return True
