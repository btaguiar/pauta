import io
import json
import logging
from typing import Any

import pytest

from pauta.observability import configure_tracing, emit, node_span, setup_logging


@pytest.fixture
def captured() -> io.StringIO:
    stream = io.StringIO()
    setup_logging(level=logging.INFO, stream=stream)
    return stream


def lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_every_record_is_one_json_line(captured: io.StringIO) -> None:
    emit("finding", run_id="r1", thread_id="t1", node="research", source="https://exemplo")
    (record,) = lines(captured)
    assert record["event"] == "finding"
    assert record["node"] == "research"
    assert record["level"] == "info"
    assert "ts" in record


def test_error_event_is_logged_at_error_level(captured: io.StringIO) -> None:
    emit("error", run_id="r1", error="tool falhou")
    (record,) = lines(captured)
    assert record["level"] == "error"


def test_node_span_carries_the_six_required_fields(captured: io.StringIO) -> None:
    with node_span("research", run_id="r1", thread_id="t1", iteration=2) as span:
        span.tokens_used = 1234
    start, end = lines(captured)
    assert start["event"] == "node_start"
    assert end["event"] == "node_end"
    for key in ("run_id", "thread_id", "node", "iteration", "tokens_used"):
        assert key in end
    assert end["tokens_used"] == 1234
    assert end["latency_ms"] >= 0


def test_node_span_emits_error_and_reraises(captured: io.StringIO) -> None:
    with pytest.raises(RuntimeError), node_span("critic", run_id="r1", thread_id="t1", iteration=1):
        raise RuntimeError("parse estruturado falhou")
    start, failure = lines(captured)
    assert start["event"] == "node_start"
    assert failure["event"] == "error"
    assert failure["error_type"] == "RuntimeError"
    assert failure["node"] == "critic"
    assert "latency_ms" in failure


def test_setup_logging_is_idempotent(captured: io.StringIO) -> None:
    setup_logging(level=logging.INFO, stream=captured)
    emit("token", run_id="r1")
    assert len(lines(captured)) == 1


def test_tracing_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert configure_tracing() is False


def test_tracing_without_key_stays_off(
    monkeypatch: pytest.MonkeyPatch, captured: io.StringIO
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    from pauta.config import get_settings

    get_settings.cache_clear()
    assert configure_tracing() is False
    assert lines(captured)[0]["level"] == "error"


def test_tracing_on_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "chave-de-teste")
    from pauta.config import get_settings

    get_settings.cache_clear()
    assert configure_tracing() is True
    import os

    assert os.environ["LANGSMITH_PROJECT"] == "pauta"
