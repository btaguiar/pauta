"""O stream de eventos: o segundo destino do mesmo emissor que escreve o log."""

import asyncio
import json
from collections.abc import Iterator

import pytest

from pauta.api.events import DONE, EventBroadcaster, frame, is_terminal
from pauta.observability import emit, setup_logging


@pytest.fixture
def broadcaster() -> Iterator[EventBroadcaster]:
    setup_logging()
    caster = EventBroadcaster(queue_size=5)
    caster.install()
    try:
        yield caster
    finally:
        caster.uninstall()


def payloads(queue: asyncio.Queue[dict[str, object]]) -> list[dict[str, object]]:
    drained: list[dict[str, object]] = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    return drained


async def test_an_event_reaches_whoever_is_listening(broadcaster: EventBroadcaster) -> None:
    with broadcaster.listen("r1") as queue:
        emit("finding", run_id="r1", node="research", source="https://a")
        got = payloads(queue)

    assert [item["event"] for item in got] == ["finding"]
    assert got[0]["source"] == "https://a"


async def test_one_run_does_not_see_the_events_of_another(broadcaster: EventBroadcaster) -> None:
    with broadcaster.listen("r1") as mine, broadcaster.listen("r2") as other:
        emit("finding", run_id="r1", node="research")
        assert len(payloads(mine)) == 1
        assert payloads(other) == []


async def test_two_clients_can_watch_the_same_run(broadcaster: EventBroadcaster) -> None:
    with broadcaster.listen("r1") as first, broadcaster.listen("r1") as second:
        emit("tool_call", run_id="r1", node="research", tool="retriever")
        assert len(payloads(first)) == 1
        assert len(payloads(second)) == 1


async def test_an_event_without_a_run_is_not_broadcast(broadcaster: EventBroadcaster) -> None:
    """Eventos de startup não pertencem a run nenhuma e não podem poluir um stream."""
    with broadcaster.listen("r1") as queue:
        emit("node_start", node="checkpointer", message="pronto")
        assert payloads(queue) == []


async def test_leaving_the_context_cancels_the_subscription(
    broadcaster: EventBroadcaster,
) -> None:
    with broadcaster.listen("r1"):
        assert broadcaster.listeners == 1
    assert broadcaster.listeners == 0
    emit("finding", run_id="r1", node="research")


async def test_the_run_closing_closes_the_stream(broadcaster: EventBroadcaster) -> None:
    """Sem este sinal o gerador esperaria para sempre por uma run que já acabou."""
    with broadcaster.listen("r1") as queue:
        emit("node_end", run_id="r1", node="runner", status="completed")
        kinds = [item["event"] for item in payloads(queue)]

    assert kinds == ["node_end", DONE]


async def test_a_failed_run_also_closes_the_stream(broadcaster: EventBroadcaster) -> None:
    with broadcaster.listen("r1") as queue:
        emit("error", run_id="r1", node="runner", error="caiu")
        assert [item["event"] for item in payloads(queue)] == ["error", DONE]


def test_only_the_runner_ends_the_run() -> None:
    assert is_terminal({"node": "runner", "event": "node_end"}) is True
    assert is_terminal({"node": "runner", "event": "error"}) is True
    assert is_terminal({"node": "research", "event": "error"}) is False
    assert is_terminal({"node": "runner", "event": "node_start"}) is False


async def test_a_slow_client_loses_events_instead_of_holding_the_run(
    broadcaster: EventBroadcaster,
) -> None:
    """Cliente lento não pode segurar a execução, e o que ele perdeu é dito a ele."""
    with broadcaster.listen("r1") as queue:
        for index in range(20):
            emit("tool_call", run_id="r1", node="research", tool=f"t{index}")
        got = payloads(queue)

    assert len(got) <= broadcaster.queue_size
    assert any(item.get("error") == "evento perdido, fila cheia" for item in got)


def test_the_frame_carries_the_contract_name_of_the_event() -> None:
    text = frame({"event": "finding", "run_id": "r1", "content": "a"})
    assert text.startswith("event: finding\ndata: ")
    assert text.endswith("\n\n")
    body = json.loads(text.split("data: ", 1)[1])
    assert body["type"] == "finding"
    assert body["content"] == "a"
    assert "event" not in body


def test_the_frame_survives_a_value_json_cannot_serialise() -> None:
    """Um evento com objeto estranho não pode derrubar o stream inteiro."""
    text = frame({"event": "critique", "run_id": "r1", "gaps": {"conjunto"}})
    assert json.loads(text.split("data: ", 1)[1])["type"] == "critique"
