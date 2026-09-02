"""Durabilidade (b): morte suja.

O processo é morto sem chance de limpar nada. O checkpoint do último superstep
concluído sobrevive, e a run retoma pelo `thread_id`. Exige Postgres: com
`InMemorySaver` o estado morre junto com o processo e o teste não provaria nada.
"""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig

from pauta.agents.supervisor import Router
from pauta.graph.builder import build_graph
from pauta.memory.checkpointer import postgres_checkpointer
from tests.fakes import FakeChatModel, fake_graph_models

requires_postgres = pytest.mark.skipif(
    os.environ.get("PAUTA_TEST_POSTGRES") != "1",
    reason="precisa de Postgres; rode docker compose up e exporte PAUTA_TEST_POSTGRES=1",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHILD = REPO_ROOT / "tests" / "durability_child.py"
SIGNAL_TIMEOUT_S = 120


def wait_for(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return True
        time.sleep(0.2)
    return False


@requires_postgres
async def test_the_run_survives_a_hard_kill(tmp_path: Path) -> None:
    thread_id = f"kill-{uuid.uuid4()}"
    signal_path = tmp_path / "sinal.txt"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    child = subprocess.Popen(
        [sys.executable, str(CHILD), thread_id, str(signal_path)],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    try:
        assert wait_for(signal_path, SIGNAL_TIMEOUT_S), "o filho não chegou ao writer"
        # Morte suja: nenhum handler roda, nada é fechado, nada é gravado na saída.
        child.kill()
        child.wait(timeout=30)
    finally:
        if child.poll() is None:  # pragma: no cover - só se o kill falhar
            child.kill()

    assert child.returncode != 0

    async with postgres_checkpointer(setup=False) as saver:
        graph = build_graph(**fake_graph_models(), checkpointer=saver)
        snapshot = await graph.aget_state(config)

        # O trabalho concluído antes da morte está no checkpoint.
        assert snapshot.values["task"] == "sobreviver ao kill"
        assert [f.content for f in snapshot.values["findings"]] == ["descoberta antes da morte"]
        assert snapshot.values["critiques"][0].verdict == "ok"
        # O writer não chegou a terminar, então não há relatório.
        assert snapshot.values.get("final_report") is None
        assert snapshot.next == ("writer",)

    # E a run retoma de onde parou, num processo novo.
    async with postgres_checkpointer(setup=False) as saver:
        resumed = build_graph(
            **fake_graph_models(
                supervisor_model=FakeChatModel(responses=[Router(next="END", rationale="pronto")]),
                writer_model=FakeChatModel(responses=["Briefing depois do kill."]),
            ),
            checkpointer=saver,
        )
        final = await resumed.ainvoke(None, config=config)

    assert final["final_report"] == "Briefing depois do kill."
    assert [f.content for f in final["findings"]] == ["descoberta antes da morte"]
