"""Checkpointer. O de Postgres exige banco e é pulado quando não há."""

import os

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from pauta.config import get_settings
from pauta.memory.checkpointer import memory_checkpointer, postgres_checkpointer

requires_postgres = pytest.mark.skipif(
    os.environ.get("PAUTA_TEST_POSTGRES") != "1",
    reason="precisa de Postgres; rode docker compose up e exporte PAUTA_TEST_POSTGRES=1",
)


def test_the_memory_checkpointer_is_the_one_for_tests() -> None:
    saver = memory_checkpointer()
    assert isinstance(saver, InMemorySaver)
    assert isinstance(saver, BaseCheckpointSaver)


def test_the_database_url_is_configured_not_hardcoded() -> None:
    assert get_settings().DATABASE_URL.startswith("postgresql://")


@requires_postgres
async def test_postgres_checkpointer_round_trips_a_thread() -> None:
    """Com banco, o estado sobrevive a fechar e reabrir a conexão."""
    from langchain_core.runnables import RunnableConfig

    from pauta.agents.supervisor import Router
    from pauta.graph.builder import build_graph
    from pauta.graph.state import new_state
    from tests.fakes import FakeChatModel, fake_graph_models

    config: RunnableConfig = {"configurable": {"thread_id": "pg-round-trip"}}
    async with postgres_checkpointer() as saver:
        graph = build_graph(
            **fake_graph_models(
                supervisor_model=FakeChatModel(responses=[Router(next="writer", rationale="x")]),
                writer_model=FakeChatModel(responses=["Briefing."]),
            ),
            checkpointer=saver,
        )
        await graph.ainvoke(new_state(task="durável", run_id="r1"), config=config)

    async with postgres_checkpointer(setup=False) as saver:
        graph = build_graph(**fake_graph_models(), checkpointer=saver)
        snapshot = await graph.aget_state(config)
        assert snapshot.values["task"] == "durável"
