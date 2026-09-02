"""Processo filho do teste de morte suja.

Roda o grafo contra o Postgres até o writer, avisa que chegou lá escrevendo um
arquivo, e trava. O teste então mata este processo sem aviso. O checkpoint do
último superstep concluído precisa sobreviver.

Uso: python tests/durability_child.py <thread_id> <arquivo_de_sinal>
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from pauta.agents.research import ResearchOutput
from pauta.agents.supervisor import Router
from pauta.config import get_settings
from pauta.graph.builder import build_graph
from pauta.graph.state import Critique, Finding, new_state
from pauta.memory.checkpointer import postgres_checkpointer, run_async
from tests.fakes import FakeChatModel, fake_graph_models

SIGNAL_TEXT = "cheguei no writer"


class BlockingFakeWriter(FakeChatModel):
    """Escreve o sinal e trava, para o teste poder matar o processo aqui."""

    signal_path: str = ""

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        Path(self.signal_path).write_text(SIGNAL_TEXT, encoding="utf-8")
        await asyncio.sleep(3600)
        raise AssertionError("não deveria acordar")


async def main(thread_id: str, signal_path: str) -> None:
    settings = get_settings().model_copy(update={"NODE_TIMEOUT_WRITER_S": 3600.0})
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    async with postgres_checkpointer() as saver:
        graph = build_graph(
            **fake_graph_models(
                supervisor_model=FakeChatModel(
                    responses=[
                        Router(next="research", rationale="reunir"),
                        Router(next="critic", rationale="validar"),
                        Router(next="writer", rationale="redigir"),
                    ]
                ),
                research_model=FakeChatModel(
                    responses=[
                        "achei",
                        ResearchOutput(
                            findings=[
                                Finding(
                                    content="descoberta antes da morte",
                                    source="https://a",
                                    agent="research",
                                )
                            ]
                        ),
                    ]
                ),
                critic_model=FakeChatModel(responses=[Critique(verdict="ok")]),
                writer_model=BlockingFakeWriter(responses=["nunca sai"], signal_path=signal_path),
            ),
            settings=settings,
            checkpointer=saver,
        )
        await graph.ainvoke(
            new_state(task="sobreviver ao kill", run_id="r1"),
            config=config,
            durability="sync",
        )


if __name__ == "__main__":
    run_async(main(sys.argv[1], sys.argv[2]))
