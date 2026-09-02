"""Fixtures comuns. Nenhum teste aqui toca provider real."""

import asyncio
import os
from collections.abc import Callable, Iterator, Mapping

import pytest

from pauta.config import Settings, get_settings
from pauta.memory.checkpointer import loop_factory

REQUIRED_ENV = {
    "MODEL_WORKER": "fake/worker",
    "MODEL_ROUTER": "fake/router",
    "MODEL_CRITIC": "fake/critic",
    "EMBEDDING_MODEL": "fake/embedding",
}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isola o ambiente do `.env` da máquina e limpa o cache de settings."""
    for key in list(os.environ):
        if key.startswith(("MODEL_", "LANGSMITH_", "EMBEDDING_", "JUDGE_", "OPENROUTER_")):
            monkeypatch.delenv(key, raising=False)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def pytest_asyncio_loop_factories(
    config: pytest.Config,
    item: pytest.Item,
) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
    """Roda os testes no mesmo loop que a aplicação usa.

    No Windows o loop padrão é o Proactor, e o psycopg async recusa rodar sobre
    ele. Sem isto, todo teste que fala com Postgres morre com um InterfaceError
    que não tem nada a ver com o que o teste queria provar.
    """
    return {"pauta": loop_factory() or asyncio.new_event_loop}
