"""Fixtures comuns. Nenhum teste aqui toca provider real."""

import os
from collections.abc import Iterator

import pytest

from pauta.config import Settings, get_settings

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
