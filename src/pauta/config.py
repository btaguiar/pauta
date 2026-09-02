"""Configuração do Pauta. Nenhuma constante mágica solta no resto do código."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HitlMode = Literal["auto", "interrupt"]


class Settings(BaseSettings):
    """Configuração lida do ambiente, com espelho vazio em `.env.example`.

    Os três modelos de papel e o embedding não têm default: um default aqui vira
    um modelo hardcoded por outro nome (ADR 007).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # Modelos por papel (ADR 007). Obrigatórios de propósito.
    MODEL_WORKER: str
    MODEL_ROUTER: str
    MODEL_CRITIC: str
    EMBEDDING_MODEL: str

    # Juiz do eval. A credencial vem da variável padrão do provider dele.
    JUDGE_MODEL: str | None = None

    # Acesso aos modelos via gateway OpenRouter, que fala o protocolo da OpenAI.
    # Uma chave, vários providers, e o juiz do eval pode ser de outra casa sem
    # exigir uma segunda conta.
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    TAVILY_API_KEY: str | None = None

    # Banco. Valor de desenvolvimento idêntico ao do docker-compose.yml.
    DATABASE_URL: str = "postgresql://pauta:pauta@localhost:5432/pauta"

    # Comportamento do grafo. Estes números são ponto de partida, não meta:
    # quem os calibra é o eval.
    HITL_MODE: HitlMode = "auto"
    MAX_SUPERVISOR_STEPS: int = Field(default=8, gt=0)
    MAX_CRITIC_LOOPS: int = Field(default=2, ge=0)
    BUDGET_TOKENS_PER_RUN: int = Field(default=60_000, gt=0)
    NODE_TIMEOUT_S: float = Field(default=20.0, gt=0)
    NODE_RETRIES: int = Field(default=2, ge=0)
    # Sem teto, um agente com tool entra em loop de chamadas sozinho.
    MAX_TOOL_ROUNDS: int = Field(default=3, gt=0)

    # Retriever.
    RETRIEVER_TOP_K: int = Field(default=5, gt=0)
    CHUNK_SIZE: int = Field(default=800, gt=0)
    CHUNK_OVERLAP: int = Field(default=100, ge=0)

    # Preço por milhão de tokens do modelo medido. Sem isto, o relatório mostra
    # tokens e diz que não calculou custo, em vez de estimar um número inventado.
    COST_PER_MTOK_USD: float | None = None

    # Guardrails da API pública.
    DAILY_BUDGET_USD: float = Field(default=5.00, gt=0)
    RATE_LIMIT_PER_IP: str = "3/hour"

    # Trace opcional, desligado por padrão.
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "pauta"

    @field_validator(
        "COST_PER_MTOK_USD",
        "JUDGE_MODEL",
        "OPENROUTER_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY",
        mode="before",
    )
    @classmethod
    def _blank_is_absent(cls, value: object) -> object:
        """Campo vazio no `.env` significa ausente, não string vazia.

        O `.env.example` deixa os opcionais em branco de propósito, e um campo
        numérico em branco quebraria o parse.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Instância única. Importar este módulo não exige ambiente configurado."""
    return Settings()  # type: ignore[call-arg]  # os obrigatórios vêm do ambiente
