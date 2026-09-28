"""Configuração do Pauta. Nenhuma constante mágica solta no resto do código."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HitlMode = Literal["auto", "interrupt"]

#: Provedor de busca web. A cota de um provedor só não pode parar o projeto,
#: então a escolha é configuração e não código (ADR 008). `ddgs` é o default
#: porque não exige chave e não custa.
SearchProvider = Literal["tavily", "ddgs"]

#: Engines do metasearch do `ddgs` para `text()`, mais `auto`. A lista vem da
#: documentação do pacote 9.16.0, e tem dez motores: a revisão do design listou
#: nove, sem `grokipedia`. Fechar em `Literal` faz o valor digitado errado falhar
#: no carregamento das settings em vez de dentro da run, depois de gastar token.
SearchBackend = Literal[
    "auto",
    "bing",
    "brave",
    "duckduckgo",
    "google",
    "grokipedia",
    "mojeek",
    "startpage",
    "yandex",
    "yahoo",
    "wikipedia",
]

#: Gateway dos modelos de chat: Model Studio da Alibaba, em modo compatível com
#: o protocolo da OpenAI. Serve três casas, Qwen, GLM e DeepSeek, então o juiz de
#: outro provider que o eval exige é uma troca de id, não uma segunda conta.
CHAT_GATEWAY_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"

#: Gateway dos embeddings, separado porque o de chat não serve nenhum. Medido em
#: 2026-09-25: `/embeddings` devolve 404 para `text-embedding-v4`,
#: `text-embedding-v3` e `text-embedding-3-small`. Naquele gateway 404 é também
#: como um modelo de chat fora do plano é recusado, então a leitura é "não está
#: no plano", e a lista de `/models` não traz embedding nenhum (ADR 005).
EMBEDDING_GATEWAY_URL = "https://openrouter.ai/api/v1"


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

    # Dois gateways, os dois falando o protocolo da OpenAI, cada um com a sua
    # chave. Chat e embedding só coincidiriam se o mesmo gateway servisse os
    # dois, e este não serve.
    # `API_KEY_ALIBABA` é aceito como segundo nome, porque é como a chave vem
    # batizada no painel do provider. Com os dois definidos, vale `CHAT_API_KEY`.
    CHAT_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CHAT_API_KEY", "API_KEY_ALIBABA"),
    )
    CHAT_BASE_URL: str = CHAT_GATEWAY_URL
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_BASE_URL: str = EMBEDDING_GATEWAY_URL
    TAVILY_API_KEY: str | None = None

    # Busca web (ADR 008). O provedor é escolhível porque a cota de um só não
    # pode parar o projeto.
    SEARCH_PROVIDER: SearchProvider = "ddgs"
    # Região da busca. O `ddgs` usa `us-en` por default, e o golden set é em
    # português: buscar em inglês americano muda o que volta e cai em `cobertura`
    # e na groundedness que o juiz avalia. Decisão de 2026-09-27.
    SEARCH_REGION: str = "br-pt"
    # Engine do metasearch. `auto` deixa o pacote escolher, e dois runs podem
    # buscar em motores diferentes, o que torna os artefatos incomparáveis.
    # Fixar o backend é o que torna a comparação honesta, pelo mesmo motivo pelo
    # qual a matriz não troca de embedding no meio.
    SEARCH_BACKEND: SearchBackend = "auto"
    # Teto da chamada de busca inteira, imposto por `asyncio.wait_for` na tool.
    # Medido em 2026-09-28: o `timeout` do `DDGS` vale por requisição HTTP, não
    # pela chamada, e com `timeout=10` uma chamada levou 31,2s. O mesmo valor
    # também vai para o `DDGS`, que assim desiste de cada requisição no prazo.
    WEB_SEARCH_TIMEOUT_S: float = Field(default=10.0, gt=0)

    # Banco. Valor de desenvolvimento idêntico ao do docker-compose.yml.
    DATABASE_URL: str = "postgresql://pauta:pauta@localhost:5432/pauta"

    # Comportamento do grafo. Estes números são ponto de partida, não meta:
    # quem os calibra é o eval.
    HITL_MODE: HitlMode = "auto"
    MAX_SUPERVISOR_STEPS: int = Field(default=8, gt=0)
    MAX_CRITIC_LOOPS: int = Field(default=2, ge=0)
    # Quantas voltas o research pode dar sem achar nada antes de a run desistir
    # e mandar o writer dizer o que faltou. Medido: sem este teto, uma tarefa
    # cuja resposta nao esta no corpus repetiu research quatro vezes e gastou
    # 58 mil tokens sem produzir briefing. Minimo 1: a primeira tentativa
    # sempre acontece.
    MAX_EMPTY_RESEARCH: int = Field(default=2, gt=0)
    BUDGET_TOKENS_PER_RUN: int = Field(default=60_000, gt=0)
    # Parte do orçamento que só o writer gasta. Os nós de trabalho param em
    # `BUDGET_TOKENS_PER_RUN - WRITER_RESERVE_TOKENS`, e o writer redige dentro
    # do que sobrou. Sem reserva, o writer rodava depois do teto e o estouro era
    # garantido. Medido em 2026-09-28 no gateway de chat atual (glm-5.3, busca
    # ddgs): o writer gastou 13.685 tokens sobre 47 descobertas (n=1). 15.000 é
    # estimativa com folga sobre essa medição, não número medido.
    WRITER_RESERVE_TOKENS: int = Field(default=15_000, ge=0)
    # Um teto por papel, porque o trabalho de cada nó tem duração diferente.
    # Medido em 2026-09-26 contra o gateway de chat atual (Model Studio da
    # Alibaba; supervisor glm-5.3, workers qwen3.6-flash), em execuções reais do
    # golden set: supervisor 2,8 a 10,7s; research completou em torno de 130s
    # com busca web, depois de estourar o teto de 90s do gateway anterior;
    # analyst estourou 90s; critic estourou 45s. Writer não foi medido neste
    # gateway: 120s é estimativa com folga sobre a medição antiga (6 a 10s),
    # não número medido.
    NODE_TIMEOUT_SUPERVISOR_S: float = Field(default=30.0, gt=0)
    NODE_TIMEOUT_RESEARCH_S: float = Field(default=240.0, gt=0)
    NODE_TIMEOUT_WRITER_S: float = Field(default=120.0, gt=0)
    NODE_TIMEOUT_CRITIC_S: float = Field(default=150.0, gt=0)
    NODE_TIMEOUT_ANALYST_S: float = Field(default=240.0, gt=0)
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
        "CHAT_API_KEY",
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

    @model_validator(mode="after")
    def _reserve_fits_the_budget(self) -> "Settings":
        """Reserva do tamanho do orçamento não deixa nada para o trabalho."""
        if self.WRITER_RESERVE_TOKENS >= self.BUDGET_TOKENS_PER_RUN:
            raise ValueError("WRITER_RESERVE_TOKENS precisa ser menor que BUDGET_TOKENS_PER_RUN")
        return self


@lru_cache
def get_settings() -> Settings:
    """Instância única. Importar este módulo não exige ambiente configurado."""
    return Settings()  # type: ignore[call-arg]  # os obrigatórios vêm do ambiente
