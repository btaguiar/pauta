"""Guardrails da API pública: camadas 2 e 3 da ADR 004.

A camada 1 é o orçamento por run, em `graph/budget.py`. Estas duas existem
porque demo pública com a chave do autor é a forma mais rápida conhecida de
queimar orçamento: um teto diário global e um rate limit por IP.

Recusa honesta é melhor que conta surpresa. Estourado o teto, a API responde 503
dizendo o que aconteceu, em vez de aceitar a run e falhar no meio dela.

O rate limit vive na memória do processo, de propósito: uma dependência a mais
para contar três requisições por hora não se paga. A consequência está escrita
em `RateLimiter`, e ela importa se um dia houver mais de uma réplica.
"""

import re
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

from ..config import Settings
from ..memory.run_store import RunStore

#: Quantos tokens cabem no preço declarado por milhão.
TOKENS_PER_MILLION = 1_000_000

#: Formato do `RATE_LIMIT_PER_IP`, o mesmo que bibliotecas de rate limit usam.
RATE_LIMIT_PATTERN = re.compile(r"^\s*(\d+)\s*/\s*(second|minute|hour|day)\s*$", re.IGNORECASE)

WINDOW_SECONDS: dict[str, int] = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


class InvalidRateLimit(ValueError):
    """`RATE_LIMIT_PER_IP` não está no formato `<número>/<janela>`."""


def parse_rate_limit(spec: str) -> tuple[int, int]:
    """`"3/hour"` vira `(3, 3600)`. Formato errado falha no startup, não na primeira request."""
    match = RATE_LIMIT_PATTERN.match(spec)
    if match is None:
        raise InvalidRateLimit(
            f"rate limit {spec!r} inválido; use <número>/<{'|'.join(WINDOW_SECONDS)}>"
        )
    allowance = int(match.group(1))
    if allowance <= 0:
        raise InvalidRateLimit(f"rate limit {spec!r} precisa permitir ao menos uma requisição")
    return allowance, WINDOW_SECONDS[match.group(2).lower()]


@dataclass(frozen=True)
class Decision:
    """Deixa passar ou não, e por quê. O motivo vai para o corpo da resposta."""

    allowed: bool
    reason: str = ""
    retry_after_s: int = 0


class RateLimiter:
    """Janela deslizante por chave, na memória deste processo.

    Duas réplicas da app são dois limitadores, e o teto efetivo dobra. Com um
    container só, que é o que o compose sobe, o número vale como está escrito.
    """

    def __init__(self, spec: str, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.allowance, self.window_s = parse_rate_limit(spec)
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> Decision:
        """Consulta e registra o acesso, quando ele passa."""
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.allowance:
            retry_after = int(self.window_s - (now - hits[0])) + 1
            return Decision(
                allowed=False,
                reason=(
                    f"limite de {self.allowance} requisição(ões) por "
                    f"{self.window_s}s atingido para este IP"
                ),
                retry_after_s=retry_after,
            )
        hits.append(now)
        return Decision(allowed=True)


@dataclass(frozen=True)
class DailySpend:
    """Quanto o dia já custou, e se dá para saber.

    `measurable` é falso quando `COST_PER_MTOK_USD` está vazio. Nesse caso o teto
    diário não protege nada, e é obrigação da app dizer isso alto em vez de
    fingir que protege.
    """

    usd: float
    tokens: int
    limit_usd: float
    measurable: bool

    @property
    def exhausted(self) -> bool:
        return self.measurable and self.usd >= self.limit_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.usd) if self.measurable else 0.0


async def spend_today(
    store: RunStore,
    settings: Settings,
    *,
    today: date | None = None,
) -> DailySpend:
    """Gasto do dia, somado das runs registradas.

    Sai da tabela de runs em vez de um contador próprio: contador na memória
    zera quando o processo reinicia, e aí o teto diário deixa de existir
    exatamente depois de um crash, que é quando ele mais importa.
    """
    price = settings.COST_PER_MTOK_USD
    reference = today or datetime.now(UTC).date()
    runs = await store.list_runs()
    tokens = sum(run.tokens_used for run in runs if run.created_at.date() == reference)
    if price is None:
        return DailySpend(
            usd=0.0, tokens=tokens, limit_usd=settings.DAILY_BUDGET_USD, measurable=False
        )
    return DailySpend(
        usd=round(tokens / TOKENS_PER_MILLION * price, 6),
        tokens=tokens,
        limit_usd=settings.DAILY_BUDGET_USD,
        measurable=True,
    )


async def check_daily_budget(
    store: RunStore,
    settings: Settings,
    *,
    today: date | None = None,
) -> Decision:
    """Recusa a run nova quando o dia já gastou o teto."""
    spend = await spend_today(store, settings, today=today)
    if not spend.exhausted:
        return Decision(allowed=True)
    return Decision(
        allowed=False,
        reason=(
            f"teto diário de {spend.limit_usd:.2f} USD esgotado; "
            f"o dia já gastou {spend.usd:.4f} USD em {spend.tokens} tokens"
        ),
    )
