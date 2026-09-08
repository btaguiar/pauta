"""Guardrails da API: teto diário e rate limit. Camadas 2 e 3 da ADR 004."""

from datetime import UTC, datetime, timedelta

import pytest

from pauta.api.limits import (
    InvalidRateLimit,
    RateLimiter,
    check_daily_budget,
    parse_rate_limit,
    spend_today,
)
from pauta.config import Settings, get_settings
from pauta.memory.run_store import InMemoryRunStore
from pauta.memory.runs import Run

TODAY = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


class FakeClock:
    """Relógio explícito. Rate limit testado com `sleep` é teste lento e instável."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(update={"COST_PER_MTOK_USD": 2.0, "DAILY_BUDGET_USD": 5.0})


@pytest.fixture
def store() -> InMemoryRunStore:
    return InMemoryRunStore()


def a_run(run_id: str, tokens: int, *, when: datetime = TODAY) -> Run:
    return Run(
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        task="vale a pena migrar?",
        tokens_used=tokens,
        created_at=when,
        updated_at=when,
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("3/hour", (3, 3600)), ("10/minute", (10, 60)), (" 1 / day ", (1, 86400))],
)
def test_the_configured_format_is_understood(spec: str, expected: tuple[int, int]) -> None:
    assert parse_rate_limit(spec) == expected


@pytest.mark.parametrize("spec", ["3 por hora", "hour/3", "3/fortnight", "0/hour", ""])
def test_a_broken_rate_limit_fails_at_startup(spec: str) -> None:
    """Formato errado precisa quebrar ao subir, não na primeira requisição real."""
    with pytest.raises(InvalidRateLimit):
        parse_rate_limit(spec)


def test_the_allowance_is_spent_then_refused() -> None:
    limiter = RateLimiter("2/hour", clock=FakeClock())
    assert limiter.check("1.2.3.4").allowed is True
    assert limiter.check("1.2.3.4").allowed is True
    refused = limiter.check("1.2.3.4")
    assert refused.allowed is False
    assert "este IP" in refused.reason


def test_one_ip_does_not_spend_the_allowance_of_another() -> None:
    limiter = RateLimiter("1/hour", clock=FakeClock())
    assert limiter.check("1.2.3.4").allowed is True
    assert limiter.check("5.6.7.8").allowed is True


def test_the_window_slides_instead_of_resetting_on_the_hour() -> None:
    clock = FakeClock()
    limiter = RateLimiter("1/hour", clock=clock)
    assert limiter.check("1.2.3.4").allowed is True
    clock.advance(3599)
    assert limiter.check("1.2.3.4").allowed is False
    clock.advance(2)
    assert limiter.check("1.2.3.4").allowed is True


def test_the_refusal_says_when_to_come_back() -> None:
    clock = FakeClock()
    limiter = RateLimiter("1/hour", clock=clock)
    limiter.check("1.2.3.4")
    clock.advance(600)
    refused = limiter.check("1.2.3.4")
    assert 3000 <= refused.retry_after_s <= 3001


async def test_the_day_adds_up_the_tokens_of_its_runs(
    store: InMemoryRunStore, settings: Settings
) -> None:
    await store.save(a_run("r1", 1_000_000))
    await store.save(a_run("r2", 500_000))
    spend = await spend_today(store, settings, today=TODAY.date())
    assert spend.tokens == 1_500_000
    assert spend.usd == pytest.approx(3.0)
    assert spend.measurable is True
    assert spend.exhausted is False


async def test_yesterday_does_not_count_against_today(
    store: InMemoryRunStore, settings: Settings
) -> None:
    await store.save(a_run("hoje", 1_000_000))
    await store.save(a_run("ontem", 9_000_000, when=TODAY - timedelta(days=1)))
    spend = await spend_today(store, settings, today=TODAY.date())
    assert spend.tokens == 1_000_000


async def test_the_cap_refuses_once_the_day_spent_it(
    store: InMemoryRunStore, settings: Settings
) -> None:
    await store.save(a_run("r1", 3_000_000))
    decision = await check_daily_budget(store, settings, today=TODAY.date())
    assert decision.allowed is False
    assert "teto diário" in decision.reason
    assert "5.00 USD" in decision.reason


async def test_the_cap_lets_the_run_through_below_the_limit(
    store: InMemoryRunStore, settings: Settings
) -> None:
    await store.save(a_run("r1", 100_000))
    assert (await check_daily_budget(store, settings, today=TODAY.date())).allowed is True


async def test_without_a_declared_price_the_cap_protects_nothing_and_says_so(
    store: InMemoryRunStore,
) -> None:
    """Fingir que o teto protege é pior que declarar que ele não protege."""
    settings = get_settings().model_copy(update={"COST_PER_MTOK_USD": None})
    await store.save(a_run("r1", 999_000_000))
    spend = await spend_today(store, settings, today=TODAY.date())
    assert spend.measurable is False
    assert spend.exhausted is False
    assert spend.tokens == 999_000_000
    assert (await check_daily_budget(store, settings, today=TODAY.date())).allowed is True


async def test_an_empty_day_costs_nothing(store: InMemoryRunStore, settings: Settings) -> None:
    spend = await spend_today(store, settings, today=TODAY.date())
    assert spend.usd == 0.0
    assert spend.remaining_usd == pytest.approx(5.0)
