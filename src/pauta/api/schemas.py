"""Contratos da API. Pydantic para tudo que cruza a fronteira HTTP.

Os nomes seguem a seção 8.6 do dossiê. Onde o corpo da resposta traz número de
custo, ele vem `None` quando `COST_PER_MTOK_USD` está vazio, nunca zero: zero
seria a afirmação de que a run foi gratuita.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from ..config import HitlMode
from ..graph.state import Critique, Finding
from ..memory.runs import Run, RunStatus

#: Teto do texto da tarefa. Sem ele, um POST de dez megabytes vira contexto de
#: LLM antes de qualquer guardrail de orçamento poder opinar.
MAX_TASK_CHARS = 2_000

TOKENS_PER_MILLION = 1_000_000


def cost_of(tokens: int, price_per_mtok: float | None) -> float | None:
    """Custo em USD, ou `None` quando o preço não foi declarado."""
    if price_per_mtok is None:
        return None
    return round(tokens / TOKENS_PER_MILLION * price_per_mtok, 6)


class RunRequest(BaseModel):
    """Corpo do `POST /runs`."""

    task: str = Field(min_length=1, max_length=MAX_TASK_CHARS)
    hitl_mode: HitlMode | None = Field(
        default=None,
        description="`interrupt` congela antes da redação; ausente usa o padrão do servidor",
    )


class RunCreated(BaseModel):
    """Resposta 201. O `thread_id` é o que retoma a run depois."""

    run_id: str
    thread_id: str
    status: RunStatus


class ResumeRequest(BaseModel):
    """Corpo do `POST /runs/{id}/resume`. É a resposta de um humano."""

    approved: bool = True
    feedback: str | None = Field(default=None, max_length=MAX_TASK_CHARS)


class RunSummary(BaseModel):
    """Uma linha do `GET /runs`."""

    run_id: str
    thread_id: str
    task: str
    status: RunStatus
    tokens_used: int
    cost_usd: float | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, run: Run, price_per_mtok: float | None) -> "RunSummary":
        return cls(
            run_id=run.run_id,
            thread_id=run.thread_id,
            task=run.task,
            status=run.status,
            tokens_used=run.tokens_used,
            cost_usd=cost_of(run.tokens_used, price_per_mtok),
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class RunDetail(RunSummary):
    """`GET /runs/{id}`, com o material que o checkpoint guardou."""

    final_report: str | None = None
    iterations: int = 0
    error: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)

    @classmethod
    def of_run(
        cls,
        run: Run,
        price_per_mtok: float | None,
        *,
        findings: list[Finding] | None = None,
        critiques: list[Critique] | None = None,
    ) -> "RunDetail":
        summary = RunSummary.of(run, price_per_mtok)
        return cls(
            **summary.model_dump(),
            final_report=run.final_report,
            iterations=run.iterations,
            error=run.error,
            findings=findings or [],
            critiques=critiques or [],
        )


class Health(BaseModel):
    """`GET /health`. Diz o que protege e o que não protege."""

    status: str = "ok"
    orphaned_at_startup: int = 0
    spent_today_usd: float | None = None
    daily_budget_usd: float
    budget_enforceable: bool = Field(
        description="falso quando COST_PER_MTOK_USD está vazio; aí o teto diário não protege nada"
    )
    hitl_mode: HitlMode


class Problem(BaseModel):
    """Corpo de toda recusa. Mensagem de erro é texto voltado ao público."""

    detail: str
