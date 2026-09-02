"""Máquina de estados da run (ADR 006).

`running` -> `interrupted` -> `running` -> `completed` | `failed` | `orphaned`.

`orphaned` é uma run com checkpoint válido e sem executor, o que acontece quando
o processo cai no meio. Ela é marcada no startup e **não** é retomada sozinha:
religar o servidor não pode gastar token de ninguém. A retomada é um POST
explícito, e por isso `orphaned` existe no contrato da API.
"""

from datetime import UTC, datetime
from typing import Literal, get_args

from pydantic import BaseModel, Field

RunStatus = Literal["running", "interrupted", "completed", "failed", "orphaned"]

#: Estados de onde nada mais sai sozinho.
TERMINAL_STATUSES: frozenset[RunStatus] = frozenset({"completed", "failed"})

#: Transições permitidas. O que não está aqui é erro de programação, não de dado.
ALLOWED: dict[RunStatus, frozenset[RunStatus]] = {
    "running": frozenset({"interrupted", "completed", "failed", "orphaned"}),
    "interrupted": frozenset({"running", "failed", "orphaned"}),
    "orphaned": frozenset({"running", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
}


class IllegalTransition(ValueError):
    """A run não pode ir do estado atual para o pedido."""


class Run(BaseModel):
    """Uma execução, do POST até o relatório."""

    run_id: str
    thread_id: str
    task: str
    status: RunStatus = "running"
    final_report: str | None = None
    tokens_used: int = 0
    iterations: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def waiting_for_human(self) -> bool:
        return self.status == "interrupted"

    @property
    def recoverable(self) -> bool:
        """Só uma órfã é retomada por `/continue`, e só por pedido explícito."""
        return self.status == "orphaned"

    def transition_to(self, status: RunStatus) -> "Run":
        """Devolve a run no novo estado, ou recusa a transição."""
        if status not in ALLOWED[self.status]:
            raise IllegalTransition(
                f"run {self.run_id} não pode ir de {self.status!r} para {status!r}; "
                f"permitidos: {sorted(ALLOWED[self.status]) or 'nenhum'}"
            )
        return self.model_copy(update={"status": status, "updated_at": datetime.now(UTC)})


def mark_orphaned(runs: list[Run]) -> tuple[list[Run], int]:
    """Marca como órfã toda run em estado não terminal. Não retoma nenhuma.

    Roda no startup. Devolve a lista atualizada e quantas foram marcadas, que é o
    número que o log do startup e o `GET /runs?status=orphaned` mostram.
    """
    updated: list[Run] = []
    count = 0
    for run in runs:
        if run.status in ("running", "interrupted"):
            updated.append(run.transition_to("orphaned"))
            count += 1
        else:
            updated.append(run)
    return updated, count


def all_statuses() -> tuple[RunStatus, ...]:
    return get_args(RunStatus)
