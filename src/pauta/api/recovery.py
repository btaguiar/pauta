"""Varredura de runs sem executor, no startup da app (ADR 006).

Uma run com checkpoint válido e sem ninguém executando é órfã. Isso acontece
quando o processo cai no meio. Ao subir, a app marca essas runs e **não** retoma
nenhuma: religar o servidor não pode gastar token de ninguém. A retomada é um
POST explícito em `/runs/{id}/continue`.

A política mora aqui e a escrita mora no store. São responsabilidades
diferentes: o store sabe gravar, este módulo sabe o que deve ser gravado ao
subir e o que nunca pode acontecer sozinho.
"""

from dataclasses import dataclass

from ..memory.run_store import RunStore, mark_orphans
from ..memory.runs import RunStatus
from ..observability import emit

#: Estados de quem estava no meio do caminho quando o processo morreu. Espelha a
#: regra de `memory.runs.mark_orphaned`, que é quem de fato transiciona.
PENDING: tuple[RunStatus, ...] = ("running", "interrupted")


@dataclass(frozen=True)
class RecoveryReport:
    """O que a varredura encontrou. Vai para o log do startup e para `/health`."""

    orphaned: int
    threads: list[str]

    @property
    def clean(self) -> bool:
        return self.orphaned == 0


async def sweep(store: RunStore) -> RecoveryReport:
    """Marca as runs sem executor e devolve quais foram.

    Devolver os `thread_id` importa: sem eles, o operador sabe que existem
    órfãs e não sabe quais retomar. O `GET /runs?status=orphaned` responde a
    mesma pergunta depois, e esta é a resposta no momento em que a app sobe.
    """
    pending = [run for run in await store.list_runs() if run.status in PENDING]
    threads = [run.thread_id for run in pending]
    orphaned = await mark_orphans(store)
    emit(
        "node_start",
        node="recovery",
        message="varredura de startup concluída",
        orphaned=orphaned,
        resumed=0,
    )
    return RecoveryReport(orphaned=orphaned, threads=threads)
