"""Espalha os eventos de uma run para quem estiver ouvindo.

O `observability.emit` já publica tudo que acontece no grafo. Este módulo é o
segundo destino que o docstring de lá promete: o mesmo evento que vira linha de
log vira frame SSE, sem um caminho paralelo de instrumentação que possa
divergir do primeiro.

Escutar o próprio logger, e não o `graph.astream`, tem um motivo: a run é
executada em segundo plano por quem chamou `POST /runs`, e o cliente do stream
chega depois. Quem observa não pode precisar ser quem executa.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Iterator
from typing import Any

from ..observability import LOGGER_NAME, PAYLOAD_KEY

#: Eventos que uma run guardados por assinante antes de o mais antigo cair. Um
#: cliente lento não pode segurar a execução, então a fila descarta em vez de
#: bloquear, e o descarte vira um evento visível.
QUEUE_SIZE = 500

#: Marca o fim do stream. Não é um evento do grafo: é o sinal de que não vem
#: mais nada, para o gerador fechar em vez de esperar para sempre.
DONE = "__done__"

#: O nó que fecha toda run. `node_end` dele é sucesso ou interrupt, `error` é
#: falha. Nos dois casos não vem mais evento nenhum daquele `run_id`.
TERMINAL_NODE = "runner"


def is_terminal(event: dict[str, Any]) -> bool:
    """O evento fecha a run, seja bem ou mal."""
    return event.get("node") == TERMINAL_NODE and event.get("event") in {"node_end", "error"}


class EventBroadcaster(logging.Handler):
    """Um handler de log que também entrega evento para assinantes vivos.

    As filas são por `run_id`: dois clientes ouvindo runs diferentes não veem o
    evento um do outro, e o mesmo cliente pode abrir o stream duas vezes.

    Tudo acontece no event loop da app, então `put_nowait` basta. Se um dia algo
    emitir de outra thread, isto precisa de `call_soon_threadsafe`.
    """

    def __init__(self, *, queue_size: int = QUEUE_SIZE) -> None:
        super().__init__()
        self.queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}

    def install(self) -> None:
        logging.getLogger(LOGGER_NAME).addHandler(self)

    def uninstall(self) -> None:
        logging.getLogger(LOGGER_NAME).removeHandler(self)

    @property
    def listeners(self) -> int:
        return sum(len(queues) for queues in self._subscribers.values())

    @contextlib.contextmanager
    def listen(self, run_id: str) -> Iterator[asyncio.Queue[dict[str, Any]]]:
        """Assina os eventos de uma run e cancela a assinatura ao sair.

        Assinar antes de consultar o estado da run evita a corrida óbvia: a run
        termina entre a consulta e a assinatura, e o cliente espera para sempre
        por um evento que já passou.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.queue_size)
        self._subscribers.setdefault(run_id, set()).add(queue)
        try:
            yield queue
        finally:
            queues = self._subscribers.get(run_id, set())
            queues.discard(queue)
            if not queues:
                self._subscribers.pop(run_id, None)

    def close_run(self, run_id: str) -> None:
        """Manda os assinantes daquela run fecharem, sem esperar mais nada."""
        self._deliver(run_id, {"event": DONE})

    def emit(self, record: logging.LogRecord) -> None:
        payload = getattr(record, PAYLOAD_KEY, None)
        if not isinstance(payload, dict):
            return
        run_id = payload.get("run_id")
        if not isinstance(run_id, str):
            return
        self._deliver(run_id, payload)
        if is_terminal(payload):
            self._deliver(run_id, {"event": DONE})

    def _deliver(self, run_id: str, payload: dict[str, Any]) -> None:
        for queue in self._subscribers.get(run_id, set()):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Cliente lento não segura a run. O que ele perde é dito para
                # ele, para o stream não mentir que entregou tudo.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(
                        {"event": "error", "run_id": run_id, "error": "evento perdido, fila cheia"}
                    )


def frame(payload: dict[str, Any]) -> str:
    """Um frame SSE.

    O campo `event` do log vira `type` no corpo, que é o nome do contrato da
    seção 8.6 do dossiê, e também o nome do evento SSE, para o cliente poder
    escutar por tipo em vez de filtrar tudo no `onmessage`.
    """
    kind = str(payload.get("event", "message"))
    body = {"type": kind, **{k: v for k, v in payload.items() if k != "event"}}
    data = json.dumps(body, ensure_ascii=False, default=str)
    return f"event: {kind}\ndata: {data}\n\n"
