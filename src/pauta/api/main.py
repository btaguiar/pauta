"""API HTTP do Pauta, com o contrato da seção 8.6 do dossiê.

    uv run uvicorn pauta.api.main:app --host 0.0.0.0 --port 8000

`POST /runs` responde 201 na hora e executa em segundo plano. Se o processo cair
no meio, a run fica com checkpoint vivo e sem executor, e a varredura de startup
a marca como `orphaned` sem retomar nada (ADR 006).

Dois endpoints de retomada, porque são duas semânticas: `/resume` responde a um
humano que estava segurando a run, `/continue` responde a uma queda.

O streaming por SSE previsto na 8.6 não está aqui. Ele é o passo seguinte, e a
demo que o consome depende dele.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..config import HitlMode, Settings, get_settings
from ..graph.builder import build_graph
from ..graph.state import Critique, Finding
from ..memory.checkpointer import postgres_checkpointer
from ..memory.run_store import RunStore, postgres_run_store
from ..memory.runs import Run, RunStatus
from ..observability import configure_tracing, emit, setup_logging
from ..runner import Graph, RunAlreadyFinished, RunNotFound, execute_run, register_run, resume_run
from ..tools import analyst_tools, research_tools
from .limits import RateLimiter, check_daily_budget, spend_today
from .recovery import sweep
from .schemas import Health, ResumeRequest, RunCreated, RunDetail, RunRequest, RunSummary

#: Um grafo compilado por modo, porque `interrupt_before` é decidido no
#: `compile()` e não dá para mudar por requisição.
Graphs = dict[HitlMode, Graph]


def build_graphs(settings: Settings, checkpointer: BaseCheckpointSaver[Any]) -> Graphs:
    """Compila os dois modos uma vez, no startup. Modelo e tool são caros de montar."""
    research = research_tools(settings)
    analyst = analyst_tools()
    return {
        mode: build_graph(
            research_tools=research,
            analyst_tools=analyst,
            settings=settings.model_copy(update={"HITL_MODE": mode}),
            checkpointer=checkpointer,
        )
        for mode in ("auto", "interrupt")
    }


class Resources:
    """O que a app precisa ter aberto para atender. Injetável, para o teste existir."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: RunStore,
        graphs: Graphs,
        limiter: RateLimiter,
    ) -> None:
        self.settings = settings
        self.store = store
        self.graphs = graphs
        self.limiter = limiter
        self.orphaned_at_startup = 0
        # Sem guardar a referência, o coletor de lixo pode recolher a task no
        # meio da run. Uma run some sem nem virar `failed`.
        self.background: set[asyncio.Task[None]] = set()

    def graph_for(self, mode: HitlMode | None) -> Graph:
        return self.graphs[mode or self.settings.HITL_MODE]


def resources_of(request: Request) -> Resources:
    return request.app.state.resources  # type: ignore[no-any-return]


#: `Annotated` em vez de default: o FastAPI recomenda, e evita chamada de
#: função em valor padrão, que o ruff recusa com razão.
ResourcesDep = Annotated[Resources, Depends(resources_of)]


async def _drive_in_background(resources: Resources, run: Run, mode: HitlMode | None) -> None:
    """Executa a run já registrada. Falha vira evento, nunca exceção solta na task."""
    try:
        await execute_run(run, graph=resources.graph_for(mode), store=resources.store)
    except Exception as exc:  # o runner já gravou `failed` antes de propagar
        emit("error", node="api", run_id=run.run_id, error_type=type(exc).__name__, error=str(exc))


def _schedule(resources: Resources, coroutine: Any) -> None:
    task = asyncio.create_task(coroutine)
    resources.background.add(task)
    task.add_done_callback(resources.background.discard)


async def _material(resources: Resources, thread_id: str) -> tuple[list[Finding], list[Critique]]:
    """Descobertas e críticas saem do checkpoint, não do ponteiro.

    O registro de runs guarda o que a API precisa para achar a run; o material
    reunido vive no estado do grafo, que é quem sabe dele.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await resources.graph_for(None).aget_state(config)
    except Exception as exc:
        emit("error", node="api", thread_id=thread_id, error=f"checkpoint ilegível: {exc}")
        return [], []
    values = snapshot.values or {}
    return list(values.get("findings", [])), list(values.get("critiques", []))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Abre banco, compila grafos e varre as órfãs antes de aceitar requisição."""
    setup_logging()
    configure_tracing()
    settings = get_settings()
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(postgres_checkpointer(settings))
        store = await stack.enter_async_context(postgres_run_store(settings))
        resources = Resources(
            settings=settings,
            store=store,
            graphs=build_graphs(settings, checkpointer),
            limiter=RateLimiter(settings.RATE_LIMIT_PER_IP),
        )
        resources.orphaned_at_startup = (await sweep(store)).orphaned
        app.state.resources = resources
        yield


def create_app(resources: Resources | None = None) -> FastAPI:
    """Monta a app. Com `resources`, o lifespan não abre banco nenhum.

    É assim que o teste roda a API inteira sem Postgres e sem provider: os
    recursos entram prontos, com store em memória e modelo falso.
    """
    app = FastAPI(
        title="Pauta",
        description="Briefings analíticos multi-agente, com custo visível.",
        lifespan=None if resources is not None else lifespan,
    )
    if resources is not None:
        app.state.resources = resources
    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=Health)
    async def health(resources: ResourcesDep) -> Health:
        spend = await spend_today(resources.store, resources.settings)
        return Health(
            orphaned_at_startup=resources.orphaned_at_startup,
            spent_today_usd=spend.usd if spend.measurable else None,
            daily_budget_usd=spend.limit_usd,
            budget_enforceable=spend.measurable,
            hitl_mode=resources.settings.HITL_MODE,
        )

    @app.post("/runs", response_model=RunCreated, status_code=status.HTTP_201_CREATED)
    async def create_run(
        body: RunRequest,
        request: Request,
        response: Response,
        resources: ResourcesDep,
    ) -> RunCreated:
        client = request.client.host if request.client else "desconhecido"
        allowed = resources.limiter.check(client)
        if not allowed.allowed:
            response.headers["Retry-After"] = str(allowed.retry_after_s)
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, allowed.reason)

        budget = await check_daily_budget(resources.store, resources.settings)
        if not budget.allowed:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, budget.reason)

        run = await register_run(body.task, store=resources.store)
        _schedule(resources, _drive_in_background(resources, run, body.hitl_mode))
        return RunCreated(run_id=run.run_id, thread_id=run.thread_id, status=run.status)

    @app.get("/runs", response_model=list[RunSummary])
    async def list_runs(
        resources: ResourcesDep,
        # `alias` porque o contrato da 8.6 é `?status=orphaned`, e `status` sozinho
        # sombrearia o módulo de códigos HTTP importado aqui.
        run_status: Annotated[RunStatus | None, Query(alias="status")] = None,
    ) -> list[RunSummary]:
        runs = await resources.store.list_runs(run_status)
        price = resources.settings.COST_PER_MTOK_USD
        return [RunSummary.of(run, price) for run in runs]

    @app.get("/runs/{run_id}", response_model=RunDetail)
    async def get_run(run_id: str, resources: ResourcesDep) -> RunDetail:
        run = await resources.store.get(run_id)
        if run is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {run_id!r} não existe")
        findings, critiques = await _material(resources, run.thread_id)
        return RunDetail.of_run(
            run,
            resources.settings.COST_PER_MTOK_USD,
            findings=findings,
            critiques=critiques,
        )

    @app.post("/runs/{run_id}/resume", response_model=RunCreated)
    async def resume(
        run_id: str,
        body: ResumeRequest,
        resources: ResourcesDep,
    ) -> RunCreated:
        """Resposta de um humano. Só vale sobre run que está esperando por ele."""
        run = await _require(resources, run_id, expected="interrupted")
        _schedule(
            resources,
            _continue_run(resources, run.thread_id, body.feedback if body.approved else None),
        )
        return RunCreated(run_id=run.run_id, thread_id=run.thread_id, status="running")

    @app.post("/runs/{run_id}/continue", response_model=RunCreated)
    async def continue_run(run_id: str, resources: ResourcesDep) -> RunCreated:
        """Retomada depois de uma queda. Nunca acontece sozinha (ADR 006)."""
        run = await _require(resources, run_id, expected="orphaned")
        _schedule(resources, _continue_run(resources, run.thread_id, None))
        return RunCreated(run_id=run.run_id, thread_id=run.thread_id, status="running")


async def _require(resources: Resources, run_id: str, *, expected: RunStatus) -> Run:
    run = await resources.store.get(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"run {run_id!r} não existe")
    if run.status != expected:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"a run {run_id} está {run.status!r} e este endpoint exige {expected!r}",
        )
    return run


async def _continue_run(resources: Resources, thread_id: str, feedback: str | None) -> None:
    try:
        await resume_run(
            thread_id,
            graph=resources.graph_for(None),
            store=resources.store,
            feedback=feedback,
        )
    except (RunNotFound, RunAlreadyFinished) as exc:
        emit("error", node="api", thread_id=thread_id, error=str(exc))
    except Exception as exc:
        emit(
            "error", node="api", thread_id=thread_id, error_type=type(exc).__name__, error=str(exc)
        )


app = create_app()
