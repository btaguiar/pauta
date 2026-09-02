import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

from pauta.agents.research import ResearchOutput
from pauta.agents.supervisor import Router
from pauta.config import Settings, get_settings
from pauta.graph.builder import build_graph, is_retryable, retry_policy
from pauta.graph.state import Finding, new_state
from tests.fakes import FakeChatModel


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@tool
def busca_fake(query: str) -> str:
    """Busca determinística para teste."""
    return f"resultado para {query}"


class FakeRateLimit(Exception):
    """Nome imita o que os SDKs levantam em 429."""


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeHttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.response = FakeResponse(status_code)


def test_rate_limit_is_retried() -> None:
    """O default do LangGraph não retenta 429; este projeto precisa retentar."""
    assert is_retryable(FakeRateLimit()) is True
    assert is_retryable(FakeHttpError(429)) is True


def test_server_error_and_network_are_retried() -> None:
    assert is_retryable(FakeHttpError(503)) is True
    assert is_retryable(ConnectionError("recusou")) is True
    assert is_retryable(TimeoutError("estourou")) is True


def test_schema_validation_is_never_retried() -> None:
    assert is_retryable(ValueError("json inválido")) is False
    assert is_retryable(TypeError("campo faltando")) is False
    assert is_retryable(FakeHttpError(400)) is False


def test_retry_policy_counts_the_first_attempt(settings: Settings) -> None:
    policy = retry_policy(settings)
    assert isinstance(policy, RetryPolicy)
    assert policy.max_attempts == settings.NODE_RETRIES + 1


async def test_graph_runs_task_to_report(settings: Settings) -> None:
    """Tarefa entra, relatório sai."""
    supervisor = FakeChatModel(
        responses=[
            Router(next="research", rationale="falta dado"),
            Router(next="writer", rationale="material reunido"),
        ]
    )
    research = FakeChatModel(
        responses=[
            "achei o preço",
            ResearchOutput(
                findings=[
                    Finding(content="A100 custa X por hora", source="https://a", agent="research")
                ]
            ),
        ]
    )
    writer = FakeChatModel(responses=["Briefing: a API sai mais barata até 1M de tokens."])

    graph = build_graph(
        supervisor_model=supervisor,
        research_model=research,
        writer_model=writer,
        tools=[busca_fake],
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    state = new_state(task="GPU dedicada ou API por token?", run_id="r1")
    final = await graph.ainvoke(state, config={"configurable": {"thread_id": "t1"}})

    assert final["final_report"].startswith("Briefing")
    assert len(final["findings"]) == 1
    assert final["iteration"] == 2
    assert final["tokens_used"] > 0


async def test_findings_accumulate_across_cycles(settings: Settings) -> None:
    supervisor = FakeChatModel(
        responses=[
            Router(next="research", rationale="primeira rodada"),
            Router(next="research", rationale="ainda falta"),
            Router(next="writer", rationale="agora chega"),
        ]
    )
    research = FakeChatModel(
        responses=[
            "primeira",
            ResearchOutput(findings=[Finding(content="a", source="https://a", agent="research")]),
            "segunda",
            ResearchOutput(findings=[Finding(content="b", source="https://b", agent="research")]),
        ]
    )
    writer = FakeChatModel(responses=["Briefing final."])
    graph = build_graph(
        supervisor_model=supervisor,
        research_model=research,
        writer_model=writer,
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    final = await graph.ainvoke(
        new_state(task="t", run_id="r1"), config={"configurable": {"thread_id": "t2"}}
    )
    assert [f.content for f in final["findings"]] == ["a", "b"]


async def test_broken_router_still_reaches_a_report(settings: Settings) -> None:
    """O grafo continua andando quando o LLM que decide o caminho falha."""
    supervisor = FakeChatModel(
        responses=[
            ValueError("json quebrado"),
            ValueError("json quebrado"),
            ValueError("json quebrado de novo"),
            ValueError("json quebrado de novo"),
        ]
    )
    research = FakeChatModel(
        responses=[
            "achei",
            ResearchOutput(findings=[Finding(content="a", source="https://a", agent="research")]),
        ]
    )
    writer = FakeChatModel(responses=["Briefing com o que houve."])
    graph = build_graph(
        supervisor_model=supervisor,
        research_model=research,
        writer_model=writer,
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    final = await graph.ainvoke(
        new_state(task="t", run_id="r1"), config={"configurable": {"thread_id": "t3"}}
    )
    assert final["final_report"] == "Briefing com o que houve."


async def test_budget_overrun_goes_straight_to_the_writer(settings: Settings) -> None:
    supervisor = FakeChatModel(responses=[])
    research = FakeChatModel(responses=[])
    writer = FakeChatModel(responses=["Briefing com o orçamento estourado."])
    graph = build_graph(
        supervisor_model=supervisor,
        research_model=research,
        writer_model=writer,
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    state = new_state(task="t", run_id="r1")
    state["tokens_used"] = settings.BUDGET_TOKENS_PER_RUN
    final = await graph.ainvoke(state, config={"configurable": {"thread_id": "t4"}})
    assert final["final_report"].startswith("Briefing")
    assert supervisor.call_count == 0
    assert research.call_count == 0


async def test_state_survives_by_thread_id(settings: Settings) -> None:
    """O checkpointer guarda o estado por thread, e ele volta depois da execução."""
    supervisor = FakeChatModel(
        responses=[
            Router(next="research", rationale="reunir"),
            Router(next="writer", rationale="material reunido"),
        ]
    )
    graph = build_graph(
        supervisor_model=supervisor,
        research_model=FakeChatModel(
            responses=[
                "achei",
                ResearchOutput(
                    findings=[Finding(content="a", source="https://a", agent="research")]
                ),
            ]
        ),
        writer_model=FakeChatModel(responses=["Briefing."]),
        settings=settings,
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "t5"}}
    await graph.ainvoke(new_state(task="t", run_id="r1"), config=config)
    snapshot = graph.get_state(config)
    assert snapshot.values["final_report"] == "Briefing."
    assert snapshot.values["task"] == "t"


def test_the_tools_reach_the_researcher(settings: Settings) -> None:
    research = FakeChatModel(responses=["sem tool", ResearchOutput(findings=[])])
    tools: list[BaseTool] = [busca_fake]
    build_graph(
        supervisor_model=FakeChatModel(responses=[]),
        research_model=research,
        writer_model=FakeChatModel(responses=[]),
        tools=tools,
        settings=settings,
    )
    assert [t.name for t in research.bound_tools] == ["busca_fake"]


async def test_node_timeout_is_not_retried() -> None:
    """Timeout de nó não é erro de rede nem rate limit, então não retenta."""
    from langgraph.errors import NodeTimeoutError

    error = NodeTimeoutError("research", 0.1, kind="run", run_timeout=0.05)
    assert is_retryable(error) is False
