import operator
from typing import Annotated, get_args, get_type_hints

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from pauta.graph.state import AgentState, Critique, Finding, new_state


def test_new_state_starts_every_counter_at_zero() -> None:
    state = new_state(task="vale a pena migrar?", run_id="r1")
    assert state["iteration"] == 0
    assert state["critic_loops"] == 0
    assert state["tokens_used"] == 0
    assert state["findings"] == []
    assert state["final_report"] is None


def test_finding_requires_a_source() -> None:
    finding = Finding(content="custo por token caiu", source="https://exemplo", agent="research")
    assert finding.source == "https://exemplo"


def test_critique_defaults_to_no_gaps() -> None:
    critique = Critique(verdict="ok")
    assert critique.gaps == []
    assert critique.delegate_to == []


def test_counters_use_addition_as_reducer() -> None:
    """Cada nó devolve o quanto gastou; o reducer soma."""
    hints = get_type_hints(AgentState, include_extras=True)
    for field in ("iteration", "critic_loops", "tokens_used"):
        assert get_args(hints[field])[1] is operator.add
    for field in ("findings", "critiques"):
        assert get_args(hints[field])[1] is operator.add
    assert get_args(hints["messages"])[1] is add_messages
    assert hints["messages"].__metadata__ is not None
    assert Annotated is not None


def test_findings_accumulate_across_nodes() -> None:
    state = new_state(task="t", run_id="r1")
    first = [Finding(content="a", source="s1", agent="research")]
    second = [Finding(content="b", source="s2", agent="analyst")]
    merged = operator.add(state["findings"] + first, second)
    assert [f.content for f in merged] == ["a", "b"]


def test_add_messages_appends() -> None:
    merged = add_messages(HumanMessage("pergunta"), AIMessage("resposta"))
    assert isinstance(merged, list)
    assert len(merged) == 2
