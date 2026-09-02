import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tests.fakes import FakeChatModel, FakeChatModelExhausted


class Answer(BaseModel):
    value: str


def test_returns_responses_in_order() -> None:
    model = FakeChatModel(responses=["um", "dois"])
    assert model.invoke([HumanMessage("a")]).content == "um"
    assert model.invoke([HumanMessage("b")]).content == "dois"
    assert model.call_count == 2


def test_records_what_it_received() -> None:
    model = FakeChatModel(responses=["ok"])
    model.invoke([HumanMessage("qual o custo?")])
    assert model.calls[0][0].content == "qual o custo?"


def test_reports_usage_metadata() -> None:
    model = FakeChatModel(responses=["ok"], input_tokens=100, output_tokens=20)
    message = model.invoke([HumanMessage("a")])
    assert message.usage_metadata is not None
    assert message.usage_metadata["total_tokens"] == 120


def test_structured_output_returns_the_model() -> None:
    model = FakeChatModel(responses=[Answer(value="pronto")])
    result = model.with_structured_output(Answer).invoke([HumanMessage("a")])
    assert isinstance(result, Answer)
    assert result.value == "pronto"


def test_queued_exception_is_raised() -> None:
    model = FakeChatModel(responses=[ValueError("parse falhou"), "recuperado"])
    with pytest.raises(ValueError, match="parse falhou"):
        model.invoke([HumanMessage("a")])
    assert model.invoke([HumanMessage("b")]).content == "recuperado"


def test_exhausted_queue_fails_loudly() -> None:
    model = FakeChatModel(responses=["so uma"])
    model.invoke([HumanMessage("a")])
    with pytest.raises(FakeChatModelExhausted):
        model.invoke([HumanMessage("b")])
