"""Modelo falso, explícito e determinístico. Nenhum teste chama LLM real.

Não use `unittest.mock` para simular modelo: o fake devolve o que foi programado,
na ordem, e registra o que recebeu. Quando a fila acaba, ele falha alto.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr


class FakeChatModelExhausted(RuntimeError):
    """A fila de respostas acabou antes de o grafo parar de perguntar."""


class FakeChatModel(BaseChatModel):
    """Devolve `responses` em ordem, uma por chamada.

    Cada item pode ser:

    - `str`, que vira o conteúdo de uma `AIMessage`;
    - instância de `BaseModel`, devolvida como está por `with_structured_output`;
    - instância de `Exception`, que é levantada, para exercitar caminho de falha.
    """

    responses: Sequence[Any]
    input_tokens: int = 10
    output_tokens: int = 5

    _cursor: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """As mensagens recebidas em cada chamada, na ordem."""
        return self._calls

    @property
    def call_count(self) -> int:
        return self._cursor

    def _next(self, messages: list[BaseMessage]) -> Any:
        self._calls.append(list(messages))
        if self._cursor >= len(self.responses):
            raise FakeChatModelExhausted(
                f"chamada {self._cursor + 1} sem resposta programada; "
                f"a fila tinha {len(self.responses)}"
            )
        value = self.responses[self._cursor]
        self._cursor += 1
        if isinstance(value, Exception):
            raise value
        return value

    def _message_from(self, value: Any) -> AIMessage:
        content = value if isinstance(value, str) else value.model_dump_json()
        return AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        value = self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=self._message_from(value))])

    def with_structured_output(
        self,
        schema: Any = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        """Devolve o próximo item da fila já tipado, sem passar por parser real."""

        def _invoke(payload: Any) -> Any:
            messages = payload if isinstance(payload, list) else [payload]
            value = self._next([m for m in messages if isinstance(m, BaseMessage)])
            if isinstance(value, BaseModel):
                return value
            raise TypeError(
                f"with_structured_output esperava BaseModel na fila, veio {type(value).__name__}"
            )

        return RunnableLambda(_invoke)
