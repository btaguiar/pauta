"""O que todo agente faz igual.

Contar token, executar tool call e fechar tool call que ficou sem resposta são
mecânica, não comportamento. Comportamento é o que cada agente define no seu
próprio módulo: o prompt, o schema de saída e o que ele faz com o resultado.

Estava duplicado em até cinco arquivos, e a duplicação cobrou: a correção de uma
conversa inválida por orçamento esgotado precisou ser escrita duas vezes.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..observability import emit


def tokens_from(message: Any) -> int:
    """Tokens que a chamada consumiu, segundo o provider.

    Zero quando o provider não declara uso. Estimar aqui seria inventar o número
    que o orçamento inteiro usa para decidir.
    """
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0))
    return 0


async def run_tools(
    tools: Sequence[BaseTool],
    message: AIMessage,
    *,
    node: str,
    run_id: str,
) -> list[ToolMessage]:
    """Executa as tool calls pedidas.

    Falha de tool vira `ToolMessage` de erro, não exceção: uma busca fora do ar
    não derruba a run, ela vira material que o agente lê e contorna.
    """
    by_name = {tool.name: tool for tool in tools}
    results: list[ToolMessage] = []
    for call in message.tool_calls:
        emit("tool_call", node=node, run_id=run_id, tool=call["name"], args=call["args"])
        tool = by_name.get(call["name"])
        if tool is None:
            results.append(
                ToolMessage(
                    content=f"tool desconhecida: {call['name']}",
                    tool_call_id=call["id"] or "",
                    status="error",
                )
            )
            continue
        try:
            results.append(await tool.ainvoke(call))
        except Exception as exc:
            emit(
                "error",
                node=node,
                run_id=run_id,
                tool=call["name"],
                error_type=type(exc).__name__,
                error=str(exc),
            )
            results.append(
                ToolMessage(
                    content=f"a tool {call['name']} falhou: {exc}",
                    tool_call_id=call["id"] or "",
                    status="error",
                )
            )
    return results


def unanswered(message: AIMessage) -> list[ToolMessage]:
    """Fecha as tool calls que ficaram sem resposta quando o orçamento acabou.

    Uma `AIMessage` com `tool_calls` sem a `ToolMessage` correspondente é uma
    conversa inválida: o provider recusa com 400 e a run inteira se perde depois
    de já ter gastado o orçamento. Parar com o parcial só é parar de verdade se
    o histórico continuar coerente.
    """
    return [
        ToolMessage(
            content="não executada: o orçamento da run acabou",
            tool_call_id=call["id"] or "",
            status="error",
        )
        for call in message.tool_calls
    ]
