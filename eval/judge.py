"""Juiz de fidelidade, binário. SIM ou NAO, nunca escala de 1 a 5.

O juiz avalia o que a regra não alcança: se cada afirmação do briefing se
sustenta no material que os agentes reuniram. A escala é binária de propósito.
Escala Likert de LLM tem âncora instável entre execuções e entre modelos, e a
média de notas instáveis parece precisa sem ser.

Dois cuidados que a metodologia exige, e que este módulo aplica:

1. O juiz vem de `JUDGE_MODEL`, que o `.env.example` manda apontar para um
   provider diferente do executor. Modelo que julga a própria saída se prefere.
2. Nenhum número do juiz vale antes de calibrado. `calibrate_judge.py` roda o
   juiz sobre casos rotulados à mão e calcula o Kappa de Cohen. Abaixo do piso,
   o número não entra em relatório nenhum.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from pauta.models import get_model
from pauta.observability import emit

JUDGE_PROMPT = """Você julga se um briefing se sustenta no material que o produziu.

Responda uma coisa só: toda afirmação factual do briefing está apoiada no
material listado?

Responda SIM quando o briefing só afirma o que o material sustenta, inclusive
quando ele diz que não foi possível responder.
Responda NAO quando o briefing afirma qualquer coisa que o material não sustenta,
ou cita fonte que não está no material.

Não julgue estilo, tamanho ou utilidade. Só a sustentação."""

#: Piso de concordância para o juiz valer. Kappa de Cohen acima de 0,60 é o
#: "substancial" de Landis e Koch. Abaixo disso o juiz discorda demais do humano
#: para o número dele significar algo, e o relatório diz isso em vez de publicá-lo.
KAPPA_FLOOR = 0.60


class Verdict(BaseModel):
    """Um julgamento. Binário, com uma frase de motivo para dar para conferir."""

    sustentado: bool
    motivo: str = Field(description="uma frase dizendo o que sustenta ou o que falta")


def get_judge() -> BaseChatModel:
    """O juiz vem de `get_model`, como todo modelo do projeto (ADR 007)."""
    return get_model("judge")


def render_material(findings: Sequence[str]) -> str:
    if not findings:
        return "- nenhuma descoberta foi registrada"
    return "\n".join(f"- {finding}" for finding in findings)


def build_messages(report: str, findings: Sequence[str]) -> list[Any]:
    return [
        SystemMessage(JUDGE_PROMPT),
        HumanMessage(
            "Material reunido:\n"
            f"{render_material(findings)}\n\n"
            "Briefing produzido:\n"
            f"{report or '(vazio)'}"
        ),
    ]


async def judge_report(
    report: str,
    findings: Sequence[str],
    *,
    model: BaseChatModel,
    task_id: str = "desconhecida",
) -> Verdict | None:
    """Julga um briefing. Devolve `None` quando o juiz não respondeu direito.

    `None` não é `NAO`. Um juiz que falhou não é evidência de briefing ruim, e
    entrar como reprovação inventaria dado. Quem agrega descarta o `None` e
    conta quantos sobraram no `_n`.
    """
    grader = model.with_structured_output(Verdict, include_raw=True)
    try:
        result = await grader.ainvoke(build_messages(report, findings))
    except Exception as exc:
        emit("error", node="judge", task_id=task_id, error_type=type(exc).__name__, error=str(exc))
        return None
    parsed = result.get("parsed") if isinstance(result, dict) else result
    if not isinstance(parsed, Verdict):
        emit("error", node="judge", task_id=task_id, error="veredito não estruturado")
        return None
    emit(
        "critique",
        node="judge",
        task_id=task_id,
        sustentado=parsed.sustentado,
        motivo=parsed.motivo,
    )
    return parsed


def cohen_kappa(human: Sequence[bool], machine: Sequence[bool]) -> float:
    """Concordância entre dois rotuladores, descontado o acaso.

    Acurácia crua engana quando uma das classes domina: um juiz que responde SIM
    sempre acerta 90% de um conjunto 90% SIM sem julgar nada. Kappa desconta
    exatamente essa concordância esperada por acaso.

    Devolve 1,0 quando os dois concordam sempre e não há acaso a descontar, que
    é o caso de um conjunto com uma classe só.
    """
    if len(human) != len(machine):
        raise ValueError(f"listas de tamanhos diferentes: {len(human)} e {len(machine)}")
    total = len(human)
    if total == 0:
        raise ValueError("não há como calibrar sem nenhum caso rotulado")

    observed = sum(1 for one, other in zip(human, machine, strict=True) if one == other) / total
    human_yes = sum(human) / total
    machine_yes = sum(machine) / total
    expected = human_yes * machine_yes + (1 - human_yes) * (1 - machine_yes)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def agreement_verdict(kappa: float) -> str:
    """O que fazer com o número, dito em uma linha no relatório de calibração."""
    if kappa >= KAPPA_FLOOR:
        return f"kappa {kappa:.4f} >= {KAPPA_FLOOR}: o juiz pode entrar no relatório"
    return f"kappa {kappa:.4f} < {KAPPA_FLOOR}: o juiz NAO pode ter número publicado"
