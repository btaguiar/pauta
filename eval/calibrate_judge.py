"""Calibra o juiz contra rótulos humanos e calcula o Kappa de Cohen.

    uv run python eval/calibrate_judge.py

Roda o juiz sobre `judge_calibration.jsonl`, compara com o rótulo `sustentado`
de cada caso e grava o resultado em `results/`. Enquanto o kappa estiver abaixo
do piso, nenhum número do juiz entra em relatório.

Limite conhecido, e ele é grande: os casos versionados foram construídos para
serem inequívocos, e um conjunto sem caso difícil superestima a concordância.
Antes de publicar qualquer número de fidelidade, acrescente casos reais, tirados
de rodadas do eval e rotulados à mão, e rode isto de novo.
"""

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from judge import KAPPA_FLOOR, Verdict, agreement_verdict, cohen_kappa, get_judge, judge_report
from pauta.config import get_settings
from pauta.memory.checkpointer import run_async
from pauta.observability import setup_logging

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = REPO_ROOT / "eval" / "judge_calibration.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

MISSING_JUDGE_MESSAGE = """Falta o JUDGE_MODEL para calibrar.

Preencha JUDGE_MODEL no .env, apontando para um provider DIFERENTE do executor.
Modelo que julga a própria saída se prefere, e a calibração deixa de medir o que
deveria.

Detalhe: {detail}
"""


@dataclass(frozen=True)
class Judged:
    """Um caso rotulado, com o que o juiz respondeu sobre ele."""

    case_id: str
    human: bool
    machine: bool | None
    motivo: str


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def judge_case(case: dict[str, Any]) -> Judged:
    verdict: Verdict | None = await judge_report(
        case["report"],
        case["findings"],
        model=get_judge(),
        task_id=case["id"],
    )
    return Judged(
        case_id=case["id"],
        human=bool(case["sustentado"]),
        machine=verdict.sustentado if verdict else None,
        motivo=verdict.motivo if verdict else "o juiz não respondeu",
    )


def summarise(judged: list[Judged]) -> dict[str, Any]:
    """Kappa sobre os casos que o juiz conseguiu julgar, com o `_n` junto."""
    answered = [item for item in judged if item.machine is not None]
    if not answered:
        return {"kappa": 0.0, "kappa_n": 0, "sem_resposta": len(judged), "veredito": "sem dado"}
    human = [item.human for item in answered]
    machine = [bool(item.machine) for item in answered]
    kappa = cohen_kappa(human, machine)
    return {
        "kappa": kappa,
        "kappa_n": len(answered),
        "sem_resposta": len(judged) - len(answered),
        "concordancia_crua": round(
            sum(1 for a, b in zip(human, machine, strict=True) if a == b) / len(answered), 4
        ),
        "piso": KAPPA_FLOOR,
        "aprovado": kappa >= KAPPA_FLOOR,
        "veredito": agreement_verdict(kappa),
    }


def write_calibration(summary: dict[str, Any], judged: list[Judged]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"judge_calibration_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "judge_model": get_settings().JUDGE_MODEL,
        "resumo": summary,
        "casos": [
            {
                "id": item.case_id,
                "humano": item.human,
                "juiz": item.machine,
                "motivo": item.motivo,
                "concordou": item.machine == item.human,
            }
            for item in judged
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return path


async def main_async() -> int:
    setup_logging()
    try:
        get_settings()
    except ValidationError as exc:
        sys.stderr.write(MISSING_JUDGE_MESSAGE.format(detail=exc))
        return 2
    if not get_settings().JUDGE_MODEL:
        sys.stderr.write(MISSING_JUDGE_MESSAGE.format(detail="JUDGE_MODEL vazio"))
        return 2

    cases = load_cases(CALIBRATION_FILE)
    judged = [await judge_case(case) for case in cases]
    summary = summarise(judged)
    path = write_calibration(summary, judged)

    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + f"\ngravado: {path}\n")
    for item in judged:
        if item.machine != item.human:
            sys.stdout.write(
                f"  discordou em {item.case_id}: humano={item.human} "
                f"juiz={item.machine} · {item.motivo}\n"
            )
    return 0 if summary.get("aprovado") else 1


def main() -> int:
    return int(run_async(main_async()))


if __name__ == "__main__":
    raise SystemExit(main())
