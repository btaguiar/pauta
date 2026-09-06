"""Juiz binário e calibração. Nenhuma chamada de modelo real aqui."""

import json
from pathlib import Path

import pytest

import calibrate_judge
from judge import (
    KAPPA_FLOOR,
    Verdict,
    agreement_verdict,
    build_messages,
    cohen_kappa,
    judge_report,
)
from tests.fakes import FakeChatModel

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_material_reaches_the_judge_with_the_briefing() -> None:
    messages = build_messages("O custo é 1800 USD.", ["custo de 1800 USD (fonte: https://a)"])
    prompt = str(messages[1].content)
    assert "1800 USD (fonte: https://a)" in prompt
    assert "O custo é 1800 USD." in prompt


def test_an_empty_material_is_said_out_loud() -> None:
    assert "nenhuma descoberta foi registrada" in str(
        build_messages("qualquer coisa", []).pop().content
    )


async def test_a_supported_briefing_comes_back_supported() -> None:
    model = FakeChatModel(responses=[Verdict(sustentado=True, motivo="cada afirmação tem fonte")])
    verdict = await judge_report("relatório", ["descoberta"], model=model)
    assert verdict is not None
    assert verdict.sustentado is True


async def test_a_judge_that_fails_returns_none_not_a_rejection() -> None:
    """Juiz que caiu não é evidência de briefing ruim, e não pode virar reprovação."""
    model = FakeChatModel(responses=[RuntimeError("provider fora do ar")])
    assert await judge_report("relatório", ["descoberta"], model=model) is None


async def test_an_unstructured_answer_returns_none() -> None:
    model = FakeChatModel(responses=["talvez sim, talvez não"])
    assert await judge_report("relatório", ["descoberta"], model=model) is None


def test_perfect_agreement_is_one() -> None:
    assert cohen_kappa([True, False, True], [True, False, True]) == 1.0


def test_total_disagreement_is_negative() -> None:
    assert cohen_kappa([True, False], [False, True]) < 0


def test_kappa_punishes_a_judge_that_always_says_yes() -> None:
    """É por isto que kappa existe: acurácia crua premiaria este juiz com 0,9."""
    human = [True] * 9 + [False]
    machine = [True] * 10
    assert sum(1 for a, b in zip(human, machine, strict=True) if a == b) / 10 == 0.9
    assert cohen_kappa(human, machine) == 0.0


def test_a_single_class_that_matches_is_full_agreement() -> None:
    assert cohen_kappa([True, True], [True, True]) == 1.0


def test_calibrating_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="sem nenhum caso"):
        cohen_kappa([], [])


def test_mismatched_lists_are_refused() -> None:
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        cohen_kappa([True], [True, False])


def test_the_floor_decides_whether_the_number_can_be_published() -> None:
    assert "pode entrar" in agreement_verdict(KAPPA_FLOOR)
    assert "NAO pode" in agreement_verdict(KAPPA_FLOOR - 0.01)


def test_the_calibration_file_is_labelled_and_balanced() -> None:
    """Conjunto só de um rótulo não calibra nada: kappa precisa das duas classes."""
    cases = calibrate_judge.load_cases(REPO_ROOT / "eval" / "judge_calibration.jsonl")
    assert len(cases) >= 8
    labels = [case["sustentado"] for case in cases]
    assert any(labels) and not all(labels)
    assert all(isinstance(case["findings"], list) for case in cases)
    assert all(case["origem"] == "construido" for case in cases)


def test_every_calibration_case_has_a_unique_id() -> None:
    cases = calibrate_judge.load_cases(REPO_ROOT / "eval" / "judge_calibration.jsonl")
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))


def test_the_summary_separates_disagreement_from_silence() -> None:
    judged = [
        calibrate_judge.Judged("cal-01", human=True, machine=True, motivo="ok"),
        calibrate_judge.Judged("cal-02", human=False, machine=False, motivo="ok"),
        calibrate_judge.Judged("cal-03", human=True, machine=None, motivo="não respondeu"),
    ]
    summary = calibrate_judge.summarise(judged)
    assert summary["kappa_n"] == 2
    assert summary["sem_resposta"] == 1
    assert summary["kappa"] == 1.0
    assert summary["aprovado"] is True


def test_a_judge_that_never_answered_reports_no_data() -> None:
    judged = [calibrate_judge.Judged("cal-01", human=True, machine=None, motivo="caiu")]
    summary = calibrate_judge.summarise(judged)
    assert summary["kappa_n"] == 0
    assert summary["veredito"] == "sem dado"


def test_the_calibration_artifact_records_every_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibrate_judge, "RESULTS_DIR", tmp_path)
    judged = [calibrate_judge.Judged("cal-01", human=True, machine=False, motivo="faltou fonte")]
    path = calibrate_judge.write_calibration(calibrate_judge.summarise(judged), judged)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["casos"][0]["concordou"] is False
    assert payload["casos"][0]["motivo"] == "faltou fonte"
    assert payload["resumo"]["aprovado"] is False
