"""A matriz de configurações: o instrumento que transforma a escolha em medição."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

import matrix
from matrix import Configuration, ConfigurationResult, InvalidSpec
from pauta.config import get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def a_spec(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "nome": "tudo-barato",
        "MODEL_ROUTER": "casa/router-barato",
        "MODEL_CRITIC": "casa/critic-barato",
        "MODEL_WORKER": "casa/worker-barato",
    }
    entry.update(overrides)
    return {"configuracoes": [entry]}


def write_spec(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def a_result(name: str, **metrics: Any) -> ConfigurationResult:
    base: dict[str, Any] = {
        "cobertura": 0.5,
        "cobertura_n": 10,
        "incerteza_sinalizada": 0.5,
        "incerteza_sinalizada_n": 4,
        "calculadora_usada": 1.0,
        "dentro_do_teto": 1.0,
        "fidelidade": 0.0,
        "fidelidade_n": 0,
        "tokens_media": 1000.0,
        "latencia_media_s": 2.0,
        "custo_usd": None,
        "falhas": 0,
    }
    base.update(metrics)
    return ConfigurationResult(
        name=name,
        models={"MODEL_ROUTER": "r", "MODEL_CRITIC": "c", "MODEL_WORKER": "w"},
        notes="",
        metrics=base,
    )


def test_a_filled_spec_loads(tmp_path: Path) -> None:
    configurations = matrix.load_spec(write_spec(tmp_path, a_spec()))
    assert [item.name for item in configurations] == ["tudo-barato"]
    assert configurations[0].models["MODEL_CRITIC"] == "casa/critic-barato"


def test_a_blank_model_is_refused_before_any_money_is_spent(tmp_path: Path) -> None:
    """Descobrir na terceira configuração que a quarta está vazia custa as três."""
    spec = write_spec(tmp_path, a_spec(MODEL_CRITIC=""))
    with pytest.raises(InvalidSpec, match="MODEL_CRITIC"):
        matrix.load_spec(spec)


def test_a_nameless_configuration_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidSpec, match="nome"):
        matrix.load_spec(write_spec(tmp_path, a_spec(nome="  ")))


def test_a_repeated_name_is_refused(tmp_path: Path) -> None:
    payload = a_spec()
    payload["configuracoes"].append(dict(payload["configuracoes"][0]))
    with pytest.raises(InvalidSpec, match="repetido"):
        matrix.load_spec(write_spec(tmp_path, payload))


def test_an_empty_matrix_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InvalidSpec, match="configuracoes"):
        matrix.load_spec(write_spec(tmp_path, {"configuracoes": []}))


def test_a_missing_file_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(InvalidSpec, match="não achei"):
        matrix.load_spec(tmp_path / "nao-existe.json")


def test_the_versioned_spec_is_a_template_nobody_can_run_by_accident() -> None:
    """`matrix.json` vai versionado vazio, pelo mesmo motivo do `.env.example`."""
    with pytest.raises(InvalidSpec, match="em branco"):
        matrix.load_spec(REPO_ROOT / "eval" / "matrix.json")


def test_the_versioned_spec_varies_one_role_at_a_time() -> None:
    raw = json.loads((REPO_ROOT / "eval" / "matrix.json").read_text(encoding="utf-8"))
    names = [entry["nome"] for entry in raw["configuracoes"]]
    assert names[0] == "tudo-barato", "a primeira linha é a base dos deltas"
    assert {"router-melhor", "critico-melhor", "worker-melhor"} <= set(names)
    assert all(entry.get("nota") for entry in raw["configuracoes"])


def test_applying_a_configuration_changes_what_the_settings_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem limpar os caches, a matriz mediria a mesma configuração várias vezes."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    before = get_settings().MODEL_CRITIC
    matrix.apply_configuration(
        Configuration(
            name="critico-melhor",
            models={
                "MODEL_ROUTER": "casa/router",
                "MODEL_CRITIC": "outra-casa/critic-forte",
                "MODEL_WORKER": "casa/worker",
            },
        )
    )
    assert get_settings().MODEL_CRITIC == "outra-casa/critic-forte"
    assert before != get_settings().MODEL_CRITIC


def test_the_embedding_is_not_tunable() -> None:
    """Trocar o embedding no meio compararia corpus diferentes, não modelos."""
    assert "EMBEDDING_MODEL" not in matrix.TUNABLE
    assert set(matrix.TUNABLE) == {"MODEL_ROUTER", "MODEL_CRITIC", "MODEL_WORKER"}


def test_the_delta_is_measured_against_the_first_line() -> None:
    baseline = a_result("tudo-barato", cobertura=0.5)
    better = a_result("critico-melhor", cobertura=0.8)
    assert matrix.delta_against(baseline, better)["cobertura"] == pytest.approx(0.3)


def test_a_metric_nobody_could_compute_stays_out_of_the_delta() -> None:
    baseline = a_result("tudo-barato", custo_usd=None)
    other = a_result("critico-melhor", custo_usd=None)
    assert "custo_usd" not in matrix.delta_against(baseline, other)


def test_the_table_puts_the_configurations_side_by_side() -> None:
    rendered = matrix.render_matrix(
        [a_result("tudo-barato", cobertura=0.5), a_result("critico-melhor", cobertura=0.8)]
    )
    assert "tudo-barato" in rendered
    assert "critico-melhor" in rendered
    assert "+0.3000" in rendered
    assert "cobertura_n=10" in rendered


def test_the_table_refuses_to_recommend() -> None:
    """A escolha dos tiers é decisão de quem lê, não do script."""
    rendered = matrix.render_matrix([a_result("tudo-barato")])
    assert "não escolhe" in rendered
    assert "ARCHITECTURE.md" in rendered


def test_an_unmeasured_cost_shows_as_unavailable() -> None:
    assert matrix.cell(None) == "n/d"
    assert matrix.cell(0.5) == "0.5000"


def test_the_artifact_records_every_configuration_with_its_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(matrix, "RESULTS_DIR", tmp_path)
    path = matrix.write_matrix(
        [a_result("tudo-barato"), a_result("critico-melhor", cobertura=0.8)],
        repeats=3,
        limit=5,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name.startswith("matriz_")
    assert payload["repeticoes"] == 3
    assert [item["nome"] for item in payload["configuracoes"]] == [
        "tudo-barato",
        "critico-melhor",
    ]
    assert payload["configuracoes"][1]["delta_vs_base"]["cobertura"] == pytest.approx(0.3)
    assert payload["configuracoes"][0]["modelos"]["MODEL_CRITIC"] == "c"


def test_the_matrix_artifact_carries_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-nunca-pode-vazar"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setattr(matrix, "RESULTS_DIR", tmp_path)
    get_settings.cache_clear()

    path = matrix.write_matrix([a_result("tudo-barato")], repeats=1, limit=None)
    assert secret not in path.read_text(encoding="utf-8")
