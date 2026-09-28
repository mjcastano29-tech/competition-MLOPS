"""El promotor decide con la evidencia emparejada que viaja dentro del paquete."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scripts.model_gate import Evaluation, evaluation_to_dict
from scripts.promote_model import legacy_report, paired_gate


WINDOW_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
WINDOW_END = WINDOW_START + timedelta(days=20)
ROWS_HASH = "c" * 64
HORIZONS = {15: 80.0, 30: 78.0, 45: 76.0, 60: 74.0}


def build_evaluation(
    accuracy_by_horizon: dict[int, float], *, version: str, **overrides
) -> Evaluation:
    accuracy = sum(accuracy_by_horizon.values()) / len(accuracy_by_horizon)
    fields = {
        "version": version,
        "accuracy": accuracy,
        "wape": (100 - accuracy) / 100,
        "validation_start": WINDOW_START,
        "validation_end": WINDOW_END,
        "history_gap_steps": 133,
        "dataset_rows_hash": ROWS_HASH,
        "accuracy_by_horizon": accuracy_by_horizon,
        "wape_by_horizon": {h: (100 - a) / 100 for h, a in accuracy_by_horizon.items()},
        "station_count": 12,
        "folds": 3,
        "rows": 144,
        "model_name": "Ensemble HGB + Seasonal Naive 7d (0.7)",
    }
    fields.update(overrides)
    return Evaluation(**fields)


def write_package(
    root: Path,
    *,
    candidate: Evaluation | None,
    incumbent: Evaluation | None,
    model_version: str | None = None,
    summary: dict[int, float] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if candidate is not None or incumbent is not None:
        payload = {
            "schema_version": 1,
            "validation_window": {
                "start": WINDOW_START.isoformat(),
                "end": WINDOW_END.isoformat(),
                "history_gap_steps": 133,
            },
            "dataset": {"name": "v0003-test", "rows_hash": ROWS_HASH},
            "candidate": evaluation_to_dict(candidate),
            "incumbent": evaluation_to_dict(incumbent),
        }
        (root / "promotion_inputs.json").write_text(json.dumps(payload), encoding="utf-8")
    if model_version is not None:
        (root / "manifest.json").write_text(
            json.dumps({"model_version": model_version}), encoding="utf-8"
        )
    if summary is not None:
        pd.DataFrame(
            {
                "horizon_minutes": list(summary),
                "accuracy": [summary[horizon] for horizon in summary],
                "history_gap_steps": [133] * len(summary),
            }
        ).to_csv(root / "best_ensemble_summary.csv", index=False)
    return root


def test_promueve_con_mejora_emparejada_y_campeon_coincidente(tmp_path: Path):
    candidate = build_evaluation({15: 81.0, 30: 79.0, 45: 77.0, 60: 75.0}, version="run-candidato")
    incumbent = build_evaluation(HORIZONS, version="run-campeon")
    package = write_package(tmp_path / "candidate", candidate=candidate, incumbent=incumbent)
    write_package(
        tmp_path / "previous", candidate=incumbent, incumbent=None, model_version="run-campeon"
    )

    decision, payload = paired_gate(package, tmp_path / "previous")

    assert decision.promote
    assert decision.decision == "paired_improvement"
    assert payload["dataset"]["rows_hash"] == ROWS_HASH


def test_bloquea_cuando_el_campeon_se_midio_en_otra_ventana(tmp_path: Path):
    candidate = build_evaluation({15: 95.0, 30: 95.0, 45: 95.0, 60: 95.0}, version="run-candidato")
    incumbent = build_evaluation(
        HORIZONS,
        version="run-campeon",
        validation_start=WINDOW_START - timedelta(days=28),
        validation_end=WINDOW_END - timedelta(days=28),
    )
    package = write_package(tmp_path / "candidate", candidate=candidate, incumbent=incumbent)
    write_package(
        tmp_path / "previous", candidate=incumbent, incumbent=None, model_version="run-campeon"
    )

    decision, _ = paired_gate(package, tmp_path / "previous")

    assert not decision.promote
    assert decision.decision == "incomparable"


def test_bloquea_si_el_campeon_emparejado_no_es_el_paquete_activo(tmp_path: Path):
    candidate = build_evaluation({15: 81.0, 30: 79.0, 45: 77.0, 60: 75.0}, version="run-candidato")
    incumbent = build_evaluation(HORIZONS, version="run-viejo")
    package = write_package(tmp_path / "candidate", candidate=candidate, incumbent=incumbent)
    write_package(
        tmp_path / "previous", candidate=incumbent, incumbent=None, model_version="run-actual"
    )

    decision, _ = paired_gate(package, tmp_path / "previous")

    assert not decision.promote
    assert decision.decision == "incumbent_mismatch"


def test_sin_evidencia_emparejada_el_resumen_antiguo_no_promueve(tmp_path: Path):
    package = write_package(
        tmp_path / "candidate",
        candidate=None,
        incumbent=None,
        summary={15: 90.0, 30: 90.0, 45: 90.0, 60: 90.0},
    )
    previous = write_package(
        tmp_path / "previous", candidate=None, incumbent=None, summary=dict(HORIZONS)
    )

    assert paired_gate(package, previous) is None
    report = legacy_report(package, previous, allow_unpaired=False)

    assert report["comparability"] == "unpaired"
    assert report["promoted"] is False, "un resumen sin ventana no justifica reemplazar el modelo"
    assert report["gate"]["decision"] == "unpaired_blocked"


def test_la_salida_de_emergencia_si_acepta_el_resumen_antiguo(tmp_path: Path):
    package = write_package(
        tmp_path / "candidate",
        candidate=None,
        incumbent=None,
        summary={15: 90.0, 30: 90.0, 45: 90.0, 60: 90.0},
    )
    previous = write_package(
        tmp_path / "previous", candidate=None, incumbent=None, summary=dict(HORIZONS)
    )

    report = legacy_report(package, previous, allow_unpaired=True)

    assert report["promoted"] is True
    assert report["gate"]["decision"] == "unpaired_override"


def test_sin_paquete_previo_solo_se_activa_si_se_autoriza_el_arranque(tmp_path: Path):
    candidate = build_evaluation({15: 90.0, 30: 90.0, 45: 90.0, 60: 90.0}, version="run-unico")
    package = write_package(tmp_path / "candidate", candidate=candidate, incumbent=None)

    assert paired_gate(package, None)[0].decision == "incumbent_missing"
    assert paired_gate(package, None, allow_first_model=True)[0].decision == "first_model"


def test_los_umbrales_de_negocio_viajan_hasta_la_compuerta_emparejada(tmp_path: Path):
    """Ganar un punto de media no salva a una estacion que se hunde cuatro puntos."""

    candidate = build_evaluation(
        {15: 81.0, 30: 79.0, 45: 77.0, 60: 75.0},
        version="run-candidato",
        accuracy_by_station={"0000": 81.0, "0100": 76.0},
    )
    incumbent = build_evaluation(
        HORIZONS, version="run-campeon", accuracy_by_station={"0000": 80.0, "0100": 80.0}
    )
    package = write_package(tmp_path / "candidate", candidate=candidate, incumbent=incumbent)
    write_package(
        tmp_path / "previous", candidate=incumbent, incumbent=None, model_version="run-campeon"
    )

    blocked, _ = paired_gate(package, tmp_path / "previous")
    assert blocked.decision == "station_regression"
    assert blocked.accuracy_delta > 0, "la media sube y aun asi no se promociona"

    tolerado, _ = paired_gate(package, tmp_path / "previous", max_station_drop=5.0)
    assert tolerado.decision == "paired_improvement"

    exigente, _ = paired_gate(
        package, tmp_path / "previous", max_station_drop=5.0, min_accuracy_gain=2.0
    )
    assert exigente.decision == "no_improvement"


def test_el_minimo_de_ganancia_exigido_es_de_medio_punto(tmp_path: Path):
    marginal = build_evaluation(
        {15: 80.4, 30: 78.4, 45: 76.4, 60: 74.4}, version="run-candidato"
    )
    incumbent = build_evaluation(HORIZONS, version="run-campeon")
    package = write_package(tmp_path / "candidate", candidate=marginal, incumbent=incumbent)
    write_package(
        tmp_path / "previous", candidate=incumbent, incumbent=None, model_version="run-campeon"
    )

    decision, _ = paired_gate(package, tmp_path / "previous")

    assert decision.decision == "no_improvement", "un salto de 0.4 pts es ruido, no mejora"
