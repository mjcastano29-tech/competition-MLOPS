"""La racha no depende de que el campeon entienda los datos: siempre sale un batch valido."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

import scripts.infer_and_submit as inf

STATIONS = ("A", "B")


class FixedModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


def write_bundle(root: Path, horizons=(15, 30, 45, 60), value: float = 100.0) -> dict:
    (root / "models").mkdir(parents=True)
    (root / "configs").mkdir()
    config = {
        "history_gap_steps": 0,
        "hgb_weight": 1.0,
        "feature_columns": ["is_weekend", *(f"station_id_{s}" for s in STATIONS)],
    }
    bundle = {}
    for minutes in horizons:
        model_path = root / "models" / f"horizon_{minutes}_hgb.pkl"
        model_path.write_bytes(pickle.dumps(FixedModel(value)))
        (root / "configs" / f"horizon_{minutes}_ensemble.json").write_text(json.dumps({**config, "horizon_minutes": minutes}))
        bundle[minutes] = (model_path, {**config, "horizon_minutes": minutes})
    return bundle


@pytest.fixture
def data(tmp_path, monkeypatch):
    def build(stations=STATIONS, freq="15min", level=100.0):
        stamps = pd.date_range("2026-09-10", "2026-09-12", freq=freq, tz="UTC", inclusive="left")
        obs = pd.concat(pd.DataFrame({"station_id": s, "observed_at": stamps, "demand": level}) for s in stations)
        obs_path, ctx_path = tmp_path / "observations.csv", tmp_path / "context.csv"
        obs.to_csv(obs_path, index=False)
        pd.DataFrame(columns=["observed_at", *inf.CONTEXT_COLUMNS]).to_csv(ctx_path, index=False)
        monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "observations", obs_path)
        monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "context", ctx_path)
        monkeypatch.setenv("BIAS_CORRECTION", "off")
        return stamps[-1].floor("h")
    return build


def cycle_for(cutoff, stations=STATIONS, horizons=(15, 30, 45, 60)):
    targets = [
        {"station_id": s, "target_at": (cutoff + pd.Timedelta(minutes=h)).isoformat().replace("+00:00", "Z")}
        for s in stations for h in horizons
    ]
    return {"cycle_id": "c1", "data_cutoff": cutoff.isoformat(), "targets": targets, "expected_predictions": len(targets)}


def run(cycle, bundle, error=None):
    predictions, report = inf.infer_predictions_with_report(cycle, bundle, error)
    inf.validate_predictions(cycle, predictions)  # el batch siempre es enviable
    return predictions, report


def test_datos_compatibles_usan_solo_el_campeon(tmp_path, data):
    cutoff = data()
    _, report = run(cycle_for(cutoff), write_bundle(tmp_path / "b"))
    assert report["compatible"] and report["sources"] == {"campeon": 8}


def test_una_estacion_nueva_va_al_respaldo_sin_perder_el_ciclo(tmp_path, data):
    cutoff = data(stations=("A", "B", "C"), level=70.0)
    predictions, report = run(cycle_for(cutoff, stations=("A", "B", "C")), write_bundle(tmp_path / "b"))
    assert not report["compatible"] and report["unknown_stations"] == ["C"]
    assert report["sources"] == {"campeon": 8, "persistencia": 4}
    assert {p["value"] for p in predictions if p["station_id"] == "C"} == {70.0}


def test_un_horizonte_nuevo_va_al_respaldo(tmp_path, data):
    cutoff = data()
    _, report = run(cycle_for(cutoff, horizons=(15, 30, 45, 60, 75)), write_bundle(tmp_path / "b"))
    assert report["unsupported_horizons"] == [75]
    assert report["sources"] == {"campeon": 8, "persistencia": 2}


def test_una_estacion_sin_historial_toma_la_mediana_del_ciclo(tmp_path, data):
    cutoff = data()
    predictions, report = run(cycle_for(cutoff, stations=("A", "B", "Z")), write_bundle(tmp_path / "b"))
    assert report["stations_without_history"] == ["Z"]
    assert report["sources"]["mediana_del_ciclo"] == 4
    assert all(p["value"] == 100.0 for p in predictions if p["station_id"] == "Z")


def test_otra_frecuencia_desactiva_el_campeon(tmp_path, data):
    cutoff = data(freq="30min")
    _, report = run(cycle_for(cutoff), write_bundle(tmp_path / "b"))
    assert not report["champion_enabled"] and report["frequency_minutes"] == 30
    assert report["sources"] == {"persistencia": 8}


def test_sin_paquete_se_entrega_con_respaldo(data):
    cutoff = data()
    predictions, report = run(cycle_for(cutoff), {}, "FileNotFoundError: no hay modelo")
    assert report["sources"] == {"persistencia": 8} and not report["compatible"]
    assert inf.model_metadata({})["version"] == inf.FALLBACK_MODEL_VERSION


def test_una_salida_absurda_del_campeon_se_descarta(tmp_path, data):
    cutoff = data()
    _, report = run(cycle_for(cutoff), write_bundle(tmp_path / "b", value=1e9))
    assert report["sources"] == {"persistencia": 8}
    assert report["champion_failures"] == {"prediccion_fuera_de_rango": 8}


def test_un_stream_con_otro_esquema_no_rompe_la_inferencia(data):
    cutoff = data()

    def handler(request):
        return httpx.Response(200, json={"data": [{"station": "A", "ts": "x", "value": 1}], "next_cursor": None})

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api") as client:
        inf.refresh_observations_from_stream(client, cutoff)  # no lanza
