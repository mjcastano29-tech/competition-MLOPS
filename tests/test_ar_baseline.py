"""AR(2) local de scripts/ar_baseline.py y su mezcla en la inferencia."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.infer_and_submit as inf
from scripts.ar_baseline import (
    AR_WINDOW,
    apply_ar_weights,
    ar2_station_forecast,
    build_normal_profile,
    choose_ar_weights,
    save_profile,
)

START = pd.Timestamp("2026-08-01", tz="UTC")


def flat_history(days=30, level=100.0, station="A"):
    stamps = pd.date_range(START, periods=days * 96, freq="15min")
    return pd.DataFrame({"station_id": station, "observed_at": stamps, "demand": level})


def test_el_perfil_es_la_media_por_dia_y_cuarto():
    profile = build_normal_profile(flat_history())
    grid = profile["stations"]["A"]
    assert len(grid) == 7 and all(len(day) == 96 for day in grid)
    assert {value for day in grid for value in day} == {100.0}


def test_el_ar_extrapola_una_oscilacion_amortiguada():
    history = flat_history()
    profile = build_normal_profile(history)
    # Desviacion log que sigue exactamente un AR(2) con raices complejas (una onda).
    a, b = 1.6, -0.8
    x = [0.2, 0.35]
    for _ in range(AR_WINDOW + 4 - 2):
        x.append(a * x[-1] + b * x[-2])
    series = history.set_index("observed_at")["demand"].copy()
    tail = series.index[-(AR_WINDOW + 4):]
    series.loc[tail] = 100.0 * np.exp(np.array(x))
    cutoff = tail[AR_WINDOW - 1]
    forecast = ar2_station_forecast(series, cutoff, profile, "A")
    np.testing.assert_allclose(forecast, 100.0 * np.exp(np.array(x[AR_WINDOW:])), rtol=1e-6)


def test_sin_ventana_completa_no_hay_pronostico():
    history = flat_history()
    profile = build_normal_profile(history)
    series = history.set_index("observed_at")["demand"].copy()
    cutoff = series.index[-1]
    series.loc[cutoff - pd.Timedelta(hours=1)] = np.nan
    assert ar2_station_forecast(series, cutoff, profile, "A") is None
    assert ar2_station_forecast(series, cutoff, profile, "desconocida") is None


def test_los_pesos_siguen_al_ar_solo_donde_ayuda():
    stations = np.array(["bueno"] * 4 + ["malo"] * 4)
    y = np.array([100.0] * 8)
    base = np.array([80.0] * 8)
    ar = np.array([100.0] * 4 + [20.0] * 4)  # en "malo" el AR empuja lejos del real
    weights = choose_ar_weights(stations, y, base, ar)
    assert weights["bueno"] == 0.5 and weights["malo"] == 0.0


def test_sin_ar_la_mezcla_deja_la_base():
    out = apply_ar_weights(np.array([10.0, 20.0]), np.array([np.nan, 40.0]), np.array(["A", "A"]), {"A": 0.5})
    np.testing.assert_allclose(out, [10.0, 30.0])


class FixedModel:
    def predict(self, frame):
        return np.full(len(frame), 100.0)


def test_la_inferencia_mezcla_igual_que_el_entrenamiento(tmp_path, monkeypatch):
    history = flat_history(level=100.0)
    # Ultima hora a 200: el AR (nivel 2x sobre lo normal) pronostica por encima del campeon.
    history.loc[history.index[-4:], "demand"] = 200.0
    obs_path, ctx_path = tmp_path / "obs.csv", tmp_path / "ctx.csv"
    history.to_csv(obs_path, index=False)
    pd.DataFrame(columns=["observed_at", *inf.CONTEXT_COLUMNS]).to_csv(ctx_path, index=False)
    monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "observations", obs_path)
    monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "context", ctx_path)
    monkeypatch.setenv("BIAS_CORRECTION", "off")
    root = tmp_path / "bundle"
    (root / "models").mkdir(parents=True)
    (root / "configs").mkdir()
    save_profile(build_normal_profile(flat_history(level=100.0)), root)
    config = {"history_gap_steps": 0, "hgb_weight": 1.0, "feature_columns": ["is_weekend"], "ar_weights": {"A": 0.4}}
    bundle = {}
    for minutes in (15, 30, 45, 60):
        path = root / "models" / f"horizon_{minutes}_hgb.pkl"
        path.write_bytes(pickle.dumps(FixedModel()))
        bundle[minutes] = (path, {**config, "horizon_minutes": minutes})
    cutoff = history["observed_at"].max()
    cycle = {
        "cycle_id": "c", "data_cutoff": cutoff.isoformat(), "expected_predictions": 4,
        "targets": [{"station_id": "A", "target_at": (cutoff + pd.Timedelta(minutes=15 * h)).isoformat()} for h in (1, 2, 3, 4)],
    }
    predictions, report = inf.infer_predictions_with_report(cycle, bundle)

    series = history.set_index("observed_at")["demand"].astype(float)
    ar = ar2_station_forecast(series, cutoff, build_normal_profile(flat_history(level=100.0)), "A")
    expected = apply_ar_weights(np.full(4, 100.0), ar, np.array(["A"] * 4), {"A": 0.4})
    np.testing.assert_allclose([p["value"] for p in predictions], np.round(expected, 4))
    assert report["ar_mixed_stations"] == ["A"] and report["sources"] == {"campeon": 4}
