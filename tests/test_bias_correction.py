"""Correccion de sesgo en linea de scripts/infer_and_submit.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.infer_and_submit import (
    BIAS_DEADZONE,
    BIAS_FACTOR_BOUNDS,
    BIAS_MIN_SAMPLES,
    bias_factor,
    guarded_factor,
    station_bias_factors,
)


def test_sin_muestras_suficientes_no_corrige():
    assert bias_factor([(100.0, 50.0)] * (BIAS_MIN_SAMPLES - 1)) == 1.0


def test_el_ruido_dentro_de_la_zona_muerta_no_se_toca():
    ratio = 1 + BIAS_DEADZONE * 0.8
    assert bias_factor([(100.0 * ratio, 100.0)] * 12) == 1.0


def test_solo_se_corrige_el_exceso_amortiguado():
    # Real 40 % por encima: exceso 0.35 sobre la zona muerta, con alpha 0.5 -> 1.175.
    assert bias_factor([(140.0, 100.0)] * 12, alpha=0.5, deadzone=0.05) == pytest.approx(1.175)
    assert bias_factor([(60.0, 100.0)] * 12, alpha=0.5, deadzone=0.05) == pytest.approx(0.825)


def test_un_colapso_de_demanda_queda_acotado():
    factor = bias_factor([(1.0, 100.0)] * 12)
    assert BIAS_FACTOR_BOUNDS[0] <= factor < 1.0


class ConstantModel:
    """Predice siempre lo mismo: el sesgo real queda definido por la serie observada."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


def history(level_by_station: dict[str, float], hours: int = 30) -> pd.DataFrame:
    stamps = pd.date_range("2026-09-10", periods=hours * 4, freq="15min", tz="UTC")
    return pd.concat(
        pd.DataFrame({"station_id": station, "observed_at": stamps, "demand": level})
        for station, level in level_by_station.items()
    ).reset_index(drop=True)


def test_la_correccion_recupera_el_sesgo_de_los_ciclos_ya_observados():
    observations = history({"A": 150.0, "B": 100.0})
    context = pd.DataFrame(columns=["observed_at", "rain_forecast", "temperature_forecast", "event_intensity"])
    data_cutoff = observations["observed_at"].max().floor("h")
    config = {"history_gap_steps": 0, "feature_columns": ["is_weekend"], "hgb_weight": 1.0}
    bundle = {minutes: (Path("unused"), config) for minutes in (15, 30, 45, 60)}
    models = {minutes: ConstantModel(100.0) for minutes in bundle}

    factors = station_bias_factors(["A", "B"], data_cutoff, observations, context, bundle, models)

    # A: real 150 vs predicho 100 -> ratio 1.5, exceso 0.45, alpha 0.5 -> 1.225.
    assert factors["A"] == pytest.approx(1.225)
    assert factors["B"] == 1.0


def test_la_correccion_no_mira_demanda_posterior_al_corte():
    observations = history({"A": 100.0})
    data_cutoff = observations["observed_at"].max().floor("h") - pd.Timedelta(hours=2)
    # Despues del corte la demanda se dispara: no debe influir en el factor.
    observations.loc[observations["observed_at"] > data_cutoff, "demand"] = 1000.0
    context = pd.DataFrame(columns=["observed_at", "rain_forecast", "temperature_forecast", "event_intensity"])
    config = {"history_gap_steps": 0, "feature_columns": ["is_weekend"], "hgb_weight": 1.0}
    bundle = {minutes: (Path("unused"), config) for minutes in (15, 30, 45, 60)}
    models = {minutes: ConstantModel(100.0) for minutes in bundle}

    assert station_bias_factors(["A"], data_cutoff, observations, context, bundle, models)["A"] == 1.0


def test_la_guardia_no_corrige_contra_la_ultima_hora():
    # Tras un pico: la ventana de 3 h pide subir, pero la ultima hora ya va de mas.
    assert guarded_factor(1.3, 0.8) == 1.0
    assert guarded_factor(1.3, None) == 1.0


def test_la_guardia_aplica_el_menor_desvio_confirmado():
    assert guarded_factor(1.3, 1.1) == pytest.approx(1.1)
    assert guarded_factor(1.1, 1.6) == pytest.approx(1.1)
    assert guarded_factor(0.7, 0.9) == pytest.approx(0.9)


def test_con_un_cambio_de_nivel_sostenido_la_guardia_deja_pasar_la_correccion():
    # El nivel real lleva horas 50 % arriba: 3 h y ultima hora coinciden.
    observations = history({"A": 150.0})
    context = pd.DataFrame(columns=["observed_at", "rain_forecast", "temperature_forecast", "event_intensity"])
    data_cutoff = observations["observed_at"].max().floor("h")
    config = {"history_gap_steps": 0, "feature_columns": ["is_weekend"], "hgb_weight": 1.0}
    bundle = {minutes: (Path("unused"), config) for minutes in (15, 30, 45, 60)}
    models = {minutes: ConstantModel(100.0) for minutes in bundle}

    assert station_bias_factors(["A"], data_cutoff, observations, context, bundle, models)["A"] == pytest.approx(1.225)


def test_en_la_bajada_de_un_pico_la_guardia_no_empuja_hacia_arriba():
    observations = history({"A": 100.0})
    data_cutoff = observations["observed_at"].max().floor("h")
    # Pico entre 3 h y 1 h antes del corte; la ultima hora ya volvio a lo normal.
    spike = (observations["observed_at"] > data_cutoff - pd.Timedelta(hours=3)) & (
        observations["observed_at"] <= data_cutoff - pd.Timedelta(hours=1)
    )
    observations.loc[spike, "demand"] = 400.0
    observations.loc[observations["observed_at"] > data_cutoff - pd.Timedelta(hours=1), "demand"] = 90.0
    context = pd.DataFrame(columns=["observed_at", "rain_forecast", "temperature_forecast", "event_intensity"])
    config = {"history_gap_steps": 0, "feature_columns": ["is_weekend"], "hgb_weight": 1.0}
    bundle = {minutes: (Path("unused"), config) for minutes in (15, 30, 45, 60)}
    models = {minutes: ConstantModel(100.0) for minutes in bundle}

    assert station_bias_factors(["A"], data_cutoff, observations, context, bundle, models)["A"] == 1.0
