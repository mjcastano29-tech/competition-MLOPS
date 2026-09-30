"""Paridad numerica entre `add_features()` (entrenamiento) y la inferencia.

`tests/test_feature_consistency.py` fija que el vocabulario exista; este archivo fija
que el NUMERO que produce cada familia del protocolo v2 sea el mismo en los dos extremos
del pipeline. Sin esta comprobacion, un `shift` mal traducido a la busqueda por marca de
tiempo de `scripts/infer_and_submit.py` da predicciones silenciosamente peores: el mismo
tipo de fallo que hizo perder accuracy cuando los lags cortos se colapsaban a 133 pasos.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.infer_and_submit import (
    CONTEXT_FALLBACK_DAYS,
    MAX_CONTEXT_AGE_MINUTES,
    PERIODS_PER_DAY,
    TARGET_SEASONAL_DAYS,
    _feature_row_for_target,
    unsupported_feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]
GAP_STEPS = 0
# Hueco del paquete heredado: la inferencia debe seguir reproduciendolo.
LEGACY_GAP_STEPS = 133
DAYS = 9
STATIONS = ("02300", "07107")


def load_training_module():
    """03 arrastra sklearn/mlflow: en CI sin el extra `ml` se omite el modulo entero."""

    pytest.importorskip("mlflow")
    path = ROOT / "examples" / "03_gradient_boosting.py"
    spec = importlib.util.spec_from_file_location("gradient_boosting_module", path)
    if spec is None or spec.loader is None:
        pytest.skip(f"No se pudo cargar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history_frame() -> pd.DataFrame:
    """Dos estaciones con series distintas y sin empates, para detectar grupos cruzados."""

    stamps = pd.date_range("2026-09-01", periods=DAYS * PERIODS_PER_DAY, freq="15min", tz="UTC")
    frames = []
    for offset, station_id in enumerate(STATIONS, start=1):
        index = np.arange(len(stamps), dtype=float)
        demand = (
            100 * offset
            + 40 * np.sin(2 * np.pi * (index % PERIODS_PER_DAY) / 96)
            + 7 * (index % 13)
            + offset
        )
        frames.append(
            pd.DataFrame({"station_id": station_id, "observed_at": stamps, "demand": demand})
        )
    return pd.concat(frames, ignore_index=True)


def context_frame(observations: pd.DataFrame, covered_days: int) -> pd.DataFrame:
    """Contexto publicable solo hasta `covered_days`; despues, nada (como el stream)."""

    stamps = observations["observed_at"].drop_duplicates().sort_values().reset_index(drop=True)
    stamps = stamps.iloc[: covered_days * PERIODS_PER_DAY]
    index = np.arange(len(stamps), dtype=float)
    return pd.DataFrame(
        {
            "observed_at": stamps,
            "rain_mm": np.round((index % 5) * 0.4, 3),
            "rain_forecast": np.round(((index + 1) % 7) * 0.6, 3),
            "temperature_c": 8.0 + (index % 11) / 5,
            "temperature_forecast": 9.0 + (index % 13) / 4,
            "event_intensity": (index % 4) * 0.5,
        }
    )


@pytest.fixture(scope="module")
def module():
    return load_training_module()


def candidate_columns(module, frame: pd.DataFrame) -> list[str]:
    return module.model_feature_columns(frame.columns)


@pytest.mark.parametrize("covered_days", [DAYS, 3], ids=["contexto_fresco", "contexto_viejo"])
def test_cada_columna_del_candidato_es_construible_en_inferencia(module, covered_days):
    observations = history_frame()
    context = context_frame(observations, covered_days)
    frame = module.add_features(observations.copy(), context.copy(), GAP_STEPS)

    assert unsupported_feature_columns(candidate_columns(module, frame)) == []
    # El stream sin contexto ya no se cae por el `dropna()`: toda fila con historial
    # suficiente queda util, incluidas las de los ultimos dias sin clima publicado.
    assert frame["observed_at"].max() == observations["observed_at"].max()
    assert len(frame) == len(observations) - 2 * PERIODS_PER_DAY * 7


@pytest.mark.parametrize("covered_days", [DAYS, 3], ids=["contexto_fresco", "contexto_viejo"])
@pytest.mark.parametrize("horizon_minutes", [15, 60])
@pytest.mark.parametrize("gap_steps", [GAP_STEPS, LEGACY_GAP_STEPS], ids=["gap0", "gap133"])
def test_entrenamiento_e_inferencia_producen_el_mismo_valor(
    module, covered_days, horizon_minutes, gap_steps
):
    observations = history_frame()
    context = context_frame(observations, covered_days)
    frame = module.add_features(observations.copy(), context.copy(), gap_steps)
    station_id = STATIONS[1]
    data_cutoff = frame["observed_at"].max()
    target_at = data_cutoff + pd.Timedelta(minutes=horizon_minutes)
    horizon_columns = module.feature_columns_for_horizon(
        candidate_columns(module, frame), horizon_minutes
    )
    config = {
        "horizon_minutes": horizon_minutes,
        "history_gap_steps": gap_steps,
        "target_seasonal_days": list(module.TARGET_SEASONAL_DAYS),
        "feature_columns": horizon_columns,
    }

    row = _feature_row_for_target(
        station_id, target_at, data_cutoff, observations, context, config
    )
    training = frame.loc[
        (frame["station_id"] == station_id) & (frame["observed_at"] == data_cutoff)
    ].iloc[-1]

    mismatches = {
        column: (float(training[column]), float(row[column]))
        for column in horizon_columns
        if abs(float(training[column]) - float(row[column])) > 1e-6
    }

    assert not mismatches, f"train/inferencia divergen: {mismatches}"
    assert len(row) == len(horizon_columns)


def test_una_columna_estacional_de_otro_horizonte_aborta(module):
    observations = history_frame()
    context = context_frame(observations, DAYS)
    config = {
        "horizon_minutes": 15,
        "history_gap_steps": GAP_STEPS,
        "feature_columns": ["target_seasonal_mean_60"],
    }

    with pytest.raises(RuntimeError, match="no coinciden"):
        _feature_row_for_target(
            STATIONS[0],
            pd.Timestamp("2026-09-09 12:00", tz="UTC"),
            pd.Timestamp("2026-09-09 11:45", tz="UTC"),
            observations,
            context,
            config,
        )


def test_el_contexto_viejo_se_imputa_igual_en_los_dos_mundos(module):
    observations = history_frame()
    context = context_frame(observations, 3)
    frame = module.add_features(observations.copy(), context.copy(), GAP_STEPS)
    station_id = STATIONS[0]
    data_cutoff = frame["observed_at"].max()
    assert (frame["context_is_fresh"] == 0.0).any(), "sin contexto la marca debe bajar a 0"

    columns = ["rain_forecast", "temperature_forecast", "event_intensity", "context_is_fresh"]
    config = {
        "horizon_minutes": 15,
        "history_gap_steps": GAP_STEPS,
        "feature_columns": [*columns, "is_weekend"],
    }
    row = _feature_row_for_target(
        station_id,
        data_cutoff + pd.Timedelta(minutes=15),
        data_cutoff,
        observations,
        context,
        config,
    )
    training = frame.loc[
        (frame["station_id"] == station_id) & (frame["observed_at"] == data_cutoff)
    ].iloc[-1]

    for column in columns:
        assert abs(float(training[column]) - float(row[column])) < 1e-9, column
    # La imputacion es la mediana de los ultimos dias, no un cero silencioso.
    assert row["temperature_forecast"] > 0.0


def test_las_constantes_del_protocolo_coinciden_en_los_dos_modulos(module):
    """Un desfase aqui es exactamente el skew que hundio la exactitud del modelo."""

    assert module.TRAINING_HISTORY_GAP_STEPS == GAP_STEPS
    assert module.TARGET_SEASONAL_DAYS == TARGET_SEASONAL_DAYS
    assert module.PERIODS_PER_DAY == PERIODS_PER_DAY
    assert module.MAX_CONTEXT_AGE_MINUTES == MAX_CONTEXT_AGE_MINUTES
    assert module.CONTEXT_FALLBACK_DAYS == CONTEXT_FALLBACK_DAYS


def test_los_lags_heredados_no_entrenan_pero_la_baseline_si_existe(module):
    observations = history_frame()
    context = context_frame(observations, DAYS)
    frame = module.add_features(observations.copy(), context.copy(), LEGACY_GAP_STEPS)
    columns = candidate_columns(module, frame)

    assert not [column for column in columns if column.startswith("demand_lag_")]
    for horizon in module.HORIZONS:
        baseline = f"demand_lag_{672 - horizon}"
        assert baseline in frame.columns, baseline
        # `target_lag_7d_*` es, paso a paso, esa misma baseline estacional.
        minutes = horizon * module.PERIOD_MINUTES
        np.testing.assert_array_equal(
            frame[f"target_lag_7d_{minutes}"].to_numpy(),
            frame[baseline].to_numpy(),
        )


def test_los_lags_cortos_heredados_eran_copias_y_los_nuevos_no(module):
    observations = history_frame()
    context = context_frame(observations, DAYS)
    frame = module.add_features(observations.copy(), context.copy(), LEGACY_GAP_STEPS)
    legacy_short = [f"demand_lag_{lag}" for lag in module.LAGS if lag < LEGACY_GAP_STEPS]

    # Fix del bug: los 11 lags cortos eran la misma columna repetida 11 veces.
    first = frame[legacy_short[0]].to_numpy()
    assert all(np.array_equal(frame[column].to_numpy(), first) for column in legacy_short[1:])
    # La escalera nueva si distingue la trayectoria reciente.
    visible = [f"visible_lag_{offset}" for offset in module.VISIBLE_LAG_OFFSETS]
    assert not frame[visible[0]].equals(frame[visible[1]])
    assert not frame[visible[0]].equals(frame[visible[-1]])



def test_con_hueco_cero_la_ultima_demanda_es_la_del_corte(module):
    """El modelo actual debe ver la demanda del propio data_cutoff, no la de 33 h antes."""

    observations = history_frame()
    context = context_frame(observations, DAYS)
    frame = module.add_features(observations.copy(), context.copy(), GAP_STEPS)
    station_id = STATIONS[0]
    data_cutoff = frame["observed_at"].max()
    expected = observations.loc[
        (observations["station_id"] == station_id)
        & (observations["observed_at"] == data_cutoff),
        "demand",
    ].iloc[0]
    config = {
        "horizon_minutes": 15,
        "history_gap_steps": GAP_STEPS,
        "feature_columns": ["visible_lag_0", "target_lag_1d_15"],
    }
    row = _feature_row_for_target(
        station_id,
        data_cutoff + pd.Timedelta(minutes=15),
        data_cutoff,
        observations,
        context,
        config,
    )
    assert row["visible_lag_0"] == pytest.approx(expected)
    yesterday = observations.loc[
        (observations["station_id"] == station_id)
        & (observations["observed_at"] == data_cutoff + pd.Timedelta(minutes=15) - pd.Timedelta(days=1)),
        "demand",
    ].iloc[0]
    assert row["target_lag_1d_15"] == pytest.approx(yesterday)


def test_la_mezcla_con_persistencia_coincide_en_los_dos_mundos(module):
    """El peso por estacion aplicado al enviar reproduce el que se valido al entrenar."""

    from scripts.infer_and_submit import blend_with_persistence

    observations = history_frame()
    context = context_frame(observations, DAYS)
    frame = module.add_features(observations.copy(), context.copy(), GAP_STEPS)
    data_cutoff = frame["observed_at"].max()
    weights = {STATIONS[0]: 0.35}
    config = {"history_gap_steps": GAP_STEPS, "persistence_weights": weights}
    rows = frame.loc[frame["observed_at"] == data_cutoff].sort_values("station_id")
    base = np.array([120.0, 250.0])

    training = module.apply_persistence_weights(
        base,
        rows[module.PERSISTENCE_COLUMN].to_numpy(),
        rows["station_id"].to_numpy(),
        weights,
    )
    inference = [
        blend_with_persistence(value, station_id, data_cutoff, observations, config)
        for value, station_id in zip(base, rows["station_id"])
    ]

    np.testing.assert_allclose(training, inference)
    # La estacion sin peso queda intacta: la mezcla nunca toca lo que no se calibro.
    assert inference[1] == base[1]


def test_el_peso_de_persistencia_solo_sube_si_mejora(module):
    stations = np.array(["A"] * 4 + ["B"] * 4)
    target = np.array([10.0, 10.0, 10.0, 10.0, 50.0, 60.0, 70.0, 80.0])
    base = np.array([10.0, 10.0, 10.0, 10.0, 40.0, 50.0, 60.0, 70.0])
    persistence = np.array([30.0, 30.0, 30.0, 30.0, 50.0, 60.0, 70.0, 80.0])

    weights = module.choose_persistence_weights(stations, target, base, persistence)

    assert weights["A"] == 0.0  # el modelo ya acierta: la persistencia solo empeora
    assert weights["B"] == max(module.PERSISTENCE_WEIGHT_GRID)


def test_un_campeon_que_vio_la_fold_se_puntua_como_receta(module):
    start = pd.Timestamp("2026-09-10 03:00", tz="UTC")

    assert module.champion_saw_fold({"training_data_end": "2026-09-16T23:00:00+00:00"}, start)
    assert not module.champion_saw_fold({"training_data_end": "2026-09-08T23:00:00+00:00"}, start)
    # Sin fecha de corte no se puede probar que este fuera de muestra: se reentrena.
    assert module.champion_saw_fold({}, start)


def test_la_receta_del_campeon_solo_ve_datos_previos_al_corte(module):
    observations = history_frame()
    context = context_frame(observations, DAYS)
    frame = module.add_features(observations.copy(), context.copy(), GAP_STEPS)
    columns = module.feature_columns_for_horizon(candidate_columns(module, frame), 15)
    config = {"hgb_model": "HGB shallow", "feature_columns": columns}
    cutoff = frame["observed_at"].max() - pd.Timedelta(days=1)

    model = module.refit_champion_recipe(config, frame, 1, cutoff)

    assert model is not None
    # 12 estaciones en produccion, 2 aqui: filas por estacion hasta el corte, sin el futuro.
    expected_rows = int((frame["observed_at"] <= cutoff).sum())
    assert model.n_features_in_ == len(columns)
    assert expected_rows < len(frame)
    assert module.refit_champion_recipe({**config, "hgb_model": "no existe"}, frame, 1, cutoff) is None


def test_la_receta_del_campeon_no_usa_los_pesos_del_config(module, monkeypatch):
    """Reentrenado, el campeon mezcla con pesos walk-forward, no con los de su ultima fold."""

    # `score` exige las 12 estaciones de produccion; aqui hay dos.
    monkeypatch.setattr(module, "EXPECTED_STATION_COUNT", len(STATIONS))
    observations = history_frame()
    context = context_frame(observations, DAYS)
    frame = module.add_features(observations.copy(), context.copy(), GAP_STEPS)
    frame["target"] = frame.groupby("station_id", sort=False)["demand"].shift(-1)
    frame = frame.dropna(subset=["target"])
    columns = module.feature_columns_for_horizon(candidate_columns(module, frame), 15)
    cutoff = frame["observed_at"].max() - pd.Timedelta(days=1)
    validation = frame.loc[frame["observed_at"] > cutoff]
    config = {
        "hgb_model": "HGB shallow",
        "hgb_weight": 1.0,
        "feature_columns": columns,
        "persistence_weights": {station: 0.5 for station in STATIONS},
    }
    model = module.refit_champion_recipe(config, frame, 1, cutoff)
    bundle = {15: (model, config)}

    frozen = module.champion_rows_for_horizon(bundle, "c", 1, 1, validation)
    refit_first_fold = module.champion_rows_for_horizon(
        bundle, "c", 1, 1, validation, model=model, persistence_weights={}
    )
    unblended = module.score(
        "c", 1, 1, validation,
        pd.Series(module.champion_base_predictions(model, config, validation, 1)),
    )

    assert [row["accuracy"] for row in refit_first_fold] == [row["accuracy"] for row in unblended]
    assert [row["accuracy"] for row in frozen] != [row["accuracy"] for row in unblended]
