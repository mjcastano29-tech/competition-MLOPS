"""Las columnas que pide un modelo guardado tienen que ser construibles en inferencia.

Regresion real: un horizonte conservado del campeon pedía una columna que la inferencia
ya no calculaba, se relleno con 0.0 en silencio y el cron siguio entregando predicciones
degradadas. Hoy una columna desconocida aborta el envio o impide empaquetar; estos tests
fijan ese contrato en los dos extremos del pipeline.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.infer_and_submit import (
    SUPPORTED_FEATURE_NAMES,
    _feature_row_for_target,
    unsupported_feature_columns,
)


def load_packaging_module():
    """04 importa mlflow/sklearn: en CI solo se ejercita cuando el extra `ml` existe."""

    pytest.importorskip("mlflow")
    path = Path(__file__).resolve().parents[1] / "examples" / "04_package_best_model.py"
    spec = importlib.util.spec_from_file_location("package_best_model", path)
    if spec is None or spec.loader is None:
        pytest.skip(f"No se pudo cargar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config(columns: list[str] | None = None, **overrides) -> dict:
    config: dict = {
        "horizon_minutes": 15,
        "history_gap_steps": 133,
        "feature_columns": ["demand_lag_4h", "is_weekend"],
    }
    if columns is not None:
        config["feature_columns"] = columns
    config.update(overrides)
    return config


def empty_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = pd.DataFrame(
        {"station_id": [], "observed_at": pd.to_datetime([], utc=True), "demand": []}
    )
    context = pd.DataFrame(
        {
            "observed_at": pd.to_datetime([], utc=True),
            "rain_forecast": [],
            "temperature_forecast": [],
            "event_intensity": [],
        }
    )
    return observations, context


def write_bundle(bundle_dir: Path, columns_by_horizon: dict[int, list[str] | None]) -> Path:
    (bundle_dir / "models").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "configs").mkdir(parents=True, exist_ok=True)
    for horizon_minutes, columns in columns_by_horizon.items():
        config: dict = {"horizon_minutes": horizon_minutes}
        if columns is not None:
            config["feature_columns"] = columns
        (bundle_dir / "models" / f"horizon_{horizon_minutes}_hgb.pkl").write_bytes(b"pickle")
        (bundle_dir / "configs" / f"horizon_{horizon_minutes}_ensemble.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
    return bundle_dir


def test_las_columnas_del_protocolo_actual_son_construibles():
    known = [
        "demand_lag_4h",
        "demand_mean_24h",
        "demand_std_1w",
        "station_id_0100",
        "target_is_weekend_1",
        "target_quarter_sin_4",
        "target_weekday_cos_2",
    ] + sorted(SUPPORTED_FEATURE_NAMES)[:2]

    assert unsupported_feature_columns(known) == []
    assert unsupported_feature_columns([]) == []


def test_una_columna_de_otro_protocolo_se_detecta():
    assert unsupported_feature_columns(["demand_lag_4h", "rolling_mean_4h"]) == [
        "rolling_mean_4h"
    ]
    assert unsupported_feature_columns(["lag_15m", "hour_of_day"]) == ["hour_of_day", "lag_15m"]


def test_la_inferencia_aborta_ante_una_columna_que_no_sabe_construir():
    observations, context = empty_frames()

    with pytest.raises(RuntimeError, match="no sabe construir"):
        _feature_row_for_target(
            "0100",
            pd.Timestamp("2026-09-22 12:00", tz="UTC"),
            pd.Timestamp("2026-09-22 11:00", tz="UTC"),
            observations,
            context,
            build_config(["demand_lag_4h", "rolling_mean_4h"]),
        )


def test_la_inferencia_aborta_si_el_config_no_declara_columnas():
    observations, context = empty_frames()

    with pytest.raises(RuntimeError, match="no declara feature_columns"):
        _feature_row_for_target(
            "0100",
            pd.Timestamp("2026-09-22 12:00", tz="UTC"),
            pd.Timestamp("2026-09-22 11:00", tz="UTC"),
            observations,
            context,
            build_config([]),
        )


def test_la_conservacion_del_campeon_descarta_el_horizonte_no_construible(tmp_path):
    packaging = load_packaging_module()
    previous = write_bundle(
        tmp_path / "previous_model",
        {
            15: ["demand_lag_4h", "is_weekend"],
            30: ["demand_lag_4h", "rolling_mean_4h"],
            45: ["demand_lag_1d", "rain_forecast"],
            60: ["demand_mean_24h", "temperature_forecast"],
        },
    )

    reusable = packaging.champion_package_horizons(previous, "Champion v9")

    assert reusable == [15, 45, 60], "el horizonte con la columna muerta se reentrena"
    assert packaging.champion_package_horizons(previous, None) == []
    assert packaging.champion_package_horizons(tmp_path / "inexistente", "Champion v9") == []


def test_un_config_heredado_sin_columnas_tambien_se_descarta(tmp_path):
    packaging = load_packaging_module()
    previous = write_bundle(
        tmp_path / "previous_model",
        {15: ["demand_lag_4h"], 30: None, 45: ["demand_lag_4h"], 60: ["demand_lag_4h"]},
    )

    assert packaging.champion_package_horizons(previous, "Champion v9") == [15, 45, 60]


def test_el_paquete_mezclado_con_columnas_muertas_no_empaqueta(tmp_path):
    packaging = load_packaging_module()
    staged = write_bundle(
        tmp_path,
        {15: ["demand_lag_4h"], 30: ["demand_lag_4h", "rolling_mean_4h"], 45: None, 60: None},
    )

    broken = packaging.unbuildable_package_features(staged)

    assert broken == {
        "horizon_30_ensemble.json": ["rolling_mean_4h"],
        "horizon_45_ensemble.json": ["<sin feature_columns>"],
        "horizon_60_ensemble.json": ["<sin feature_columns>"],
    }


def test_un_paquete_sano_no_bloquea_el_empaquetado(tmp_path):
    packaging = load_packaging_module()
    staged = write_bundle(
        tmp_path,
        {
            15: ["demand_lag_4h"],
            30: ["demand_mean_24h", "is_weekend"],
            45: ["station_id_0100", "event_intensity"],
            60: ["demand_std_1w"],
        },
    )

    assert packaging.unbuildable_package_features(staged) == {}
