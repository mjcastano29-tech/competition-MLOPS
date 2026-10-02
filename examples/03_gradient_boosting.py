from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from scripts.ar_baseline import (
        apply_ar_weights,
        ar2_station_forecast,
        build_normal_profile,
        choose_ar_weights,
    )
    from scripts.relative_model import (
        RELATIVE_CONFIGS,
        RELATIVE_OFFSET,
        RELATIVE_REFERENCE,
        RELATIVE_SUFFIX,
        RELATIVE_WEIGHT,
        MEMORY_SUFFIX,
        SHORT_HALF_LIFE_DAYS,
        RelativeBlend,
        from_relative,
        parse_candidate,
        relative_model_path,
        relative_target,
        split_candidate_name,
    )
except ImportError:  # ejecutado como `python examples/03_gradient_boosting.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.ar_baseline import (
        apply_ar_weights,
        ar2_station_forecast,
        build_normal_profile,
        choose_ar_weights,
    )
    from scripts.relative_model import (
        RELATIVE_CONFIGS,
        RELATIVE_OFFSET,
        RELATIVE_REFERENCE,
        RELATIVE_SUFFIX,
        RELATIVE_WEIGHT,
        MEMORY_SUFFIX,
        SHORT_HALF_LIFE_DAYS,
        RelativeBlend,
        from_relative,
        parse_candidate,
        relative_model_path,
        relative_target,
        split_candidate_name,
    )


TARGET = "demand"
PERIOD_MINUTES = 15
PERIODS_PER_DAY = 96
# Lags heredados. Con un hueco de 133 pasos, `shift(max(lag, 133))` convertía los 11
# lags cortos (1, 2, 3, 4, 8, 12, 92..96) en 11 copias exactas de la última fila
# visible: el modelo perdía toda la trayectoria reciente. Se siguen calculando porque
# el campeón los necesita para la comparación emparejada (`champion_rows_for_horizon`)
# y porque la baseline estacional se referencia a `demand_lag_{672 - horizon}`, pero
# `model_feature_columns()` ya no los entrega al candidato.
LAGS = (1, 2, 3, 4, 8, 12, 92, 93, 94, 95, 96, 668, 669, 670, 671, 672)
RETIRED_FEATURE_PREFIXES = ("demand_lag_",)
# Antigüedad, en pasos adicionales sobre la última fila visible, de la escalera de
# recencia. `visible_lag_0` es la última demanda publicable; `visible_lag_96` es la
# misma hora de ayer relativo a esa última fila. Todos caen dentro del hueco.
VISIBLE_LAG_OFFSETS = (0, 1, 2, 3, 4, 6, 8, 12, 24, 48, 96)
# Días antes del objetivo cuya demanda del mismo cuarto horario se usa como referencia
# (hace falta 96*d - horizon >= hueco para que sea publicable). d=7 reproduce
# exactamente la baseline estacional de 7 días que usa el ensemble.
TARGET_SEASONAL_DAYS = (1, 2, 3, 4, 5, 6, 7)
CONTEXT_FEATURES = ("rain_forecast", "temperature_forecast", "event_intensity")
# Deben coincidir con MAX_CONTEXT_AGE_MINUTES / CONTEXT_FALLBACK_DAYS de
# scripts/infer_and_submit.py: el contexto se imputa en entrenamiento igual que en
# producción, o el modelo aprende un régimen de clima que al enviar nunca ve.
MAX_CONTEXT_AGE_MINUTES = 60
CONTEXT_FALLBACK_DAYS = 7
FEATURE_PROTOCOL = "v3-fresh-stream"
ROLLING_WINDOWS = (4, 16, 96)
ROLLING_STD_WINDOWS = (4, 16)
HORIZONS = (1, 2, 3, 4)
# `/v1/stream/observations` publica la demanda del propio `data_cutoff` en el mismo
# tick en que abre el ciclo, y la inferencia la lee directo de la API: no hay hueco.
# Los 133 pasos heredados median la distancia al dataset inicial, no al stream, y
# dejaban al modelo ciego ~33 h justo cuando el profesor cambia el patron.
TRAINING_HISTORY_GAP_STEPS = 0
# Hueco con el que se entreno el paquete gap133 original cuando su config no lo declara.
LEGACY_HISTORY_GAP_STEPS = 133
ENSEMBLE_WEIGHTS = (0.85, 0.9, 0.95, 1.0)
EXPECTED_STATION_COUNT = 12
# El stream trae cambios de nivel por estacion en cuestion de dias: tres dias de vida
# media pesan mas el regimen actual sin tirar la estacionalidad semanal.
RECENCY_HALF_LIFE_DAYS = 3.0
MLFLOW_EXPERIMENT = "pulso-transmi-forecasting"
BEST_MODEL_DIR = Path("artifacts/models")
# Validation is a fixed absolute window, not "the last 21 days of whatever is
# on disk", so a candidate and the champion can be scored on identical folds.
HOLDOUT_DAYS = 21
VALIDATION_FOLD_DAYS = 7
SNAPSHOT_POINTER = Path("data/snapshot.json")
VALIDATION_WINDOW_REPORT = Path("reports/validation_window.json")
# Mezcla final con persistencia (la demanda del corte): con drift reacciona antes que el
# arbol. El peso se elige por estacion en la fold anterior (walk-forward) y en estaciones
# estables converge a 0, asi que ahi no cambia nada.
PERSISTENCE_COLUMN = "visible_lag_0"
PERSISTENCE_WEIGHT_GRID = tuple(round(0.05 * step, 2) for step in range(11))
PERSISTENCE_WEIGHTS_REPORT = Path("reports/persistence_weights.json")
# Mezcla final con el AR(2) local de scripts/ar_baseline.py: peso por estacion elegido en
# la fold anterior, igual que la persistencia; en estaciones donde el AR no sirve queda 0.
AR_WEIGHTS_REPORT = Path("reports/ar_weights.json")

MODEL_CONFIGS = {
    "HGB baseline": {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    "HGB more leaves": {
        "learning_rate": 0.04,
        "max_iter": 450,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    "HGB more regularized": {
        "learning_rate": 0.04,
        "max_iter": 450,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 10.0,
    },
    "HGB shallow": {
        "learning_rate": 0.04,
        "max_iter": 450,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    "HGB small leaves": {
        "learning_rate": 0.03,
        "max_iter": 600,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
    },
    "HGB strong regularization": {
        "learning_rate": 0.03,
        "max_iter": 600,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 40,
        "l2_regularization": 10.0,
    },
    "HGB absolute error": {
        "loss": "absolute_error",
        "learning_rate": 0.04,
        "max_iter": 450,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 2.0,
    },
}


def _context_window_median(ordered_context: pd.DataFrame, moments: pd.DatetimeIndex) -> np.ndarray:
    """Mediana por feature del contexto publicado en la ventana previa de varios dias.

    Es el mismo relleno que aplica la inferencia cuando el contexto esta viejo o no
    existe: mediana hacia atras de `CONTEXT_FALLBACK_DAYS` y 0.0 si tampoco hay filas.
    """

    times = pd.to_datetime(ordered_context["observed_at"], utc=True).to_numpy(dtype="datetime64[ns]")
    values = ordered_context.loc[:, list(CONTEXT_FEATURES)].to_numpy(dtype=float)
    window = np.timedelta64(CONTEXT_FALLBACK_DAYS * 24 * 60, "m")
    result = np.zeros((len(moments), len(CONTEXT_FEATURES)), dtype=float)
    moments_ns = pd.DatetimeIndex(moments).to_numpy(dtype="datetime64[ns]")
    for position, moment in enumerate(moments_ns):
        start = int(np.searchsorted(times, moment - window, side="left"))
        end = int(np.searchsorted(times, moment, side="right"))
        block = values[start:end]
        for column in range(block.shape[1]):
            samples = block[:, column]
            samples = samples[~np.isnan(samples)]
            result[position, column] = float(np.median(samples)) if samples.size else 0.0
    return result


def _context_lookup(context: pd.DataFrame, available_times: pd.Series) -> pd.DataFrame:
    """Contexto reconstruido con las reglas exactas de `scripts/infer_and_submit.py`.

    (1) Vale la ultima fila de contexto completa anterior a `available_times` y solo
    cuenta como fresca si tiene <= `MAX_CONTEXT_AGE_MINUTES` minutos; (2) si esta vieja
    o no existe, se imputa la mediana de los ultimos `CONTEXT_FALLBACK_DAYS` dias. Con
    el merge exacto anterior quedaba NaN y el `dropna()` final se llevaba por delante
    cada fila del stream de la competencia (la API no publica contexto despues del
    dataset inicial): el modelo nunca veia la demanda nueva y aprendia un regimen de
    clima que al enviar nunca recibe. `context_is_fresh` deja que el arbol sepa cuando
    esas tres columnas van en serio y cuando son el relleno.
    """

    times = pd.to_datetime(available_times, utc=True)
    unique_times = pd.DatetimeIndex(pd.unique(times)).sort_values()
    ordered = context.copy()
    ordered["observed_at"] = pd.to_datetime(ordered["observed_at"], utc=True)
    ordered = ordered.sort_values("observed_at")
    complete = ordered.dropna(subset=list(CONTEXT_FEATURES))
    table = pd.DataFrame(index=unique_times, columns=list(CONTEXT_FEATURES), dtype=float)
    table["context_is_fresh"] = 0.0
    if not complete.empty:
        # `DatetimeIndex.to_frame(name)` pone las marcas en el INDICE, no en una columna,
        # y `merge_asof` terminaba buscando `feature_available_at` en el indice y fallando.
        left = pd.DataFrame({"feature_available_at": unique_times})
        merged = pd.merge_asof(
            left,
            complete.loc[:, ["observed_at", *CONTEXT_FEATURES]].rename(
                columns={"observed_at": "context_observed_at"}
            ),
            left_on="feature_available_at",
            right_on="context_observed_at",
            direction="backward",
        )
        age_minutes = (
            (merged["feature_available_at"] - merged["context_observed_at"]).dt.total_seconds()
            / 60.0
        ).to_numpy(dtype=float)
        table.loc[:, list(CONTEXT_FEATURES)] = merged[list(CONTEXT_FEATURES)].to_numpy(dtype=float)
        table["context_is_fresh"] = (age_minutes <= MAX_CONTEXT_AGE_MINUTES).astype(float)
    stale = table.index[table["context_is_fresh"] == 0.0]
    if len(stale):
        table.loc[stale, list(CONTEXT_FEATURES)] = _context_window_median(ordered, stale)
    aligned = table.fillna(0.0).reindex(times.to_numpy())
    aligned.index = times.index
    return aligned



def add_features(
    observations: pd.DataFrame,
    context: pd.DataFrame,
    history_gap_steps: int = TRAINING_HISTORY_GAP_STEPS,
) -> pd.DataFrame:
    if history_gap_steps < 0:
        raise ValueError("history_gap_steps no puede ser negativo.")
    gap = int(history_gap_steps)
    frame = observations.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame = frame.sort_values(["station_id", "observed_at"]).copy()
    grouped_demand = frame.groupby("station_id", sort=False)[TARGET]

    for lag in LAGS:
        frame[f"demand_lag_{lag}"] = grouped_demand.shift(max(lag, gap))

    for window in ROLLING_WINDOWS:
        frame[f"demand_mean_{window}"] = grouped_demand.transform(
            lambda values: values.shift(max(1, gap)).rolling(window).mean()
        )

    for window in ROLLING_STD_WINDOWS:
        frame[f"demand_std_{window}"] = grouped_demand.transform(
            lambda values: values.shift(max(1, gap)).rolling(window).std()
        )

    # Escalera de recencia: los lags heredados eran copias del primero; esta escala si
    # recorre la trayectoria publicable (ultima fila .. 24 h antes que esa misma fila).
    for offset in VISIBLE_LAG_OFFSETS:
        frame[f"visible_lag_{offset}"] = grouped_demand.shift(gap + offset)

    # Referencias estacionales alineadas al instante objetivo, solo con dias que a esa
    # altura ya estan publicados. d=7 es, paso a paso, la baseline del ensemble.
    rolling_reference = f"demand_mean_{max(ROLLING_WINDOWS)}"
    for horizon in HORIZONS:
        target_minutes = horizon * PERIOD_MINUTES
        references: list[str] = []
        for days in TARGET_SEASONAL_DAYS:
            shift_steps = PERIODS_PER_DAY * days - horizon
            if shift_steps < gap:
                continue
            column = f"target_lag_{days}d_{target_minutes}"
            frame[column] = grouped_demand.shift(shift_steps)
            references.append(column)
        if not references:
            continue
        frame[f"target_seasonal_mean_{target_minutes}"] = frame[references].mean(axis=1)
        frame[f"target_seasonal_std_{target_minutes}"] = frame[references].std(axis=1)
        # Cuanto se separa el nivel reciente del nivel historico de este mismo cuarto
        # horario: senal de deriva que un arbol no reconstruye restando dos columnas.
        frame[f"level_gap_{target_minutes}"] = (
            frame[rolling_reference] - frame[f"target_seasonal_mean_{target_minutes}"]
        )

    frame["demand_level_shift_4_96"] = frame["demand_mean_4"] - frame[rolling_reference]
    frame["demand_level_shift_16_96"] = frame["demand_mean_16"] - frame[rolling_reference]

    for horizon in HORIZONS:
        target_minutes = horizon * 15
        target_at = frame["observed_at"] + pd.Timedelta(minutes=target_minutes)
        quarter = target_at.dt.hour * 4 + target_at.dt.minute // 15
        weekday = target_at.dt.dayofweek
        frame[f"target_is_weekend_{target_minutes}"] = (weekday >= 5).astype(int)
        frame[f"target_quarter_sin_{target_minutes}"] = np.sin(2 * np.pi * quarter / 96)
        frame[f"target_quarter_cos_{target_minutes}"] = np.cos(2 * np.pi * quarter / 96)
        frame[f"target_weekday_sin_{target_minutes}"] = np.sin(2 * np.pi * weekday / 7)
        frame[f"target_weekday_cos_{target_minutes}"] = np.cos(2 * np.pi * weekday / 7)

    frame["quarter_of_day"] = frame["observed_at"].dt.hour * 4 + frame["observed_at"].dt.minute // 15
    frame["day_of_week"] = frame["observed_at"].dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["quarter_sin"] = np.sin(2 * np.pi * frame["quarter_of_day"] / 96)
    frame["quarter_cos"] = np.cos(2 * np.pi * frame["quarter_of_day"] / 96)
    frame["weekday_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)

    # El contexto se reconstruye con las reglas exactas de la inferencia: fresco si tiene
    # <= 60 minutos, si no mediana de los ultimos 7 dias (ver `_context_lookup`).
    frame["feature_available_at"] = frame["observed_at"] - pd.Timedelta(
        minutes=gap * PERIOD_MINUTES
    )
    context_frame = _context_lookup(context, frame["feature_available_at"])
    for column in CONTEXT_FEATURES:
        frame[column] = context_frame[column].to_numpy(dtype=float)
    frame["context_is_fresh"] = context_frame["context_is_fresh"].to_numpy(dtype=float)

    seasonal_columns = [
        f"target_lag_{days}d_{horizon * PERIOD_MINUTES}"
        for horizon in HORIZONS
        for days in TARGET_SEASONAL_DAYS
        if PERIODS_PER_DAY * days - horizon >= gap
    ]
    seasonal_aggregates = [
        f"{aggregate}_{horizon * PERIOD_MINUTES}"
        for horizon in HORIZONS
        for aggregate in ("target_seasonal_mean", "target_seasonal_std", "level_gap")
        if f"target_seasonal_mean_{horizon * PERIOD_MINUTES}" in frame.columns
    ]
    feature_columns = [
        # Los lags heredados siguen en el frame: `demand_lag_{672 - horizon}` es la
        # baseline estacional del ensemble y el campeon los necesita al emparejar, pero
        # `model_feature_columns()` los deja fuera del candidato.
        *(f"demand_lag_{lag}" for lag in LAGS),
        *(f"visible_lag_{offset}" for offset in VISIBLE_LAG_OFFSETS),
        *(f"demand_mean_{window}" for window in ROLLING_WINDOWS),
        *(f"demand_std_{window}" for window in ROLLING_STD_WINDOWS),
        "demand_level_shift_4_96",
        "demand_level_shift_16_96",
        *seasonal_columns,
        *seasonal_aggregates,
        *(
            f"target_{feature}_{horizon * 15}"
            for horizon in HORIZONS
            for feature in ("is_weekend", "quarter_sin", "quarter_cos", "weekday_sin", "weekday_cos")
        ),
        *CONTEXT_FEATURES,
        "context_is_fresh",
        "is_weekend",
        "quarter_sin",
        "quarter_cos",
        "weekday_sin",
        "weekday_cos",
        "station_id",
    ]
    frame = frame[["observed_at", TARGET, *feature_columns]].dropna().copy()
    station_ids = frame["station_id"].copy()
    encoded = pd.get_dummies(frame, columns=["station_id"], dtype=float)
    encoded.insert(encoded.columns.get_loc(TARGET) + 1, "station_id", station_ids)
    return encoded


def model_feature_columns(columns: Any) -> list[str]:
    """Columnas que ven los modelos candidatos: el frame completo menos los heredados.

    `demand_lag_*` se calcula igual que siempre para no romper la comparacion emparejada
    con el campeon ni la baseline estacional, pero los 11 lags cortos eran copias exactas
    entre si y `demand_lag_668..672` quedan cubiertos por `target_lag_*d_*`, que esta
    alineada al instante objetivo. Un duplicado exacto no aporta señal y si diluye las
    divisiones del arbol, asi que el candidato entrena sin ellos.
    """

    return [
        column
        for column in columns
        if column not in {"observed_at", TARGET, "station_id"}
        and not str(column).startswith(RETIRED_FEATURE_PREFIXES)
    ]


def feature_columns_for_horizon(feature_columns: list[str], horizon_minutes: int) -> list[str]:
    target_calendar_prefixes = (
        "target_is_weekend_",
        "target_quarter_sin_",
        "target_quarter_cos_",
        "target_weekday_sin_",
        "target_weekday_cos_",
        "target_lag_",
        "target_seasonal_mean_",
        "target_seasonal_std_",
        "level_gap_",
    )
    return [
        column for column in feature_columns
        if not column.startswith(target_calendar_prefixes) or column.endswith(f"_{horizon_minutes}")
    ]


def station_balanced_weights(
    frame: pd.DataFrame,
    target_column: str = "target",
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> np.ndarray:
    if half_life_days <= 0:
        raise ValueError("half_life_days debe ser positivo.")
    observed_at = pd.to_datetime(frame["observed_at"], utc=True)
    age_days = (observed_at.max() - observed_at).dt.total_seconds() / 86400
    recency = np.power(0.5, age_days.to_numpy(dtype=float) / half_life_days)
    weighted_target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float) * recency
    station_target_sum = pd.Series(weighted_target, index=frame.index).groupby(
        frame["station_id"], sort=False
    ).transform("sum")
    if (station_target_sum <= 0).any():
        raise ValueError("No se pueden calcular pesos WAPE con suma de demanda no positiva.")
    weights = recency / station_target_sum.to_numpy(dtype=float)
    # Equalize station-level weighted WAPE while preserving the estimator scale.
    weights *= len(weights) / weights.sum()
    return weights


def accuracy_by_station(frame: pd.DataFrame) -> pd.Series:
    absolute_error = (frame[TARGET] - frame["prediction"]).abs()
    return 100 * (
        1
        - absolute_error.groupby(frame["station_id"]).sum()
        / frame[TARGET].groupby(frame["station_id"]).sum()
    ).clip(lower=0)


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def load_dataset_frames(
    snapshot_pointer: Path = SNAPSHOT_POINTER,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Carga el snapshot versionado cuando existe; si no, los CSV sueltos de `data/`.

    El manifiesto del snapshot se devuelve tal cual para que cada fila de metrica
    quede ligada al hash de filas y al `dataset_id` exactos del entrenamiento.
    """

    try:
        from scripts.snapshot_dataset import load_snapshot_manifest
    except ImportError:  # ejecutado como `python examples/03_gradient_boosting.py`
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.snapshot_dataset import load_snapshot_manifest

    manifest = load_snapshot_manifest(snapshot_pointer)
    if manifest:
        observations = pd.read_csv(
            manifest["observations"], dtype={"station_id": "string"}, parse_dates=["observed_at"]
        )
        context = pd.read_csv(manifest["context"], parse_dates=["observed_at"])
        print(
            f"Entrenando sobre snapshot {manifest['dataset_name']} "
            f"(hash {str(manifest['rows_hash'])[:12]}, {manifest['row_count']} filas)."
        )
        return observations, context, manifest
    observations = pd.read_csv("data/observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"])
    context = pd.read_csv("data/context.csv", parse_dates=["observed_at"])
    print(
        "Aviso: no hay data/snapshot.json; se entrena sobre data/*.csv y las metricas "
        "quedan sin dataset_id ni hash de filas."
    )
    return observations, context, {}


def resolve_fold_starts(
    frame: pd.DataFrame,
    validation_anchor: str | datetime | None = None,
    *,
    holdout_days: int = HOLDOUT_DAYS,
    fold_days: int = VALIDATION_FOLD_DAYS,
) -> tuple[pd.Timestamp, list[pd.Timestamp]]:
    """Ventanas de validacion absolutas y comparables entre modelos.

    Sin ancla se conserva el comportamiento historico (ultimos `holdout_days` dias).
    Con ancla —el `validation_start` del campeon— las folds se repiten exactas, que
    es lo unico que hace valida una comparacion de promocion.
    """

    if holdout_days <= 0 or holdout_days % fold_days:
        raise ValueError("holdout_days debe ser positivo y multiplo de fold_days.")
    latest = frame["observed_at"].max() - timedelta(minutes=max(HORIZONS) * 15)
    if validation_anchor is None:
        anchor = latest - timedelta(days=holdout_days)
    else:
        anchor = pd.Timestamp(validation_anchor)
        anchor = anchor.tz_localize("UTC") if anchor.tzinfo is None else anchor.tz_convert("UTC")
    fold_starts = [
        anchor + timedelta(days=fold_days * index) for index in range(holdout_days // fold_days)
    ]
    if fold_starts[0] <= frame["observed_at"].min():
        raise ValueError(
            f"La ventana anclada {fold_starts[0]} no deja historial para entrenar "
            f"(primera fila util {frame['observed_at'].min()})."
        )
    if fold_starts[-1] > latest:
        raise ValueError(
            f"La ventana anclada {fold_starts[-1]} esta mas alla del ultimo dato util {latest}."
        )
    return anchor, fold_starts


def load_champion_bundle(
    bundle_dir: Path,
) -> tuple[dict[int, tuple[Any, dict[str, Any]]], dict[str, Any]]:
    """Lee `models/*.pkl` + `configs/*_ensemble.json` del paquete del campeon."""

    root = Path(bundle_dir)
    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    bundle: dict[int, tuple[Any, dict[str, Any]]] = {}
    for config_dir in (root / "configs", root):
        for config_path in sorted(config_dir.glob("horizon_*_ensemble.json")):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            horizon_minutes = int(config["horizon_minutes"])
            if horizon_minutes in bundle:
                continue
            model_path = root / "models" / f"horizon_{horizon_minutes}_hgb.pkl"
            if not model_path.exists():
                model_path = root / f"horizon_{horizon_minutes}_hgb.pkl"
            if not model_path.exists():
                continue
            with model_path.open("rb") as stream:
                model = pickle.load(stream)
            if config.get("relative_weight"):
                with relative_model_path(model_path).open("rb") as stream:
                    model = RelativeBlend(model, pickle.load(stream), float(config["relative_weight"]))
            bundle[horizon_minutes] = (model, config)
        if bundle:
            break
    if not bundle:
        raise ValueError(
            f"No hay modelos del campeon en {root}; esperaba models/horizon_*_hgb.pkl."
        )
    return bundle, manifest


def choose_persistence_weights(
    station_ids: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, float]:
    """Peso de persistencia por estacion que minimiza su WAPE en una fold ya observada.

    Recorre `PERSISTENCE_WEIGHT_GRID` en orden y solo cambia ante una mejora estricta:
    en empate gana el peso menor, que deja la prediccion del modelo intacta.
    """

    frame = pd.DataFrame(
        {"station_id": station_ids, "target": target, "base": base, "persistence": persistence}
    )
    weights: dict[str, float] = {}
    for station_id, block in frame.groupby("station_id", sort=True):
        denominator = float(block["target"].sum())
        best_weight, best_error = 0.0, float("inf")
        for weight in PERSISTENCE_WEIGHT_GRID:
            blended = np.clip(
                (1 - weight) * block["base"].to_numpy() + weight * block["persistence"].to_numpy(),
                0,
                None,
            )
            error = float(np.abs(block["target"].to_numpy() - blended).sum())
            if error < best_error - 1e-9:
                best_weight, best_error = weight, error
        weights[str(station_id)] = best_weight if denominator > 0 else 0.0
    return weights


def apply_persistence_weights(
    base: np.ndarray,
    persistence: np.ndarray,
    station_ids: np.ndarray,
    weights: dict[str, float] | None,
) -> np.ndarray:
    """`(1 - w) * base + w * persistencia` con el peso de cada estacion (0 si no esta)."""

    if not weights:
        return np.asarray(base, dtype=float)
    station_weights = np.array(
        [float(weights.get(str(station_id), 0.0)) for station_id in station_ids], dtype=float
    )
    return (1 - station_weights) * np.asarray(base, dtype=float) + station_weights * np.asarray(
        persistence, dtype=float
    )


def champion_saw_fold(champion_manifest: dict[str, Any], validation_start: pd.Timestamp) -> bool:
    """True si el campeon se entreno con datos de la fold (o no se sabe hasta cuando).

    Puntuar su modelo congelado ahi seria medirlo dentro de la muestra: un campeon
    entrenado hasta ayer "acierta" las tres folds de memoria y ningun candidato le gana.
    """

    training_end = champion_manifest.get("training_data_end")
    if not training_end:
        return True
    end = pd.Timestamp(training_end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    return end >= validation_start


def fit_relative_model(
    model_config: dict[str, Any],
    train: pd.DataFrame,
    feature_columns: list[str],
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> HistGradientBoostingRegressor:
    """Arbol del candidato relativo: mismo modelo y pesos, target log-razon contra el corte."""

    model = HistGradientBoostingRegressor(**model_config, early_stopping=False, random_state=42)
    model.fit(
        train[feature_columns],
        relative_target(train["target"], train[RELATIVE_REFERENCE]),
        sample_weight=station_balanced_weights(train, half_life_days=half_life_days),
    )
    return model


def refit_champion_recipe(
    config: dict[str, Any],
    champion_frame: pd.DataFrame,
    horizon: int,
    train_cutoff: pd.Timestamp,
) -> Any | None:
    """Reentrena la receta del campeon (modelo, columnas y hueco) con datos previos a la fold.

    Es la misma regla que sigue el candidato, asi la comparacion mide receta contra receta
    con la misma informacion. Devuelve `None` si la receta ya no existe en MODEL_CONFIGS.
    """

    model_config = MODEL_CONFIGS.get(str(config.get("hgb_model")))
    if model_config is None:
        return None
    train = champion_frame.copy()
    train["target"] = train.groupby("station_id", sort=False)[TARGET].shift(-horizon)
    train = train.dropna(subset=["target"])
    train = train.loc[train["observed_at"] <= train_cutoff]
    model = HistGradientBoostingRegressor(**model_config, early_stopping=False, random_state=42)
    model.fit(
        train.loc[:, list(config["feature_columns"])],
        train["target"],
        sample_weight=station_balanced_weights(
            train,
            half_life_days=float(config.get("recency_half_life_days") or RECENCY_HALF_LIFE_DAYS),
        ),
    )
    if config.get("relative_weight"):
        relative = fit_relative_model(
            model_config,
            train,
            list(config["feature_columns"]),
            float(config.get("recency_half_life_days") or RECENCY_HALF_LIFE_DAYS),
        )
        return RelativeBlend(model, relative, float(config["relative_weight"]))
    return model


def champion_base_predictions(
    model: Any, config: dict[str, Any], validation: pd.DataFrame, horizon: int
) -> np.ndarray:
    """Ensemble del campeon (arbol + baseline estacional) antes de mezclar persistencia."""

    feature_columns = list(config["feature_columns"])
    missing = [column for column in feature_columns if column not in validation.columns]
    if missing:
        raise KeyError(
            f"El campeon exige {len(missing)} columnas ausentes ({missing[:5]}...): cambio el "
            "protocolo de features y la comparacion deja de ser valida."
        )
    hgb_weight = float(config["hgb_weight"])
    baseline_lag = int(config.get("baseline_lag", 672 - horizon))
    weekly_baseline = validation[f"demand_lag_{baseline_lag}"].to_numpy()
    # El DataFrame se pasa con el orden exacto del fit: scikit-learn valida nombres
    # y orden de columnas antes de predecir.
    return hgb_weight * model.predict(validation.loc[:, feature_columns]) + (
        1 - hgb_weight
    ) * weekly_baseline


def ar_forecast_table(
    observations: pd.DataFrame, profile: dict[str, Any], cutoffs: pd.DatetimeIndex
) -> dict[tuple[str, pd.Timestamp], np.ndarray]:
    """Pronostico AR(2) de 4 pasos por estacion y corte, para los cortes de validacion."""

    frame = observations.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    table: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
    for station_id, block in frame.groupby("station_id"):
        series = block.set_index("observed_at")["demand"].astype(float).sort_index()
        for cutoff in cutoffs:
            forecast = ar2_station_forecast(series, cutoff, profile, str(station_id))
            if forecast is not None:
                table[(str(station_id), pd.Timestamp(cutoff))] = forecast
    return table


def ar_values_for(
    validation: pd.DataFrame, horizon: int, table: dict[tuple[str, pd.Timestamp], np.ndarray]
) -> np.ndarray:
    """Columna AR para las filas de validacion de un horizonte (NaN si no hay pronostico)."""

    out = np.full(len(validation), np.nan)
    for position, (station_id, cutoff) in enumerate(
        zip(validation["station_id"].astype(str), pd.to_datetime(validation["observed_at"], utc=True))
    ):
        forecast = table.get((station_id, pd.Timestamp(cutoff)))
        if forecast is not None:
            out[position] = forecast[horizon - 1]
    return out


def champion_rows_for_horizon(
    bundle: dict[int, tuple[Any, dict[str, Any]]],
    label: str,
    fold: int,
    horizon: int,
    validation: pd.DataFrame,
    model: Any | None = None,
    persistence_weights: dict[str, float] | None = None,
    ar_values: np.ndarray | None = None,
    ar_weights: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Puntua al campeon sobre la MISMA fold del candidato: comparacion emparejada.

    `model` sustituye al modelo congelado cuando se puntua la receta reentrenada, y
    entonces `persistence_weights` trae los pesos walk-forward de la fold anterior: los
    del config se eligieron sobre la ultima fold y le darian ventaja dentro de la muestra.
    Con el modelo congelado se usan los del config, que es lo que sirve en produccion.
    """

    frozen_model, config = bundle[horizon * 15]
    predictions = champion_base_predictions(
        frozen_model if model is None else model, config, validation, horizon
    )
    predictions = apply_persistence_weights(
        predictions,
        validation[PERSISTENCE_COLUMN].to_numpy(),
        validation["station_id"].to_numpy(),
        config.get("persistence_weights") if model is None else persistence_weights,
    )
    if ar_values is not None:
        predictions = apply_ar_weights(
            predictions,
            ar_values,
            validation["station_id"].to_numpy(),
            config.get("ar_weights") if model is None else ar_weights,
        )
    return score(label, fold, horizon, validation, pd.Series(predictions))



def champion_history_gap(config: dict[str, Any]) -> int:
    """Hueco con el que se entreno (y se sirve) un horizonte del campeon."""

    return int(config.get("history_gap_steps", LEGACY_HISTORY_GAP_STEPS))


def align_champion_validation(
    champion_frame: pd.DataFrame, validation: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Filas del campeon construidas con SU hueco, para los mismos objetivos del candidato.

    El candidato y el campeon pueden leer historia con huecos distintos (0 frente al
    133 heredado). Cada uno se puntua con la informacion que tendria al enviar y sobre
    exactamente las mismas parejas estacion + instante: esa es la comparacion que
    describe produccion.
    """

    keyed = champion_frame.copy()
    keyed["target"] = keyed.groupby("station_id", sort=False)[TARGET].shift(-horizon)
    keyed = keyed.dropna(subset=["target"]).set_index(["station_id", "observed_at"])
    keys = pd.MultiIndex.from_frame(validation[["station_id", "observed_at"]])
    missing = keys.difference(keyed.index)
    if len(missing):
        raise ValueError(
            f"El campeon no tiene {len(missing)} filas de la fold del candidato con su hueco; "
            "la comparacion emparejada no seria sobre los mismos objetivos."
        )
    aligned = keyed.loc[keys].reset_index()
    aligned.index = validation.index
    return aligned


def score(
    name: str,
    fold: int,
    horizon: int,
    frame: pd.DataFrame,
    predictions: pd.Series,
) -> list[dict[str, object]]:
    scored = frame[["station_id", "target"]].copy()
    scored = scored.rename(columns={"target": TARGET})
    scored["prediction"] = predictions.clip(lower=0).to_numpy()
    station_count = scored["station_id"].nunique()
    if station_count != EXPECTED_STATION_COUNT:
        raise ValueError(
            f"Se esperaban {EXPECTED_STATION_COUNT} estaciones, pero la predicción "
            f"contiene {station_count}."
        )
    station_scores = accuracy_by_station(scored)
    rows = []
    for station_id, accuracy in station_scores.items():
        station_frame = scored.loc[scored["station_id"] == station_id]
        wape = (
            (station_frame[TARGET] - station_frame["prediction"]).abs().sum()
            / station_frame[TARGET].sum()
        )
        rows.append(
            {
                "fold": fold,
                "horizon_minutes": horizon * 15,
                "model": name,
                "station_id": station_id,
                "rows": len(station_frame),
                "wape": wape,
                "accuracy": accuracy,
            }
        )
    return rows


def log_evaluation(
    rows: list[dict[str, object]],
    horizon: int,
    fold: int,
    training_rows: int,
    validation_rows: int,
) -> None:
    station_metrics = pd.DataFrame(rows)
    mlflow.log_params(
        {
            "horizon_minutes": horizon * 15,
            "fold": fold,
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "station_count": station_metrics["station_id"].nunique(),
        }
    )
    mlflow.log_metrics(
        {
            "wape_mean_station": station_metrics["wape"].mean(),
            "accuracy_mean_station": station_metrics["accuracy"].mean(),
            "prediction_rows": validation_rows,
        }
    )
    mlflow.set_tag("metric_definition", "WAPE per station, then mean across 12 stations")


def save_best_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    best_model_names: dict[int, str],
    ensemble_metrics: pd.DataFrame,
    parent_run_id: str,
    persistence_weights: dict[str, dict[str, dict[str, float]]] | None = None,
    ar_weights: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for horizon_minutes, ensemble_name in best_model_names.items():
        model_name, is_relative, candidate_half_life = parse_candidate(
            ensemble_name.split(" + Seasonal Naive 7d", 1)[0].replace("Ensemble ", "")
        )
        half_life = candidate_half_life or RECENCY_HALF_LIFE_DAYS
        hgb_weight = float(ensemble_name.rsplit("(", 1)[1].rstrip(")"))
        horizon = horizon_minutes // 15
        horizon_frame = frame.copy()
        horizon_frame["target"] = horizon_frame.groupby("station_id", sort=False)[TARGET].shift(-horizon)
        horizon_frame = horizon_frame.dropna(subset=["target"])
        horizon_feature_columns = feature_columns_for_horizon(feature_columns, horizon_minutes)
        model = HistGradientBoostingRegressor(
            **MODEL_CONFIGS[model_name],
            early_stopping=False,
            random_state=42,
        )
        model.fit(
            horizon_frame[horizon_feature_columns],
            horizon_frame["target"],
            sample_weight=station_balanced_weights(horizon_frame, half_life_days=half_life),
        )
        model_path = BEST_MODEL_DIR / f"horizon_{horizon_minutes}_hgb.pkl"
        config_path = BEST_MODEL_DIR / f"horizon_{horizon_minutes}_ensemble.json"
        with model_path.open("wb") as output:
            pickle.dump(model, output)
        relative_fields: dict[str, Any] = {}
        if is_relative:
            relative = fit_relative_model(
                MODEL_CONFIGS[model_name], horizon_frame, horizon_feature_columns, half_life
            )
            with relative_model_path(model_path).open("wb") as output:
                pickle.dump(relative, output)
            relative_fields = {
                "relative_weight": RELATIVE_WEIGHT,
                "relative_offset": RELATIVE_OFFSET,
                "relative_reference": RELATIVE_REFERENCE,
            }
        config_path.write_text(
            json.dumps(
                {
                    **relative_fields,
                    "recency_half_life_days": half_life,
                    "horizon_minutes": horizon_minutes,
                    "hgb_model": model_name,
                    "hgb_weight": hgb_weight,
                    "baseline": "Seasonal Naive 7d",
                    "baseline_lag": 672 - horizon,
                    "history_gap_steps": TRAINING_HISTORY_GAP_STEPS,
                    "feature_protocol": FEATURE_PROTOCOL,
                    "target_seasonal_days": list(TARGET_SEASONAL_DAYS),
                    "persistence_weights": (persistence_weights or {})
                    .get(str(horizon_minutes), {})
                    .get(ensemble_name, {}),
                    "ar_weights": (ar_weights or {}).get(str(horizon_minutes), {}).get(ensemble_name, {}),
                    "station_count": EXPECTED_STATION_COUNT,
                    "feature_columns": horizon_feature_columns,
                    "training_rows": len(horizon_frame),
                },
                indent=2,
            )
        )
        best_rows = ensemble_metrics.loc[
            (ensemble_metrics["horizon_minutes"] == horizon_minutes)
            & (ensemble_metrics["model"] == ensemble_name)
        ]
        with mlflow.start_run(
            run_name=f"best-ensemble-h{horizon_minutes}",
            nested=True,
        ):
            mlflow.log_params(
                {
                    "horizon_minutes": horizon_minutes,
                    "hgb_model": model_name,
                    "hgb_weight": hgb_weight,
                    "training_rows": len(horizon_frame),
                    "station_count": EXPECTED_STATION_COUNT,
                }
            )
            mlflow.log_metrics(
                {
                    "wape_mean_station": best_rows["wape"].mean(),
                    "accuracy_mean_station": best_rows["accuracy"].mean(),
                }
            )
            mlflow.set_tag("model_role", "best_ensemble_retrained_on_all_data")
            mlflow.set_tag("parent_run_id", parent_run_id)
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(config_path), artifact_path="model")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validacion temporal multiorizonte con comparacion emparejada."
    )
    parser.add_argument(
        "--validation-anchor",
        help="validation_start absoluto (ISO). Sin el, se usan los ultimos HOLDOUT_DAYS dias.",
    )
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    parser.add_argument(
        "--score-champion",
        type=Path,
        help="Directorio del bundle del campeon, para puntuarlo en las mismas folds.",
    )
    parser.add_argument("--champion-version", help="Version del campeon a rotular en las metricas.")
    args = parser.parse_args()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="rolling-multihorizon-validation") as parent_run:
        run_experiment(
            parent_run.info.run_id,
            validation_anchor=args.validation_anchor,
            holdout_days=args.holdout_days,
            champion_dir=args.score_champion,
            champion_version=args.champion_version,
        )


def run_experiment(
    parent_run_id: str,
    *,
    validation_anchor: str | None = None,
    holdout_days: int = HOLDOUT_DAYS,
    champion_dir: Path | None = None,
    champion_version: str | None = None,
) -> None:
    observations, context, snapshot = load_dataset_frames()
    frame = add_features(observations, context)

    feature_columns = model_feature_columns(frame.columns)
    anchor, fold_starts = resolve_fold_starts(
        frame, validation_anchor, holdout_days=holdout_days
    )
    fold_windows = {
        fold: (start, start + timedelta(days=VALIDATION_FOLD_DAYS))
        for fold, start in enumerate(fold_starts, start=1)
    }
    champion_bundle: dict[int, tuple[Any, dict[str, Any]]] | None = None
    champion_label: str | None = None
    champion_manifest: dict[str, Any] = {}
    champion_scoring: set[str] = set()
    if champion_dir is not None:
        champion_bundle, champion_manifest = load_champion_bundle(champion_dir)
        champion_version = (
            champion_version or champion_manifest.get("model_version") or "desconocida"
        )
        champion_label = f"Champion {champion_version}"
    print(
        f"Validacion anclada en {anchor} -> {fold_windows[len(fold_starts)][1]} "
        f"({len(fold_starts)} folds de {VALIDATION_FOLD_DAYS} dias); "
        f"campeon: {champion_label or 'sin emparejar'}."
    )
    metric_rows: list[dict[str, object]] = []
    # Frames de features por hueco: el del candidato y, si difiere, el del campeon.
    frames_by_gap: dict[int, pd.DataFrame] = {TRAINING_HISTORY_GAP_STEPS: frame}
    champion_gaps: set[int] = set()
    # Pesos de persistencia de cada ensemble elegidos en la ultima fold: los que empaqueta
    # 04_package_best_model.py para la inferencia.
    persistence_weights: dict[str, dict[str, dict[str, float]]] = {}
    ar_weights: dict[str, dict[str, dict[str, float]]] = {}
    profile = build_normal_profile(observations)
    window_cutoffs = pd.DatetimeIndex(
        sorted(frame.loc[(frame["observed_at"] >= anchor) & (frame["observed_at"] < fold_windows[len(fold_starts)][1]), "observed_at"].unique())
    )
    ar_table = ar_forecast_table(observations, profile, window_cutoffs)
    print(f"AR(2) local: {len(ar_table)} pronosticos para {len(window_cutoffs)} cortes de validacion.")

    for horizon in HORIZONS:
        horizon_frame = frame.copy()
        horizon_frame["target"] = horizon_frame.groupby("station_id", sort=False)[TARGET].shift(-horizon)
        horizon_frame = horizon_frame.dropna(subset=["target"])
        horizon_feature_columns = feature_columns_for_horizon(feature_columns, horizon * 15)
        # Pesos de persistencia de la fold anterior por ensemble; la primera fold va sin
        # mezcla, porque elegirlos sobre la misma fold que se puntua seria hacer trampa.
        previous_weights: dict[str, dict[str, float]] = {}
        previous_ar_weights: dict[str, dict[str, float]] = {}
        champion_previous_weights: dict[str, float] = {}
        champion_previous_ar_weights: dict[str, float] = {}
        for fold, validation_start in enumerate(fold_starts, start=1):
            validation_end = validation_start + timedelta(days=VALIDATION_FOLD_DAYS)
            # Leave a horizon-sized embargo so training labels cannot overlap validation.
            train_cutoff = validation_start - timedelta(minutes=(horizon + 1) * 15)
            train = horizon_frame.loc[horizon_frame["observed_at"] <= train_cutoff].copy()
            validation = horizon_frame.loc[
                (horizon_frame["observed_at"] >= validation_start)
                & (horizon_frame["observed_at"] < validation_end)
            ].copy()
            if train.empty or validation.empty:
                raise ValueError(
                    f"Fold {fold} del horizonte {horizon * 15} quedo vacia "
                    f"(entrenamiento={len(train)}, validacion={len(validation)}); la ventana "
                    "anclada no corresponde a los datos del snapshot."
                )
            print(
                f"Horizonte {horizon * 15} min, fold {fold}: "
                f"entrenamiento={len(train)}; validación={len(validation)}"
            )
            metric_rows.extend(
                (baseline_rows := score(
                    "Seasonal Naive 24h",
                    fold,
                    horizon,
                    validation,
                    validation[f"demand_lag_{96 - horizon}"],
                ))
            )
            with mlflow.start_run(
                run_name=f"Seasonal-Naive-24h-h{horizon * 15}-fold{fold}",
                nested=True,
            ):
                log_evaluation(baseline_rows, horizon, fold, len(train), len(validation))
            metric_rows.extend(
                (baseline_rows := score(
                    "Seasonal Naive 7d",
                    fold,
                    horizon,
                    validation,
                    validation[f"demand_lag_{672 - horizon}"],
                ))
            )
            with mlflow.start_run(
                run_name=f"Seasonal-Naive-7d-h{horizon * 15}-fold{fold}",
                nested=True,
            ):
                log_evaluation(baseline_rows, horizon, fold, len(train), len(validation))

            if champion_bundle is not None and horizon * 15 in champion_bundle:
                # El campeon se evalua sobre estas mismas filas: el delta deja de
                # ser "dos epocas distintas" y pasa a ser una comparacion emparejada.
                champion_gap = champion_history_gap(champion_bundle[horizon * 15][1])
                champion_gaps.add(champion_gap)
                if champion_gap not in frames_by_gap:
                    frames_by_gap[champion_gap] = add_features(
                        observations, context, champion_gap
                    )
                champion_config = champion_bundle[horizon * 15][1]
                refit_model = None
                if champion_saw_fold(champion_manifest, validation_start):
                    refit_model = refit_champion_recipe(
                        champion_config, frames_by_gap[champion_gap], horizon, train_cutoff
                    )
                    if refit_model is None:
                        print(
                            f"Aviso: la receta {champion_config.get('hgb_model')!r} del campeon ya "
                            "no existe; se puntua su modelo congelado dentro de la muestra."
                        )
                champion_scoring.add("frozen" if refit_model is None else "refit")
                champion_validation = align_champion_validation(
                    frames_by_gap[champion_gap], validation, horizon
                )
                champion_ar = ar_values_for(champion_validation, horizon, ar_table)
                incumbent_rows = champion_rows_for_horizon(
                    champion_bundle,
                    str(champion_label),
                    fold,
                    horizon,
                    champion_validation,
                    model=refit_model,
                    persistence_weights=champion_previous_weights,
                    ar_values=champion_ar,
                    ar_weights=champion_previous_ar_weights,
                )
                if refit_model is not None:
                    # Mismo walk-forward que el candidato: estos pesos valen para la
                    # siguiente fold, nunca para la que acaba de puntuarse.
                    champion_base = champion_base_predictions(
                        refit_model, champion_config, champion_validation, horizon
                    )
                    champion_stations = champion_validation["station_id"].to_numpy()
                    champion_blended = apply_persistence_weights(
                        champion_base,
                        champion_validation[PERSISTENCE_COLUMN].to_numpy(),
                        champion_stations,
                        champion_previous_weights,
                    )
                    champion_previous_ar_weights = choose_ar_weights(
                        champion_stations,
                        champion_validation["target"].to_numpy(),
                        champion_blended,
                        champion_ar,
                    )
                    champion_previous_weights = choose_persistence_weights(
                        champion_stations,
                        champion_validation["target"].to_numpy(),
                        champion_base,
                        champion_validation[PERSISTENCE_COLUMN].to_numpy(),
                    )
                metric_rows.extend(incumbent_rows)
                with mlflow.start_run(
                    run_name=f"champion-{champion_version}-h{horizon * 15}-fold{fold}",
                    nested=True,
                ):
                    log_evaluation(incumbent_rows, horizon, fold, len(train), len(validation))
                    mlflow.set_tag("model_role", "incumbent_scored_on_candidate_window")
                    mlflow.set_tag("champion_version", str(champion_version))

            hgb_predictions: dict[str, pd.Series] = {}
            for model_name, model_config in MODEL_CONFIGS.items():
                with mlflow.start_run(
                    run_name=f"{model_name}-h{horizon * 15}-fold{fold}",
                    nested=True,
                ):
                    model = HistGradientBoostingRegressor(
                        **model_config,
                        early_stopping=False,
                        random_state=42,
                    )
                    model.fit(
                        train[horizon_feature_columns],
                        train["target"],
                        sample_weight=station_balanced_weights(train),
                    )
                    predictions = pd.Series(model.predict(validation[horizon_feature_columns]))
                    hgb_predictions[model_name] = predictions
                    if model_name in RELATIVE_CONFIGS:
                        relative = fit_relative_model(model_config, train, horizon_feature_columns)
                        relative_predictions = from_relative(
                            relative.predict(validation[horizon_feature_columns]),
                            validation[RELATIVE_REFERENCE].to_numpy(),
                        )
                        hgb_predictions[f"{model_name}{RELATIVE_SUFFIX}"] = pd.Series(
                            (1 - RELATIVE_WEIGHT) * predictions.to_numpy()
                            + RELATIVE_WEIGHT * relative_predictions
                        )
                        # Misma receta con memoria corta: candidata para regimenes que cambian.
                        short_level = HistGradientBoostingRegressor(
                            **model_config, early_stopping=False, random_state=42
                        ).fit(
                            train[horizon_feature_columns],
                            train["target"],
                            sample_weight=station_balanced_weights(train, half_life_days=SHORT_HALF_LIFE_DAYS),
                        )
                        short_relative = fit_relative_model(
                            model_config, train, horizon_feature_columns, SHORT_HALF_LIFE_DAYS
                        )
                        hgb_predictions[f"{model_name}{RELATIVE_SUFFIX}{MEMORY_SUFFIX}"] = pd.Series(
                            (1 - RELATIVE_WEIGHT) * short_level.predict(validation[horizon_feature_columns])
                            + RELATIVE_WEIGHT * from_relative(
                                short_relative.predict(validation[horizon_feature_columns]),
                                validation[RELATIVE_REFERENCE].to_numpy(),
                            )
                        )
                    rows = score(model_name, fold, horizon, validation, predictions)
                    metric_rows.extend(rows)
                    log_evaluation(rows, horizon, fold, len(train), len(validation))
                    mlflow.log_param("feature_count", len(horizon_feature_columns))
                    mlflow.set_tag("parent_run_id", parent_run_id)
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                    )

            weekly_baseline = validation[f"demand_lag_{672 - horizon}"].to_numpy()
            persistence = validation[PERSISTENCE_COLUMN].to_numpy()
            station_ids = validation["station_id"].to_numpy()
            ar_values = ar_values_for(validation, horizon, ar_table)
            fold_weights: dict[str, dict[str, float]] = {}
            fold_ar_weights: dict[str, dict[str, float]] = {}
            for model_name, predictions in hgb_predictions.items():
                for hgb_weight in ENSEMBLE_WEIGHTS:
                    ensemble_name = f"Ensemble {model_name} + Seasonal Naive 7d ({hgb_weight:.1f})"
                    base_predictions = (
                        hgb_weight * predictions.to_numpy()
                        + (1 - hgb_weight) * weekly_baseline
                    )
                    fold_weights[ensemble_name] = choose_persistence_weights(
                        station_ids,
                        validation["target"].to_numpy(),
                        base_predictions,
                        persistence,
                    )
                    ensemble_predictions = apply_persistence_weights(
                        base_predictions,
                        persistence,
                        station_ids,
                        previous_weights.get(ensemble_name),
                    )
                    fold_ar_weights[ensemble_name] = choose_ar_weights(
                        station_ids,
                        validation["target"].to_numpy(),
                        ensemble_predictions,
                        ar_values,
                    )
                    ensemble_predictions = apply_ar_weights(
                        ensemble_predictions,
                        ar_values,
                        station_ids,
                        previous_ar_weights.get(ensemble_name),
                    )
                    ensemble_rows = score(
                        ensemble_name,
                        fold,
                        horizon,
                        validation,
                        pd.Series(ensemble_predictions),
                    )
                    metric_rows.extend(ensemble_rows)
                    with mlflow.start_run(
                        run_name=f"ensemble-{model_name}-h{horizon * 15}-fold{fold}-w{hgb_weight:.1f}",
                        nested=True,
                    ):
                        log_evaluation(ensemble_rows, horizon, fold, len(train), len(validation))
                        mlflow.log_param("hgb_model", model_name)
                        mlflow.log_param("hgb_weight", hgb_weight)
                        mlflow.set_tag("ensemble", "HistGradientBoosting + Seasonal Naive 7d")
            previous_weights = fold_weights
            previous_ar_weights = fold_ar_weights
        persistence_weights[str(horizon * 15)] = previous_weights
        ar_weights[str(horizon * 15)] = previous_ar_weights

    metrics = pd.DataFrame(metric_rows)
    # `validation_start/end` es la VENTANA COMPLETA de evaluacion (la union de
    # folds), identica para candidato y campeon: es la clave con la que
    # scripts/model_gate.py verifica que ambas metricas son comparables. El detalle
    # por fold se conserva en `fold_start/fold_end`.
    window_start = anchor
    window_end = fold_windows[len(fold_starts)][1]
    metrics["validation_start"] = window_start
    metrics["validation_end"] = window_end
    metrics["fold_start"] = metrics["fold"].map(
        {fold: window[0] for fold, window in fold_windows.items()}
    )
    metrics["fold_end"] = metrics["fold"].map(
        {fold: window[1] for fold, window in fold_windows.items()}
    )
    metrics["history_gap_steps"] = TRAINING_HISTORY_GAP_STEPS
    if champion_label is not None and champion_gaps:
        # Un campeon que conservo horizontes viejos puede mezclar huecos; cada horizonte ya
        # se puntuo con el suyo, y la compuerta necesita un solo valor: el mas atrasado.
        metrics.loc[metrics["model"] == champion_label, "history_gap_steps"] = max(champion_gaps)
        # "refit" solo si todas sus filas se midieron como receta: la compuerta relaja la
        # ganancia minima nada mas cuando la comparacion es enteramente fuera de muestra.
        metrics.loc[metrics["model"] == champion_label, "incumbent_scoring"] = (
            "refit" if champion_scoring == {"refit"} else "frozen"
        )
    metrics["dataset_name"] = snapshot.get("dataset_name", "")
    metrics["dataset_id"] = snapshot.get("dataset_id") or ""
    metrics["dataset_rows_hash"] = snapshot.get("rows_hash") or ""
    metrics["git_commit"] = git_commit() or ""
    window_report = {
        "validation_start": anchor.isoformat(),
        "validation_end": fold_windows[len(fold_starts)][1].isoformat(),
        "folds": [
            {"fold": fold, "start": window[0].isoformat(), "end": window[1].isoformat()}
            for fold, window in sorted(fold_windows.items())
        ],
        "holdout_days": holdout_days,
        "fold_days": VALIDATION_FOLD_DAYS,
        "history_gap_steps": TRAINING_HISTORY_GAP_STEPS,
        "feature_protocol": FEATURE_PROTOCOL,
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "dataset_name": snapshot.get("dataset_name"),
        "dataset_id": snapshot.get("dataset_id"),
        "dataset_rows_hash": snapshot.get("rows_hash"),
        "dataset_cutoff_at": snapshot.get("cutoff_at"),
        "git_commit": git_commit(),
        "champion_version": champion_version,
        "champion_scored": champion_bundle is not None,
        "mlflow_run_id": parent_run_id,
    }
    report_path = Path("reports/ml_validation_metrics.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_path, index=False)
    VALIDATION_WINDOW_REPORT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_WINDOW_REPORT.write_text(
        json.dumps(window_report, indent=2) + "\n", encoding="utf-8"
    )
    mlflow.log_artifact(str(VALIDATION_WINDOW_REPORT), artifact_path="validation")
    PERSISTENCE_WEIGHTS_REPORT.write_text(
        json.dumps(persistence_weights, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mlflow.log_artifact(str(PERSISTENCE_WEIGHTS_REPORT), artifact_path="validation")
    AR_WEIGHTS_REPORT.write_text(json.dumps(ar_weights, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(AR_WEIGHTS_REPORT), artifact_path="validation")
    mlflow.log_artifact(str(report_path), artifact_path="validation")
    ensemble_metrics = metrics[metrics["model"].str.startswith("Ensemble ")]
    ensemble_summary = (
        ensemble_metrics.groupby(["horizon_minutes", "model"], as_index=False)[["accuracy", "wape"]]
        .mean()
        .sort_values(["horizon_minutes", "accuracy"], ascending=[True, False])
    )
    best_ensemble = ensemble_summary.groupby("horizon_minutes", as_index=False).first()
    best_model_names = dict(zip(best_ensemble["horizon_minutes"], best_ensemble["model"]))
    best_station_metrics = ensemble_metrics.merge(
        best_ensemble[["horizon_minutes", "model"]],
        on=["horizon_minutes", "model"],
        how="inner",
    )
    best_station_metrics = (
        best_station_metrics.groupby(["horizon_minutes", "model", "station_id"], as_index=False)[
            ["wape", "accuracy"]
        ]
        .mean()
        .sort_values(["horizon_minutes", "station_id"])
    )
    station_report_path = Path("reports/best_ensemble_wape_by_station.csv")
    best_station_metrics.to_csv(station_report_path, index=False)
    mlflow.log_artifact(str(station_report_path), artifact_path="validation")
    save_best_models(
        frame,
        feature_columns,
        best_model_names,
        ensemble_metrics,
        parent_run_id,
        persistence_weights,
        ar_weights,
    )
    summary = (
        metrics.groupby(["horizon_minutes", "model"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["horizon_minutes", "accuracy"], ascending=[True, False])
    )
    print("\nAccuracy promedio por horizonte y modelo:")
    print(summary.to_string(index=False, formatters={"accuracy": "{:.2f}".format}))
    print("\nMejor ensemble por horizonte:")
    print(best_ensemble.to_string(index=False, formatters={"accuracy": "{:.2f}".format}))
    print("\nWAPE promedio por estación del mejor ensemble:")
    print(
        best_station_metrics.pivot(index="station_id", columns="horizon_minutes", values="wape")
        .mul(100)
        .round(2)
        .to_string()
    )
    print(f"\nMétricas detalladas guardadas en {report_path}")
    print(f"WAPE por estación guardado en {station_report_path}")
    print(f"Mejores modelos guardados en {BEST_MODEL_DIR}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow parent run: {parent_run_id}")


if __name__ == "__main__":
    main()