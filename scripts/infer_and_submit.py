from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
import pandas as pd

try:
    from scripts.ar_baseline import ar2_station_forecast, load_profile
    from scripts.leader_model import leader_forecasts
    from scripts.relative_model import RelativeBlend, relative_model_path
except ImportError:  # ejecutado como `python scripts/infer_and_submit.py`
    from ar_baseline import ar2_station_forecast, load_profile
    from leader_model import leader_forecasts
    from relative_model import RelativeBlend, relative_model_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
BUNDLE_DIR = ROOT / "artifacts" / "pulso_transmi_best_models"
BUNDLE_ZIP = ROOT / "artifacts" / "pulso_transmi_best_models.zip"
# Legacy Actions cache entries use the gap133 cache namespace but predate
# history_gap_steps in each per-horizon JSON config.
DEFAULT_HISTORY_GAP_STEPS = 133
MAX_CONTEXT_AGE_MINUTES = 60
CONTEXT_FALLBACK_DAYS = 7
PERIOD_MINUTES = 15
PERIODS_PER_DAY = 96
# Ventana de media movil que usa `level_gap_*` en entrenamiento (demand_mean_96).
LEVEL_REFERENCE_WINDOW = 96
# Por defecto el protocolo actual de 03_gradient_boosting.py; cada config puede declarar
# los suyos en `target_seasonal_days`.
TARGET_SEASONAL_DAYS = (1, 2, 3, 4, 5, 6, 7)
STREAM_PAGE_SIZE = 5000
# Correccion de sesgo en linea: el factor de cada estacion sale de lo que el campeon habria
# predicho en los ciclos de las ultimas BIAS_WINDOW_HOURS horas que ya tienen demanda real.
# Backtest walk-forward (stream real + choques sinteticos de -50 %..+100 %): +0.8 pts en el
# escenario real, +1.2 con choques y 63.5 -> 76.5 en las 6 h posteriores a un choque; con la
# zona muerta, la peor estacion estable cede 0.15 pts.
BIAS_WINDOW_HOURS = 3
BIAS_ALPHA = 0.5
BIAS_DEADZONE = 0.05
BIAS_FACTOR_BOUNDS = (0.5, 2.0)
BIAS_MIN_SAMPLES = 8
# Guardia de direccion: tras un pico transitorio la ventana de 3 h sigue diciendo "sube"
# mientras la demanda ya cae, y empujaba justo en la bajada (backtest con picos x6: 53.6 ->
# 37.8 en la bajada). Solo se corrige si la ultima hora (h = 15 min en los 4 cuartos
# previos) va en el mismo sentido, y como mucho lo que esa ultima hora respalda. Con la
# guardia la bajada queda en 51.3 y la ganancia en cambios de nivel sostenidos se mantiene.
SURPRISE_QUARTERS = 4
SURPRISE_MIN_SAMPLES = 3
# Respaldo cuando el campeon no puede o no debe responder un target (datos incompatibles).
CONTEXT_COLUMNS = ("rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity")
PLAUSIBLE_MAX_RATIO = 10.0
FALLBACK_CONSTANT = 1.0
FALLBACK_MODEL_VERSION = "respaldo-persistencia"
COMPATIBILITY_REPORT = ROOT / "artifacts" / "compatibility_report.json"
SAMPLE_DATA_PATHS = {
    "observations": ROOT / "data" / "observations.csv",
    "context": ROOT / "data" / "context.csv",
}

# Vocabulario de variables que esta inferencia sabe construir. Un paquete puede mezclar
# horizontes de origenes distintos (el candidato conserva los horizontes que gana el
# campeon) y cada horizonte trae su propia lista en su config: si una columna no esta
# aqui, rellenarla con 0.0 daria predicciones malas sin ningun aviso.
SUPPORTED_FEATURE_PREFIXES = (
    "station_id_",
    "demand_lag_",
    "demand_mean_",
    "demand_std_",
    "demand_level_shift_",
    "visible_lag_",
    "target_lag_",
    "target_seasonal_mean_",
    "target_seasonal_std_",
    "level_gap_",
    "target_is_weekend_",
    "target_quarter_sin_",
    "target_quarter_cos_",
    "target_weekday_sin_",
    "target_weekday_cos_",
)
SUPPORTED_FEATURE_NAMES = {
    "rain_forecast",
    "temperature_forecast",
    "event_intensity",
    "context_is_fresh",
    "is_weekend",
    "quarter_sin",
    "quarter_cos",
    "weekday_sin",
    "weekday_cos",
}


def unsupported_feature_columns(columns: Iterable[str]) -> list[str]:
    """Columnas que la inferencia no puede calcular para ningun horizonte."""

    return sorted(
        {
            str(column)
            for column in columns
            if not str(column).startswith(SUPPORTED_FEATURE_PREFIXES)
            and str(column) not in SUPPORTED_FEATURE_NAMES
        }
    )


def request_headers(api_key: str, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def load_cycle(client: httpx.Client, api_key: str, attempts: int = 5) -> dict[str, Any] | None:
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    for attempt in range(attempts):
        try:
            response = client.get("/v1/forecast-cycles/current", headers=request_headers(api_key))
        except httpx.TransportError as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"No se pudo consultar el ciclo tras {attempts} intentos: {exc}") from exc
            delay = min(2 ** attempt, 30)
            print(f"Fallo de red consultando el ciclo; reintento {attempt + 2}/{attempts} en {delay}s.")
            time.sleep(delay)
            continue
        if response.status_code == 404:
            return None
        if response.status_code not in retryable_statuses:
            if response.is_error:
                raise RuntimeError(f"No se pudo consultar el ciclo actual: HTTP {response.status_code} {response.text}")
            return response.json()
        if attempt + 1 == attempts:
            raise RuntimeError(f"No se pudo consultar el ciclo actual tras {attempts} intentos: HTTP {response.status_code} {response.text}")
        delay = min(2 ** attempt, 30)
        print(f"API respondió HTTP {response.status_code} al consultar el ciclo; reintento {attempt + 2}/{attempts} en {delay}s.")
        time.sleep(delay)
    raise RuntimeError("Se agotaron los intentos de consulta del ciclo.")


def ensure_bundle_ready() -> dict[int, tuple[Path, dict[str, Any]]]:
    if BUNDLE_DIR.exists():
        bundle = BUNDLE_DIR
    else:
        if not BUNDLE_ZIP.exists():
            raise FileNotFoundError(
                "No existe el paquete del modelo. Ejecuta: PYTHONPATH=src python examples/04_package_best_model.py"
            )
        bundle = ROOT / "artifacts" / "pulso_transmi_best_models_extracted"
        if bundle.exists():
            for child in sorted(bundle.iterdir(), reverse=True):
                if child.is_dir():
                    continue
        bundle.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BUNDLE_ZIP, "r") as archive:
            archive.extractall(bundle)

    model_paths: dict[int, tuple[Path, dict[str, Any]]] = {}
    # Solo los arboles de nivel: los relativos (horizon_*_rel.pkl) se cargan junto a su
    # horizonte segun el config.
    for model_path in sorted((bundle / "models").glob("horizon_*_hgb.pkl")):
        is_horizon = "horizon_" in model_path.name
        if not is_horizon:
            continue
        horizon = int(model_path.name.split("horizon_")[1].split("_hgb")[0])
        config_path = bundle / "configs" / f"horizon_{horizon}_ensemble.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Falta configuracion para {horizon} minutos: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_paths[horizon] = (model_path, config)
    if not model_paths:
        raise FileNotFoundError("El paquete no trae modelos serializados para predicción.")
    return model_paths


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _last_known_row(history: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any] | None:
    if history.empty:
        return None
    eligible = history[history["observed_at"] <= cutoff]
    if eligible.empty:
        return None
    return eligible.iloc[-1].to_dict()


def _feature_row_for_target(
    station_id: str,
    target_at: pd.Timestamp,
    data_cutoff: pd.Timestamp,
    observations: pd.DataFrame,
    context: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float]:
    # Training features are indexed by the observation time (forecast origin).
    # add_features() deliberately reads demand/context only as far as
    # observed_at - history_gap_steps * 15 minutes; mirror that at inference.
    if "history_gap_steps" not in config:
        history_gap_steps = DEFAULT_HISTORY_GAP_STEPS
        print(
            "Bundle heredado sin history_gap_steps; uso 133 pasos, "
            "el desfase del paquete gap133 de Actions."
        )
    else:
        history_gap_steps = int(config["history_gap_steps"])
    if history_gap_steps < 0:
        raise ValueError("history_gap_steps no puede ser negativo.")
    feature_data_cutoff = data_cutoff - pd.Timedelta(minutes=history_gap_steps * 15)
    history = observations[
        (observations["station_id"] == station_id)
        & (observations["observed_at"] <= feature_data_cutoff)
    ].sort_values("observed_at").copy()
    feature_columns = list(config.get("feature_columns") or [])
    if not feature_columns:
        raise RuntimeError(
            f"El config del horizonte {config.get('horizon_minutes', '?')} no declara "
            "feature_columns; no se puede reconstruir su fila de variables."
        )
    unknown = unsupported_feature_columns(feature_columns)
    if unknown:
        raise RuntimeError(
            f"El modelo de {config.get('horizon_minutes', '?')} min pide columnas que esta "
            f"inferencia no sabe construir: {unknown}. Un paquete puede mezclar horizontes "
            "de origenes distintos; ante una columna desconocida se prefiere no enviar a "
            "mandar predicciones con ceros silenciosos."
        )
    context_features = sorted(
        {"rain_forecast", "temperature_forecast", "event_intensity"}.intersection(feature_columns)
    )
    if "context_is_fresh" in feature_columns and not context_features:
        # La marca de frescura se evalua sobre las tres variables canonicas: si el config
        # solo pide la marca, hay que leer el contexto igualmente.
        context_features = sorted({"rain_forecast", "temperature_forecast", "event_intensity"})
    context_values: dict[str, float] = {}
    context_is_fresh = False
    if context_features:
        historic_context = context[
            context["observed_at"] <= feature_data_cutoff
        ].sort_values("observed_at")
        complete_context = historic_context.dropna(subset=context_features)
        context_is_fresh = False
        if not complete_context.empty:
            latest_context = complete_context.iloc[-1]
            age_minutes = (feature_data_cutoff - latest_context["observed_at"]).total_seconds() / 60
            context_is_fresh = age_minutes <= MAX_CONTEXT_AGE_MINUTES
            if context_is_fresh:
                context_values = {
                    feature: safe_float(latest_context.get(feature), 0.0)
                    for feature in context_features
                }
        if not context_is_fresh:
            # Old weather/event context must not be carried forward indefinitely.
            # Replace it with recent historical medians, and use neutral defaults
            # only when a feature has no usable history. This path never aborts
            # inference or submission.
            fallback_start = feature_data_cutoff - pd.Timedelta(days=CONTEXT_FALLBACK_DAYS)
            fallback_context = historic_context[
                historic_context["observed_at"] >= fallback_start
            ]
            for feature in context_features:
                values = pd.to_numeric(fallback_context[feature], errors="coerce").dropna()
                context_values[feature] = safe_float(values.median(), 0.0) if not values.empty else 0.0
            age_text = f"{age_minutes:.0f} min" if not complete_context.empty else "sin registro completo"
            print(
                f"Contexto {station_id} obsoleto ({age_text}); uso medianas de "
                f"hasta {CONTEXT_FALLBACK_DAYS} días. La inferencia continúa."
            )
    if history.empty:
        raise RuntimeError(
            f"No hay historial de demanda para {station_id} hasta {feature_data_cutoff}."
        )
    row: dict[str, float] = {column: 0.0 for column in feature_columns}
    horizon_steps = int(round((target_at - data_cutoff).total_seconds() / (PERIOD_MINUTES * 60)))
    # Mismo filtro que add_features(): solo dias cuyo cuarto horario ya es publicable.
    seasonal_days = tuple(
        int(days)
        for days in config.get("target_seasonal_days") or TARGET_SEASONAL_DAYS
        if PERIODS_PER_DAY * int(days) - horizon_steps >= history_gap_steps
    )
    seasonal_cache: dict[int, list[float]] = {}

    def demand_at_steps(steps: int, feature: str) -> float:
        """Ultima demanda publicada `steps` pasos antes del corte, sin imputar ceros."""

        lag_time = data_cutoff - pd.Timedelta(minutes=steps * PERIOD_MINUTES)
        last_row = _last_known_row(history, lag_time)
        if last_row is None or pd.isna(last_row.get("demand")):
            raise RuntimeError(
                f"Falta {feature} para {station_id} hasta {lag_time}; "
                "no se enviará una predicción con variables imputadas."
            )
        return safe_float(last_row["demand"])

    def rolling_stat(window: int, feature: str, kind: str) -> float:
        # add_features() usa shift(max(1, gap)).rolling(window): la ventana termina en la
        # ultima observacion publicable, no en el corte.
        rolling_cutoff = data_cutoff - pd.Timedelta(
            minutes=max(1, history_gap_steps) * PERIOD_MINUTES
        )
        recent = history[history["observed_at"] <= rolling_cutoff].tail(window)["demand"]
        if len(recent) < window:
            raise RuntimeError(
                f"Historial insuficiente para {feature} de {station_id}: "
                f"{len(recent)}/{window} observaciones."
            )
        statistic = recent.mean() if kind == "mean" else recent.std()
        if not math.isfinite(float(statistic)):
            raise RuntimeError(f"Variable no finita al calcular {feature} para {station_id}.")
        return float(statistic)

    def seasonal_references(minutes: int, feature: str) -> list[float]:
        """Demanda del mismo cuarto horario los dias de `TARGET_SEASONAL_DAYS` antes.

        Es el espejo de `shift(96*d - horizon)` en `add_features()`: el instante pedido es
        `target_at - d dias`, que equivale a retroceder `96*d - horizon` pasos desde el
        corte. d=7 es exactamente la baseline estacional del ensemble.
        """

        steps = minutes // PERIOD_MINUTES
        if steps != horizon_steps:
            raise RuntimeError(
                f"{feature} apunta al horizonte {minutes} min pero el objetivo esta a "
                f"{horizon_steps * PERIOD_MINUTES} min del corte; el config del modelo y "
                "el objetivo del ciclo no coinciden."
            )
        cached = seasonal_cache.get(minutes)
        if cached is None:
            cached = [
                demand_at_steps(
                    PERIODS_PER_DAY * days - steps, f"target_lag_{days}d_{minutes}"
                )
                for days in seasonal_days
            ]
            seasonal_cache[minutes] = cached
        return cached

    def seasonal_stat(minutes: int, feature: str, kind: str) -> float:
        values = seasonal_references(minutes, feature)
        if kind == "mean":
            return float(sum(values) / len(values))
        # `add_features()` agrega con pandas, cuya desviacion tipica usa ddof=1.
        return float(pd.Series(values).std())


    for feature in feature_columns:
        if feature.startswith("station_id_"):
            row[feature] = 1.0 if station_id == feature.removeprefix("station_id_") else 0.0
            continue
        if feature.startswith("demand_lag_"):
            lag = int(feature.split("_")[-1])
            lag_time = data_cutoff - pd.Timedelta(minutes=max(lag, history_gap_steps) * 15)
            last_row = _last_known_row(history, lag_time)
            if last_row is None or pd.isna(last_row.get("demand")):
                raise RuntimeError(
                    f"Falta {feature} para {station_id} hasta {lag_time}; "
                    "no se enviará una predicción con variables imputadas."
                )
            row[feature] = safe_float(last_row["demand"])
            continue
        if feature.startswith(("demand_mean_", "demand_std_")):
            window = int(feature.split("_")[-1])
            # add_features() uses shift(max(1, gap)).rolling(window), so the
            # window ends at the last observation available at the gap cutoff.
            rolling_cutoff = data_cutoff - pd.Timedelta(minutes=max(1, history_gap_steps) * 15)
            recent = history[history["observed_at"] <= rolling_cutoff].tail(window)["demand"]
            if len(recent) < window:
                raise RuntimeError(
                    f"Historial insuficiente para {feature} de {station_id}: "
                    f"{len(recent)}/{window} observaciones."
                )
            statistic = recent.mean() if feature.startswith("demand_mean_") else recent.std()
            if not math.isfinite(float(statistic)):
                raise RuntimeError(f"Variable no finita al calcular {feature} para {station_id}.")
            row[feature] = float(statistic)
            continue
        if feature.startswith("visible_lag_"):
            # visible_lag_0 es la ultima fila publicable: data_cutoff - gap pasos.
            offset = int(feature.rsplit("_", 1)[-1])
            row[feature] = demand_at_steps(history_gap_steps + offset, feature)
            continue
        if feature.startswith("target_lag_"):
            try:
                _, _, day_token, minute_token = feature.split("_")
                days, minutes = int(day_token[:-1]), int(minute_token)
            except ValueError as exc:
                raise RuntimeError(
                    f"Nombre de variable estacional no reconocido: {feature!r}"
                ) from exc
            if not day_token.endswith("d") or days not in seasonal_days:
                raise RuntimeError(
                    f"{feature!r} pide un dia fuera de {seasonal_days}; el protocolo "
                    "de features cambio y este paquete no se construyo con el."
                )
            row[feature] = seasonal_references(minutes, feature)[seasonal_days.index(days)]
            continue
        if feature.startswith(("target_seasonal_mean_", "target_seasonal_std_")):
            minutes = int(feature.rsplit("_", 1)[-1])
            kind = "mean" if feature.startswith("target_seasonal_mean_") else "std"
            row[feature] = seasonal_stat(minutes, feature, kind)
            continue
        if feature.startswith("level_gap_"):
            # media de las ultimas 96 filas publicables - media historica del mismo cuarto.
            minutes = int(feature.rsplit("_", 1)[-1])
            row[feature] = rolling_stat(
                LEVEL_REFERENCE_WINDOW, feature, "mean"
            ) - seasonal_stat(minutes, feature, "mean")
            continue
        if feature.startswith("demand_level_shift_"):
            windows = feature.removeprefix("demand_level_shift_").split("_")
            if len(windows) != 2 or not all(token.isdigit() for token in windows):
                raise RuntimeError(f"Nombre de salto de nivel no reconocido: {feature!r}")
            recent_window, reference_window = (int(token) for token in windows)
            row[feature] = rolling_stat(recent_window, feature, "mean") - rolling_stat(
                reference_window, feature, "mean"
            )
            continue
        if feature.startswith((
            "target_is_weekend_",
            "target_quarter_sin_",
            "target_quarter_cos_",
            "target_weekday_sin_",
            "target_weekday_cos_",
        )):
            if feature.startswith("target_is_weekend_"):
                row[feature] = 1.0 if target_at.dayofweek >= 5 else 0.0
            elif "quarter_sin" in feature:
                quarter = target_at.hour * 4 + target_at.minute // 15
                row[feature] = float(np.sin(2 * np.pi * quarter / 96))
            elif "quarter_cos" in feature:
                quarter = target_at.hour * 4 + target_at.minute // 15
                row[feature] = float(np.cos(2 * np.pi * quarter / 96))
            elif "weekday_sin" in feature:
                row[feature] = float(np.sin(2 * np.pi * target_at.dayofweek / 7))
            else:
                row[feature] = float(np.cos(2 * np.pi * target_at.dayofweek / 7))
            continue
        if feature in {"rain_forecast", "temperature_forecast", "event_intensity"}:
            row[feature] = context_values.get(feature, 0.0)
            continue
        if feature == "context_is_fresh":
            row[feature] = 1.0 if context_is_fresh else 0.0
            continue
        if feature in {"is_weekend", "quarter_sin", "quarter_cos", "weekday_sin", "weekday_cos"}:
            day_of_week = data_cutoff.dayofweek
            quarter_of_day = data_cutoff.hour * 4 + data_cutoff.minute // 15
            if feature == "is_weekend":
                row[feature] = 1.0 if day_of_week >= 5 else 0.0
            elif feature == "quarter_sin":
                row[feature] = float(np.sin(2 * np.pi * quarter_of_day / 96))
            elif feature == "quarter_cos":
                row[feature] = float(np.cos(2 * np.pi * quarter_of_day / 96))
            elif feature == "weekday_sin":
                row[feature] = float(np.sin(2 * np.pi * day_of_week / 7))
            elif feature == "weekday_cos":
                row[feature] = float(np.cos(2 * np.pi * day_of_week / 7))
            continue
        raise RuntimeError(
            f"Feature {feature!r} no esta en el vocabulario construible de la inferencia; "
            "rellenarla con 0.0 cambiaria la prediccion sin avisar."
        )

    return row


def refresh_observations_from_stream(client: httpx.Client, data_cutoff: pd.Timestamp) -> None:
    """Envoltorio que nunca falla: un stream con otro esquema no puede costar el ciclo."""

    try:
        _refresh_observations_from_stream(client, data_cutoff)
    except Exception as exc:
        print(f"::warning::No se pudo completar con el stream de la API ({type(exc).__name__}: {exc}); se sigue con los datos locales.")


def _refresh_observations_from_stream(client: httpx.Client, data_cutoff: pd.Timestamp) -> None:
    """Completa `data/observations.csv` con el stream de la API hasta `data_cutoff`.

    Supabase va hasta 30 min atrasado (el colector corre antes del tick que abre el
    ciclo) y el modelo usa la demanda del propio `data_cutoff`, que la API publica en ese
    mismo tick. Si el stream falla se sigue con lo de Supabase: los lags toman la ultima
    fila conocida y la entrega no se pierde.
    """

    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    try:
        while True:
            params: dict[str, Any] = {"limit": STREAM_PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            response = client.get("/v1/stream/observations", params=params)
            response.raise_for_status()
            page = response.json()
            rows.extend(page.get("data", []))
            next_cursor = page.get("next_cursor")
            if not next_cursor or next_cursor == cursor or not page.get("data"):
                break
            cursor = next_cursor
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Aviso: no se pudo leer el stream de la API ({exc}); se usa solo Supabase.")
        return
    if not rows:
        return
    stream = pd.DataFrame(rows)[["station_id", "observed_at", "demand"]]
    stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)
    stream = stream[stream["observed_at"] <= data_cutoff]
    path = SAMPLE_DATA_PATHS["observations"]
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = (
        pd.read_csv(path, dtype={"station_id": "string"})
        if path.exists()
        else pd.DataFrame(columns=["station_id", "observed_at", "demand"])
    )
    stored["observed_at"] = pd.to_datetime(stored["observed_at"], utc=True)
    before = stored["observed_at"].max()
    merged = pd.concat([stored, stream], ignore_index=True)
    merged["station_id"] = merged["station_id"].map(normalize_station_id)
    merged = merged.drop_duplicates(subset=["station_id", "observed_at"], keep="last")
    merged = merged.sort_values(["station_id", "observed_at"])
    merged.to_csv(path, index=False)
    print(
        f"Stream de la API: ultima observacion {before} -> {merged['observed_at'].max()} "
        f"(data_cutoff {data_cutoff})."
    )


def blend_with_persistence(
    value: float,
    station_id: str,
    data_cutoff: pd.Timestamp,
    observations: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    """Mezcla el ensemble con la demanda del corte usando el peso de la estacion.

    Espejo de `apply_persistence_weights()` en 03_gradient_boosting.py: la persistencia
    es `visible_lag_0`, la ultima demanda conocida en `data_cutoff - hueco`. Un config sin
    `persistence_weights` (campeones anteriores) devuelve el valor intacto.
    """

    weight = float((config.get("persistence_weights") or {}).get(station_id, 0.0))
    if weight <= 0.0:
        return value
    history_gap_steps = int(config.get("history_gap_steps", DEFAULT_HISTORY_GAP_STEPS))
    lag_time = data_cutoff - pd.Timedelta(minutes=history_gap_steps * PERIOD_MINUTES)
    history = observations[observations["station_id"] == station_id].sort_values("observed_at")
    last_row = _last_known_row(history, lag_time)
    if last_row is None or pd.isna(last_row.get("demand")):
        raise RuntimeError(
            f"Falta la demanda de persistencia para {station_id} hasta {lag_time}; "
            "no se enviará una predicción con variables imputadas."
        )
    return (1 - weight) * value + weight * safe_float(last_row["demand"])


def normalize_station_id(value: Any) -> str:
    station_id = str(value).strip()
    return station_id.zfill(5) if station_id.isdigit() else station_id


def normalize_target_key(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("Z", "+00:00")
    return str(value)


def predict_value(
    station_id: str,
    target_at: pd.Timestamp,
    data_cutoff: pd.Timestamp,
    observations: pd.DataFrame,
    context: pd.DataFrame,
    model: Any,
    config: dict[str, Any],
) -> float:
    """Ensemble del campeon (arbol + baseline estacional + persistencia) para un target."""

    feature_row = _feature_row_for_target(
        station_id, target_at, data_cutoff, observations, context, config
    )
    hgb_value = float(model.predict(pd.DataFrame([feature_row]))[0])
    hgb_weight = float(config.get("hgb_weight", 1.0))
    horizon_minutes = int((target_at - data_cutoff).total_seconds() // 60)
    baseline_lag = int(config.get("baseline_lag", 672 - horizon_minutes // 15))
    seasonal_value = safe_float(feature_row.get(f"demand_lag_{baseline_lag}"), 0.0)
    value = hgb_weight * hgb_value + (1 - hgb_weight) * seasonal_value
    return blend_with_persistence(value, station_id, data_cutoff, observations, config)


def bias_factor(
    matured: list[tuple[float, float]],
    *,
    alpha: float = BIAS_ALPHA,
    deadzone: float = BIAS_DEADZONE,
    bounds: tuple[float, float] = BIAS_FACTOR_BOUNDS,
    min_samples: int = BIAS_MIN_SAMPLES,
) -> float:
    """Factor multiplicativo desde pares (real, predicho) recientes de una estacion.

    Con pocas muestras o sin demanda devuelve 1. El cociente real/predicho se recorta a
    `bounds`, solo se corrige la parte del sesgo que excede `deadzone` (el ruido normal de
    una estacion estable queda intacto) y `alpha` amortigua la reaccion.
    """

    if len(matured) < min_samples:
        return 1.0
    actual = sum(pair[0] for pair in matured)
    predicted = sum(pair[1] for pair in matured)
    if actual <= 0 or predicted <= 0:
        return 1.0
    ratio = min(max(actual / predicted, bounds[0]), bounds[1])
    deviation = ratio - 1
    excess = math.copysign(max(abs(deviation) - deadzone, 0.0), deviation)
    return min(max(1 + alpha * excess, bounds[0]), bounds[1])


def guarded_factor(slow: float, recent: float | None) -> float:
    """Factor de 3 h acotado por lo que confirma la ultima hora.

    Si la ultima hora no alcanza para medir o va en sentido contrario, no se corrige; si
    coincide, se aplica el menor de los dos desvios.
    """

    if recent is None or (slow - 1) * (recent - 1) <= 0:
        return 1.0
    return 1 + math.copysign(min(abs(slow - 1), abs(recent - 1)), slow - 1)


def recent_surprise(
    station_id: str,
    data_cutoff: pd.Timestamp,
    observations: pd.DataFrame,
    context: pd.DataFrame,
    model: Any,
    config: dict[str, Any],
    actual_by_key: dict[tuple[str, pd.Timestamp], float],
) -> float | None:
    """Real / predicho a 15 min en los ultimos cuartos: que tan atrasado va el campeon ahora."""

    actual_sum = predicted_sum = 0.0
    samples = 0
    for quarters_back in range(1, SURPRISE_QUARTERS + 1):
        past_cutoff = data_cutoff - pd.Timedelta(minutes=PERIOD_MINUTES * quarters_back)
        target = past_cutoff + pd.Timedelta(minutes=PERIOD_MINUTES)
        actual = actual_by_key.get((station_id, target))
        if actual is None:
            continue
        try:
            predicted = predict_value(station_id, target, past_cutoff, observations, context, model, config)
        except (RuntimeError, ValueError, KeyError):
            continue
        actual_sum += actual
        predicted_sum += max(predicted, 0.0)
        samples += 1
    if samples < SURPRISE_MIN_SAMPLES or predicted_sum <= 0:
        return None
    return actual_sum / predicted_sum


def station_bias_factors(
    stations: Iterable[str],
    data_cutoff: pd.Timestamp,
    observations: pd.DataFrame,
    context: pd.DataFrame,
    bundle: dict[int, tuple[Path, dict[str, Any]]],
    loaded_models: dict[int, Any],
) -> dict[str, float]:
    """Factor de correccion por estacion con el campeon actual en los ciclos ya observados.

    Recalcula lo que este mismo paquete habria enviado en los cortes horarios de las ultimas
    `BIAS_WINDOW_HOURS` horas y lo compara con la demanda real de esos targets, que ya esta
    publicada. No depende de Supabase ni de predicciones de campeones anteriores.
    """

    actual_by_key = {
        (row.station_id, row.observed_at): float(row.demand)
        for row in observations.loc[
            (observations["observed_at"] <= data_cutoff)
            & (observations["observed_at"] > data_cutoff - pd.Timedelta(hours=BIAS_WINDOW_HOURS)),
            ["station_id", "observed_at", "demand"],
        ].itertuples(index=False)
        if pd.notna(row.demand)
    }
    factors: dict[str, float] = {}
    for station_id in sorted(set(stations)):
        matured: list[tuple[float, float]] = []
        for hours_back in range(1, BIAS_WINDOW_HOURS + 1):
            past_cutoff = data_cutoff - pd.Timedelta(hours=hours_back)
            for horizon_minutes, (_, config) in bundle.items():
                past_target = past_cutoff + pd.Timedelta(minutes=horizon_minutes)
                if not (data_cutoff - pd.Timedelta(hours=BIAS_WINDOW_HOURS) < past_target <= data_cutoff):
                    continue
                actual = actual_by_key.get((station_id, past_target))
                if actual is None:
                    continue
                try:
                    predicted = predict_value(
                        station_id, past_target, past_cutoff, observations, context,
                        loaded_models[horizon_minutes], config,
                    )
                except (RuntimeError, ValueError, KeyError):
                    continue
                matured.append((actual, max(predicted, 0.0)))
        slow = bias_factor(matured)
        recent = None
        if slow != 1.0 and PERIOD_MINUTES in bundle:
            recent = recent_surprise(
                station_id, data_cutoff, observations, context,
                loaded_models[PERIOD_MINUTES], bundle[PERIOD_MINUTES][1], actual_by_key,
            )
        factors[station_id] = guarded_factor(slow, recent)
    return factors


def load_inference_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observaciones y contexto locales; vacios (con columnas) si no hay archivos."""

    obs_path, ctx_path = SAMPLE_DATA_PATHS["observations"], SAMPLE_DATA_PATHS["context"]
    if obs_path.exists():
        observations = pd.read_csv(obs_path, dtype={"station_id": "string"})
    else:
        observations = pd.DataFrame(columns=["station_id", "observed_at", "demand"])
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    observations["station_id"] = observations["station_id"].map(normalize_station_id)
    observations["demand"] = pd.to_numeric(observations["demand"], errors="coerce")
    if ctx_path.exists():
        context = pd.read_csv(ctx_path)
    else:
        context = pd.DataFrame(columns=["observed_at", *CONTEXT_COLUMNS])
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)
    return observations, context


def champion_stations(bundle: dict[int, tuple[Path, dict[str, Any]]]) -> set[str] | None:
    """Estaciones que el campeon conoce por su one-hot; None si no usa one-hot."""

    stations = {
        column.removeprefix("station_id_")
        for _, config in bundle.values()
        for column in config.get("feature_columns") or []
        if column.startswith("station_id_")
    }
    return stations or None


def compatibility_report(
    cycle: dict[str, Any],
    bundle: dict[int, tuple[Path, dict[str, Any]]],
    observations: pd.DataFrame,
    bundle_error: str | None = None,
) -> dict[str, Any]:
    """Que partes del ciclo puede atender el campeon y por que no las demas.

    Distingue incompatibilidad (el campeon no sabe o no deberia responder) de drift (sabe
    responder aunque el nivel cambie): solo lo primero manda targets al respaldo.
    """

    targets = cycle.get("targets") or []
    data_cutoff = pd.to_datetime(cycle["data_cutoff"], utc=True)
    reasons: list[str] = []
    if bundle_error:
        reasons.append(f"el paquete del campeon no carga: {bundle_error}")
    horizons = sorted({
        int((pd.to_datetime(t["target_at"], utc=True) - data_cutoff).total_seconds() // 60) for t in targets
    })
    unsupported = [h for h in horizons if bundle and h not in bundle]
    if unsupported:
        reasons.append(f"horizontes sin modelo: {unsupported} min")
    off_grid = [h for h in horizons if h <= 0 or h % PERIOD_MINUTES]
    if off_grid or data_cutoff.minute % PERIOD_MINUTES:
        reasons.append(f"targets fuera de la grilla de {PERIOD_MINUTES} min: {off_grid or 'corte'}")
    target_stations = {normalize_station_id(t["station_id"]) for t in targets}
    known = champion_stations(bundle)
    unknown = sorted(target_stations - known) if known is not None else []
    if unknown:
        reasons.append(f"estaciones que el campeon no conoce: {unknown}")
    history = observations.loc[observations["observed_at"] <= data_cutoff]
    with_history = set(history["station_id"].dropna())
    without_history = sorted(target_stations - with_history)
    if without_history:
        reasons.append(f"estaciones sin historial hasta el corte: {without_history}")
    frequency = None
    recent = history.loc[history["observed_at"] > data_cutoff - pd.Timedelta(days=2)]
    if not recent.empty:
        steps = (
            recent.sort_values("observed_at").groupby("station_id")["observed_at"].diff().dropna()
            .dt.total_seconds().div(60)
        )
        if not steps.empty:
            frequency = float(steps.median())
            if abs(frequency - PERIOD_MINUTES) > 0.5:
                reasons.append(f"frecuencia de observaciones de {frequency:g} min (el campeon usa {PERIOD_MINUTES})")
    expected = cycle.get("expected_predictions")
    if expected is not None and int(expected) != len(targets):
        reasons.append(f"el ciclo pide {expected} predicciones pero trae {len(targets)} targets")
    return {
        "cycle_id": cycle.get("cycle_id"),
        "data_cutoff": data_cutoff.isoformat(),
        "compatible": not reasons,
        "reasons": reasons,
        "unsupported_horizons": unsupported,
        "unknown_stations": unknown,
        "stations_without_history": without_history,
        "frequency_minutes": frequency,
        # Con otra frecuencia o sin paquete, los lags del campeon no significan lo mismo:
        # no se le confia ningun target, aunque el horizonte y la estacion existan.
        "champion_enabled": bool(bundle) and not off_grid and (
            frequency is None or abs(frequency - PERIOD_MINUTES) <= 0.5
        ),
    }


def fallback_value(
    station_id: str, data_cutoff: pd.Timestamp, observations: pd.DataFrame
) -> tuple[float | None, str]:
    """Persistencia: ultima demanda publicada de la estacion hasta el corte."""

    history = observations.loc[
        (observations["station_id"] == station_id) & (observations["observed_at"] <= data_cutoff)
    ].dropna(subset=["demand"])
    if history.empty:
        return None, "sin_historial"
    value = float(history.sort_values("observed_at")["demand"].iloc[-1])
    return (value, "persistencia") if math.isfinite(value) and value >= 0 else (None, "sin_historial")


def plausible(value: float, station_id: str, data_cutoff: pd.Timestamp, observations: pd.DataFrame) -> bool:
    """Descarta salidas absurdas del campeon: no finitas, negativas o 10x el maximo reciente."""

    if not math.isfinite(value) or value < 0:
        return False
    recent = observations.loc[
        (observations["station_id"] == station_id)
        & (observations["observed_at"] <= data_cutoff)
        & (observations["observed_at"] > data_cutoff - pd.Timedelta(days=7)),
        "demand",
    ].dropna()
    ceiling = max(float(recent.max()) * PLAUSIBLE_MAX_RATIO, 50.0) if not recent.empty else 1e5
    return value <= ceiling


def infer_predictions_with_report(
    cycle: dict[str, Any],
    bundle: dict[int, tuple[Path, dict[str, Any]]],
    bundle_error: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Una prediccion valida por target, pase lo que pase: campeon y, si no, respaldo.

    El batch es atomico en la API: un target sin prediccion cuesta el ciclo completo y la
    racha. Cada target intenta el campeon; si el campeon no aplica (horizonte o estacion
    desconocidos, otra frecuencia, paquete roto), falla o devuelve algo absurdo, se usa la
    persistencia; sin historial de la estacion, la mediana del resto del ciclo; y como
    ultimo recurso una constante. El reporte dice cuantos targets fueron por cada via.
    """

    observations, context = load_inference_frames()
    targets = cycle["targets"]
    data_cutoff = pd.to_datetime(cycle["data_cutoff"], utc=True)
    report = compatibility_report(cycle, bundle, observations, bundle_error)
    for reason in report["reasons"]:
        print(f"::warning::Compatibilidad: {reason}")
    latest_observation = observations.loc[observations["observed_at"] <= data_cutoff, "observed_at"].max()
    if pd.notna(latest_observation):
        history_age_minutes = (data_cutoff - latest_observation).total_seconds() / 60
        print(f"Antigüedad de la última observación al data_cutoff: {history_age_minutes:.0f} minutos.")

    loaded_models: dict[int, Any] = {}
    if report["champion_enabled"]:
        for horizon, (model_path, _) in bundle.items():
            try:
                with model_path.open("rb") as stream:
                    model = pickle.load(stream)
                config = bundle[horizon][1]
                if config.get("relative_weight"):
                    with relative_model_path(model_path).open("rb") as stream:
                        model = RelativeBlend(model, pickle.load(stream), float(config["relative_weight"]))
                loaded_models[horizon] = model
            except Exception as exc:  # un pickle roto no puede costar el ciclo
                report["reasons"].append(f"modelo de {horizon} min no carga: {exc}")
                report["compatible"] = False
    known = champion_stations(bundle)
    target_stations = {normalize_station_id(t["station_id"]) for t in targets}
    factors: dict[str, float] = {}
    if loaded_models and os.getenv("BIAS_CORRECTION", "on").lower() not in {"off", "0", "false"}:
        try:
            factors = station_bias_factors(
                target_stations if known is None else target_stations & known,
                data_cutoff, observations, context,
                {h: bundle[h] for h in loaded_models}, loaded_models,
            )
            corrected = {station: round(f, 3) for station, f in factors.items() if f != 1.0}
            print(f"Correccion de sesgo ({BIAS_WINDOW_HOURS} h): {corrected or 'ninguna estacion fuera de la zona muerta'}")
        except Exception as exc:  # la correccion nunca debe costar un ciclo
            factors = {}
            print(f"Aviso: correccion de sesgo desactivada en este ciclo ({exc}).")

    # AR(2) local por estacion (scripts/ar_baseline.py), mezclado con el peso por estacion
    # que el entrenamiento eligio en validacion. Sin perfil en el paquete o sin datos
    # suficientes no se mezcla: la prediccion del campeon queda tal cual.
    ar_forecasts: dict[str, np.ndarray] = {}
    if loaded_models:
        try:
            profile = load_profile(next(iter(bundle.values()))[0].parent.parent)
            if profile is not None:
                for station_id in sorted(target_stations):
                    series = (
                        observations.loc[observations["station_id"] == station_id]
                        .set_index("observed_at")["demand"].astype(float).sort_index()
                    )
                    series = series[~series.index.duplicated(keep="last")]
                    forecast = ar2_station_forecast(series, data_cutoff, profile, station_id)
                    if forecast is not None:
                        ar_forecasts[station_id] = forecast
        except Exception as exc:  # el AR nunca debe costar un ciclo
            ar_forecasts = {}
            print(f"Aviso: mezcla AR desactivada en este ciclo ({exc}).")
    ar_mixed: set[str] = set()

    # Regimen de ondas: cada estacion copia a otra con 1-4 h de desfase (scripts/leader_model.py).
    # Solo para estaciones que pasan la compuerta doble; el resto sigue con el campeon.
    leaders: dict[str, dict[str, Any]] = {}
    if os.getenv("LEADER_MODEL", "on").lower() not in {"off", "0", "false"}:
        try:
            leaders = leader_forecasts(observations, data_cutoff, target_stations)
            if leaders:
                print("Estacion lider activa: " + ", ".join(
                    f"{s}<-{v['leader']} {v['lag_quarters'] * PERIOD_MINUTES}min (corr {v['corr']:.3f}, prueba {v['holdout_accuracy']:.0%})"
                    for s, v in sorted(leaders.items())
                ))
        except Exception as exc:  # nunca debe costar un ciclo
            leaders = {}
            print(f"Aviso: estacion lider desactivada en este ciclo ({exc}).")

    predictions: list[dict[str, Any]] = []
    sources: list[str] = []
    failures: dict[str, int] = {}
    for target in targets:
        station_id = normalize_station_id(target["station_id"])
        target_at = pd.to_datetime(target["target_at"], utc=True)
        horizon_minutes = int((target_at - data_cutoff).total_seconds() // 60)
        value: float | None = None
        source = "campeon"
        step = horizon_minutes // PERIOD_MINUTES
        if station_id in leaders and 1 <= step <= len(leaders[station_id]["forecast"]):
            candidate = float(leaders[station_id]["forecast"][step - 1])
            if plausible(candidate, station_id, data_cutoff, observations):
                value, source = candidate, "lider"
        if value is None and horizon_minutes in loaded_models and (known is None or station_id in known):
            try:
                candidate = predict_value(
                    station_id, target_at, data_cutoff, observations, context,
                    loaded_models[horizon_minutes], bundle[horizon_minutes][1],
                ) * factors.get(station_id, 1.0)
                ar_weight = float((bundle[horizon_minutes][1].get("ar_weights") or {}).get(station_id, 0.0))
                step = horizon_minutes // PERIOD_MINUTES
                if ar_weight > 0 and station_id in ar_forecasts and 1 <= step <= len(ar_forecasts[station_id]):
                    candidate = (1 - ar_weight) * candidate + ar_weight * float(ar_forecasts[station_id][step - 1])
                    ar_mixed.add(station_id)
                if plausible(candidate, station_id, data_cutoff, observations):
                    value = candidate
                else:
                    failures["prediccion_fuera_de_rango"] = failures.get("prediccion_fuera_de_rango", 0) + 1
            except Exception as exc:
                key = type(exc).__name__
                failures[key] = failures.get(key, 0) + 1
        if value is None:
            value, source = fallback_value(station_id, data_cutoff, observations)
        predictions.append({"station_id": station_id, "target_at": target["target_at"], "value": value})
        sources.append(source)

    finite = [item["value"] for item in predictions if item["value"] is not None]
    cycle_median = float(pd.Series(finite).median()) if finite else FALLBACK_CONSTANT
    for item, source_index in zip(predictions, range(len(sources))):
        if item["value"] is None:
            item["value"] = cycle_median
            sources[source_index] = "mediana_del_ciclo" if finite else "constante"
        item["value"] = round(max(float(item["value"]), 0.0), 4)

    counts = {name: sources.count(name) for name in dict.fromkeys(sources)}
    modeled = counts.get("campeon", 0) + counts.get("lider", 0)
    report["leader_stations"] = sorted(leaders)
    report.update({
        "total_targets": len(predictions),
        "champion_targets": modeled,
        "fallback_targets": len(predictions) - modeled,
        "sources": counts,
        "champion_failures": failures,
    })
    if report["fallback_targets"] and report["compatible"]:
        # Todo parecia compatible, pero el campeon fallo en targets concretos: tambien es
        # una senal de que el modelo ya no sirve para estos datos.
        report["compatible"] = False
        report["reasons"].append(
            f"el campeon fallo en {report['fallback_targets']} targets: {failures or 'sin detalle'}"
        )
    report["ar_mixed_stations"] = sorted(ar_mixed)
    if ar_mixed:
        print(f"Mezcla AR(2) aplicada en {len(ar_mixed)} estaciones: {sorted(ar_mixed)}")
    print(
        f"Fuentes de prediccion: {counts} · compatible={report['compatible']}"
        + (f" · fallas del campeon {failures}" if failures else "")
    )
    return predictions, report


def infer_predictions(cycle: dict[str, Any], bundle: dict[int, tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    return infer_predictions_with_report(cycle, bundle)[0]


def persist_compatibility(report: dict[str, Any], model_version: str | None) -> None:
    """Guarda el chequeo en Supabase para el dashboard; nunca interrumpe la entrega."""

    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not service_key:
        return
    supabase_url = os.getenv("SUPABASE_URL", "https://jwlgxabibcticikhjhzf.supabase.co").rstrip("/")
    row = {
        "cycle_id": report.get("cycle_id"),
        "data_cutoff": report.get("data_cutoff"),
        "compatible": bool(report.get("compatible")),
        "total_targets": report.get("total_targets"),
        "champion_targets": report.get("champion_targets"),
        "fallback_targets": report.get("fallback_targets"),
        "reasons": report.get("reasons", []),
        "sources": report.get("sources", {}),
        "model_version": model_version,
    }
    try:
        response = httpx.post(
            f"{supabase_url}/rest/v1/model_compatibility",
            params={"on_conflict": "cycle_id"},
            json=[row],
            headers={**supabase_request_headers(service_key), "Prefer": "resolution=merge-duplicates,return=minimal"},
            timeout=30.0,
        )
        if response.is_error:
            print(f"Aviso: no se guardo el chequeo de compatibilidad (HTTP {response.status_code}).")
    except httpx.HTTPError as exc:
        print(f"Aviso: no se guardo el chequeo de compatibilidad ({exc}).")


def model_metadata(bundle: dict[int, tuple[Path, dict[str, Any]]]) -> dict[str, str | None]:
    if not bundle:
        return {"version": FALLBACK_MODEL_VERSION, "training_data_end": None}
    digest = hashlib.sha256()
    seen: set[Path] = set()
    for model_path, config in bundle.values():
        for path in (model_path, relative_model_path(model_path), model_path.parent.parent / "configs" / f"horizon_{int(model_path.name.split('horizon_')[1].split('_hgb')[0])}_ensemble.json"):
            if path in seen or not path.exists():
                continue
            seen.add(path)
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    model_version = f"pulso-hgb-{digest.hexdigest()[:16]}"
    manifest_path = next(iter(bundle.values()))[0].parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    training_end = manifest.get("training_data_end")
    if not training_end and SAMPLE_DATA_PATHS["observations"].exists():
        frame = pd.read_csv(SAMPLE_DATA_PATHS["observations"], usecols=["observed_at"], parse_dates=["observed_at"])
        latest = pd.to_datetime(frame["observed_at"], utc=True).max()
        training_end = latest.isoformat() if pd.notna(latest) else None
    return {"version": model_version, "training_data_end": training_end}


def payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def find_confirmed_submission(cycle_id: str) -> dict[str, Any] | None:
    supabase_url = os.getenv("SUPABASE_URL", "https://jwlgxabibcticikhjhzf.supabase.co").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not service_key:
        raise RuntimeError("Falta SUPABASE_SERVICE_ROLE_KEY; no se enviará sin poder registrar el recibo y evitar duplicados.")
    response = httpx.get(
        f"{supabase_url}/rest/v1/forecast_predictions",
        params={"select": "submission_id,model_version,payload_hash", "cycle_id": f"eq.{cycle_id}", "submission_id": "not.is.null", "limit": "1"},
        headers=supabase_request_headers(service_key),
        timeout=30.0,
    )
    if response.is_error:
        raise RuntimeError(f"No se pudo comprobar el recibo en Supabase: HTTP {response.status_code} {response.text}")
    rows = response.json()
    return rows[0] if rows else None

def build_payload(cycle: dict[str, Any], predictions: list[dict[str, Any]], metadata: dict[str, str | None] | None = None) -> dict[str, Any]:
    cycle_fingerprint = hashlib.sha256(str(cycle["cycle_id"]).encode("utf-8")).hexdigest()[:32]
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"gha-cycle-{cycle_fingerprint}",
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": (metadata or {}).get("version", "pulso-transmi-history-gap-aware-hgb"),
            "training_data_end": (metadata or {}).get("training_data_end"),
            "git_commit": os.getenv("GITHUB_SHA", "unknown"),
        },
        "predictions": predictions,
    }
    return payload


def validate_predictions(cycle: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    expected = cycle.get("targets", [])
    expected_keys = {
        (normalize_station_id(target["station_id"]), pd.to_datetime(target["target_at"], utc=True).isoformat())
        for target in expected
    }
    received_keys = [
        (normalize_station_id(item["station_id"]), pd.to_datetime(item["target_at"], utc=True).isoformat())
        for item in predictions
    ]
    expected_count = cycle.get("expected_predictions", len(expected))
    if len(predictions) != expected_count or len(set(received_keys)) != len(received_keys):
        raise ValueError(
            f"Cantidad de predicciones inválida: se esperaban {expected_count} "
            f"y se generaron {len(predictions)}."
        )
    if set(received_keys) != expected_keys:
        raise ValueError("Las estaciones y fechas de las predicciones no coinciden con los objetivos del ciclo.")
    for item in predictions:
        value = item.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 100000:
            raise ValueError(f"Predicción inválida para {item.get('station_id')}: {value}")


def submit_with_retries(
    client: httpx.Client, api_key: str, payload: dict[str, Any], attempts: int = 3
) -> httpx.Response:
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    headers = request_headers(api_key, payload["client_run_id"])
    for attempt in range(attempts):
        try:
            response = client.post("/v1/submissions", headers=headers, json=payload)
        except httpx.TransportError as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"No se pudo confirmar el envío tras {attempts} intentos: {exc}") from exc
            delay = min(2 ** attempt, 30)
            print(f"Fallo de red al enviar; reintento {attempt + 2}/{attempts} en {delay}s.")
            time.sleep(delay)
            continue
        if response.status_code not in retryable_statuses:
            return response
        if attempt + 1 == attempts:
            return response
        retry_after = response.headers.get("Retry-After")
        try:
            delay = min(max(int(retry_after), 1), 60) if retry_after else min(2 ** attempt, 30)
        except ValueError:
            delay = min(2 ** attempt, 30)
        print(f"API respondió HTTP {response.status_code}; reintento {attempt + 2}/{attempts} en {delay}s.")
        time.sleep(delay)
    raise RuntimeError("Se agotaron los intentos de envío.")


def write_github_output(name: str, value: str) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")


def supabase_request_headers(api_key: str) -> dict[str, str]:
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    # New sb_secret keys are API keys, not JWTs; legacy service_role keys are JWTs.
    if not api_key.startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def persist_confirmed_predictions(payload: dict[str, Any], submission_id: str | None) -> bool:
    """Persist only predictions confirmed as official by the competition API."""
    supabase_url = os.getenv("SUPABASE_URL", "https://jwlgxabibcticikhjhzf.supabase.co").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not service_key:
        print("Aviso: falta SUPABASE_SERVICE_ROLE_KEY; el envío fue oficial, pero no se guardará para medir WAPE.")
        write_github_output("monitoring_persisted", "false")
        return False
    cutoff = pd.to_datetime(payload["data_cutoff"], utc=True)
    digest = payload_hash(payload)
    rows = []
    for prediction in payload["predictions"]:
        target_at = pd.to_datetime(prediction["target_at"], utc=True)
        rows.append({
            "cycle_id": str(payload["cycle_id"]),
            "client_run_id": payload["client_run_id"],
            "submission_id": submission_id,
            "station_id": normalize_station_id(prediction["station_id"]),
            "target_at": target_at.isoformat(),
            "data_cutoff": cutoff.isoformat(),
            "horizon_minutes": int((target_at - cutoff).total_seconds() // 60),
            "predicted_demand": prediction["value"],
            "payload_hash": digest,
            "model_version": payload["model"]["version"],
            "git_commit": payload["model"]["git_commit"],
            "training_data_end": payload["model"]["training_data_end"],
        })
    response = httpx.post(
        f"{supabase_url}/rest/v1/forecast_predictions",
        params={"on_conflict": "cycle_id,station_id,target_at"},
        headers={**supabase_request_headers(service_key), "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=rows,
        timeout=60.0,
    )
    if response.is_error:
        print(f"Error guardando predicciones para monitoreo WAPE: HTTP {response.status_code} {response.text}")
        write_github_output("monitoring_persisted", "false")
        return False
    print(f"Predicciones oficiales guardadas para monitoreo WAPE: {len(rows)}")
    write_github_output("monitoring_persisted", "true")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta el ciclo activo, genera predicciones y envía la submission si el ciclo está abierto.")
    parser.add_argument("--dry-run", action="store_true", help="No envía la submission; solo genera el payload local.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "current_submission.json")
    parser.add_argument("--wait-seconds", type=int, default=0, help="Espera este tiempo buscando un ciclo abierto.")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Intervalo entre consultas del ciclo.")
    args = parser.parse_args()

    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        raise RuntimeError("Falta PULSO_API_KEY. Configúralo como secreto en GitHub Actions.")

    base_url = os.getenv("PULSO_API_URL", DEFAULT_API_URL).rstrip("/")
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        deadline = time.monotonic() + max(args.wait_seconds, 0)
        cycle = load_cycle(client, api_key)
        while (cycle is None or cycle.get("state") != "open") and time.monotonic() < deadline:
            remaining = int(deadline - time.monotonic())
            print(f"No hay ciclo abierto; reintentando en {args.poll_seconds}s (quedan {remaining}s).")
            time.sleep(min(args.poll_seconds, max(remaining, 1)))
            cycle = load_cycle(client, api_key)
        if cycle is None or cycle.get("state") != "open":
            print("No hay un ciclo abierto; la acción termina sin enviar predicciones.")
            write_github_output("submitted", "false")
            return 0

        if not args.dry_run:
            receipt = find_confirmed_submission(str(cycle["cycle_id"]))
            if receipt:
                print(f"El ciclo ya tiene recibo oficial {receipt.get('submission_id')}; se omite el POST duplicado.")
                write_github_output("submitted", "true")
                write_github_output("already_submitted", "true")
                write_github_output("submission_id", str(receipt.get("submission_id", "")))
                write_github_output("monitoring_persisted", "true")
                return 0

        cutoff = pd.to_datetime(cycle["data_cutoff"], utc=True)
        start_at = (cutoff - pd.Timedelta(days=30)).isoformat()
        try:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "download_supabase_data.py"), "--start-at", start_at],
                check=True,
                cwd=ROOT,
            )
        except subprocess.CalledProcessError as exc:
            # Si el colector se rompe (por ejemplo, con un esquema nuevo de la API), el
            # stream de la API todavia alcanza para enviar.
            print(f"::warning::No se pudieron leer los datos de Supabase (codigo {exc.returncode}); se usa el stream de la API.")
        refresh_observations_from_stream(client, cutoff)
        bundle_error = None
        try:
            bundle = ensure_bundle_ready()
        except Exception as exc:
            bundle, bundle_error = {}, f"{type(exc).__name__}: {exc}"
        predictions, report = infer_predictions_with_report(cycle, bundle, bundle_error)
        validate_predictions(cycle, predictions)
        metadata = model_metadata(bundle)
        if bundle and report["fallback_targets"]:
            metadata["version"] = f"{metadata['version']}+respaldo"
        COMPATIBILITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
        COMPATIBILITY_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        write_github_output("compatible", str(report["compatible"]).lower())
        write_github_output("fallback_targets", str(report["fallback_targets"]))
        write_github_output("total_targets", str(report["total_targets"]))
        if not args.dry_run:
            persist_compatibility(report, metadata["version"])
        payload = build_payload(cycle, predictions, metadata)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Payload generado en {args.output} con {len(predictions)} predicciones.")

        if args.dry_run:
            write_github_output("submitted", "false")
            return 0

        response = submit_with_retries(client, api_key, payload)
        if response.is_error:
            raise RuntimeError(f"Submission rechazada: HTTP {response.status_code} {response.text}")
        result = response.json()
        print(json.dumps(result, indent=2))
        is_official = result.get("is_official") is True
        write_github_output("submitted", str(is_official).lower())
        if is_official:
            submission_id = str(result.get("submission_id", "")) or None
            write_github_output("submission_id", submission_id or "")
            if not persist_confirmed_predictions(payload, submission_id):
                raise RuntimeError("La submission fue oficial, pero falló el registro del recibo; el siguiente intento reutilizará la misma Idempotency-Key.")
        if not is_official:
            raise RuntimeError("La API respondió sin confirmar is_official=true; revisar la respuesta antes de darlo por entregado.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
