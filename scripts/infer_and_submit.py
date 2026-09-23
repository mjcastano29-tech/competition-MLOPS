from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
BUNDLE_DIR = ROOT / "artifacts" / "pulso_transmi_best_models"
BUNDLE_ZIP = ROOT / "artifacts" / "pulso_transmi_best_models.zip"
SAMPLE_DATA_PATHS = {
    "observations": ROOT / "data" / "observations.csv",
    "context": ROOT / "data" / "context.csv",
}


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
    for model_path in sorted((bundle / "models").glob("*.pkl")):
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
        return float(value)
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
    # Training features are indexed by the observation time (forecast origin),
    # so inference must use the cycle cutoff, not the future target timestamp.
    history = observations[
        (observations["station_id"] == station_id) & (observations["observed_at"] <= data_cutoff)
    ].sort_values("observed_at").copy()
    historic_context = context[context["observed_at"] <= data_cutoff].sort_values("observed_at")
    latest_context = historic_context.iloc[-1].to_dict() if not historic_context.empty else {}
    feature_columns = config.get("feature_columns", [])
    row: dict[str, float] = {column: 0.0 for column in feature_columns}

    for feature in feature_columns:
        if feature.startswith("station_id_"):
            row[feature] = 1.0 if station_id == feature.removeprefix("station_id_") else 0.0
            continue
        if feature.startswith("demand_lag_"):
            lag = int(feature.split("_")[-1])
            lag_time = data_cutoff - pd.Timedelta(minutes=lag * 15)
            last_row = _last_known_row(history, lag_time)
            row[feature] = safe_float(last_row.get("demand"), 0.0) if last_row else 0.0
            continue
        if feature.startswith(("demand_mean_", "demand_std_")):
            window = int(feature.split("_")[-1])
            # add_features() uses shift(1).rolling(window), excluding the value
            # at the forecast origin itself.
            recent = history[history["observed_at"] < data_cutoff].tail(window)["demand"]
            statistic = recent.mean() if feature.startswith("demand_mean_") else recent.std()
            row[feature] = safe_float(statistic, 0.0)
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
            row[feature] = safe_float(latest_context.get(feature), 0.0)
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
        row[feature] = 0.0

    return row


def normalize_station_id(value: Any) -> str:
    station_id = str(value).strip()
    return station_id.zfill(5) if station_id.isdigit() else station_id


def infer_predictions(cycle: dict[str, Any], bundle: dict[int, tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    observations = pd.read_csv(
        SAMPLE_DATA_PATHS["observations"],
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    )
    context = pd.read_csv(SAMPLE_DATA_PATHS["context"], parse_dates=["observed_at"])
    observations["station_id"] = observations["station_id"].map(normalize_station_id)

    targets = cycle["targets"]
    data_cutoff = pd.to_datetime(cycle["data_cutoff"])
    target_stations = {normalize_station_id(target["station_id"]) for target in targets}
    available_stations = set(
        observations.loc[observations["observed_at"] <= data_cutoff, "station_id"].dropna()
    )
    missing_stations = sorted(target_stations - available_stations)
    if missing_stations:
        raise RuntimeError(
            "No hay historial de observaciones hasta el data_cutoff para estas estaciones: "
            + ", ".join(missing_stations)
        )
    print(f"Historial disponible para {len(target_stations)} estaciones objetivo.")
    latest_observation = observations.loc[
        observations["observed_at"] <= data_cutoff, "observed_at"
    ].max()
    if pd.notna(latest_observation):
        history_age_minutes = (data_cutoff - latest_observation).total_seconds() / 60
        print(f"Antigüedad de la última observación al data_cutoff: {history_age_minutes:.0f} minutos.")
    predictions: list[dict[str, Any]] = []
    loaded_models: dict[int, Any] = {}
    for horizon, (model_path, _) in bundle.items():
        with model_path.open("rb") as stream:
            loaded_models[horizon] = pickle.load(stream)
    for target in targets:
        station_id = normalize_station_id(target["station_id"])
        target_at = pd.to_datetime(target["target_at"])
        horizon_minutes = int((target_at - data_cutoff).total_seconds() // 60)
        if horizon_minutes not in bundle:
            raise ValueError(f"No hay modelo empaquetado para horizonte de {horizon_minutes} minutos.")
        horizon_key = horizon_minutes
        _, config = bundle[horizon_key]
        model = loaded_models[horizon_key]
        feature_row = _feature_row_for_target(
            station_id, target_at, data_cutoff, observations, context, config
        )
        frame = pd.DataFrame([feature_row])
        hgb_value = float(model.predict(frame)[0])
        hgb_weight = float(config.get("hgb_weight", 1.0))
        baseline_lag = int(config.get("baseline_lag", 672 - horizon_minutes // 15))
        seasonal_value = safe_float(feature_row.get(f"demand_lag_{baseline_lag}"), 0.0)
        value = hgb_weight * hgb_value + (1 - hgb_weight) * seasonal_value
        predictions.append({
            "station_id": station_id,
            "target_at": target["target_at"],
            "value": round(max(value, 0.0), 4),
        })

    return predictions


def build_payload(cycle: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"gha-{uuid.uuid4().hex}",
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": "pulso-transmi-history-gap-aware-hgb",
            "training_data_end": None,
            "git_commit": None,
        },
        "predictions": predictions,
    }
    return payload


def validate_predictions(cycle: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    expected = cycle.get("targets", [])
    expected_keys = {(str(target["station_id"]), target["target_at"]) for target in expected}
    received_keys = [(str(item["station_id"]), item["target_at"]) for item in predictions]
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
    client: httpx.Client, api_key: str, payload: dict[str, Any], attempts: int = 5
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

        bundle = ensure_bundle_ready()
        predictions = infer_predictions(cycle, bundle)
        validate_predictions(cycle, predictions)
        payload = build_payload(cycle, predictions)
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
            write_github_output("submission_id", str(result.get("submission_id", "")))
        if not is_official:
            raise RuntimeError("La API respondió sin confirmar is_official=true; revisar la respuesta antes de darlo por entregado.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
