from __future__ import annotations

import argparse
import json
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


def load_cycle(client: httpx.Client, api_key: str) -> dict[str, Any] | None:
    response = client.get("/v1/forecast-cycles/current", headers=request_headers(api_key))
    if response.status_code == 404:
        return None
    if response.is_error:
        raise RuntimeError(f"No se pudo consultar el ciclo actual: HTTP {response.status_code} {response.text}")
    return response.json()


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
    observations: pd.DataFrame,
    context: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float]:
    history = observations[observations["station_id"] == station_id].sort_values("observed_at").copy()
    historic_context = context[context["observed_at"] <= target_at].sort_values("observed_at")
    latest_context = historic_context.iloc[-1].to_dict() if not historic_context.empty else {}
    feature_columns = config.get("feature_columns", [])
    row: dict[str, float] = {column: 0.0 for column in feature_columns}

    for feature in feature_columns:
        if feature.startswith("station_id_"):
            row[feature] = 1.0 if station_id == feature.removeprefix("station_id_") else 0.0
            continue
        if feature.startswith("demand_lag_"):
            lag = int(feature.split("_")[-1])
            lag_time = target_at - pd.Timedelta(minutes=lag * 15)
            last_row = _last_known_row(history, lag_time)
            row[feature] = safe_float(last_row.get("demand"), 0.0) if last_row else 0.0
            continue
        if feature.startswith("demand_mean_"):
            window = int(feature.split("_")[-1])
            recent = history[history["observed_at"] <= target_at].tail(window)
            row[feature] = safe_float(recent["demand"].mean(), 0.0)
            continue
        if feature in {"rain_forecast", "temperature_forecast", "event_intensity"}:
            row[feature] = safe_float(latest_context.get(feature), 0.0)
            continue
        if feature in {"is_weekend", "quarter_sin", "quarter_cos", "weekday_sin", "weekday_cos"}:
            day_of_week = target_at.dayofweek
            quarter_of_day = target_at.hour * 4 + target_at.minute // 15
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


def infer_predictions(cycle: dict[str, Any], bundle: dict[int, tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    observations = pd.read_csv(SAMPLE_DATA_PATHS["observations"], parse_dates=["observed_at"])
    context = pd.read_csv(SAMPLE_DATA_PATHS["context"], parse_dates=["observed_at"])
    observations["station_id"] = observations["station_id"].astype(str)

    targets = cycle["targets"]
    data_cutoff = pd.to_datetime(cycle["data_cutoff"])
    predictions: list[dict[str, Any]] = []
    for target in targets:
        station_id = str(target["station_id"])
        target_at = pd.to_datetime(target["target_at"])
        horizon_minutes = int((target_at - data_cutoff).total_seconds() // 60)
        if horizon_minutes not in bundle:
            raise ValueError(f"No hay modelo empaquetado para horizonte de {horizon_minutes} minutos.")
        horizon_key = horizon_minutes
        model_path, config = bundle[horizon_key]
        with model_path.open("rb") as stream:
            model = pickle.load(stream)

        feature_row = _feature_row_for_target(station_id, target_at, observations, context, config)
        frame = pd.DataFrame([feature_row])
        value = float(model.predict(frame)[0])
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
            "version": "pulso-transmi-best-ensemble-gha",
            "training_data_end": None,
            "git_commit": None,
        },
        "predictions": predictions,
    }
    return payload


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
            return 0

        bundle = ensure_bundle_ready()
        predictions = infer_predictions(cycle, bundle)
        payload = build_payload(cycle, predictions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Payload generado en {args.output} con {len(predictions)} predicciones.")

        if args.dry_run:
            return 0

        response = client.post(
            "/v1/submissions",
            headers=request_headers(api_key, payload["client_run_id"]),
            json=payload,
        )
        if response.is_error:
            raise RuntimeError(f"Submission rechazada: HTTP {response.status_code} {response.text}")
        print(json.dumps(response.json(), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
