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
from typing import Any

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
BUNDLE_DIR = ROOT / "artifacts" / "pulso_transmi_best_models"
BUNDLE_ZIP = ROOT / "artifacts" / "pulso_transmi_best_models.zip"
# Legacy Actions cache entries use the gap133 cache namespace but predate
# history_gap_steps in each per-horizon JSON config.
DEFAULT_HISTORY_GAP_STEPS = 133
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
    feature_columns = config.get("feature_columns", [])
    context_features = sorted(
        {"rain_forecast", "temperature_forecast", "event_intensity"}.intersection(feature_columns)
    )
    available_context = {}
    if context_features:
        historic_context = context[
            (context["observed_at"] <= feature_data_cutoff)
            & context[context_features].notna().all(axis=1)
        ].sort_values("observed_at")
        if not historic_context.empty:
            available_context = historic_context.iloc[-1].to_dict()
            context_at = available_context["observed_at"]
            if context_at < feature_data_cutoff:
                age_minutes = (feature_data_cutoff - context_at).total_seconds() / 60
                print(
                    f"Contexto {station_id}: uso el último registro completo, "
                    f"{age_minutes:.0f} min anterior al corte de variables."
                )
        else:
            print(
                f"Advertencia: sin contexto completo para {station_id} hasta "
                f"{feature_data_cutoff}; las variables de contexto usarán 0."
            )
    if history.empty:
        raise RuntimeError(
            f"No hay historial de demanda para {station_id} hasta {feature_data_cutoff}."
        )
    row: dict[str, float] = {column: 0.0 for column in feature_columns}

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
            row[feature] = safe_float(available_context.get(feature), 0.0)
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


def normalize_target_key(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("Z", "+00:00")
    return str(value)


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


def model_metadata(bundle: dict[int, tuple[Path, dict[str, Any]]]) -> dict[str, str | None]:
    digest = hashlib.sha256()
    seen: set[Path] = set()
    for model_path, config in bundle.values():
        for path in (model_path, model_path.parent.parent / "configs" / f"horizon_{int(model_path.name.split('horizon_')[1].split('_hgb')[0])}_ensemble.json"):
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
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "download_supabase_data.py"), "--start-at", start_at],
            check=True,
            cwd=ROOT,
        )
        bundle = ensure_bundle_ready()
        predictions = infer_predictions(cycle, bundle)
        validate_predictions(cycle, predictions)
        payload = build_payload(cycle, predictions, model_metadata(bundle))
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
