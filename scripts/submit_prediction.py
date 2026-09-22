from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

DEFAULT_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"


class SubmissionError(RuntimeError):
    pass


def request_headers(api_key: str, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def load_cycle(client: httpx.Client, api_key: str) -> dict[str, Any]:
    response = client.get("/v1/forecast-cycles/current", headers=request_headers(api_key))
    if response.is_error:
        raise SubmissionError(f"No se pudo consultar el ciclo actual: HTTP {response.status_code} {response.text}")
    return response.json()


def build_template(cycle: dict[str, Any], model_version: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"local-{uuid.uuid4().hex}",
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": model_version,
            "training_data_end": None,
            "git_commit": None,
        },
        "predictions": [
            {
                "station_id": target["station_id"],
                "target_at": target["target_at"],
                "value": 0.0,
            }
            for target in cycle["targets"]
        ],
    }


def validate_payload(payload: dict[str, Any], cycle: dict[str, Any]) -> None:
    required = {"schema_version", "cycle_id", "client_run_id", "data_cutoff", "model", "predictions"}
    missing = required - payload.keys()
    if missing:
        raise SubmissionError(f"Faltan campos requeridos: {sorted(missing)}")
    if payload["schema_version"] != "1.0":
        raise SubmissionError("schema_version debe ser '1.0'.")
    if payload["cycle_id"] != cycle["cycle_id"]:
        raise SubmissionError(
            f"El payload usa {payload['cycle_id']}, pero el ciclo actual es {cycle['cycle_id']}."
        )
    if payload["data_cutoff"] != cycle["data_cutoff"]:
        raise SubmissionError("data_cutoff debe coincidir exactamente con el ciclo actual.")

    expected = {(target["station_id"], target["target_at"]) for target in cycle["targets"]}
    received = {(prediction.get("station_id"), prediction.get("target_at")) for prediction in payload["predictions"]}
    if len(payload["predictions"]) != cycle["expected_predictions"]:
        raise SubmissionError(
            f"Se esperaban {cycle['expected_predictions']} predicciones, pero llegaron {len(payload['predictions'])}."
        )
    if received != expected:
        raise SubmissionError("Las estaciones o fechas objetivo no coinciden con el ciclo actual.")
    for prediction in payload["predictions"]:
        value = prediction.get("value")
        if not isinstance(value, (int, float)) or value < 0 or value > 100000:
            raise SubmissionError(f"Valor inválido para {prediction.get('station_id')}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida y envía una submission de Pulso TransMi.")
    parser.add_argument("--payload", type=Path, help="JSON de submission ya completado.")
    parser.add_argument("--template", type=Path, help="Escribe una plantilla para el ciclo actual.")
    parser.add_argument("--model-version", default="pulso-transmi-best-ensemble-20260918")
    args = parser.parse_args()

    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        raise SubmissionError("Falta PULSO_API_KEY en el entorno local.")
    base_url = os.getenv("PULSO_API_URL", DEFAULT_API_URL).rstrip("/")

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        cycle = load_cycle(client, api_key)
        if cycle.get("state") != "open":
            raise SubmissionError(f"El ciclo actual no está abierto: {cycle.get('state')}")

        if args.template:
            args.template.write_text(json.dumps(build_template(cycle, args.model_version), indent=2) + "\n")
            print(f"Plantilla escrita en {args.template}. Completa los valores de predictions[].value.")
            return
        if not args.payload:
            raise SubmissionError("Usa --template para crear una plantilla o --payload para enviar un JSON.")

        payload = json.loads(args.payload.read_text())
        validate_payload(payload, cycle)
        idempotency_key = f"{payload['client_run_id']}-{uuid.uuid4().hex}"
        response = client.post(
            "/v1/submissions",
            headers=request_headers(api_key, idempotency_key),
            json=payload,
        )
        if response.is_error:
            raise SubmissionError(f"Submission rechazada: HTTP {response.status_code} {response.text}")
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
