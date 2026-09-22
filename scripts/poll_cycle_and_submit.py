from __future__ import annotations

import argparse
import importlib.util
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"


def _load_submit_module() -> Any:
    module_path = ROOT / "scripts" / "submit_prediction.py"
    spec = importlib.util.spec_from_file_location("submit_prediction", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta el ciclo activo y envía una submission cuando está abierto.")
    parser.add_argument("--payload", type=Path, default=ROOT / "artifacts" / "current_submission.json")
    parser.add_argument("--template-out", type=Path, default=ROOT / "artifacts" / "next_cycle_template.json")
    parser.add_argument("--model-version", default="pulso-transmi-best-ensemble-20260918")
    args = parser.parse_args()

    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        raise RuntimeError("Falta PULSO_API_KEY en el entorno. Configúralo como secreto en GitHub Actions.")

    base_url = os.getenv("PULSO_API_URL", DEFAULT_API_URL).rstrip("/")
    module = _load_submit_module()

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        cycle = load_cycle(client, api_key)
        if cycle is None or cycle.get("state") != "open":
            print("No hay un ciclo abierto en este momento; la acción termina sin enviar predicciones.")
            return 0

        payload_path = args.payload
        payload_path.parent.mkdir(parents=True, exist_ok=True)

        if payload_path.exists():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            module.validate_payload(payload, cycle)
            idempotency_key = f"{payload['client_run_id']}-{uuid.uuid4().hex}"
            response = client.post(
                "/v1/submissions",
                headers=request_headers(api_key, idempotency_key),
                json=payload,
            )
            if response.is_error:
                raise RuntimeError(f"Submission rechazada: HTTP {response.status_code} {response.text}")
            print(json.dumps(response.json(), indent=2))
            return 0

        template = module.build_template(cycle, args.model_version)
        args.template_out.parent.mkdir(parents=True, exist_ok=True)
        args.template_out.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
        print(
            f"El ciclo está abierto, pero no hay un payload listo en {payload_path}. "
            f"Se generó una plantilla en {args.template_out}."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
