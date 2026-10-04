"""Envio de ultimo recurso: persistencia pura con solo la libreria estandar de Python.

Corre en el workflow cuando infer_and_submit.py no confirmo la entrega (fallo la
instalacion, el paquete del modelo, la inferencia o la API rechazo el batch). No depende de
pandas, scikit-learn, Supabase ni del codigo del campeon: lo que pudo romperse. Responde
cada target con la ultima demanda observada de su estacion (o la mediana del corte).

    python3 scripts/last_resort_submit.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:  # solo libreria estandar; si aun asi fallara, se lee `demand` a secas
    from scripts.stream_schema import row_demand
except Exception:  # pragma: no cover
    def row_demand(row: dict[str, Any]) -> float | None:
        try:
            return float(row.get("demand"))
        except (TypeError, ValueError):
            return None

DEFAULT_API_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
VERSION = "respaldo-ultimo-recurso"
MAX_STREAM_PAGES = 60
Request = Callable[[str, str, dict[str, Any] | None, dict[str, str]], tuple[int, Any]]


def http_request(method: str, url: str, body: dict[str, Any] | None, headers: dict[str, str]) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


def call(request: Request, method: str, url: str, body: dict[str, Any] | None, headers: dict[str, str], attempts: int = 4) -> tuple[int, Any]:
    for attempt in range(attempts):
        try:
            status, data = request(method, url, body, headers)
        except Exception as exc:  # red caida, timeout
            status, data = 0, str(exc)
        if status not in {0, 408, 425, 429, 500, 502, 503, 504} or attempt + 1 == attempts:
            return status, data
        time.sleep(min(2 ** attempt, 10))
    return status, data


def parse_time(value: Any) -> float | None:
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def norm(station: Any) -> str:
    text = str(station).strip()
    return text.zfill(5) if text.isdigit() else text


def last_demand(request: Request, base: str, headers: dict[str, str], cutoff: float | None) -> dict[str, float]:
    latest: dict[str, tuple[float, float]] = {}
    cursor = None
    for _ in range(MAX_STREAM_PAGES):
        params = {"limit": 5000, **({"cursor": cursor} if cursor else {})}
        status, data = call(request, "GET", f"{base}/v1/stream/observations?{urllib.parse.urlencode(params)}", None, headers)
        if status != 200 or not isinstance(data, dict):
            break
        for row in data.get("data") or []:
            if not isinstance(row, dict):
                continue
            stamp, value = parse_time(row.get("observed_at")), row_demand(row)
            if stamp is None or value is None or not math.isfinite(value) or value < 0:
                continue
            if cutoff is not None and stamp > cutoff:
                continue
            key = norm(row.get("station_id"))
            if key not in latest or stamp > latest[key][0]:
                latest[key] = (stamp, value)
        cursor = data.get("next_cursor")
        if not cursor or not data.get("data"):
            break
    return {k: v for k, (_, v) in latest.items()}


def git_commit() -> str:
    sha = os.getenv("GITHUB_SHA", "")
    return sha if re.fullmatch(r"[0-9a-fA-F]{7,40}", sha) else "0000000"


def build_payload(cycle: dict[str, Any], last: dict[str, float]) -> dict[str, Any]:
    default = statistics.median(last.values()) if last else 1.0
    predictions = [
        {"station_id": t["station_id"], "target_at": t["target_at"], "value": round(float(last.get(norm(t["station_id"]), default)), 2)}
        for t in cycle["targets"]
    ]
    fingerprint = hashlib.sha256(str(cycle["cycle_id"]).encode("utf-8")).hexdigest()[:32]
    return {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": f"gha-cycle-{fingerprint}-lr",
        "data_cutoff": cycle["data_cutoff"],
        "model": {"version": VERSION, "training_data_end": cycle["data_cutoff"], "git_commit": git_commit()},
        "predictions": predictions,
    }


def run(request: Request = http_request) -> int:
    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        print("::error::Falta PULSO_API_KEY.")
        return 1
    base = os.getenv("PULSO_API_URL", DEFAULT_API_URL).rstrip("/")
    auth = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    status, cycle = call(request, "GET", f"{base}/v1/forecast-cycles/current", None, auth)
    if status == 404 or not isinstance(cycle, dict) or cycle.get("state") != "open":
        print(f"No hay ciclo abierto (HTTP {status}); nada que enviar.")
        return 0
    last = last_demand(request, base, auth, parse_time(cycle.get("data_cutoff")))
    payload = build_payload(cycle, last)
    print(f"Ultimo recurso: {len(payload['predictions'])} predicciones por persistencia ({len(last)} estaciones con dato).")
    status, data = call(request, "POST", f"{base}/v1/submissions", payload, {**auth, "Idempotency-Key": payload["client_run_id"]})
    if status in {400, 422}:
        # Ultima variante: valores enteros, por si la regla nueva es sobre decimales.
        print(f"::warning::HTTP {status} {data}; se reintenta con valores enteros.")
        for p in payload["predictions"]:
            p["value"] = int(round(p["value"]))
        payload["client_run_id"] += "i"
        status, data = call(request, "POST", f"{base}/v1/submissions", payload, {**auth, "Idempotency-Key": payload["client_run_id"]})
    if status == 409:
        print("El ciclo ya tenia una entrega con esta clave; queda cubierto.")
        return 0
    if 200 <= status < 300:
        print(json.dumps(data, indent=2))
        return 0
    print(f"::error::El envio de ultimo recurso fallo: HTTP {status} {data}")
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
