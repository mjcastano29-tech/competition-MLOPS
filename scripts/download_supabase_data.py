from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

DEFAULT_URL = "https://jwlgxabibcticikhjhzf.supabase.co"
PAGE_SIZE = 1000
ROOT = Path(__file__).resolve().parents[1]


def headers(key: str) -> dict[str, str]:
    result = {"apikey": key, "Content-Type": "application/json"}
    if not key.startswith("sb_secret_"):
        result["Authorization"] = f"Bearer {key}"
    return result


def fetch_table(client: httpx.Client, table: str, columns: str, order: str, start_at: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = client.get(
            f"/{table}",
            params={"select": columns, "order": order, "limit": str(PAGE_SIZE), "offset": str(offset), **({"observed_at": f"gte.{start_at}"} if start_at else {})},
        )
        if response.is_error:
            raise RuntimeError(f"Supabase {table}: HTTP {response.status_code}: {response.text}")
        page = response.json()
        rows.extend(page)
        print(f"{table}: descargadas {len(rows)} filas")
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", help="Limita observaciones/contexto a este timestamp inclusivo.")
    args = parser.parse_args()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise RuntimeError("Falta SUPABASE_SERVICE_ROLE_KEY; la inferencia debe usar la misma fuente persistida por el colector.")
    base_url = os.getenv("SUPABASE_URL", DEFAULT_URL).rstrip("/") + "/rest/v1"
    with httpx.Client(base_url=base_url, headers=headers(key), timeout=60) as client:
        observations = fetch_table(client, "observations", "station_id,observed_at,demand", "station_id.asc,observed_at.asc", args.start_at)
        context = fetch_table(client, "context", "observed_at,rain_mm,rain_forecast,temperature_c,temperature_forecast,event_intensity", "observed_at.asc", args.start_at)
    if not observations or not context:
        raise RuntimeError("Supabase aún no tiene observaciones y contexto; ejecuta el colector antes de inferir.")
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(observations).to_csv(data_dir / "observations.csv", index=False)
    pd.DataFrame(context).to_csv(data_dir / "context.csv", index=False)
    latest = pd.to_datetime(pd.DataFrame(observations)["observed_at"], utc=True).max()
    print(f"Datos de inferencia actualizados desde Supabase: {len(observations)} observaciones, {len(context)} contextos; último observado {latest}.")


if __name__ == "__main__":
    main()
