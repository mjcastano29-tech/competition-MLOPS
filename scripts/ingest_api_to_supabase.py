from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from pulso_transmi import PulsoTransmiClient


BATCH_SIZE = 500
STREAM_PAGE_SIZE = 5000
DEFAULT_SUPABASE_URL = "https://jwlgxabibcticikhjhzf.supabase.co"


class SupabaseIngestionError(RuntimeError):
    pass


def supabase_request_headers(api_key: str) -> dict[str, str]:
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    # New sb_secret keys are API keys, not JWTs; legacy service_role keys are JWTs.
    if not api_key.startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def require_environment() -> tuple[str, str]:
    supabase_url = os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL).rstrip("/")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not service_role_key:
        raise SupabaseIngestionError(
            "Falta SUPABASE_SERVICE_ROLE_KEY. Configúrala como secreto SUPABASE_SERVICE_ROLE_KEY en GitHub Actions o como variable local segura."
        )
    return supabase_url, service_role_key


class SupabaseRestClient:
    def __init__(self, base_url: str, service_role_key: str) -> None:
        self._client = httpx.Client(
            base_url=f"{base_url}/rest/v1",
            headers=supabase_request_headers(service_role_key),
            timeout=60.0,
        )

    def close(self) -> None:
        self._client.close()

    def upsert(
        self, table: str, rows: list[dict[str, Any]], conflict_columns: str, *, ignore_duplicates: bool = False
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        response = self._client.post(
            f"/{table}",
            params={"on_conflict": conflict_columns},
            headers={
                "Prefer": f"resolution={'ignore' if ignore_duplicates else 'merge'}-duplicates,return=representation"
            },
            json=rows,
        )
        if response.is_error:
            raise SupabaseIngestionError(
                f"Supabase rechazó {table}: HTTP {response.status_code} {response.text}"
            )
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def find_dataset(self, dataset_name: str, cutoff_at: str) -> dict[str, Any]:
        response = self._client.get(
            "/datasets",
            params={
                "select": "dataset_id,dataset_name,cutoff_at",
                "dataset_name": f"eq.{dataset_name}",
                "cutoff_at": f"eq.{cutoff_at}",
            },
        )
        if response.is_error:
            raise SupabaseIngestionError(
                f"No se pudo consultar datasets: HTTP {response.status_code} {response.text}"
            )
        rows = response.json()
        if not rows:
            raise SupabaseIngestionError("Supabase no devolvió el dataset creado.")
        return rows[0]

    def has_observations(self) -> bool:
        response = self._client.get("/observations", params={"select": "observation_id", "limit": "1"})
        if response.is_error:
            raise SupabaseIngestionError(
                f"No se pudo comprobar la carga inicial: HTTP {response.status_code} {response.text}"
            )
        return bool(response.json())

    def get_watermark(self, stream_name: str) -> datetime | None:
        response = self._client.get(
            "/api_cursors",
            params={"select": "cursor_value", "stream_name": f"eq.{stream_name}", "limit": "1"},
        )
        if response.is_error:
            raise SupabaseIngestionError(
                f"No se pudo leer api_cursors: HTTP {response.status_code} {response.text}"
            )
        rows = response.json()
        return datetime.fromisoformat(rows[0]["cursor_value"].replace("Z", "+00:00")) if rows else None

    def save_watermark(self, stream_name: str, value: datetime) -> None:
        response = self._client.post(
            "/api_cursors",
            params={"on_conflict": "stream_name"},
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            json={"stream_name": stream_name, "cursor_value": value.isoformat()},
        )
        if response.is_error:
            raise SupabaseIngestionError(
                f"No se pudo avanzar el cursor {stream_name}: HTTP {response.status_code} {response.text}"
            )


def isoformat(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def canonical_timestamp(value: Any) -> str:
    raw = isoformat(value).replace("Z", "+00:00")
    return datetime.fromisoformat(raw).astimezone(timezone.utc).isoformat()


def chunks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [rows[start : start + BATCH_SIZE] for start in range(0, len(rows), BATCH_SIZE)]


def upsert_in_batches(
    supabase: SupabaseRestClient,
    table: str,
    rows: list[dict[str, Any]],
    conflict_columns: str,
    *,
    ignore_duplicates: bool = False,
) -> None:
    batches = chunks(rows)
    for index, batch in enumerate(batches, start=1):
        supabase.upsert(table, batch, conflict_columns, ignore_duplicates=ignore_duplicates)
        print(f"{table}: lote {index}/{len(batches)} ({len(batch)} filas)")


def build_dataset_row(metadata: dict[str, Any]) -> dict[str, Any]:
    dataset = metadata["dataset"]
    files = dataset["files"]
    manifest = json.dumps(files, sort_keys=True).encode("utf-8")
    return {
        "dataset_name": dataset["dataset"],
        "source_url": os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io"),
        "api_version": metadata["api_version"],
        "content_hash": hashlib.sha256(manifest).hexdigest(),
        "cutoff_at": f"{dataset['history_end']}" if dataset.get("history_end") else datetime.now().isoformat(),
    }


def main() -> None:
    supabase_url, service_role_key = require_environment()
    supabase = SupabaseRestClient(supabase_url, service_role_key)
    try:
        with PulsoTransmiClient() as api:
            metadata = api.meta()
            dataset_row = build_dataset_row(metadata)
            dataset_response = supabase.upsert(
                "datasets", [dataset_row], "dataset_name,cutoff_at"
            )
            dataset = dataset_response[0] if dataset_response else supabase.find_dataset(
                dataset_row["dataset_name"], dataset_row["cutoff_at"]
            )
            dataset_id = dataset["dataset_id"]
            print(f"dataset: {dataset_row['dataset_name']} ({dataset_id})")

            stations = api.stations().to_dict("records")
            station_rows = [
                {
                    "station_id": row["station_id"],
                    "station_name": row["station_name"],
                    "corridor": row["corridor"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "active": True,
                }
                for row in stations
            ]
            upsert_in_batches(supabase, "stations", station_rows, "station_id")

            context_frame = api.context_dataframe()
            context_rows = [
                {
                    "observed_at": isoformat(row["observed_at"]),
                    "dataset_id": dataset_id,
                    "rain_mm": row["rain_mm"],
                    "rain_forecast": row["rain_forecast"],
                    "temperature_c": row["temperature_c"],
                    "temperature_forecast": row["temperature_forecast"],
                    "event_intensity": row["event_intensity"],
                }
                for row in context_frame.to_dict("records")
            ]
            upsert_in_batches(supabase, "context", context_rows, "observed_at")
            known_context_times = {canonical_timestamp(row["observed_at"]) for row in context_rows}

            observation_count = 0
            if not supabase.has_observations():
                # Bootstrap the complete history once; subsequent runs consume
                # released stream rows and advance the watermark only after writes.
                observations_frame = api.observations_dataframe()
                observation_rows = [
                    {
                        "dataset_id": dataset_id,
                        "station_id": str(row["station_id"]),
                        "observed_at": isoformat(row["observed_at"]),
                        "demand": int(row["demand"]),
                    }
                    for row in observations_frame.to_dict("records")
                ]
                upsert_in_batches(supabase, "observations", observation_rows, "station_id,observed_at")
                observation_count += len(observation_rows)

            watermark = supabase.get_watermark("observations")
            cursor = None
            seen_cursors: set[str] = set()
            latest_release = watermark
            while True:
                page = api.stream_observations_page(cursor=cursor, limit=STREAM_PAGE_SIZE)
                stream_rows = []
                missing_context: dict[str, dict[str, Any]] = {}
                for row in page.get("data", []):
                    released_at = datetime.fromisoformat(row["released_at"].replace("Z", "+00:00"))
                    if latest_release is None or released_at > latest_release:
                        latest_release = released_at
                    # Re-read equal-time rows so a partial failure cannot skip
                    # records sharing a release timestamp. Upserts are idempotent.
                    if watermark is not None and released_at < watermark:
                        continue
                    observed_at = canonical_timestamp(row["observed_at"])
                    stream_rows.append({
                        "dataset_id": dataset_id,
                        "station_id": str(row["station_id"]),
                        "observed_at": observed_at,
                        "demand": int(row["demand"]),
                    })
                    if observed_at not in known_context_times:
                        # The API stream can publish demand before its context
                        # series catches up. Store an explicit all-NULL context
                        # row to satisfy the FK; never carry stale weather forward.
                        missing_context[observed_at] = {
                            "observed_at": observed_at,
                            "dataset_id": dataset_id,
                            "rain_mm": None,
                            "rain_forecast": None,
                            "temperature_c": None,
                            "temperature_forecast": None,
                            "event_intensity": None,
                        }
                if missing_context:
                    upsert_in_batches(
                        supabase,
                        "context",
                        list(missing_context.values()),
                        "observed_at",
                        ignore_duplicates=True,
                    )
                    known_context_times.update(missing_context)
                upsert_in_batches(supabase, "observations", stream_rows, "station_id,observed_at")
                observation_count += len(stream_rows)
                next_cursor = page.get("next_cursor")
                if next_cursor is None:
                    break
                if next_cursor in seen_cursors:
                    raise SupabaseIngestionError("La API devolvió un cursor repetido en el stream.")
                seen_cursors.add(next_cursor)
                cursor = next_cursor

            if latest_release is not None and latest_release != watermark:
                # Move the timestamp cursor only after all page writes succeed.
                supabase.save_watermark("observations", latest_release)
            print(
                f"Carga completa: {len(station_rows)} estaciones, "
                f"{len(context_rows)} contextos, {observation_count} observaciones procesadas; "
                f"watermark={latest_release.isoformat() if latest_release else 'sin cambios'}."
            )
    finally:
        supabase.close()


if __name__ == "__main__":
    main()