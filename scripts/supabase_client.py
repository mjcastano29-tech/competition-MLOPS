"""Cliente REST minimo de Supabase (PostgREST + Storage) sin dependencias extra.

Se apoya solo en `httpx`, que ya es traduccion directa del SDK usado en el
repositorio, y centraliza las convenciones que antes estaban duplicadas en los
scripts de ingesta y descarga.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import httpx

DEFAULT_SUPABASE_URL = "https://jwlgxabibcticikhjhzf.supabase.co"
REQUIRED_KEY_PREFIXES = ("sb_secret_", "eyJ")
SERVICE_KEY_MISSING = (
    "Falta SUPABASE_SERVICE_ROLE_KEY. Defina la service key de Supabase como "
    "variable de entorno antes de escribir en el registro de modelos."
)


def resolve_url(url: str | None = None) -> str:
    return (url or os.getenv("SUPABASE_URL") or DEFAULT_SUPABASE_URL).rstrip("/")


def resolve_key(key: str | None = None) -> str:
    value = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
    if not value:
        raise SystemExit(SERVICE_KEY_MISSING)
    if not value.startswith(REQUIRED_KEY_PREFIXES):
        raise SystemExit(
            "SUPABASE_SERVICE_ROLE_KEY debe ser una service key (sb_secret_... o JWT). "
            "La anon key no puede escribir en el registro de modelos."
        )
    return value


def service_headers(key: str) -> dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


class SupabaseRest:
    """Accesores delgados sobre PostgREST y Storage con autenticacion de servicio."""

    def __init__(self, url: str | None = None, key: str | None = None, *, timeout: float = 180.0) -> None:
        self.url = resolve_url(url)
        self.key = resolve_key(key)
        self.headers = service_headers(self.key)
        self.timeout = timeout

    def _client(self, extra: dict[str, str] | None = None) -> httpx.Client:
        headers = dict(self.headers)
        headers.update(extra or {})
        return httpx.Client(headers=headers, timeout=self.timeout)

    @staticmethod
    def _query(
        columns: str,
        filters: dict[str, str] | None,
        order: str | None,
        limit: int | None,
        offset: int | None,
    ) -> dict[str, str]:
        params: dict[str, str] = {"select": columns}
        params.update(filters or {})
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        return params

    def select(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: dict[str, str] | None = None,
        order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        params = self._query(columns, filters, order, limit, offset)
        with self._client() as client:
            response = client.get(f"{self.url}/rest/v1/{table}", params=params)
            response.raise_for_status()
            return response.json()

    def pages(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: dict[str, str] | None = None,
        order: str | None = None,
        page_size: int = 1000,
    ) -> Iterator[list[dict[str, Any]]]:
        offset = 0
        while True:
            rows = self.select(
                table,
                columns=columns,
                filters=filters,
                order=order,
                limit=page_size,
                offset=offset,
            )
            if not rows:
                return
            yield rows
            if len(rows) < page_size:
                return
            offset += page_size

    def insert(
        self,
        table: str,
        rows: list[dict[str, Any]],
        *,
        on_conflict: str | None = None,
        return_minimal: bool = False,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        params = {"on_conflict": on_conflict} if on_conflict else None
        headers = {"Prefer": "return=minimal" if return_minimal else "return=representation"}
        with self._client(headers) as client:
            response = client.post(f"{self.url}/rest/v1/{table}", params=params, json=rows)
            response.raise_for_status()
            return response.json() if response.content else []

    def upsert(self, table: str, rows: list[dict[str, Any]], *, on_conflict: str) -> list[dict[str, Any]]:
        with self._client({"Prefer": "return=representation,resolution=merge-duplicates"}) as client:
            response = client.post(
                f"{self.url}/rest/v1/{table}", params={"on_conflict": on_conflict}, json=rows
            )
            response.raise_for_status()
            return response.json() if response.content else []

    def update(self, table: str, values: dict[str, Any], *, filters: dict[str, str]) -> list[dict[str, Any]]:
        with self._client({"Prefer": "return=representation"}) as client:
            response = client.patch(
                f"{self.url}/rest/v1/{table}",
                params=self._query("*", filters, None, None, None),
                json=values,
            )
            response.raise_for_status()
            return response.json() if response.content else []

    def rpc(self, function: str, arguments: dict[str, Any]) -> Any:
        with self._client() as client:
            response = client.post(f"{self.url}/rest/v1/rpc/{function}", json=arguments)
            response.raise_for_status()
            return response.json() if response.content else None

    def upload_object(self, bucket: str, object_key: str, payload: bytes, *, content_type: str) -> str:
        with self._client({"Content-Type": content_type, "x-upsert": "true"}) as client:
            response = client.put(
                f"{self.url}/storage/v1/object/{bucket}/{quote(object_key)}",
                content=payload,
            )
            response.raise_for_status()
            return response.json().get("Key", object_key)

    def download_object(self, bucket: str, object_key: str, destination: str | Path) -> Path:
        path = Path(destination)
        with self._client({"Accept": "*/*"}) as client:
            response = client.get(f"{self.url}/storage/v1/object/{bucket}/{quote(object_key)}")
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        return path

