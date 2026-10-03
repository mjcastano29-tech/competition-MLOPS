"""Lectura de la demanda de una fila del stream, en cualquiera de sus esquemas.

Esquema 1: {"station_id", "observed_at", "released_at", "demand": 360}
Esquema 2 (desde 2026-09-20 12:15 virtual, liberado el 2026-10-03):
    {"station_id", "observed_at", "released_at", "schema_version": 2,
     "measurement": {"value": "546.00", "unit": "passengers", "quality": "observed"}}

`demand` viene vacio en el esquema 2: leerlo a secas dejaba al modelo sin la demanda del
corte. La inferencia y el colector usan esta misma funcion para no divergir.
"""

from __future__ import annotations

import math
from typing import Any

# Unidad -> factor para expresar la demanda en pasajeros por cuarto de hora.
UNIT_FACTORS = {
    "passengers": 1.0,
    "passenger": 1.0,
    "pax": 1.0,
    "pasajeros": 1.0,
    "hundred_passengers": 100.0,
    "hundreds_of_passengers": 100.0,
    "thousand_passengers": 1000.0,
    "thousands_of_passengers": 1000.0,
    "kpassengers": 1000.0,
}
# Calidades que no representan una medicion utilizable.
UNUSABLE_QUALITY = {"missing", "invalid", "rejected", "error", "null", "unavailable"}
_warned: set[str] = set()


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(" ", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(f"::warning::{message}")


def row_demand(row: dict[str, Any]) -> float | None:
    """Demanda en pasajeros de una fila del stream, o None si no hay una medicion valida."""

    direct = _to_float(row.get("demand"))
    if direct is not None:
        return direct
    measurement = row.get("measurement")
    if not isinstance(measurement, dict):
        return None
    quality = str(measurement.get("quality") or "observed").strip().lower()
    if quality in UNUSABLE_QUALITY:
        return None
    value = _to_float(measurement.get("value"))
    if value is None:
        return None
    unit = str(measurement.get("unit") or "passengers").strip().lower()
    factor = UNIT_FACTORS.get(unit)
    if factor is None:
        _warn_once(f"Unidad de demanda desconocida en el stream: {unit!r}; se usa el valor tal cual.")
        factor = 1.0
    if quality not in {"observed", "estimated", "imputed", "provisional"}:
        _warn_once(f"Calidad de medicion desconocida en el stream: {quality!r}; se usa el valor.")
    return value * factor
