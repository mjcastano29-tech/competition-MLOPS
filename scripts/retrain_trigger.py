"""Decide si hay que adelantar un reentrenamiento: drift fuerte o campeon viejo.

Corre despues de cada colector (una vez por hora, disparado desde Supabase), asi que no
depende del cron de GitHub, que llega con horas de retraso o no llega. Lee las mismas
senales que el dashboard (`forecast_dashboard()`, ancladas al reloj virtual) y el
historial de runs de `retrain_on_drift.yml` que le pasa el workflow.

Senales:
  * cambio de nivel reciente: la demanda de las ultimas 24 h de una estacion se separa
    >= LEVEL_SHIFT de su nivel de los 7 dias previos (drift que el campeon no vio);
  * degradacion: la accuracy de las ultimas 6 h cae >= DROP_6H pts bajo la de 24 h, o la
    de 24 h cae >= DROP_24H pts bajo las 24 h anteriores;
  * campeon viejo: ningun reentrenamiento empezo en las ultimas STALE_HOURS horas.

Nunca despacha si hay un reentrenamiento en curso o si el ultimo empezo hace menos de
COOLDOWN_HOURS: un candidato tarda ~25 min y la compuerta ya protege al campeon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

LEVEL_SHIFT = 0.25
DROP_6H = 5.0
DROP_24H = 3.0
STALE_HOURS = 6.0
COOLDOWN_HOURS = 3.0
DEFAULT_URL = "https://jwlgxabibcticikhjhzf.supabase.co"


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def drift_signals(dashboard: dict[str, Any]) -> list[str]:
    """Razones de drift en el payload v2 de `forecast_dashboard()`; vacio si no hay."""

    reasons: list[str] = []
    for station in dashboard.get("stations") or []:
        recent, previous = station.get("level_24h"), station.get("level_prev_7d")
        if recent is None or not previous:
            continue
        shift = recent / previous - 1
        if abs(shift) >= LEVEL_SHIFT:
            reasons.append(
                f"{station.get('station_name') or station.get('station_id')}: nivel {shift:+.0%} "
                "frente a sus 7 dias previos"
            )
    windows = dashboard.get("windows") or {}
    last_6h = (windows.get("last_6h") or {}).get("accuracy")
    rolling = (windows.get("rolling_24h") or {}).get("accuracy")
    previous = (windows.get("previous_24h") or {}).get("accuracy")
    if last_6h is not None and rolling is not None and rolling - last_6h >= DROP_6H:
        reasons.append(f"accuracy 6 h {last_6h:.1f} vs 24 h {rolling:.1f}")
    if rolling is not None and previous is not None and previous - rolling >= DROP_24H:
        reasons.append(f"accuracy 24 h {rolling:.1f} vs 24 h previas {previous:.1f}")
    return reasons


def decide(
    dashboard: dict[str, Any], runs: list[dict[str, Any]], now: datetime
) -> tuple[bool, str]:
    """`(despachar, motivo)` a partir de las senales y del historial de reentrenamientos."""

    if any(run.get("status") in {"queued", "in_progress", "waiting", "pending"} for run in runs):
        return False, "hay un reentrenamiento en curso"
    starts = [start for run in runs if (start := _timestamp(run.get("createdAt")))]
    last_start = max(starts) if starts else None
    since_last = (now - last_start) if last_start else None
    if since_last is not None and since_last < timedelta(hours=COOLDOWN_HOURS):
        return False, f"ultimo reentrenamiento hace {since_last.total_seconds() / 3600:.1f} h (enfriamiento {COOLDOWN_HOURS:.0f} h)"
    reasons = drift_signals(dashboard)
    if reasons:
        return True, "drift: " + "; ".join(reasons[:4])
    if since_last is None or since_last >= timedelta(hours=STALE_HOURS):
        age = "nunca" if since_last is None else f"hace {since_last.total_seconds() / 3600:.1f} h"
        return True, f"campeon viejo: ultimo reentrenamiento {age}"
    return False, "sin drift y campeon reciente"


def fetch_dashboard() -> dict[str, Any]:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise RuntimeError("Falta SUPABASE_SERVICE_ROLE_KEY para leer las senales de drift.")
    headers = {"apikey": key, "Content-Type": "application/json"}
    if not key.startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {key}"
    url = os.getenv("SUPABASE_URL", DEFAULT_URL).rstrip("/") + "/rest/v1/rpc/forecast_dashboard"
    response = httpx.post(url, headers=headers, content="{}", timeout=60.0)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="JSON de `gh run list` del workflow de reentrenamiento.")
    args = parser.parse_args()
    runs = json.loads(args.runs.read_text(encoding="utf-8") or "[]")
    dispatch, reason = decide(fetch_dashboard(), runs, datetime.now(timezone.utc))
    print(f"Reentrenar: {'si' if dispatch else 'no'} ({reason})")
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"dispatch={'true' if dispatch else 'false'}\n")
            stream.write(f"reason={reason.replace(chr(10), ' ')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
