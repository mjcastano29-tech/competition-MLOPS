from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

DEFAULT_URL = "https://jwlgxabibcticikhjhzf.supabase.co"
PAGE_SIZE = 1000


def supabase_request_headers(api_key: str) -> dict[str, str]:
    headers = {"apikey": api_key, "Content-Type": "application/json"}
    # New sb_secret keys are API keys, not JWTs; legacy service_role keys are JWTs.
    if not api_key.startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class Supabase:
    def __init__(self) -> None:
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not key:
            raise RuntimeError("Falta SUPABASE_SERVICE_ROLE_KEY en GitHub Actions.")
        self.client = httpx.Client(
            base_url=os.getenv("SUPABASE_URL", DEFAULT_URL).rstrip("/") + "/rest/v1",
            headers=supabase_request_headers(key),
            timeout=60,
        )

    def rows(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        start = 0
        while True:
            response = self.client.get(f"/{table}", params={**params, "limit": str(PAGE_SIZE), "offset": str(start)})
            if response.is_error:
                raise RuntimeError(f"Supabase {table}: HTTP {response.status_code}: {response.text}")
            page = response.json()
            result.extend(page)
            if len(page) < PAGE_SIZE:
                return result
            start += PAGE_SIZE

    def insert(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(f"/{table}", headers={"Prefer": "return=representation"}, json=row)
        if response.is_error:
            raise RuntimeError(f"Supabase {table}: HTTP {response.status_code}: {response.text}")
        payload = response.json()
        return payload[0] if payload else {}


def wape(pairs: list[tuple[str, float, float]]) -> float | None:
    by_station: dict[str, list[tuple[float, float]]] = {}
    for station_id, predicted, actual in pairs:
        by_station.setdefault(station_id, []).append((predicted, actual))
    scores = []
    for station_pairs in by_station.values():
        denominator = sum(actual for _, actual in station_pairs)
        if denominator > 0:
            scores.append(sum(abs(actual - predicted) for predicted, actual in station_pairs) / denominator)
    return sum(scores) / len(scores) if scores else None


def monitor(args: argparse.Namespace) -> dict[str, Any]:
    supabase = Supabase()
    now = datetime.now(timezone.utc)
    # Allow time for the half-hour collector to ingest the realized target.
    mature_before = now - timedelta(minutes=args.maturity_minutes)
    current_start = mature_before - timedelta(days=args.window_days)
    reference_start = current_start - timedelta(days=args.window_days)
    predictions = supabase.rows("forecast_predictions", {
        "select": "cycle_id,station_id,target_at,predicted_demand",
        "submission_id": "not.is.null",
        "target_at": f"gte.{reference_start.isoformat()}",
        "order": "target_at.asc",
    })
    actuals = supabase.rows("observations", {
        "select": "station_id,observed_at,demand",
        "observed_at": f"gte.{reference_start.isoformat()}",
        "and": f"(observed_at.lt.{mature_before.isoformat()})",
        "order": "observed_at.asc",
    })
    actual_by_key = {(str(row["station_id"]), timestamp(row["observed_at"])): float(row["demand"]) for row in actuals}
    reference: list[tuple[str, float, float]] = []
    current: list[tuple[str, float, float]] = []
    for prediction in predictions:
        target = timestamp(prediction["target_at"])
        key = (str(prediction["station_id"]), target)
        actual = actual_by_key.get(key)
        if actual is None or target >= mature_before:
            continue
        pair = (key[0], float(prediction["predicted_demand"]), actual)
        (current if target >= current_start else reference).append(pair)

    reference_wape = wape(reference)
    current_wape = wape(current)
    relative = None
    if reference_wape is not None and reference_wape > 0 and current_wape is not None:
        relative = current_wape / reference_wape - 1
    reference_stations = len({station for station, _, _ in reference})
    current_stations = len({station for station, _, _ in current})
    enough = (
        len(reference) >= args.min_samples and len(current) >= args.min_samples
        and reference_stations >= args.min_stations and current_stations >= args.min_stations
    )
    detected = bool(enough and relative is not None and relative >= args.threshold)
    result: dict[str, Any] = {
        "metric": "WAPE",
        "threshold_relative_increase": args.threshold,
        "window_days": args.window_days,
        "maturity_minutes": args.maturity_minutes,
        "reference_window": {"start": reference_start.isoformat(), "end": current_start.isoformat(), "samples": len(reference), "stations": reference_stations, "wape": reference_wape},
        "current_window": {"start": current_start.isoformat(), "end": mature_before.isoformat(), "samples": len(current), "stations": current_stations, "wape": current_wape},
        "relative_increase": relative,
        "minimum_samples_per_window": args.min_samples,
        "minimum_stations_per_window": args.min_stations,
        "drift_detected": detected,
        "reason": "threshold_exceeded" if detected else ("insufficient_matured_samples" if not enough else "within_threshold"),
        "created_at": now.isoformat(),
    }
    supabase.insert("wape_drift_checks", {
        "current_window_start": current_start.isoformat(),
        "current_window_end": mature_before.isoformat(),
        "reference_wape": reference_wape,
        "current_wape": current_wape,
        "relative_increase": relative,
        "threshold": args.threshold,
        "reference_count": len(reference),
        "current_count": len(current),
        "drift_detected": detected,
        "details": result,
    })
    return result


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pending() -> dict[str, Any] | None:
    rows = Supabase().rows("wape_drift_checks", {
        "select": "drift_check_id,current_wape,reference_wape,relative_increase,threshold,created_at",
        "drift_detected": "eq.true", "processed_at": "is.null", "order": "created_at.asc",
    })
    return rows[0] if rows else None


def mark(check_id: int) -> None:
    supabase = Supabase()
    response = supabase.client.patch(
        "/wape_drift_checks", params={"drift_check_id": f"lte.{check_id}", "processed_at": "is.null"},
        headers={"Prefer": "return=minimal"}, json={"processed_at": datetime.now(timezone.utc).isoformat()},
    )
    if response.is_error:
        raise RuntimeError(f"No se pudo cerrar la alerta de drift: HTTP {response.status_code}: {response.text}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitorea degradación de WAPE con predicciones oficiales ya maduras.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--threshold", type=float, default=0.05, help="Aumento relativo mínimo de WAPE; 0.05 activa con WAPE reciente >= 105% de la referencia.")
    check.add_argument("--window-days", type=int, default=7)
    check.add_argument("--min-samples", type=int, default=120)
    check.add_argument("--min-stations", type=int, default=10)
    check.add_argument("--maturity-minutes", type=int, default=45)
    check.add_argument("--output", default="artifacts/wape_drift_report.json")
    sub.add_parser("pending")
    finish = sub.add_parser("mark")
    finish.add_argument("--id", type=int, required=True)
    args = parser.parse_args()
    if args.command == "check":
        result = monitor(args)
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(json.dumps(result, indent=2))
    elif args.command == "pending":
        row = pending()
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(f"drift_pending={'true' if row else 'false'}\n")
            stream.write(f"drift_check_id={row['drift_check_id'] if row else ''}\n")
        if row:
            print(json.dumps(row, indent=2))
    else:
        mark(args.id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
