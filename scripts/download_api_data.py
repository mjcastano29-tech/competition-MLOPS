"""Refresca `data/` desde la API pública, incluido el stream de la competencia.

`/v1/observations` y `/v1/downloads/*` siguen sirviendo solo el dataset inicial
(`pulso-transmi-starter-v1`). La demanda que se libera durante la competencia sale
exclusivamente por `/v1/stream/observations`, así que descargar únicamente el
starter deja el entrenamiento anclado al patrón viejo: el modelo se mide contra
datos que nunca vio. Aquí se descargan ambos y se fusionan.

Las marcas de tiempo se escriben en ISO-8601 UTC, que es el formato que ya usan
`scripts/download_supabase_data.py` y la inferencia. Si se guardaran con el offset
local (`-05:00`), las variables de calendario (`quarter_of_day`, `day_of_week`)
saldrían corridas 5 horas frente a las que calcula `scripts/infer_and_submit.py`
a partir de `data_cutoff`/`target_at` en UTC.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from pulso_transmi import PulsoTransmiClient

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STARTER_FILES = ("stations.csv", "observations.csv", "context.csv", "metadata.json")
STREAM_PAGE_SIZE = 5000


def fetch_stream_observations(client: PulsoTransmiClient) -> pd.DataFrame:
    """Recorre `/v1/stream/observations` completo y devuelve demanda + release."""

    rows: list[dict] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        page = client.stream_observations_page(cursor=cursor, limit=STREAM_PAGE_SIZE)
        rows.extend(page.get("data", []))
        cursor = page.get("next_cursor")
        if cursor is None:
            break
        if cursor in seen:
            raise RuntimeError("La API devolvió un cursor repetido en el stream.")
        seen.add(cursor)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["station_id", "observed_at", "demand", "released_at"])
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["released_at"] = pd.to_datetime(frame["released_at"], utc=True)
    frame["station_id"] = frame["station_id"].astype(str)
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
    return frame.dropna(subset=["demand", "observed_at"]).copy()


def merge_observations(starter: pd.DataFrame, stream: pd.DataFrame) -> pd.DataFrame:
    """Une starter y stream por (estación, instante); el stream manda en empates."""

    starter = starter.copy()
    starter["observed_at"] = pd.to_datetime(starter["observed_at"], utc=True)
    starter["station_id"] = starter["station_id"].astype(str)
    combined = pd.concat(
        [
            starter[["station_id", "observed_at", "demand"]],
            stream[["station_id", "observed_at", "demand"]],
        ],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset=["station_id", "observed_at"], keep="last")
    return combined.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def _as_utc_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station_id": str})
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    return frame


def _write_utc(frame: pd.DataFrame, destination: Path) -> None:
    frame.assign(
        observed_at=pd.to_datetime(frame["observed_at"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        )
    ).to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Descarga solo el dataset inicial, sin el stream de la competencia.",
    )
    args = parser.parse_args()
    api_key = os.getenv("PULSO_API_KEY")

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    with PulsoTransmiClient(api_key=api_key) as client:
        for filename in STARTER_FILES:
            destination = client.download(filename, data_dir / filename)
            print(f"Descargado {filename}: {destination}")
        if args.no_stream:
            return
        starter = _as_utc_frame(data_dir / "observations.csv")
        stream = fetch_stream_observations(client)
        context = _as_utc_frame(data_dir / "context.csv")

    merged = merge_observations(starter, stream)
    _write_utc(merged, data_dir / "observations.csv")
    _write_utc(context, data_dir / "context.csv")

    stream_start = stream["observed_at"].min() if not stream.empty else None
    print(
        f"Starter: {len(starter)} observaciones hasta {starter['observed_at'].max()}; "
        f"stream: {len(stream)} filas desde {stream_start}."
    )
    print(
        f"data/observations.csv fusionado: {len(merged)} filas "
        f"({merged['station_id'].nunique()} estaciones) hasta {merged['observed_at'].max()}."
    )
    print(
        f"Atención: /v1/context no tiene filas después de {context['observed_at'].max()}; "
        "el clima/eventos del periodo del stream quedan sin publicar."
    )


if __name__ == "__main__":
    main()
