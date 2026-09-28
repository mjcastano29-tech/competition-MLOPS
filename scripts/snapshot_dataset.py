"""Congela el dataset de entrenamiento en un snapshot inmutable y versionado.

Sin esto, "reentrenar" significa entrenar sobre lo que Supabase tenga en ese
instante: dos ejecuciones del mismo workflow producen modelos con datos
distintos y metricas incomparables. El snapshot genera un hash canonico de las
filas, lo guarda en `public.datasets` y deja un puntero local (`data/snapshot.json`)
que el entrenamiento y el empaquetado propagan hasta el registro de modelos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "datasets"
DEFAULT_POINTER = ROOT / "data" / "snapshot.json"
DEFAULT_DATA_DIR = ROOT / "data"
SNAPSHOT_BUCKET = os.getenv("SNAPSHOT_BUCKET", "dataset-snapshots")
OBSERVATION_COLUMNS = ["station_id", "observed_at", "demand"]
CONTEXT_COLUMNS = [
    "observed_at",
    "rain_mm",
    "rain_forecast",
    "temperature_c",
    "temperature_forecast",
    "event_intensity",
]
EXPECTED_STATION_COUNT = 12
MINIMUM_ROWS = 5000


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _normalized_observations(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    observations = frame.loc[:, OBSERVATION_COLUMNS].copy()
    observations["station_id"] = observations["station_id"].astype(str)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    observations["demand"] = pd.to_numeric(observations["demand"], errors="coerce")
    if observations["demand"].isna().any():
        raise ValueError("Hay observaciones con demanda no numerica; el snapshot seria ambiguo.")
    observations = observations.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    duplicates = int(observations.duplicated(subset=["station_id", "observed_at"]).sum())
    if duplicates:
        observations = observations.drop_duplicates(subset=["station_id", "observed_at"], keep="last")
    return observations, duplicates


def _normalized_context(frame: pd.DataFrame) -> pd.DataFrame:
    context = frame.loc[:, CONTEXT_COLUMNS].copy()
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)
    for column in CONTEXT_COLUMNS[1:]:
        context[column] = pd.to_numeric(context[column], errors="coerce").fillna(0.0)
    context = context.sort_values("observed_at").reset_index(drop=True)
    duplicates = int(context.duplicated(subset=["observed_at"]).sum())
    if duplicates:
        context = context.drop_duplicates(subset=["observed_at"], keep="last")
    return context, duplicates


def _canonical_hash(observations: pd.DataFrame, context: pd.DataFrame) -> str:
    """Hash sha256 sobre las filas normalizadas, independiente del orden de lectura."""

    digest = hashlib.sha256()
    digest.update(b"observations:v1\n")
    for station_id, observed_at, demand in observations.itertuples(index=False, name=None):
        digest.update(f"{station_id}|{observed_at.isoformat()}|{float(demand):.6f}\n".encode())
    digest.update(b"context:v1\n")
    for row in context.itertuples(index=False, name=None):
        stamp = row[0].isoformat()
        values = "|".join(f"{float(value):.6f}" for value in row[1:])
        digest.update(f"{stamp}|{values}\n".encode())
    return digest.hexdigest()


def load_supabase(start_at: str | None = None, url: str | None = None, key: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.supabase_client import SupabaseRest

    rest = SupabaseRest(url, key)
    observation_filters = {"observed_at": f"gte.{start_at}"} if start_at else None
    context_filters = {"observed_at": f"gte.{start_at}"} if start_at else None
    observation_rows: list[dict[str, Any]] = []
    for page in rest.pages(
        "observations",
        columns="station_id,observed_at,demand",
        filters=observation_filters,
        order="station_id.asc,observed_at.asc",
    ):
        observation_rows.extend(page)
    context_rows: list[dict[str, Any]] = []
    for page in rest.pages(
        "context",
        columns=",".join(CONTEXT_COLUMNS),
        filters=context_filters,
        order="observed_at.asc",
    ):
        context_rows.extend(page)
    if not observation_rows or not context_rows:
        raise SystemExit("Supabase no tiene observaciones o contexto; ejecuta el colector antes de crear el snapshot.")
    print(f"Observaciones leidas: {len(observation_rows)}; contexto: {len(context_rows)}.")
    return pd.DataFrame(observation_rows), pd.DataFrame(context_rows)


def load_local(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations_path = data_dir / "observations.csv"
    context_path = data_dir / "context.csv"
    if not observations_path.exists() or not context_path.exists():
        raise SystemExit(f"No hay CSV en {data_dir}; no se puede crear un snapshot local.")
    return (
        pd.read_csv(observations_path, dtype={"station_id": str}),
        pd.read_csv(context_path),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot(
    observations: pd.DataFrame,
    context: pd.DataFrame,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source: str = "supabase",
    start_at: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Escribe `observations.csv`, `context.csv` y `manifest.json` de un snapshot."""

    observations, observation_duplicates = _normalized_observations(observations)
    context, context_duplicates = _normalized_context(context)
    rows_hash = _canonical_hash(observations, context)
    cutoff_at = observations["observed_at"].max()
    observations_start = observations["observed_at"].min()
    stations = sorted(observations["station_id"].unique())
    created = created_at or datetime.now(timezone.utc)

    snapshot_dir = output_root / rows_hash[:16]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    observations_path = snapshot_dir / "observations.csv"
    context_path = snapshot_dir / "context.csv"
    observations.assign(observed_at=observations["observed_at"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")).to_csv(
        observations_path, index=False
    )
    context.assign(observed_at=context["observed_at"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")).to_csv(
        context_path, index=False
    )

    if len(stations) < EXPECTED_STATION_COUNT:
        raise ValueError(
            f"El snapshot tiene {len(stations)} estaciones y se exigen {EXPECTED_STATION_COUNT}: "
            "el entrenamiento las rechazaría."
        )
    if len(observations) < MINIMUM_ROWS:
        raise ValueError(f"El snapshot tiene {len(observations)} filas, menos del minimo {MINIMUM_ROWS}.")

    manifest = {
        "schema_version": 1,
        "dataset_name": f"obs-{cutoff_at.strftime('%Y%m%dT%H%M%SZ')}-{rows_hash[:8]}",
        "rows_hash": rows_hash,
        "cutoff_at": cutoff_at.isoformat(),
        "observations_start": observations_start.isoformat(),
        "row_count": int(len(observations)),
        "context_rows": int(len(context)),
        "station_count": int(len(stations)),
        "stations": stations,
        "duplicate_rows_dropped": observation_duplicates + context_duplicates,
        "source": source,
        "start_at": start_at,
        "created_at": created.isoformat(),
        "git_commit": git_commit(),
        "directory": str(snapshot_dir.relative_to(ROOT)) if snapshot_dir.is_relative_to(ROOT) else str(snapshot_dir),
        "files": {
            "observations.csv": _file_sha256(observations_path),
            "context.csv": _file_sha256(context_path),
        },
        "supabase": {"dataset_id": None, "snapshot_uri": None},
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def register_snapshot(manifest: dict[str, Any], *, url: str | None = None, key: str | None = None) -> str:
    """Registra el snapshot en `public.datasets`; el hash de filas evita duplicados."""

    from scripts.supabase_client import SupabaseRest

    rest = SupabaseRest(url, key)
    existing = rest.select(
        "datasets",
        columns="dataset_id",
        filters={"rows_hash": f"eq.{manifest['rows_hash']}"},
        limit=1,
    )
    if existing:
        print(f"El snapshot {manifest['rows_hash'][:12]} ya estaba registrado como dataset {existing[0]['dataset_id']}.")
        return str(existing[0]["dataset_id"])

    row = {
        "dataset_name": manifest["dataset_name"],
        "cutoff_at": manifest["cutoff_at"],
        "content_hash": manifest["rows_hash"],
        "rows_hash": manifest["rows_hash"],
        "api_version": manifest["source"],
        "row_count": manifest["row_count"],
        "context_rows": manifest["context_rows"],
        "station_count": manifest["station_count"],
        "observations_start": manifest["observations_start"],
        "git_commit": manifest["git_commit"],
        "snapshot_uri": manifest["supabase"].get("snapshot_uri"),
    }
    created = rest.insert("datasets", [row])
    if not created:
        raise SystemExit("No se pudo registrar el snapshot en public.datasets.")
    dataset_id = str(created[0]["dataset_id"])
    print(f"Snapshot {manifest['dataset_name']} registrado como dataset {dataset_id}.")
    return dataset_id


def publish_snapshot(
    manifest: dict[str, Any],
    *,
    bucket: str = SNAPSHOT_BUCKET,
    url: str | None = None,
    key: str | None = None,
) -> str:
    """Sube el snapshot comprimido a Storage para poder reproducirlo despues."""

    from scripts.supabase_client import SupabaseRest

    snapshot_dir = (ROOT / manifest["directory"]).resolve()
    object_key = f"snapshots/{manifest['rows_hash'][:16]}/snapshot.tar.gz"
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "snapshot.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(snapshot_dir, arcname=manifest["rows_hash"][:16])
        payload = archive.read_bytes()
    rest = SupabaseRest(url, key)
    rest.upload_object(bucket, object_key, payload, content_type="application/gzip")
    print(f"Snapshot publicado en {bucket}/{object_key} ({len(payload)} bytes).")
    return object_key


def write_pointer(manifest: dict[str, Any], pointer: Path = DEFAULT_POINTER) -> Path:
    """Deja el puntero que el entrenamiento y el empaquetado deben leer."""

    directory = ROOT / manifest["directory"]
    pointer = Path(pointer)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps(
            {
                "dataset_name": manifest["dataset_name"],
                "rows_hash": manifest["rows_hash"],
                "cutoff_at": manifest["cutoff_at"],
                "observations_start": manifest["observations_start"],
                "row_count": manifest["row_count"],
                "dataset_id": manifest["supabase"].get("dataset_id"),
                "directory": manifest["directory"],
                "observations": str(directory / "observations.csv"),
                "context": str(directory / "context.csv"),
                "manifest": str(directory / "manifest.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return pointer


def load_snapshot_manifest(pointer: Path = DEFAULT_POINTER) -> dict[str, Any] | None:
    """Lee el puntero de snapshot; `None` cuando se entrena con `data/*.csv`.

    Si el puntero apunta a un snapshot que ya no existe localmente se falla en
    vez de caer silenciosamente a los CSV sueltos, que pueden ser de otra fecha.
    """

    pointer = Path(pointer)
    if not pointer.exists():
        return None
    manifest = json.loads(pointer.read_text())
    missing = [name for name in ("observations", "context") if not Path(manifest[name]).exists()]
    if missing:
        raise SystemExit(
            f"data/snapshot.json apunta a un snapshot con archivos faltantes: {missing}. "
            "Vuelva a crear el snapshot antes de entrenar."
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("supabase", "local"), default="supabase")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Origen cuando --source=local.")
    parser.add_argument("--start-at", help="Recorta el snapshot a observaciones >= este timestamp ISO.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports" / "dataset_snapshot.json")
    parser.add_argument("--no-register", action="store_true", help="No escribe en public.datasets.")
    parser.add_argument("--no-pointer", action="store_true", help="No escribe data/snapshot.json.")
    parser.add_argument("--publish", action="store_true", help="Sube el snapshot comprimido a Storage.")
    parser.add_argument("--bucket", default=SNAPSHOT_BUCKET)
    parser.add_argument("--url", default=None)
    parser.add_argument("--key", default=None)
    args = parser.parse_args()

    if args.source == "local":
        observations, context = load_local(args.data_dir)
    else:
        observations, context = load_supabase(args.start_at, args.url, args.key)

    manifest = build_snapshot(
        observations,
        context,
        output_root=args.output_root,
        source=args.source,
        start_at=args.start_at,
    )
    print(
        f"Snapshot {manifest['dataset_name']}: {manifest['row_count']} observaciones, "
        f"{manifest['station_count']} estaciones, corte {manifest['cutoff_at']}, "
        f"hash {manifest['rows_hash'][:12]}."
    )
    if not args.no_register and args.source == "supabase":
        manifest["supabase"]["dataset_id"] = register_snapshot(manifest, url=args.url, key=args.key)
    if args.publish:
        manifest["supabase"]["snapshot_uri"] = publish_snapshot(
            manifest, bucket=args.bucket, url=args.url, key=args.key
        )
        if not args.no_register and manifest["supabase"]["dataset_id"]:
            from scripts.supabase_client import SupabaseRest

            SupabaseRest(args.url, args.key).update(
                "datasets",
                {"snapshot_uri": manifest["supabase"]["snapshot_uri"]},
                filters={"dataset_id": f"eq.{manifest['supabase']['dataset_id']}"},
            )
    if not args.no_pointer:
        write_pointer(manifest, args.pointer)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Manifiesto escrito en {args.output_json}.")


if __name__ == "__main__":
    main()


