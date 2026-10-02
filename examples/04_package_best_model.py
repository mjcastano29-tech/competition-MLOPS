from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ar_baseline import build_normal_profile, save_profile  # noqa: E402
from scripts.relative_model import (  # noqa: E402
    RELATIVE_OFFSET,
    RELATIVE_REFERENCE,
    RELATIVE_WEIGHT,
    relative_model_path,
    split_candidate_name,
)
from scripts.infer_and_submit import unsupported_feature_columns  # noqa: E402
from scripts.model_gate import (  # noqa: E402
    CANDIDATE_SOURCE,
    CHAMPION_SOURCE,
    REQUIRED_HORIZONS,
    evaluation_from_frame,
    evaluation_to_dict,
    select_per_horizon,
)


REPORT_PATH = ROOT / "reports/ml_validation_metrics.csv"
WINDOW_REPORT_PATH = ROOT / "reports/validation_window.json"
PERSISTENCE_WEIGHTS_PATH = ROOT / "reports/persistence_weights.json"
AR_WEIGHTS_PATH = ROOT / "reports/ar_weights.json"
PACKAGE_DIR = ROOT / "artifacts/pulso_transmi_best_models"
PACKAGE_PATH = ROOT / "artifacts/pulso_transmi_best_models.zip"
PROMOTION_INPUTS_PATH = PACKAGE_DIR / "promotion_inputs.json"
EXPERIMENT = "pulso-transmi-forecasting"
# El run de entrenamiento puntua al campeon sobre las mismas folds y lo etiqueta asi.
INCUMBENT_PREFIX = "Champion "
# Paquete del campeon activo que el workflow restaura antes de entrenar. De aqui se
# copian los modelos de los horizontes que el campeon gane en la ventana emparejada.
PREVIOUS_DIR = ROOT / "artifacts" / "previous_model"


def load_training_module():
    module_path = ROOT / "examples/03_gradient_boosting.py"
    spec = importlib.util.spec_from_file_location("gradient_boosting", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_window_report(path: Path = WINDOW_REPORT_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empaqueta el mejor modelo por horizonte sin quedar bajo el campeon."
    )
    parser.add_argument(
        "--previous-model",
        type=Path,
        default=PREVIOUS_DIR,
        help="Paquete del campeon activo, para conservar los horizontes que gane.",
    )
    return parser.parse_args()


def incumbent_label(metrics: pd.DataFrame) -> str | None:
    """Etiqueta `Champion <version>` con la que 03 guardo las filas del campeon."""

    scored = metrics.loc[metrics["model"].astype(str).str.startswith(INCUMBENT_PREFIX)]
    if scored.empty:
        return None
    return str(scored["model"].dropna().unique()[0])


def champion_package_horizons(previous_dir: Path, champion_label: str | None) -> list[int]:
    """Horizontes del campeon reutilizables: modelo, config y columnas construibles.

    Si falta el modelo o la config se deja fuera el horizonte: conservar a medias
    rompería la inferencia, y es preferible reentrenar ese horizonte a empaquetar un
    binomio roto. Tambien se descarta cuando el config pide columnas que la inferencia
    de hoy no sabe construir: un horizonte conservado con ceros silenciosos es justo lo
    que dejo al cron de envios sin predicciones durante una hora.
    """

    if champion_label is None or not previous_dir.is_dir():
        return []
    reusable: list[int] = []
    for horizon_minutes in REQUIRED_HORIZONS:
        model_path = previous_dir / "models" / f"horizon_{horizon_minutes}_hgb.pkl"
        config_path = previous_dir / "configs" / f"horizon_{horizon_minutes}_ensemble.json"
        if not (model_path.exists() and config_path.exists()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("relative_weight") and not relative_model_path(model_path).exists():
            print(f"  - horizonte {horizon_minutes} min fuera de la conservacion: falta su arbol relativo.")
            continue
        columns = list(config.get("feature_columns") or [])
        unknown = unsupported_feature_columns(columns)
        if not columns or unknown:
            print(
                f"  - horizonte {horizon_minutes} min fuera de la conservacion: columnas no "
                f"construibles en inferencia ({unknown or 'sin feature_columns'})."
            )
            continue
        reusable.append(int(horizon_minutes))
    return reusable


def copy_champion_horizon(previous_dir: Path, horizon_minutes: int) -> list[Path]:
    """Copia modelo y config del campeon en un horizonte que el candidato no supero.

    Se copian tal cual, con sus propias `feature_columns` y `baseline_lag`: es el mismo
    binomio que ya esta sirviendo predicciones, asi que la entrega no cambia de
    protocolo y medir ese horizonte vuelve a dar exactamente la metrica del campeon.
    """

    sources = [
        previous_dir / "models" / f"horizon_{horizon_minutes}_hgb.pkl",
        previous_dir / "configs" / f"horizon_{horizon_minutes}_ensemble.json",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise RuntimeError(f"El campeon no trae el horizonte {horizon_minutes}: {missing}.")
    # Un horizonte relativo trae un segundo arbol: sin el, la inferencia no podria servirlo.
    relative = relative_model_path(sources[0])
    if json.loads(sources[1].read_text(encoding="utf-8")).get("relative_weight"):
        if not relative.exists():
            raise RuntimeError(f"El campeon declara un modelo relativo sin {relative}.")
        sources.append(relative)
    copied: list[Path] = []
    for source in sources:
        destination = PACKAGE_DIR / source.relative_to(previous_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def unbuildable_package_features(package_dir: Path) -> dict[str, list[str]]:
    """Configs del paquete que piden columnas que la inferencia no sabe construir.

    Un paquete puede mezclar horizontes del candidato y del campeon, cada uno con su
    propia lista de columnas; esta es la lista negra por nombre/prefijo que comparte la
    inferencia. Se devuelve el mapa `config -> columnas rotas` para que el llamador decida.
    """

    broken: dict[str, list[str]] = {}
    for config_path in sorted((package_dir / "configs").glob("horizon_*_ensemble.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        columns = list(config.get("feature_columns") or [])
        unknown = unsupported_feature_columns(columns)
        if not columns or unknown:
            broken[config_path.name] = unknown or ["<sin feature_columns>"]
    return broken


def promotion_inputs(
    metrics: pd.DataFrame,
    best_names: dict[int, str],
    *,
    candidate_version: str,
    window_report: dict,
    snapshot: dict,
) -> dict:
    """Empareja candidato y campeon sobre las MISMAS filas de validacion.

    El run de entrenamiento deja en `ml_validation_metrics.csv` las filas
    `Champion <version>`: con ellas la promocion se decide comparando dos modelos
    sobre ventanas identicas en lugar de comparar metricas de epocas distintas,
    que fue la causa de la degradacion silenciosa. El resultado viaja dentro del
    paquete para que la decision sea auditable despues.
    """

    chosen = metrics.apply(
        lambda row: str(row["model"]) == best_names.get(int(row["horizon_minutes"])), axis=1
    )
    candidate_frame = metrics.loc[chosen]
    if candidate_frame.empty:
        raise RuntimeError("El reporte no contiene metricas del ensemble elegido por horizonte.")

    hashes: set[str] = set()
    for column in ("dataset_rows_hash", "rows_hash"):
        if column in metrics.columns:
            hashes.update(str(value) for value in metrics[column].dropna().unique() if str(value))
    rows_hash = hashes.pop() if len(hashes) == 1 else None
    if snapshot.get("rows_hash") and rows_hash and snapshot["rows_hash"] != rows_hash:
        raise RuntimeError(
            "Las metricas son de un snapshot distinto al puntero actual: "
            f"{rows_hash[:12]} != {str(snapshot['rows_hash'])[:12]}. Reemplace el reporte."
        )

    incumbent_frame = metrics.loc[metrics["model"].astype(str).str.startswith(INCUMBENT_PREFIX)]
    incumbent = None
    if not incumbent_frame.empty:
        label = str(incumbent_frame["model"].dropna().unique()[0])
        incumbent = evaluation_to_dict(
            evaluation_from_frame(
                incumbent_frame,
                version=label[len(INCUMBENT_PREFIX) :] or "desconocida",
                dataset_rows_hash=rows_hash,
            )
        )

    return {
        "schema_version": 1,
        "validation_window": {
            "start": window_report.get("validation_start"),
            "end": window_report.get("validation_end"),
            "folds": window_report.get("folds", []),
            "history_gap_steps": window_report.get("history_gap_steps"),
            "recency_half_life_days": window_report.get("recency_half_life_days"),
            "training_mlflow_run_id": window_report.get("mlflow_run_id"),
        },
        "dataset": {
            "name": snapshot.get("dataset_name") or window_report.get("dataset_name"),
            "id": snapshot.get("dataset_id") or window_report.get("dataset_id"),
            "rows_hash": rows_hash or snapshot.get("rows_hash"),
            "cutoff_at": snapshot.get("cutoff_at") or window_report.get("dataset_cutoff_at"),
        },
        "candidate": evaluation_to_dict(
            evaluation_from_frame(
                candidate_frame,
                version=candidate_version,
                dataset_rows_hash=rows_hash,
            )
        ),
        "incumbent": incumbent,
    }


def main() -> None:
    args = parse_args()
    module = load_training_module()
    metrics = pd.read_csv(REPORT_PATH, dtype={"station_id": "string"})

    # El campeon entra en la eleccion solo como techo: si gana un horizonte en estas
    # mismas folds, ese horizonte se conserva de el en vez de reentrenarlo.
    champion_label = incumbent_label(metrics)
    reusable_horizons = champion_package_horizons(args.previous_model, champion_label)
    best_summary, best_names, horizon_sources = select_per_horizon(
        metrics,
        champion_label=champion_label,
        champion_horizons=reusable_horizons,
    )
    retained = sorted(
        int(horizon) for horizon, source in horizon_sources.items() if source == CHAMPION_SOURCE
    )
    if retained:
        print(
            "Horizontes conservados del campeon (el candidato no lo supero en esta ventana): "
            + ", ".join(f"{horizon} min" for horizon in retained)
        )
    best_summary["history_gap_steps"] = module.TRAINING_HISTORY_GAP_STEPS
    selected_metrics = metrics.merge(
        best_summary[["horizon_minutes", "model"]],
        on=["horizon_minutes", "model"],
        how="inner",
    )
    station_wape = (
        selected_metrics.groupby(["horizon_minutes", "model", "station_id"], as_index=False)[
            ["wape", "accuracy"]
        ]
        .mean()
        .sort_values(["horizon_minutes", "station_id"])
    )
    if station_wape.groupby("horizon_minutes")["station_id"].nunique().ne(12).any():
        raise ValueError("El paquete no contiene métricas para las 12 estaciones.")

    window_report = load_window_report()
    # Pesos de persistencia por estacion elegidos en la ultima fold de 03; sin reporte
    # (metricas antiguas) el paquete queda sin mezcla, igual que antes.
    persistence_weights = (
        json.loads(PERSISTENCE_WEIGHTS_PATH.read_text(encoding="utf-8"))
        if PERSISTENCE_WEIGHTS_PATH.exists()
        else {}
    )
    ar_weights = json.loads(AR_WEIGHTS_PATH.read_text(encoding="utf-8")) if AR_WEIGHTS_PATH.exists() else {}
    # Se entrena y empaqueta sobre el snapshot versionado (data/snapshot.json) para
    # que el hash de filas del manifiesto coincida con el de las metricas.
    observations, context, snapshot = module.load_dataset_frames()
    frame = module.add_features(observations, context)
    feature_columns = module.model_feature_columns(frame.columns)

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)
    (PACKAGE_DIR / "models").mkdir()
    (PACKAGE_DIR / "configs").mkdir()
    station_wape.to_csv(PACKAGE_DIR / "wape_by_station.csv", index=False)
    best_summary.to_csv(PACKAGE_DIR / "best_ensemble_summary.csv", index=False)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="packaged-best-ensemble") as run:
        model_files = []
        for horizon_minutes, ensemble_name in best_names.items():
            if horizon_sources.get(str(horizon_minutes)) == CHAMPION_SOURCE:
                # El campeon gano este horizonte en esta ventana: se reutiliza su modelo,
                # no se reentrena, para que el paquete final no pueda quedar por debajo.
                retained = copy_champion_horizon(args.previous_model, int(horizon_minutes))
                model_files.extend(retained)
                mlflow.log_param(f"h{horizon_minutes}_source", CHAMPION_SOURCE)
                continue
            model_name, is_relative = split_candidate_name(
                ensemble_name.split(" + Seasonal Naive 7d", 1)[0].replace("Ensemble ", "")
            )
            hgb_weight = float(ensemble_name.rsplit("(", 1)[1].rstrip(")"))
            horizon = horizon_minutes // 15
            horizon_frame = frame.copy()
            horizon_frame["target"] = horizon_frame.groupby("station_id", sort=False)[module.TARGET].shift(-horizon)
            horizon_frame = horizon_frame.dropna(subset=["target"])
            horizon_feature_columns = module.feature_columns_for_horizon(feature_columns, int(horizon_minutes))
            model = HistGradientBoostingRegressor(
                **module.MODEL_CONFIGS[model_name],
                early_stopping=False,
                random_state=42,
            )
            model.fit(
                horizon_frame[horizon_feature_columns],
                horizon_frame["target"],
                sample_weight=module.station_balanced_weights(horizon_frame),
            )

            model_path = PACKAGE_DIR / "models" / f"horizon_{horizon_minutes}_hgb.pkl"
            with model_path.open("wb") as output:
                pickle.dump(model, output)
            relative_fields: dict = {}
            if is_relative:
                relative = module.fit_relative_model(
                    module.MODEL_CONFIGS[model_name], horizon_frame, horizon_feature_columns
                )
                relative_path = relative_model_path(model_path)
                with relative_path.open("wb") as output:
                    pickle.dump(relative, output)
                model_files.append(relative_path)
                relative_fields = {
                    "relative_weight": RELATIVE_WEIGHT,
                    "relative_offset": RELATIVE_OFFSET,
                    "relative_reference": RELATIVE_REFERENCE,
                }
            config = {
                **relative_fields,
                "horizon_minutes": int(horizon_minutes),
                "hgb_model": model_name,
                "hgb_weight": hgb_weight,
                "baseline": "Seasonal Naive 7d",
                "baseline_lag": 672 - horizon,
                "feature_columns": horizon_feature_columns,
                "training_rows": len(horizon_frame),
                "station_count": 12,
                "history_gap_steps": module.TRAINING_HISTORY_GAP_STEPS,
                "feature_protocol": module.FEATURE_PROTOCOL,
                "target_seasonal_days": list(module.TARGET_SEASONAL_DAYS),
                "persistence_weights": persistence_weights.get(str(int(horizon_minutes)), {}).get(
                    ensemble_name, {}
                ),
                "ar_weights": ar_weights.get(str(int(horizon_minutes)), {}).get(ensemble_name, {}),
                "recency_half_life_days": module.RECENCY_HALF_LIFE_DAYS,
                "validation_start": window_report.get("validation_start"),
                "validation_end": window_report.get("validation_end"),
                "dataset_name": snapshot.get("dataset_name"),
                "dataset_id": snapshot.get("dataset_id"),
                "dataset_rows_hash": snapshot.get("rows_hash"),
                "mlflow_run_id": run.info.run_id,
            }
            config_path = PACKAGE_DIR / "configs" / f"horizon_{horizon_minutes}_ensemble.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            mlflow.log_param(f"h{horizon_minutes}_hgb_model", model_name)
            mlflow.log_param(f"h{horizon_minutes}_hgb_weight", hgb_weight)
            mlflow.log_param(f"h{horizon_minutes}_source", CANDIDATE_SOURCE)
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"model_h{horizon_minutes}",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
            model_files.extend([model_path, config_path])

        inputs = promotion_inputs(
            metrics,
            best_names,
            candidate_version=run.info.run_id,
            window_report=window_report,
            snapshot=snapshot,
        )
        PROMOTION_INPUTS_PATH.write_text(
            json.dumps(inputs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Perfil normal para el AR(2) de la inferencia: viaja con el paquete, asi la mezcla
        # no depende de que la inferencia descargue los primeros 28 dias de historia.
        profile_path = save_profile(build_normal_profile(observations), PACKAGE_DIR)
        files = [
            *model_files,
            profile_path,
            PACKAGE_DIR / "wape_by_station.csv",
            PACKAGE_DIR / "best_ensemble_summary.csv",
            PROMOTION_INPUTS_PATH,
        ]
        training_data_end = pd.to_datetime(observations["observed_at"], utc=True).max().isoformat()
        manifest = {
            "experiment": EXPERIMENT,
            "model_version": run.info.run_id,
            "training_data_end": training_data_end,
            "mlflow_run_id": run.info.run_id,
            "training_mlflow_run_id": window_report.get("mlflow_run_id"),
            "metric": "WAPE per station, mean across 12 stations",
            "recency_half_life_days": module.RECENCY_HALF_LIFE_DAYS,
            "history_gap_steps": module.TRAINING_HISTORY_GAP_STEPS,
            "validation_window": inputs["validation_window"],
            "dataset": inputs["dataset"],
            "incumbent": inputs["incumbent"],
            "horizons_minutes": sorted(int(value) for value in best_names),
            "horizon_sources": horizon_sources,
            "files": {str(path.relative_to(PACKAGE_DIR)): sha256(path) for path in files},
        }
        manifest_path = PACKAGE_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(PACKAGE_DIR / "wape_by_station.csv"), artifact_path="package")
        mlflow.log_artifact(str(PROMOTION_INPUTS_PATH), artifact_path="package")
        mlflow.log_artifact(str(manifest_path), artifact_path="package")
        mlflow.set_tag("dataset_rows_hash", str(inputs["dataset"].get("rows_hash") or ""))
        mlflow.set_tag(
            "validation_window",
            f"{inputs['validation_window']['start']}..{inputs['validation_window']['end']}",
        )

    # Antes de escribir el zip: si algun horizonte (del candidato o conservado del
    # campeon) pide columnas que la inferencia no construye, el paquete no sale. Es
    # mejor repetir el reentrenamiento que entregar un modelo que falle en plena
    # ventana de envio, cuando ya no hay tiempo de reaccion.
    broken_configs = unbuildable_package_features(PACKAGE_DIR)
    if broken_configs:
        raise RuntimeError(
            "El paquete pide columnas que la inferencia no construye: "
            f"{broken_configs}. No se empaqueta; el campeon sigue vigente."
        )

    with zipfile.ZipFile(PACKAGE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in PACKAGE_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(PACKAGE_DIR))

    print("Mejor ensemble por horizonte:")
    print(best_summary.to_string(index=False))
    print("\nWAPE (%) por estación:")
    print(
        station_wape.pivot(index="station_id", columns="horizon_minutes", values="wape")
        .mul(100)
        .round(2)
        .to_string()
    )
    incumbent = inputs["incumbent"]
    print("\nComparacion emparejada (misma ventana de validacion):")
    if incumbent is None:
        print(
            "  Sin campeon emparejado: la compuerta bloqueara la promocion "
            "por falta de comparacion."
        )
    else:
        print(
            f"  campeon   {incumbent['version']}: accuracy {incumbent['accuracy']:.2f} "
            f"| WAPE {incumbent['wape']:.4f}"
        )
        print(
            f"  candidato {inputs['candidate']['version']}: accuracy {inputs['candidate']['accuracy']:.2f} "
            f"| WAPE {inputs['candidate']['wape']:.4f}"
        )
    print(f"\nPaquete: {PACKAGE_PATH}")
    print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
