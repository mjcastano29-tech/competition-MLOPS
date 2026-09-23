from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import shutil
import zipfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/ml_validation_metrics.csv"
PACKAGE_DIR = ROOT / "artifacts/pulso_transmi_best_models"
PACKAGE_PATH = ROOT / "artifacts/pulso_transmi_best_models.zip"
EXPERIMENT = "pulso-transmi-forecasting"


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


def main() -> None:
    module = load_training_module()
    metrics = pd.read_csv(REPORT_PATH, dtype={"station_id": "string"})
    ensemble_metrics = metrics[metrics["model"].str.startswith("Ensemble ")].copy()
    if ensemble_metrics.empty:
        raise RuntimeError("No hay métricas de ensemble disponibles.")

    best_summary = (
        ensemble_metrics.groupby(["horizon_minutes", "model"], as_index=False)[["accuracy", "wape"]]
        .mean()
        .sort_values(["horizon_minutes", "accuracy"], ascending=[True, False])
        .groupby("horizon_minutes", as_index=False)
        .first()
    )
    best_summary["history_gap_steps"] = module.TRAINING_HISTORY_GAP_STEPS
    best_names = dict(zip(best_summary["horizon_minutes"], best_summary["model"]))
    selected_metrics = ensemble_metrics.merge(
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

    observations = pd.read_csv(
        ROOT / "data/observations.csv",
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    )
    context = pd.read_csv(ROOT / "data/context.csv", parse_dates=["observed_at"])
    frame = module.add_features(observations, context)
    feature_columns = [
        column for column in frame.columns if column not in {"observed_at", module.TARGET, "station_id"}
    ]

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
            model_name = ensemble_name.split(" + Seasonal Naive 7d", 1)[0].replace("Ensemble ", "")
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
            config = {
                "horizon_minutes": int(horizon_minutes),
                "hgb_model": model_name,
                "hgb_weight": hgb_weight,
                "baseline": "Seasonal Naive 7d",
                "baseline_lag": 672 - horizon,
                "feature_columns": horizon_feature_columns,
                "training_rows": len(horizon_frame),
                "station_count": 12,
                "history_gap_steps": module.TRAINING_HISTORY_GAP_STEPS,
                "mlflow_run_id": run.info.run_id,
            }
            config_path = PACKAGE_DIR / "configs" / f"horizon_{horizon_minutes}_ensemble.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            mlflow.log_param(f"h{horizon_minutes}_hgb_model", model_name)
            mlflow.log_param(f"h{horizon_minutes}_hgb_weight", hgb_weight)
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"model_h{horizon_minutes}",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
            model_files.extend([model_path, config_path])

        files = [*model_files, PACKAGE_DIR / "wape_by_station.csv", PACKAGE_DIR / "best_ensemble_summary.csv"]
        observations = pd.read_csv(ROOT / "data/observations.csv", parse_dates=["observed_at"])
        training_data_end = pd.to_datetime(observations["observed_at"], utc=True).max().isoformat()
        manifest = {
            "experiment": EXPERIMENT,
            "training_data_end": training_data_end,
            "mlflow_run_id": run.info.run_id,
            "metric": "WAPE per station, mean across 12 stations",
            "horizons_minutes": sorted(int(value) for value in best_names),
            "files": {str(path.relative_to(PACKAGE_DIR)): sha256(path) for path in files},
        }
        manifest_path = PACKAGE_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(PACKAGE_DIR / "wape_by_station.csv"), artifact_path="package")
        mlflow.log_artifact(str(manifest_path), artifact_path="package")

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
    print(f"\nPaquete: {PACKAGE_PATH}")
    print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
