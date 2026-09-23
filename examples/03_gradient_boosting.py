from __future__ import annotations

import json
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingRegressor


TARGET = "demand"
LAGS = (1, 2, 3, 4, 8, 12, 92, 93, 94, 95, 96, 668, 669, 670, 671, 672)
ROLLING_WINDOWS = (4, 16, 96)
ROLLING_STD_WINDOWS = (4, 16)
HORIZONS = (1, 2, 3, 4)
# The official cycle cutoff is currently 133 fifteen-minute periods newer
# than the latest public observation; train against that same information lag.
TRAINING_HISTORY_GAP_STEPS = 133
ENSEMBLE_WEIGHTS = (0.85, 0.9, 0.95, 1.0)
EXPECTED_STATION_COUNT = 12
MLFLOW_EXPERIMENT = "pulso-transmi-forecasting"
BEST_MODEL_DIR = Path("artifacts/models")
MODEL_CONFIGS = {
    "HGB baseline": {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    "HGB more leaves": {
        "learning_rate": 0.04,
        "max_iter": 450,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    "HGB more regularized": {
        "learning_rate": 0.04,
        "max_iter": 450,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 10.0,
    },
    "HGB shallow": {
        "learning_rate": 0.04,
        "max_iter": 450,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
    },
    "HGB small leaves": {
        "learning_rate": 0.03,
        "max_iter": 600,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
    },
    "HGB strong regularization": {
        "learning_rate": 0.03,
        "max_iter": 600,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 40,
        "l2_regularization": 10.0,
    },
    "HGB absolute error": {
        "loss": "absolute_error",
        "learning_rate": 0.04,
        "max_iter": 450,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 2.0,
    },
}


def add_features(
    observations: pd.DataFrame,
    context: pd.DataFrame,
    history_gap_steps: int = TRAINING_HISTORY_GAP_STEPS,
) -> pd.DataFrame:
    if history_gap_steps < 0:
        raise ValueError("history_gap_steps no puede ser negativo.")
    frame = observations.copy()
    frame["feature_available_at"] = frame["observed_at"] - pd.Timedelta(
        minutes=history_gap_steps * 15
    )
    available_context = context.rename(columns={"observed_at": "feature_available_at"})
    frame = frame.merge(
        available_context,
        on="feature_available_at",
        how="left",
        validate="many_to_one",
    )
    frame = frame.sort_values(["station_id", "observed_at"]).copy()
    grouped_demand = frame.groupby("station_id", sort=False)[TARGET]

    for lag in LAGS:
        frame[f"demand_lag_{lag}"] = grouped_demand.shift(max(lag, history_gap_steps))

    for window in ROLLING_WINDOWS:
        frame[f"demand_mean_{window}"] = grouped_demand.transform(
            lambda values: values.shift(max(1, history_gap_steps)).rolling(window).mean()
        )

    for window in ROLLING_STD_WINDOWS:
        frame[f"demand_std_{window}"] = grouped_demand.transform(
            lambda values: values.shift(max(1, history_gap_steps)).rolling(window).std()
        )

    for horizon in HORIZONS:
        target_minutes = horizon * 15
        target_at = frame["observed_at"] + pd.Timedelta(minutes=target_minutes)
        quarter = target_at.dt.hour * 4 + target_at.dt.minute // 15
        weekday = target_at.dt.dayofweek
        frame[f"target_is_weekend_{target_minutes}"] = (weekday >= 5).astype(int)
        frame[f"target_quarter_sin_{target_minutes}"] = np.sin(2 * np.pi * quarter / 96)
        frame[f"target_quarter_cos_{target_minutes}"] = np.cos(2 * np.pi * quarter / 96)
        frame[f"target_weekday_sin_{target_minutes}"] = np.sin(2 * np.pi * weekday / 7)
        frame[f"target_weekday_cos_{target_minutes}"] = np.cos(2 * np.pi * weekday / 7)

    frame["quarter_of_day"] = frame["observed_at"].dt.hour * 4 + frame["observed_at"].dt.minute // 15
    frame["day_of_week"] = frame["observed_at"].dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["quarter_sin"] = np.sin(2 * np.pi * frame["quarter_of_day"] / 96)
    frame["quarter_cos"] = np.cos(2 * np.pi * frame["quarter_of_day"] / 96)
    frame["weekday_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)

    feature_columns = [
        *(f"demand_lag_{lag}" for lag in LAGS),
        *(f"demand_mean_{window}" for window in ROLLING_WINDOWS),
        *(f"demand_std_{window}" for window in ROLLING_STD_WINDOWS),
        *(
            f"target_{feature}_{horizon * 15}"
            for horizon in HORIZONS
            for feature in ("is_weekend", "quarter_sin", "quarter_cos", "weekday_sin", "weekday_cos")
        ),
        "rain_forecast",
        "temperature_forecast",
        "event_intensity",
        "is_weekend",
        "quarter_sin",
        "quarter_cos",
        "weekday_sin",
        "weekday_cos",
        "station_id",
    ]
    frame = frame[["observed_at", TARGET, *feature_columns]].dropna().copy()
    station_ids = frame["station_id"].copy()
    encoded = pd.get_dummies(frame, columns=["station_id"], dtype=float)
    encoded.insert(encoded.columns.get_loc(TARGET) + 1, "station_id", station_ids)
    return encoded


def feature_columns_for_horizon(feature_columns: list[str], horizon_minutes: int) -> list[str]:
    target_calendar_prefixes = (
        "target_is_weekend_",
        "target_quarter_sin_",
        "target_quarter_cos_",
        "target_weekday_sin_",
        "target_weekday_cos_",
    )
    return [
        column for column in feature_columns
        if not column.startswith(target_calendar_prefixes) or column.endswith(f"_{horizon_minutes}")
    ]


def station_balanced_weights(frame: pd.DataFrame, target_column: str = "target") -> np.ndarray:
    station_target_sum = frame.groupby("station_id", sort=False)[target_column].transform("sum")
    if (station_target_sum <= 0).any():
        raise ValueError("No se pueden calcular pesos WAPE con suma de demanda no positiva.")
    weights = (1.0 / station_target_sum).to_numpy(dtype=float)
    # Preserve the effective regularization scale: sample weights should sum to N.
    weights *= len(weights) / weights.sum()
    return weights


def accuracy_by_station(frame: pd.DataFrame) -> pd.Series:
    absolute_error = (frame[TARGET] - frame["prediction"]).abs()
    return 100 * (
        1
        - absolute_error.groupby(frame["station_id"]).sum()
        / frame[TARGET].groupby(frame["station_id"]).sum()
    ).clip(lower=0)


def score(
    name: str,
    fold: int,
    horizon: int,
    frame: pd.DataFrame,
    predictions: pd.Series,
) -> list[dict[str, object]]:
    scored = frame[["station_id", "target"]].copy()
    scored = scored.rename(columns={"target": TARGET})
    scored["prediction"] = predictions.clip(lower=0).to_numpy()
    station_count = scored["station_id"].nunique()
    if station_count != EXPECTED_STATION_COUNT:
        raise ValueError(
            f"Se esperaban {EXPECTED_STATION_COUNT} estaciones, pero la predicción contiene {station_count}."
        )
    station_scores = accuracy_by_station(scored)
    rows = []
    for station_id, accuracy in station_scores.items():
        station_frame = scored.loc[scored["station_id"] == station_id]
        wape = (station_frame[TARGET] - station_frame["prediction"]).abs().sum() / station_frame[TARGET].sum()
        rows.append(
            {
                "fold": fold,
                "horizon_minutes": horizon * 15,
                "model": name,
                "station_id": station_id,
                "rows": len(station_frame),
                "wape": wape,
                "accuracy": accuracy,
            }
        )
    return rows


def log_evaluation(
    rows: list[dict[str, object]],
    horizon: int,
    fold: int,
    training_rows: int,
    validation_rows: int,
) -> None:
    station_metrics = pd.DataFrame(rows)
    mlflow.log_params(
        {
            "horizon_minutes": horizon * 15,
            "fold": fold,
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "station_count": station_metrics["station_id"].nunique(),
        }
    )
    mlflow.log_metrics(
        {
            "wape_mean_station": station_metrics["wape"].mean(),
            "accuracy_mean_station": station_metrics["accuracy"].mean(),
            "prediction_rows": validation_rows,
        }
    )
    mlflow.set_tag("metric_definition", "WAPE per station, then mean across 12 stations")


def save_best_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    best_model_names: dict[int, str],
    ensemble_metrics: pd.DataFrame,
    parent_run_id: str,
) -> None:
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for horizon_minutes, ensemble_name in best_model_names.items():
        model_name = ensemble_name.split(" + Seasonal Naive 7d", 1)[0].replace("Ensemble ", "")
        hgb_weight = float(ensemble_name.rsplit("(", 1)[1].rstrip(")"))
        horizon = horizon_minutes // 15
        horizon_frame = frame.copy()
        horizon_frame["target"] = horizon_frame.groupby("station_id", sort=False)[TARGET].shift(-horizon)
        horizon_frame = horizon_frame.dropna(subset=["target"])
        horizon_feature_columns = feature_columns_for_horizon(feature_columns, horizon_minutes)
        model = HistGradientBoostingRegressor(
            **MODEL_CONFIGS[model_name],
            early_stopping=False,
            random_state=42,
        )
        model.fit(
            horizon_frame[horizon_feature_columns],
            horizon_frame["target"],
            sample_weight=station_balanced_weights(horizon_frame),
        )
        model_path = BEST_MODEL_DIR / f"horizon_{horizon_minutes}_hgb.pkl"
        config_path = BEST_MODEL_DIR / f"horizon_{horizon_minutes}_ensemble.json"
        with model_path.open("wb") as output:
            pickle.dump(model, output)
        config_path.write_text(
            json.dumps(
                {
                    "horizon_minutes": horizon_minutes,
                    "hgb_model": model_name,
                    "hgb_weight": hgb_weight,
                    "baseline": "Seasonal Naive 7d",
                    "feature_columns": horizon_feature_columns,
                    "training_rows": len(horizon_frame),
                },
                indent=2,
            )
        )
        best_rows = ensemble_metrics.loc[
            (ensemble_metrics["horizon_minutes"] == horizon_minutes)
            & (ensemble_metrics["model"] == ensemble_name)
        ]
        with mlflow.start_run(
            run_name=f"best-ensemble-h{horizon_minutes}",
            nested=True,
        ):
            mlflow.log_params(
                {
                    "horizon_minutes": horizon_minutes,
                    "hgb_model": model_name,
                    "hgb_weight": hgb_weight,
                    "training_rows": len(horizon_frame),
                    "station_count": EXPECTED_STATION_COUNT,
                }
            )
            mlflow.log_metrics(
                {
                    "wape_mean_station": best_rows["wape"].mean(),
                    "accuracy_mean_station": best_rows["accuracy"].mean(),
                }
            )
            mlflow.set_tag("model_role", "best_ensemble_retrained_on_all_data")
            mlflow.set_tag("parent_run_id", parent_run_id)
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(config_path), artifact_path="model")


def main() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="rolling-multihorizon-validation") as parent_run:
        run_experiment(parent_run.info.run_id)


def run_experiment(parent_run_id: str) -> None:
    observations = pd.read_csv(
        "data/observations.csv",
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    )
    context = pd.read_csv("data/context.csv", parse_dates=["observed_at"])
    frame = add_features(observations, context)

    feature_columns = [
        column for column in frame.columns if column not in {"observed_at", TARGET, "station_id"}
    ]
    latest_timestamp = frame["observed_at"].max() - timedelta(minutes=max(HORIZONS) * 15)
    fold_starts = [
        latest_timestamp - timedelta(days=21),
        latest_timestamp - timedelta(days=14),
        latest_timestamp - timedelta(days=7),
    ]
    metric_rows: list[dict[str, object]] = []

    for horizon in HORIZONS:
        horizon_frame = frame.copy()
        horizon_frame["target"] = horizon_frame.groupby("station_id", sort=False)[TARGET].shift(-horizon)
        horizon_frame = horizon_frame.dropna(subset=["target"])
        horizon_feature_columns = feature_columns_for_horizon(feature_columns, horizon * 15)
        for fold, validation_start in enumerate(fold_starts, start=1):
            validation_end = validation_start + timedelta(days=7)
            # Leave a horizon-sized embargo so training labels cannot overlap validation.
            train_cutoff = validation_start - timedelta(minutes=(horizon + 1) * 15)
            train = horizon_frame.loc[horizon_frame["observed_at"] <= train_cutoff].copy()
            validation = horizon_frame.loc[
                (horizon_frame["observed_at"] >= validation_start)
                & (horizon_frame["observed_at"] < validation_end)
            ].copy()
            print(
                f"Horizonte {horizon * 15} min, fold {fold}: "
                f"entrenamiento={len(train)}; validación={len(validation)}"
            )
            metric_rows.extend(
                (baseline_rows := score(
                    "Seasonal Naive 24h",
                    fold,
                    horizon,
                    validation,
                    validation[f"demand_lag_{96 - horizon}"],
                ))
            )
            with mlflow.start_run(
                run_name=f"Seasonal-Naive-24h-h{horizon * 15}-fold{fold}",
                nested=True,
            ):
                log_evaluation(baseline_rows, horizon, fold, len(train), len(validation))
            metric_rows.extend(
                (baseline_rows := score(
                    "Seasonal Naive 7d",
                    fold,
                    horizon,
                    validation,
                    validation[f"demand_lag_{672 - horizon}"],
                ))
            )
            with mlflow.start_run(
                run_name=f"Seasonal-Naive-7d-h{horizon * 15}-fold{fold}",
                nested=True,
            ):
                log_evaluation(baseline_rows, horizon, fold, len(train), len(validation))

            hgb_predictions: dict[str, pd.Series] = {}
            for model_name, model_config in MODEL_CONFIGS.items():
                with mlflow.start_run(
                    run_name=f"{model_name}-h{horizon * 15}-fold{fold}",
                    nested=True,
                ):
                    model = HistGradientBoostingRegressor(
                        **model_config,
                        early_stopping=False,
                        random_state=42,
                    )
                    model.fit(
                        train[horizon_feature_columns],
                        train["target"],
                        sample_weight=station_balanced_weights(train),
                    )
                    predictions = pd.Series(model.predict(validation[horizon_feature_columns]))
                    hgb_predictions[model_name] = predictions
                    rows = score(model_name, fold, horizon, validation, predictions)
                    metric_rows.extend(rows)
                    log_evaluation(rows, horizon, fold, len(train), len(validation))
                    mlflow.log_param("feature_count", len(horizon_feature_columns))
                    mlflow.set_tag("parent_run_id", parent_run_id)
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                    )

            weekly_baseline = validation[f"demand_lag_{672 - horizon}"].to_numpy()
            for model_name, predictions in hgb_predictions.items():
                for hgb_weight in ENSEMBLE_WEIGHTS:
                    ensemble_name = f"Ensemble {model_name} + Seasonal Naive 7d ({hgb_weight:.1f})"
                    ensemble_predictions = (
                        hgb_weight * predictions.to_numpy()
                        + (1 - hgb_weight) * weekly_baseline
                    )
                    ensemble_rows = score(
                        ensemble_name,
                        fold,
                        horizon,
                        validation,
                        pd.Series(ensemble_predictions),
                    )
                    metric_rows.extend(ensemble_rows)
                    with mlflow.start_run(
                        run_name=f"ensemble-{model_name}-h{horizon * 15}-fold{fold}-w{hgb_weight:.1f}",
                        nested=True,
                    ):
                        log_evaluation(ensemble_rows, horizon, fold, len(train), len(validation))
                        mlflow.log_param("hgb_model", model_name)
                        mlflow.log_param("hgb_weight", hgb_weight)
                        mlflow.set_tag("ensemble", "HistGradientBoosting + Seasonal Naive 7d")

    metrics = pd.DataFrame(metric_rows)
    report_path = Path("reports/ml_validation_metrics.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_path, index=False)
    mlflow.log_artifact(str(report_path), artifact_path="validation")
    ensemble_metrics = metrics[metrics["model"].str.startswith("Ensemble ")]
    ensemble_summary = (
        ensemble_metrics.groupby(["horizon_minutes", "model"], as_index=False)[["accuracy", "wape"]]
        .mean()
        .sort_values(["horizon_minutes", "accuracy"], ascending=[True, False])
    )
    best_ensemble = ensemble_summary.groupby("horizon_minutes", as_index=False).first()
    best_model_names = dict(zip(best_ensemble["horizon_minutes"], best_ensemble["model"]))
    best_station_metrics = ensemble_metrics.merge(
        best_ensemble[["horizon_minutes", "model"]],
        on=["horizon_minutes", "model"],
        how="inner",
    )
    best_station_metrics = (
        best_station_metrics.groupby(["horizon_minutes", "model", "station_id"], as_index=False)[
            ["wape", "accuracy"]
        ]
        .mean()
        .sort_values(["horizon_minutes", "station_id"])
    )
    station_report_path = Path("reports/best_ensemble_wape_by_station.csv")
    best_station_metrics.to_csv(station_report_path, index=False)
    mlflow.log_artifact(str(station_report_path), artifact_path="validation")
    save_best_models(
        frame,
        feature_columns,
        best_model_names,
        ensemble_metrics,
        parent_run_id,
    )
    summary = (
        metrics.groupby(["horizon_minutes", "model"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["horizon_minutes", "accuracy"], ascending=[True, False])
    )
    print("\nAccuracy promedio por horizonte y modelo:")
    print(summary.to_string(index=False, formatters={"accuracy": "{:.2f}".format}))
    print("\nMejor ensemble por horizonte:")
    print(best_ensemble.to_string(index=False, formatters={"accuracy": "{:.2f}".format}))
    print("\nWAPE promedio por estación del mejor ensemble:")
    print(
        best_station_metrics.pivot(index="station_id", columns="horizon_minutes", values="wape")
        .mul(100)
        .round(2)
        .to_string()
    )
    print(f"\nMétricas detalladas guardadas en {report_path}")
    print(f"WAPE por estación guardado en {station_report_path}")
    print(f"Mejores modelos guardados en {BEST_MODEL_DIR}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow parent run: {parent_run_id}")


if __name__ == "__main__":
    main()