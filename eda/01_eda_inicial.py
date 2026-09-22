from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "stations.csv": {"station_id", "station_name", "corridor", "latitude", "longitude"},
    "observations.csv": {"observed_at", "station_id", "demand"},
    "context.csv": {
        "observed_at",
        "rain_mm",
        "rain_forecast",
        "temperature_c",
        "temperature_forecast",
        "event_intensity",
    },
}


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {
        "stations": pd.read_csv(data_dir / "stations.csv", dtype={"station_id": "string"}),
        "observations": pd.read_csv(
            data_dir / "observations.csv",
            dtype={"station_id": "string"},
            parse_dates=["observed_at"],
        ),
        "context": pd.read_csv(data_dir / "context.csv", parse_dates=["observed_at"]),
    }
    for filename, required in REQUIRED_COLUMNS.items():
        frame = frames[filename.removesuffix(".csv")]
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{filename} no contiene las columnas: {sorted(missing)}")
    return frames


def quality_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        for column in frame.columns:
            rows.append(
                {
                    "dataset": name,
                    "column": column,
                    "rows": len(frame),
                    "missing": int(frame[column].isna().sum()),
                    "missing_pct": round(frame[column].isna().mean() * 100, 3),
                    "unique": frame[column].nunique(dropna=True),
                }
            )
    return pd.DataFrame(rows)


def station_summary(observations: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    summary = observations.groupby("station_id").agg(
        observations=("demand", "size"),
        mean_demand=("demand", "mean"),
        median_demand=("demand", "median"),
        std_demand=("demand", "std"),
        min_demand=("demand", "min"),
        max_demand=("demand", "max"),
        first_observation=("observed_at", "min"),
        last_observation=("observed_at", "max"),
    ).reset_index()
    return summary.merge(
        stations[["station_id", "station_name", "corridor"]],
        on="station_id",
        how="left",
    )


def correlation_table(observations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    merged = observations.merge(context, on="observed_at", how="left")
    numeric = merged.select_dtypes(include="number")
    return numeric.corr().round(3)


def target_correlations(observations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    correlations = correlation_table(observations, context)["demand"].drop("demand")
    return (
        correlations.rename("correlation")
        .to_frame()
        .assign(abs_correlation=lambda frame: frame["correlation"].abs())
        .sort_values("abs_correlation", ascending=False)
    )


def save_plots(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    observations = frames["observations"].copy()
    context = frames["context"]
    station_info = frames["stations"]
    observations["hour"] = observations["observed_at"].dt.hour
    observations["weekday"] = observations["observed_at"].dt.dayofweek

    counts = observations.groupby("station_id").size().sort_values()
    labels = station_info.set_index("station_id")["station_name"].reindex(counts.index)
    labels = labels.fillna(pd.Series(counts.index, index=counts.index))
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels, counts)
    axis.set(title="Observaciones por estación", xlabel="Número de observaciones")
    fig.tight_layout()
    fig.savefig(output_dir / "01_observaciones_por_estacion.png", dpi=150)
    plt.close(fig)

    mean_demand = observations.groupby("station_id", as_index=False)["demand"].mean()
    geographic = station_info.merge(mean_demand, on="station_id", how="left")
    fig, axis = plt.subplots(figsize=(10, 8))
    points = axis.scatter(
        geographic["longitude"],
        geographic["latitude"],
        s=geographic["demand"] * 0.7,
        c=geographic["demand"],
        cmap="viridis",
        alpha=0.88,
        edgecolors="white",
        linewidths=1,
    )
    for _, station in geographic.iterrows():
        axis.annotate(
            station["station_id"],
            (station["longitude"], station["latitude"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        title="Demanda promedio por ubicación",
        xlabel="Longitud",
        ylabel="Latitud",
    )
    axis.grid(alpha=0.25)
    fig.colorbar(points, ax=axis, label="Demanda promedio")
    fig.tight_layout()
    fig.savefig(output_dir / "09_mapa_demanda_estaciones.png", dpi=150)
    plt.close(fig)

    total_by_time = observations.groupby("observed_at", as_index=False)["demand"].sum()
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(total_by_time["observed_at"], total_by_time["demand"], linewidth=1)
    axis.set(title="Demanda total a través del tiempo", xlabel="Fecha", ylabel="Demanda")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "02_demanda_total_temporal.png", dpi=150)
    plt.close(fig)

    grouped = observations.groupby("station_id")["demand"]
    station_ids = list(grouped.groups)
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.boxplot([grouped.get_group(station_id) for station_id in station_ids], tick_labels=station_ids)
    axis.set(title="Distribución de demanda por estación", xlabel="Estación", ylabel="Demanda")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "03_distribucion_demanda.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.hist(observations["demand"], bins=40, color="#2563eb", alpha=0.82, edgecolor="white")
    axis.axvline(observations["demand"].mean(), color="#dc2626", linestyle="--", label="Media")
    axis.axvline(observations["demand"].median(), color="#111827", linestyle=":", label="Mediana")
    axis.set(title="Distribución de la demanda observada", xlabel="Demanda", ylabel="Frecuencia")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "06_distribucion_demanda_observada.png", dpi=150)
    plt.close(fig)

    hourly = observations.groupby(["hour", "station_id"], as_index=False)["demand"].mean()
    fig, axis = plt.subplots(figsize=(12, 6))
    for station_id, group in hourly.groupby("station_id"):
        axis.plot(group["hour"], group["demand"], marker=".", label=station_id)
    axis.set(title="Perfil horario medio por estación", xlabel="Hora local", ylabel="Demanda media")
    axis.set_xticks(range(0, 24, 2))
    axis.grid(alpha=0.25)
    axis.legend(title="Estación", ncol=2, fontsize="small")
    fig.tight_layout()
    fig.savefig(output_dir / "04_perfil_horario.png", dpi=150)
    plt.close(fig)

    hourly_total = observations.groupby("hour", as_index=False)["demand"].mean()
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(hourly_total["hour"], hourly_total["demand"], marker="o", color="#0891b2")
    axis.set(title="Demanda promedio por hora", xlabel="Hora local", ylabel="Demanda promedio")
    axis.set_xticks(range(24))
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "07_demanda_promedio_por_hora.png", dpi=150)
    plt.close(fig)

    weekday_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    weekday_total = observations.groupby("weekday", as_index=False)["demand"].mean()
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(weekday_total["weekday"], weekday_total["demand"], color="#7c3aed")
    axis.set(title="Demanda promedio por día de la semana", xlabel="Día", ylabel="Demanda promedio")
    axis.set_xticks(range(7), weekday_names)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "08_demanda_promedio_por_dia.png", dpi=150)
    plt.close(fig)

    correlations = correlation_table(observations, context)
    ranked = target_correlations(observations, context)
    selected = ["demand", *ranked.head(5).index]
    target_matrix = correlations.loc[selected, selected]
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(target_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(target_matrix)), target_matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(target_matrix)), target_matrix.index)
    for row in range(len(target_matrix)):
        for column in range(len(target_matrix)):
            axis.text(column, row, f"{target_matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Variables más correlacionadas con la demanda")
    fig.colorbar(image, ax=axis, label="Correlación de Pearson")
    fig.tight_layout()
    fig.savefig(output_dir / "05_correlaciones.png", dpi=150)
    plt.close(fig)


def run(data_dir: Path, output_dir: Path) -> None:
    frames = load_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_summary(frames).to_csv(output_dir / "calidad_columnas.csv", index=False)
    station_summary(frames["observations"], frames["stations"]).to_csv(
        output_dir / "resumen_estaciones.csv", index=False
    )
    correlation_table(frames["observations"], frames["context"]).to_csv(
        output_dir / "correlaciones.csv"
    )
    target_correlations(frames["observations"], frames["context"]).to_csv(
        output_dir / "correlaciones_con_demanda.csv"
    )
    save_plots(frames, output_dir)

    observations = frames["observations"]
    print(f"Observaciones: {len(observations):,}")
    print(f"Estaciones: {observations['station_id'].nunique()}")
    print(f"Periodo: {observations['observed_at'].min()} -> {observations['observed_at'].max()}")
    print(f"Resultados escritos en: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA inicial del dataset Pulso TransMi")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("eda_outputs"))
    arguments = parser.parse_args()
    run(arguments.data_dir, arguments.output_dir)


if __name__ == "__main__":
    main()