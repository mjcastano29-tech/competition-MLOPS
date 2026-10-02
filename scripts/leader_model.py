"""Pronostico por estacion lider: en el regimen de ondas cada estacion copia a otra.

Desde el 18-sep (virtual) la demanda de cada estacion es una copia escalada de la de otra
estacion desplazada 1-4 h (correlacion del log de la demanda 0.987-0.989 en las 12). Con
un desfase de al menos 4 cuartos, el valor de la lider que predice cada target ya esta
publicado en el corte: y_a(t + h) ~ razon * y_b(t + h - L).

Solo se usa cuando el regimen esta presente, con una compuerta doble por estacion:
  * la mejor pareja (lider, desfase) en las ultimas WINDOW_HOURS horas tiene correlacion
    >= MIN_CORR (en dias normales casi nunca pasa de 0.98; en ondas la mediana es 0.992);
  * ajustada con datos hasta una hora antes del corte, habria acertado la ultima hora ya
    observada con accuracy >= MIN_HOLDOUT_ACCURACY.
Backtest walk-forward en el regimen de ondas: 90.6 (persistencia 49.2, modelo ~68).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PERIOD_MINUTES = 15
WINDOW_HOURS = 12
LAGS = range(4, 17)          # >= 4 cuartos: el dato de la lider ya esta publicado
RATIO_POINTS = 16
MIN_CORR = 0.985
MIN_HOLDOUT_ACCURACY = 0.85
# pandas da correlacion 1.0 entre series constantes: sin variacion real no hay nada que copiar.
MIN_LOG_STD = 0.1
STEPS = 4


def _best_pair(log_window: pd.DataFrame, station: str) -> tuple[float, str, int] | None:
    best = None
    target = log_window[station]
    if target.std() < MIN_LOG_STD:
        return None
    for other in log_window.columns:
        if other == station or log_window[other].std() < MIN_LOG_STD:
            continue
        for lag in LAGS:
            corr = target.corr(log_window[other].shift(lag))
            if np.isfinite(corr) and (best is None or corr > best[0]):
                best = (float(corr), str(other), int(lag))
    return best


def _fit(grid: pd.DataFrame, station: str, cutoff: pd.Timestamp) -> tuple[float, str, int, float] | None:
    window = grid.loc[cutoff - pd.Timedelta(hours=WINDOW_HOURS): cutoff]
    if len(window) < WINDOW_HOURS * 4 or window.isna().any().any():
        return None
    pair = _best_pair(np.log(window.clip(lower=1)), station)
    if pair is None:
        return None
    corr, leader, lag = pair
    ratios = (window[station] / window[leader].shift(lag)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratios) < RATIO_POINTS // 2:
        return None
    return corr, leader, lag, float(np.median(ratios.tail(RATIO_POINTS)))


def _forecast(grid: pd.DataFrame, cutoff: pd.Timestamp, leader: str, lag: int, ratio: float) -> np.ndarray | None:
    times = [cutoff + pd.Timedelta(minutes=PERIOD_MINUTES * (h - lag)) for h in range(1, STEPS + 1)]
    values = grid[leader].reindex(times).to_numpy(dtype=float)
    return None if np.isnan(values).any() else ratio * values


def leader_forecasts(observations: pd.DataFrame, data_cutoff: pd.Timestamp, stations: Any) -> dict[str, dict[str, Any]]:
    """Pronostico de 4 pasos por estacion que pasa la compuerta doble; vacio si ninguna."""

    cutoff = pd.Timestamp(data_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    frame = observations.loc[:, ["station_id", "observed_at", "demand"]].dropna().copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame = frame.loc[(frame["observed_at"] <= cutoff) & (frame["observed_at"] > cutoff - pd.Timedelta(hours=WINDOW_HOURS + 6))]
    grid = (
        frame.drop_duplicates(["station_id", "observed_at"], keep="last")
        .pivot(index="observed_at", columns="station_id", values="demand")
        .sort_index()
        .asfreq(f"{PERIOD_MINUTES}min")
        .astype(float)
    )
    grid.columns = grid.columns.astype(str)
    out: dict[str, dict[str, Any]] = {}
    previous_cutoff = cutoff - pd.Timedelta(hours=1)
    for station in sorted({str(s) for s in stations} & set(grid.columns)):
        fitted = _fit(grid, station, cutoff)
        if fitted is None or fitted[0] < MIN_CORR:
            continue
        # Prueba fuera de muestra: la receta ajustada una hora antes, sobre la ultima hora.
        past = _fit(grid, station, previous_cutoff)
        if past is None:
            continue
        holdout = _forecast(grid, previous_cutoff, past[1], past[2], past[3])
        actual = grid[station].reindex(
            [previous_cutoff + pd.Timedelta(minutes=PERIOD_MINUTES * h) for h in range(1, STEPS + 1)]
        ).to_numpy(dtype=float)
        if holdout is None or np.isnan(actual).any() or actual.sum() <= 0:
            continue
        holdout_accuracy = 1 - np.abs(actual - holdout).sum() / actual.sum()
        if holdout_accuracy < MIN_HOLDOUT_ACCURACY:
            continue
        corr, leader, lag, ratio = fitted
        forecast = _forecast(grid, cutoff, leader, lag, ratio)
        if forecast is None:
            continue
        out[station] = {
            "forecast": np.clip(forecast, 0, None),
            "leader": leader,
            "lag_quarters": lag,
            "ratio": ratio,
            "corr": corr,
            "holdout_accuracy": float(holdout_accuracy),
        }
    return out
