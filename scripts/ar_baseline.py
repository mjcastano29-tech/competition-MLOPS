"""Pronostico local AR(2) sobre la desviacion de la demanda frente a su perfil normal.

Lo usan el entrenamiento (examples/03_gradient_boosting.py, para elegir el peso de la
mezcla en validacion) y la inferencia (scripts/infer_and_submit.py): un solo codigo
garantiza que ambos calculan el mismo numero.

Para cada estacion se toma x_t = log(demanda_t / normal_t) en las ultimas AR_WINDOW
observaciones (normal = media de esa estacion en ese dia de la semana y cuarto de hora
durante los primeros PROFILE_DAYS dias publicados), se ajusta x_t = a x_{t-1} + b x_{t-2}
+ c por minimos cuadrados y se itera 4 pasos. Un AR(2) puede representar una oscilacion
amortiguada, que el arbol (entrenado sobre dias normales) no extrapola. Si el ajuste es
inestable devuelve la persistencia; si faltan datos, None (no se mezcla).

Backtest walk-forward de 6 dias sobre el stream real, mezclado con el campeon con un
peso por estacion elegido en la fold anterior: 84.64 -> 85.91, mejora en las 5 folds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AR_WINDOW = 24
AR_STEPS = 4
PERIOD_MINUTES = 15
PROFILE_DAYS = 28
PROFILE_TIMEZONE = "America/Bogota"
AR_WEIGHT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
PROFILE_FILENAME = "normal_profile.json"


def _slots(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    local = pd.DatetimeIndex(times).tz_convert(PROFILE_TIMEZONE)
    return np.asarray(local.dayofweek), np.asarray(local.hour * 4 + local.minute // PERIOD_MINUTES)


def build_normal_profile(observations: pd.DataFrame) -> dict[str, Any]:
    """Media por estacion, dia de la semana y cuarto en los primeros PROFILE_DAYS dias."""

    frame = observations.loc[:, ["station_id", "observed_at", "demand"]].dropna().copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    start = frame["observed_at"].min()
    frame = frame.loc[frame["observed_at"] < start + pd.Timedelta(days=PROFILE_DAYS)]
    dow, slot = _slots(pd.DatetimeIndex(frame["observed_at"]))
    frame["dow"], frame["slot"] = dow, slot
    means = frame.groupby(["station_id", "dow", "slot"])["demand"].mean()
    profile: dict[str, list[list[float | None]]] = {}
    for station_id in sorted(frame["station_id"].astype(str).unique()):
        grid = [[None] * 96 for _ in range(7)]
        for (day, quarter), value in means.xs(station_id, level=0).items():
            grid[int(day)][int(quarter)] = float(value)
        profile[str(station_id)] = grid
    return {
        "profile_start": start.isoformat(),
        "profile_days": PROFILE_DAYS,
        "timezone": PROFILE_TIMEZONE,
        "stations": profile,
    }


def save_profile(profile: dict[str, Any], directory: Path) -> Path:
    path = Path(directory) / PROFILE_FILENAME
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def load_profile(directory: Path) -> dict[str, Any] | None:
    path = Path(directory) / PROFILE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def normal_values(profile: dict[str, Any], station_id: str, times: pd.DatetimeIndex) -> np.ndarray:
    grid = (profile.get("stations") or {}).get(str(station_id))
    if grid is None:
        return np.full(len(times), np.nan)
    dow, slot = _slots(times)
    values = [grid[d][s] for d, s in zip(dow, slot)]
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def ar2_station_forecast(
    history: pd.Series, data_cutoff: pd.Timestamp, profile: dict[str, Any], station_id: str
) -> np.ndarray | None:
    """Pronostico de los AR_STEPS cuartos siguientes al corte, o None si no se puede.

    `history` es la demanda de la estacion indexada por instante (UTC). Se exige la
    ventana completa de AR_WINDOW cuartos que termina en el corte, sin huecos.
    """

    cutoff = pd.Timestamp(data_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    window = pd.date_range(end=cutoff, periods=AR_WINDOW, freq=f"{PERIOD_MINUTES}min")
    values = history.reindex(window).to_numpy(dtype=float)
    targets = pd.date_range(cutoff + pd.Timedelta(minutes=PERIOD_MINUTES), periods=AR_STEPS, freq=f"{PERIOD_MINUTES}min")
    normal_hist = normal_values(profile, station_id, window)
    normal_target = normal_values(profile, station_id, targets)
    if np.isnan(values).any() or np.isnan(normal_hist).any() or np.isnan(normal_target).any():
        return None
    if (normal_hist <= 0).any() or (normal_target <= 0).any():
        return None
    x = np.log(np.clip(values, 1, None) / normal_hist)
    y, design = x[2:], np.c_[x[1:-1], x[:-2], np.ones(len(x) - 2)]
    (a, b, c), *_ = np.linalg.lstsq(design, y, rcond=None)
    if np.max(np.abs(np.roots([1.0, -a, -b]))) >= 1.0:
        return np.repeat(values[-1], AR_STEPS)
    previous, current = x[-2], x[-1]
    forecast = []
    for _ in range(AR_STEPS):
        previous, current = current, a * current + b * previous + c
        forecast.append(current)
    return normal_target * np.exp(np.array(forecast))


def choose_ar_weights(
    station_ids: np.ndarray, target: np.ndarray, base: np.ndarray, ar: np.ndarray
) -> dict[str, float]:
    """Peso del AR por estacion que minimiza su error absoluto en una fold ya observada.

    Filas sin AR (NaN) no cuentan. En empate gana el peso menor: sin evidencia, el
    campeon queda intacto.
    """

    frame = pd.DataFrame({"station_id": station_ids, "y": target, "base": base, "ar": ar}).dropna()
    weights: dict[str, float] = {}
    for station_id, block in frame.groupby("station_id", sort=True):
        best_weight, best_error = 0.0, float("inf")
        for weight in AR_WEIGHT_GRID:
            blended = np.clip((1 - weight) * block["base"].to_numpy() + weight * block["ar"].to_numpy(), 0, None)
            error = float(np.abs(block["y"].to_numpy() - blended).sum())
            if error < best_error - 1e-9:
                best_weight, best_error = weight, error
        weights[str(station_id)] = best_weight
    return weights


def apply_ar_weights(
    base: np.ndarray, ar: np.ndarray, station_ids: np.ndarray, weights: dict[str, float] | None
) -> np.ndarray:
    """`(1 - w) * base + w * ar` por estacion; sin AR (NaN) o sin peso, base intacta."""

    base = np.asarray(base, dtype=float)
    if not weights:
        return base
    w = np.array([float(weights.get(str(s), 0.0)) for s in station_ids], dtype=float)
    ar = np.asarray(ar, dtype=float)
    w = np.where(np.isnan(ar), 0.0, w)
    return (1 - w) * base + w * np.nan_to_num(ar, nan=0.0)
