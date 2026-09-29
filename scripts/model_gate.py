"""Criterio de promocion de modelos: puro, determinista y testeable.

Toda comparacion exige que candidato y campeon esten medidos sobre la MISMA
ventana de validacion, el MISMO protocolo de hueco de historia y las mismas
estaciones. Si algo de eso no se cumple, la decision es NO promocionar: una
comparacion incompleta no puede justificar reemplazar el modelo en produccion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd

EXPECTED_STATION_COUNT = 12
REQUIRED_HORIZONS: tuple[int, ...] = (15, 30, 45, 60)

# Reglas de la compuerta: el candidato debe traer al menos MEDIO PUNTO de accuracy
# promedio medido sobre la MISMA ventana, y NINGUNA estacion puede caer mas de DOS
# PUNTOS. Una ganancia menor se pierde en el ruido de las folds, y una caida grande en
# una sola estacion se disfraza de mejoria cuando se promedia con las otras once.
MIN_ACCURACY_GAIN = 0.5
MAX_STATION_DROP = 2.0

# Pasos de 15 min entre el ultimo dato observado y el primer paso pronosticado.
# Debe coincidir con examples/03_gradient_boosting.py. Un campeon con hueco mayor sigue
# siendo comparable (ver `gap_is_comparable`): se puntua con su propio hueco.
TRAINING_HISTORY_GAP_STEPS = 0


@dataclass(frozen=True)
class Evaluation:
    """Metricas agregadas de un modelo sobre una ventana temporal concreta."""

    version: str
    accuracy: float
    wape: float
    validation_start: datetime | None
    validation_end: datetime | None
    history_gap_steps: int | None
    dataset_rows_hash: str | None
    accuracy_by_horizon: dict[int, float] = field(default_factory=dict)
    wape_by_horizon: dict[int, float] = field(default_factory=dict)
    accuracy_by_station: dict[str, float] = field(default_factory=dict)
    station_count: int = 0
    folds: int = 0
    rows: int = 0
    model_name: str | None = None
    # "refit": el campeon se midio reentrenando su receta con datos previos a cada fold;
    # "frozen": con su modelo tal cual (puede haber visto esas filas al entrenar).
    scoring: str | None = None

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.accuracy_by_horizon))

    @property
    def has_window(self) -> bool:
        return self.validation_start is not None and self.validation_end is not None

    def same_window(self, other: "Evaluation") -> bool:
        return bool(
            self.has_window
            and other.has_window
            and self.validation_start == other.validation_start
            and self.validation_end == other.validation_end
        )


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    decision: str
    reasons: tuple[str, ...]
    accuracy_delta: float | None = None
    wape_delta: float | None = None
    horizon_deltas: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "promote": self.promote,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "accuracy_delta": self.accuracy_delta,
            "wape_delta": self.wape_delta,
            "horizon_deltas": {str(key): value for key, value in self.horizon_deltas.items()},
        }

    def describe(self) -> str:
        detail = "; ".join(self.reasons) if self.reasons else self.decision
        gain = "" if self.accuracy_delta is None else f" (delta {self.accuracy_delta:+.2f} pts)"
        return f"{'promover' if self.promote else 'mantener campeon'}: {detail}{gain}"


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.casefold() in {"none", "nat", "nan"}:
            return None
        return pd.Timestamp(text).to_pydatetime()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if pd.isna(value):
        return None
    return None


def _model_name(frame: pd.DataFrame, horizon_minutes: int) -> str | None:
    if "model" not in frame.columns:
        return None
    models = frame.loc[frame["horizon_minutes"] == horizon_minutes, "model"].dropna().unique()
    return str(models[0]) if len(models) else None


def evaluation_from_frame(
    frame: pd.DataFrame,
    *,
    version: str,
    model: str | None = None,
    dataset_rows_hash: str | None = None,
) -> Evaluation:
    """Agrega metricas por estacion/horizonte en una `Evaluation` comparable.

    Espera `horizon_minutes`, `accuracy`, `wape` y `station_id`; aprovecha
    `validation_start`, `validation_end`, `history_gap_steps`, `model` y `fold`
    cuando existen. La exactitud agregada es la media de las medias por
    horizonte, la misma definicion que usa el WAPE oficial del dashboard.
    """

    required = {"horizon_minutes", "accuracy", "wape", "station_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para evaluar: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No hay filas de metricas para evaluar.")

    selected = frame.copy()
    if model is not None:
        if "model" not in selected.columns:
            raise ValueError(
                "Se pidio un modelo concreto pero las metricas no tienen columna `model`."
            )
        selected = selected.loc[selected["model"] == model]
        if selected.empty:
            raise ValueError(f"No hay metricas para el modelo {model!r}.")

    selected["horizon_minutes"] = selected["horizon_minutes"].astype(int)

    starts = {
        _timestamp(value) for value in selected.get("validation_start", pd.Series(dtype=object))
    }
    ends = {_timestamp(value) for value in selected.get("validation_end", pd.Series(dtype=object))}
    starts.discard(None)
    ends.discard(None)
    if len(starts) > 1 or len(ends) > 1:
        raise ValueError("Las metricas mezclan mas de una ventana de validacion.")

    gaps = {
        int(value)
        for value in selected.get("history_gap_steps", pd.Series(dtype=object)).dropna().unique()
    }
    if len(gaps) > 1:
        raise ValueError(f"Las metricas mezclan protocolos de hueco de historia: {sorted(gaps)}")

    hashes: set[str] = set()
    for column in ("rows_hash", "dataset_rows_hash"):
        if column in selected.columns:
            hashes.update(str(value) for value in selected[column].dropna().unique())
    if dataset_rows_hash:
        hashes.add(dataset_rows_hash)
    if len(hashes) > 1:
        raise ValueError(f"Las metricas mezclan snapshots del dataset: {sorted(hashes)}")

    scorings = {
        str(value)
        for value in selected.get("incumbent_scoring", pd.Series(dtype=object)).dropna().unique()
    }

    accuracy_by_horizon: dict[int, float] = {}
    wape_by_horizon: dict[int, float] = {}
    for horizon_minutes, horizon_frame in selected.groupby("horizon_minutes"):
        accuracy_by_horizon[int(horizon_minutes)] = float(
            horizon_frame.groupby("station_id")["accuracy"].mean().mean()
        )
        wape_by_horizon[int(horizon_minutes)] = float(
            horizon_frame.groupby("station_id")["wape"].mean().mean()
        )

    # Media de cada estacion sobre todos los horizontes: la compuerta tambien mira el
    # peor caso individual, porque una estacion que se hunde se diluye al promediar once.
    accuracy_by_station = {
        str(station): float(values.mean())
        for station, values in selected.groupby("station_id")["accuracy"]
    }

    return Evaluation(
        version=version,
        accuracy=sum(accuracy_by_horizon.values()) / len(accuracy_by_horizon),
        wape=sum(wape_by_horizon.values()) / len(wape_by_horizon),
        validation_start=next(iter(starts), None),
        validation_end=next(iter(ends), None),
        history_gap_steps=next(iter(gaps), None),
        dataset_rows_hash=next(iter(hashes), None),
        accuracy_by_horizon=accuracy_by_horizon,
        wape_by_horizon=wape_by_horizon,
        accuracy_by_station=accuracy_by_station,
        station_count=int(selected["station_id"].nunique()),
        folds=int(selected["fold"].nunique()) if "fold" in selected.columns else 1,
        rows=int(len(selected)),
        model_name=_model_name(selected, min(accuracy_by_horizon)),
        scoring=next(iter(scorings)) if len(scorings) == 1 else ("frozen" if scorings else None),
    )


def gap_is_comparable(candidate_gap: int | None, incumbent_gap: int | None) -> bool:
    if candidate_gap is None or incumbent_gap is None:
        return candidate_gap == incumbent_gap
    return candidate_gap <= incumbent_gap


def evaluate_promotion(
    candidate: Evaluation | None,
    incumbent: Evaluation | None,
    *,
    min_accuracy_gain: float = MIN_ACCURACY_GAIN,
    max_horizon_drop: float = 0.0,
    max_station_drop: float = MAX_STATION_DROP,
    expected_stations: int = EXPECTED_STATION_COUNT,
    required_horizons: Sequence[int] = REQUIRED_HORIZONS,
    allow_first_model: bool = False,
) -> PromotionDecision:
    """Decide si `candidate` reemplaza a `incumbent` midiendo la misma ventana."""

    if candidate is None:
        return PromotionDecision(
            False, "candidate_metrics_missing", ("El candidato no dejo metricas evaluables.",)
        )

    if incumbent is None:
        if allow_first_model:
            return PromotionDecision(
                True,
                "first_model",
                ("No hay campeon registrado y se autorizo promocionar el primer modelo.",),
            )
        return PromotionDecision(
            False,
            "incumbent_missing",
            ("No hay campeon evaluable: sin comparacion emparejada no se promociona.",),
        )

    reasons: list[str] = []
    if not candidate.has_window or not incumbent.has_window:
        reasons.append(
            "Las metricas no declaran validation_start/validation_end; no son comparables."
        )
    elif not candidate.same_window(incumbent):
        reasons.append(
            "Ventana de validacion distinta: candidato "
            f"{candidate.validation_start}..{candidate.validation_end} vs campeon "
            f"{incumbent.validation_start}..{incumbent.validation_end}."
        )
    # Cada modelo se puntua sobre los mismos objetivos con el hueco con que se sirve, asi
    # que un candidato que lee historia mas fresca es comparable: es justo la mejora que se
    # quiere medir. Un candidato mas atrasado que el campeon, en cambio, no se acepta.
    if not gap_is_comparable(candidate.history_gap_steps, incumbent.history_gap_steps):
        reasons.append(
            f"Protocolo de hueco distinto: candidato {candidate.history_gap_steps} vs "
            f"campeon {incumbent.history_gap_steps} pasos."
        )

    missing_horizons = [
        horizon for horizon in required_horizons if horizon not in candidate.accuracy_by_horizon
    ]
    if missing_horizons:
        reasons.append(f"El candidato no cubre los horizontes {missing_horizons}.")
    if candidate.station_count < expected_stations:
        reasons.append(
            f"Cobertura insuficiente: {candidate.station_count} estaciones frente a "
            f"{expected_stations} esperadas."
        )

    comparable = [
        horizon
        for horizon in required_horizons
        if horizon in candidate.accuracy_by_horizon and horizon in incumbent.accuracy_by_horizon
    ]
    horizon_deltas = {
        horizon: candidate.accuracy_by_horizon[horizon] - incumbent.accuracy_by_horizon[horizon]
        for horizon in comparable
    }
    accuracy_delta = candidate.accuracy - incumbent.accuracy
    wape_delta = candidate.wape - incumbent.wape

    if reasons:
        return PromotionDecision(
            False,
            "incomparable",
            tuple(reasons),
            accuracy_delta=accuracy_delta,
            wape_delta=wape_delta,
            horizon_deltas=horizon_deltas,
        )

    # Ninguna estacion puede hundirse aunque la media suba: doce promediadas taparían el
    # problema de una, y en produccion se nota primero en esa estacion.
    station_deltas = {
        station: candidate.accuracy_by_station[station] - incumbent.accuracy_by_station[station]
        for station in candidate.accuracy_by_station
        if station in incumbent.accuracy_by_station
    }
    broken_stations = {
        station: delta for station, delta in station_deltas.items() if delta < -max_station_drop
    }
    if broken_stations:
        worst_station = min(broken_stations, key=lambda station: broken_stations[station])
        return PromotionDecision(
            False,
            "station_regression",
            tuple(
                f"Estacion {station} cae {abs(delta):.2f} pts."
                for station, delta in sorted(broken_stations.items())
            )
            + (
                f"Peor caso en {worst_station} ({broken_stations[worst_station]:+.2f} pts) y el "
                f"limite por estacion es {max_station_drop:.2f} pts, aunque la media vaya "
                f"{accuracy_delta:+.2f}.",
            ),
            accuracy_delta=accuracy_delta,
            wape_delta=wape_delta,
            horizon_deltas=horizon_deltas,
        )

    regressed = {
        horizon: delta for horizon, delta in horizon_deltas.items() if delta < -max_horizon_drop
    }
    if regressed:
        worst = min(regressed, key=lambda horizon: regressed[horizon])
        return PromotionDecision(
            False,
            "horizon_regression",
            tuple(
                f"Horizonte {horizon} retrocede {abs(delta):.2f} pts."
                for horizon, delta in sorted(regressed.items())
            )
            + (
                f"Peor caso en horizonte {worst}; la media {accuracy_delta:+.2f} pts no compensa "
                "la regresion de un horizonte.",
            ),
            accuracy_delta=accuracy_delta,
            wape_delta=wape_delta,
            horizon_deltas=horizon_deltas,
        )

    if incumbent.scoring == "refit" and accuracy_delta >= 0:
        # Receta contra receta con la misma informacion: si la del candidato no es peor,
        # su modelo gana por construccion, porque se entreno con datos mas nuevos que el
        # campeon. Exigir +0.5 aqui congelaria al campeon mientras el drift avanza.
        return PromotionDecision(
            True,
            "paired_refresh",
            (
                f"Campeon medido como receta reentrenada en la misma ventana "
                f"{candidate.validation_start}..{candidate.validation_end}; hueco "
                f"{incumbent.history_gap_steps} -> {candidate.history_gap_steps} pasos.",
                f"Exactitud {incumbent.accuracy:.2f} -> {candidate.accuracy:.2f} "
                f"({accuracy_delta:+.2f} pts) con datos de entrenamiento mas recientes.",
            ),
            accuracy_delta=accuracy_delta,
            wape_delta=wape_delta,
            horizon_deltas=horizon_deltas,
        )

    if accuracy_delta <= min_accuracy_gain:
        return PromotionDecision(
            False,
            "no_improvement",
            (
                f"Exactitud {candidate.accuracy:.2f} frente a {incumbent.accuracy:.2f}: ganancia "
                f"{accuracy_delta:+.2f} pts no supera el minimo {min_accuracy_gain:+.2f}.",
            ),
            accuracy_delta=accuracy_delta,
            wape_delta=wape_delta,
            horizon_deltas=horizon_deltas,
        )

    return PromotionDecision(
        True,
        "paired_improvement",
        (
            f"Misma ventana {candidate.validation_start}..{candidate.validation_end}; hueco "
            f"{incumbent.history_gap_steps} -> {candidate.history_gap_steps} pasos.",
            f"Exactitud {incumbent.accuracy:.2f} -> {candidate.accuracy:.2f} "
            f"({accuracy_delta:+.2f} pts).",
            f"WAPE {incumbent.wape:.4f} -> {candidate.wape:.4f} ({wape_delta:+.4f}).",
        ),
        accuracy_delta=accuracy_delta,
        wape_delta=wape_delta,
        horizon_deltas=horizon_deltas,
    )


def evaluation_to_dict(evaluation: Evaluation | None) -> dict[str, Any] | None:
    """Serializa una `Evaluation` para dejarla dentro del paquete del modelo.

    El paquete se convierte asi en su propia evidencia: lleva las metricas con las
    que fue promovido y la ventana exacta sobre la que se midieron.
    """

    if evaluation is None:
        return None
    return {
        "version": evaluation.version,
        "model_name": evaluation.model_name,
        "accuracy": evaluation.accuracy,
        "wape": evaluation.wape,
        "validation_start": evaluation.validation_start.isoformat()
        if evaluation.validation_start
        else None,
        "validation_end": evaluation.validation_end.isoformat()
        if evaluation.validation_end
        else None,
        "history_gap_steps": evaluation.history_gap_steps,
        "dataset_rows_hash": evaluation.dataset_rows_hash,
        "accuracy_by_horizon": {
            str(key): value for key, value in evaluation.accuracy_by_horizon.items()
        },
        "wape_by_horizon": {str(key): value for key, value in evaluation.wape_by_horizon.items()},
        "accuracy_by_station": {
            str(key): value for key, value in evaluation.accuracy_by_station.items()
        },
        "station_count": evaluation.station_count,
        "folds": evaluation.folds,
        "rows": evaluation.rows,
        "scoring": evaluation.scoring,
    }


def evaluation_from_dict(payload: dict[str, Any] | None) -> Evaluation | None:
    """Reconstruye una `Evaluation` guardada por `evaluation_to_dict`."""

    if not payload:
        return None
    return Evaluation(
        version=str(payload["version"]),
        accuracy=float(payload["accuracy"]),
        wape=float(payload["wape"]),
        validation_start=_timestamp(payload.get("validation_start")),
        validation_end=_timestamp(payload.get("validation_end")),
        history_gap_steps=(
            int(payload["history_gap_steps"])
            if payload.get("history_gap_steps") is not None
            else None
        ),
        dataset_rows_hash=payload.get("dataset_rows_hash"),
        accuracy_by_horizon={
            int(key): float(value)
            for key, value in (payload.get("accuracy_by_horizon") or {}).items()
        },
        wape_by_horizon={
            int(key): float(value) for key, value in (payload.get("wape_by_horizon") or {}).items()
        },
        accuracy_by_station={
            str(key): float(value)
            for key, value in (payload.get("accuracy_by_station") or {}).items()
        },
        station_count=int(payload.get("station_count") or 0),
        folds=int(payload.get("folds") or 0),
        rows=int(payload.get("rows") or 0),
        model_name=payload.get("model_name"),
        scoring=payload.get("scoring"),
    )


def decision_to_dict(decision: PromotionDecision) -> dict[str, Any]:
    payload = decision.as_dict()
    payload["description"] = decision.describe()
    return payload


CHAMPION_SOURCE = "champion_retained"
CANDIDATE_SOURCE = "candidate_refit"


def mean_metrics_by_horizon(frame: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Accuracy y WAPE medios por horizonte, con la misma formula que agrega
    `evaluation_from_frame`: media de las medias por estacion.

    Sirve para comparar candidato y campeon dentro del empaquetado con la misma
    definicion con la que despues decide la compuerta.
    """

    required = {"horizon_minutes", "accuracy", "wape", "station_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para agregar por horizonte: {sorted(missing)}")
    aggregated: dict[int, dict[str, float]] = {}
    for horizon_minutes, horizon_frame in frame.groupby("horizon_minutes"):
        aggregated[int(horizon_minutes)] = {
            "accuracy": float(horizon_frame.groupby("station_id")["accuracy"].mean().mean()),
            "wape": float(horizon_frame.groupby("station_id")["wape"].mean().mean()),
        }
    return aggregated


def horizons_behind_champion(
    candidate_accuracy_by_horizon: Mapping[int, float],
    champion_accuracy_by_horizon: Mapping[int, float],
    *,
    tolerance: float = 0.0,
) -> list[int]:
    """Horizontes en los que el mejor candidato nuevo queda por debajo del campeon.

    La comparacion solo tiene sentido cuando ambos numeros salen de las mismas filas
    de validacion: `examples/03_gradient_boosting.py` puntua al campeon sobre las
    folds del candidato justamente para eso.
    """

    return [
        int(horizon)
        for horizon, accuracy in sorted(candidate_accuracy_by_horizon.items())
        if horizon in champion_accuracy_by_horizon
        and float(accuracy) < float(champion_accuracy_by_horizon[horizon]) - tolerance
    ]


def select_per_horizon(
    metrics: pd.DataFrame,
    *,
    candidate_pattern: str = "Ensemble ",
    champion_label: str | None = None,
    champion_horizons: Sequence[int] = (),
) -> tuple[pd.DataFrame, dict[int, str], dict[str, str]]:
    """Elige el mejor modelo por horizonte sin poder quedar por debajo del campeon.

    La busqueda del candidato es solo entre los ensambles del run actual. Si el
    campeon, medido sobre estas mismas folds, supera al mejor candidato en un
    horizonte y su modelo esta disponible para reutilizar, ese horizonte se conserva
    del campeon. El paquete resultante no puede registrar regresion en ninguno de los
    horizontes que `evaluate_promotion` penaliza, de modo que promocionar deja de
    depender de ganar en los cuatro horizontes a la vez y pasa a depender de ganar
    en al menos uno.

    Devuelve `(mejor_resumen, mejor_modelo_por_horizonte, origen_por_horizonte)`.
    """

    required = {"horizon_minutes", "model", "accuracy", "wape", "station_id"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Faltan columnas para elegir por horizonte: {sorted(missing)}")

    candidates = metrics.loc[metrics["model"].astype(str).str.startswith(candidate_pattern)]
    if candidates.empty:
        raise ValueError("No hay metricas de ensambles para elegir el mejor por horizonte.")
    summary = (
        candidates.groupby(["horizon_minutes", "model"], as_index=False)[["accuracy", "wape"]]
        .mean()
        .sort_values(["horizon_minutes", "accuracy", "model"], ascending=[True, False, True])
    )
    rows = {
        int(row.horizon_minutes): {
            "horizon_minutes": int(row.horizon_minutes),
            "model": str(row.model),
            "accuracy": float(row.accuracy),
            "wape": float(row.wape),
        }
        for row in summary.groupby("horizon_minutes", as_index=False).first().itertuples()
    }
    sources = {str(horizon): CANDIDATE_SOURCE for horizon in rows}

    if champion_label and champion_horizons:
        champion_frame = metrics.loc[metrics["model"].astype(str) == champion_label]
        champion_metrics = mean_metrics_by_horizon(champion_frame)
        candidate_accuracy = {
            horizon: values["accuracy"] for horizon, values in rows.items()
        }
        reusable = {int(horizon) for horizon in champion_horizons}
        for horizon in horizons_behind_champion(candidate_accuracy, {
            horizon: values["accuracy"] for horizon, values in champion_metrics.items()
        }):
            if horizon not in reusable or horizon not in rows:
                continue
            rows[horizon] = {
                "horizon_minutes": horizon,
                "model": champion_label,
                "accuracy": champion_metrics[horizon]["accuracy"],
                "wape": champion_metrics[horizon]["wape"],
            }
            sources[str(horizon)] = CHAMPION_SOURCE

    best_summary = pd.DataFrame([rows[horizon] for horizon in sorted(rows)])
    best_names = {horizon: rows[horizon]["model"] for horizon in sorted(rows)}
    return best_summary, best_names, sources
