"""Candidato "relativo": un segundo arbol que predice el cambio frente al nivel del corte.

El arbol de nivel predice la demanda y no puede salir del rango de valores que vio al
entrenar. Este predice `log((y + OFFSET) / (visible_lag_0 + OFFSET))`, la razon contra la
demanda publicada en el corte, y separa "cuanto cambia" de "en que nivel esta" la
estacion. Por si solo pierde en el regimen de ondas (se ancla a picos y valles), pero
promediado 50/50 con el de nivel es robusto: backtest walk-forward de 7 dias, +0.8 pts en
dias normales (peor estacion -0.09) y neutro en el regimen de ondas (+0.01).

Lo usan el entrenamiento (examples/03 y 04) y la inferencia: un solo codigo garantiza la
misma transformacion en los dos extremos.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

RELATIVE_REFERENCE = "visible_lag_0"
RELATIVE_OFFSET = 10.0
RELATIVE_WEIGHT = 0.5
RELATIVE_SUFFIX = " + Relativo"
# Solo la configuracion validada en el backtest: cada configuracion relativa suma ~6 min
# al reentrenamiento, y con dos el job rozaba su limite.
RELATIVE_CONFIGS = ("HGB more leaves",)


def relative_target(target: pd.Series | np.ndarray, reference: pd.Series | np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return np.log((target + RELATIVE_OFFSET) / (np.clip(reference, 0, None) + RELATIVE_OFFSET))


def from_relative(prediction: np.ndarray, reference: pd.Series | np.ndarray) -> np.ndarray:
    reference = np.clip(np.asarray(reference, dtype=float), 0, None)
    return (reference + RELATIVE_OFFSET) * np.exp(np.asarray(prediction, dtype=float)) - RELATIVE_OFFSET


def split_candidate_name(model_name: str) -> tuple[str, bool]:
    """`("HGB more leaves", True)` para "HGB more leaves + Relativo"."""

    if model_name.endswith(RELATIVE_SUFFIX):
        return model_name[: -len(RELATIVE_SUFFIX)], True
    return model_name, False


class RelativeBlend:
    """Promedia el arbol de nivel con el relativo; misma interfaz `predict(frame)`.

    El resto del pipeline (baseline estacional, persistencia, AR, correccion de sesgo)
    lo trata como un modelo mas.
    """

    def __init__(self, level: Any, relative: Any, weight: float = RELATIVE_WEIGHT) -> None:
        self.level = level
        self.relative = relative
        self.weight = float(weight)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        level = np.asarray(self.level.predict(frame), dtype=float)
        if RELATIVE_REFERENCE not in frame.columns:
            raise KeyError(f"El candidato relativo necesita la columna {RELATIVE_REFERENCE!r}.")
        relative = from_relative(self.relative.predict(frame), frame[RELATIVE_REFERENCE].to_numpy())
        return (1 - self.weight) * level + self.weight * relative


def relative_model_path(model_path: Any) -> Any:
    """Ruta del arbol relativo junto al de nivel: horizon_15_hgb.pkl -> horizon_15_rel.pkl."""

    return model_path.with_name(model_path.name.replace("_hgb.pkl", "_rel.pkl"))
