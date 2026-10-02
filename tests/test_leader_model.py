"""Estacion lider: solo se activa cuando una estacion de verdad copia a otra."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.leader_model import leader_forecasts

STAMPS = pd.date_range("2026-09-18", periods=96, freq="15min", tz="UTC")


def frame(series: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat(
        pd.DataFrame({"station_id": s, "observed_at": STAMPS, "demand": v}) for s, v in series.items()
    )


def wave(shift=0):
    t = np.arange(len(STAMPS)) - shift
    return 300 * np.exp(1.2 * np.sin(2 * np.pi * t / 16) + 0.4 * np.sin(2 * np.pi * t / 37))


def test_una_estacion_que_copia_a_otra_se_pronostica_con_su_lider():
    # B es A x 0.5, una hora (4 cuartos) despues.
    obs = frame({"A": wave(), "B": 0.5 * wave(shift=4), "C": np.random.default_rng(1).uniform(50, 150, 96)})
    cutoff = STAMPS[-5]
    out = leader_forecasts(obs[obs.observed_at <= cutoff], cutoff, ["A", "B", "C"])
    assert "B" in out and out["B"]["leader"] == "A" and out["B"]["lag_quarters"] == 4
    expected = 0.5 * wave(shift=4)[-4:]
    np.testing.assert_allclose(out["B"]["forecast"], expected, rtol=1e-6)
    assert "C" not in out  # ruido: nadie la anticipa


def test_series_planas_no_activan_el_lider():
    obs = frame({"A": np.full(96, 100.0), "B": np.full(96, 100.0)})
    assert leader_forecasts(obs, STAMPS[-1], ["A", "B"]) == {}


def test_estaciones_parecidas_pero_no_copias_no_activan_el_lider():
    rng = np.random.default_rng(7)
    base = 200 + 80 * np.sin(2 * np.pi * np.arange(96) / 96)
    obs = frame({s: base * rng.lognormal(0, 0.25, 96) for s in ("A", "B", "C")})
    assert leader_forecasts(obs, STAMPS[-1], ["A", "B", "C"]) == {}
