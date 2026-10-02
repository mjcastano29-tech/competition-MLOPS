"""Candidato relativo: transformacion del target y mezcla con el arbol de nivel."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import scripts.infer_and_submit as inf
from scripts.relative_model import (
    RELATIVE_REFERENCE,
    RelativeBlend,
    from_relative,
    relative_model_path,
    relative_target,
    split_candidate_name,
)


def test_la_transformacion_es_reversible():
    y = np.array([0.0, 5.0, 120.0, 2888.0])
    ref = np.array([0.0, 40.0, 100.0, 400.0])
    np.testing.assert_allclose(from_relative(relative_target(y, ref), ref), y)


def test_un_choque_x7_queda_en_rango_para_el_arbol():
    # Para el arbol de nivel 2888 esta fuera de rango si solo vio ~400; en la escala
    # relativa es log(7) desde el corte, un valor acotado.
    assert relative_target([2898.0], [404.0])[0] < 2.0


def test_el_nombre_del_candidato_se_separa():
    assert split_candidate_name("HGB more leaves + Relativo") == ("HGB more leaves", True)
    assert split_candidate_name("HGB more leaves") == ("HGB more leaves", False)


class Const:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def test_la_mezcla_promedia_nivel_y_relativo():
    frame = pd.DataFrame({RELATIVE_REFERENCE: [100.0, 200.0]})
    blend = RelativeBlend(Const(150.0), Const(np.log(1.5)), weight=0.5)
    relative = from_relative(np.full(2, np.log(1.5)), frame[RELATIVE_REFERENCE].to_numpy())
    np.testing.assert_allclose(blend.predict(frame), 0.5 * 150.0 + 0.5 * relative)


def test_la_inferencia_sirve_la_mezcla_del_paquete(tmp_path, monkeypatch):
    stamps = pd.date_range("2026-09-10", periods=4 * 96, freq="15min", tz="UTC")
    obs = pd.DataFrame({"station_id": "A", "observed_at": stamps, "demand": 100.0})
    obs.to_csv(tmp_path / "obs.csv", index=False)
    pd.DataFrame(columns=["observed_at", *inf.CONTEXT_COLUMNS]).to_csv(tmp_path / "ctx.csv", index=False)
    monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "observations", tmp_path / "obs.csv")
    monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "context", tmp_path / "ctx.csv")
    monkeypatch.setenv("BIAS_CORRECTION", "off")
    (tmp_path / "models").mkdir()
    config = {"history_gap_steps": 0, "hgb_weight": 1.0, "feature_columns": [RELATIVE_REFERENCE], "relative_weight": 0.5}
    bundle = {}
    for minutes in (15, 30, 45, 60):
        path = tmp_path / "models" / f"horizon_{minutes}_hgb.pkl"
        path.write_bytes(pickle.dumps(Const(80.0)))
        relative_model_path(path).write_bytes(pickle.dumps(Const(np.log(1.2))))
        bundle[minutes] = (path, {**config, "horizon_minutes": minutes})
    cutoff = stamps[-1]
    cycle = {"cycle_id": "c", "data_cutoff": cutoff.isoformat(), "expected_predictions": 4,
             "targets": [{"station_id": "A", "target_at": (cutoff + pd.Timedelta(minutes=15 * h)).isoformat()} for h in (1, 2, 3, 4)]}
    predictions, report = inf.infer_predictions_with_report(cycle, bundle)
    expected = 0.5 * 80.0 + 0.5 * from_relative(np.log(1.2), 100.0)
    assert report["sources"] == {"campeon": 4}
    np.testing.assert_allclose([p["value"] for p in predictions], round(float(expected), 4))
    # La version del modelo cambia si cambia el arbol relativo.
    assert "relative" not in inf.model_metadata(bundle)["version"]


def test_el_paquete_con_arboles_relativos_carga_un_modelo_por_horizonte(tmp_path, monkeypatch):
    import json

    (tmp_path / "models").mkdir()
    (tmp_path / "configs").mkdir()
    for minutes in (15, 30):
        (tmp_path / "models" / f"horizon_{minutes}_hgb.pkl").write_bytes(pickle.dumps(Const(1.0)))
        (tmp_path / "models" / f"horizon_{minutes}_rel.pkl").write_bytes(pickle.dumps(Const(0.0)))
        (tmp_path / "configs" / f"horizon_{minutes}_ensemble.json").write_text(json.dumps({"relative_weight": 0.5}))
    monkeypatch.setattr(inf, "BUNDLE_DIR", tmp_path)
    bundle = inf.ensure_bundle_ready()
    assert sorted(bundle) == [15, 30]
    assert all(path.name.endswith("_hgb.pkl") for path, _ in bundle.values())


def test_el_candidato_de_memoria_corta_se_reconoce():
    from scripts.relative_model import SHORT_HALF_LIFE_DAYS, parse_candidate

    assert parse_candidate("HGB more leaves + Relativo + Memoria 1d") == ("HGB more leaves", True, SHORT_HALF_LIFE_DAYS)
    assert parse_candidate("HGB more leaves + Relativo") == ("HGB more leaves", True, None)
    assert parse_candidate("HGB shallow") == ("HGB shallow", False, None)
