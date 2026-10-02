"""main() completo ante datos incompatibles: siempre debe salir un POST valido.

Se simula la API (ciclo, stream y submissions) y se rompen Supabase, el paquete o el
formato de los datos. En todos los casos el batch enviado debe responder exactamente los
targets que pidio el ciclo, con valores finitos y no negativos.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import httpx
import numpy as np
import pandas as pd
import pytest

import scripts.infer_and_submit as inf

CUTOFF = pd.Timestamp("2026-09-20T05:00:00Z")


class Fixed:
    def predict(self, frame):
        return np.full(len(frame), 120.0)


def stream_rows(stations, freq="15min", scale=1.0, hours=48):
    stamps = pd.date_range(CUTOFF - pd.Timedelta(hours=hours), CUTOFF, freq=freq)
    rows = []
    for i, s in enumerate(stations):
        for t in stamps:
            rows.append({"station_id": s, "observed_at": t.isoformat().replace("+00:00", "Z"),
                         "demand": float((100 + 10 * i + 30 * np.sin(t.hour / 3)) * scale), "released_at": t.isoformat()})
    return rows


def cycle(stations, horizons=(15, 30, 45, 60), expected=None):
    targets = [{"station_id": s, "target_at": (CUTOFF + pd.Timedelta(minutes=h)).isoformat().replace("+00:00", "Z"), "horizon_minutes": h}
               for s in stations for h in horizons]
    return {"cycle_id": "cyc_test_1", "state": "open", "data_cutoff": CUTOFF.isoformat().replace("+00:00", "Z"),
            "targets": targets, "expected_predictions": len(targets) if expected is None else expected}


@pytest.fixture
def world(tmp_path, monkeypatch):
    sent = {}

    def install(cyc, rows, bundle=True, supabase_receipt_fails=False):
        root = tmp_path / "bundle"
        if bundle:
            (root / "models").mkdir(parents=True)
            (root / "configs").mkdir()
            config = {"history_gap_steps": 0, "hgb_weight": 1.0, "feature_columns": ["is_weekend", "station_id_02300", "station_id_03000"]}
            for m in (15, 30, 45, 60):
                (root / "models" / f"horizon_{m}_hgb.pkl").write_bytes(pickle.dumps(Fixed()))
                (root / "configs" / f"horizon_{m}_ensemble.json").write_text(json.dumps({**config, "horizon_minutes": m}))
        monkeypatch.setattr(inf, "BUNDLE_DIR", root)
        monkeypatch.setattr(inf, "BUNDLE_ZIP", tmp_path / "no.zip")
        monkeypatch.setattr(inf, "COMPATIBILITY_REPORT", tmp_path / "compat.json")
        monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "observations", tmp_path / "data" / "observations.csv")
        monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "context", tmp_path / "data" / "context.csv")
        monkeypatch.setenv("PULSO_API_KEY", "test"); monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test")
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        monkeypatch.setattr(sys, "argv", ["infer", "--output", str(tmp_path / "payload.json")])

        def receipt(_):
            if supabase_receipt_fails:
                raise RuntimeError("Supabase caido")
            return None
        monkeypatch.setattr(inf, "find_confirmed_submission", receipt)
        monkeypatch.setattr(inf, "persist_confirmed_predictions", lambda *a: True)
        monkeypatch.setattr(inf, "persist_compatibility", lambda *a: None)

        def supabase_down(*a, **k):  # el colector y Supabase rotos: solo queda el stream
            raise subprocess.CalledProcessError(1, "download_supabase_data.py")
        monkeypatch.setattr(inf.subprocess, "run", supabase_down)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/forecast-cycles/current":
                return httpx.Response(200, json=cyc)
            if request.url.path == "/v1/stream/observations":
                return httpx.Response(200, json={"data": rows, "next_cursor": None})
            if request.url.path == "/v1/submissions" and request.method == "POST":
                sent["payload"] = json.loads(request.content)
                return httpx.Response(201, json={"submission_id": "sub_1", "status": "accepted", "is_official": True})
            return httpx.Response(404, json={})

        real_client = httpx.Client
        monkeypatch.setattr(inf.httpx, "Client", lambda *a, **k: real_client(*a, transport=httpx.MockTransport(handler), **k))
        return sent
    return install


def assert_valid_batch(sent, cyc):
    payload = sent["payload"]
    got = {(p["station_id"], p["target_at"]) for p in payload["predictions"]}
    want = {(t["station_id"], t["target_at"]) for t in cyc["targets"]}
    assert got == want and len(payload["predictions"]) == len(cyc["targets"])
    assert all(np.isfinite(p["value"]) and p["value"] >= 0 for p in payload["predictions"])


BASE = ["02300", "03000"]


@pytest.mark.parametrize("name,cyc,rows,kwargs", [
    ("compatible", cycle(BASE), stream_rows(BASE), {}),
    ("estacion y horizonte nuevos", cycle(BASE + ["11020"], horizons=(15, 30, 45, 60, 75)), stream_rows(BASE + ["11020"]), {}),
    ("estacion nueva sin historial", cycle(BASE + ["99999"]), stream_rows(BASE), {}),
    ("ids con otro formato", cycle(["2300", "3000"]), stream_rows(["2300", "3000"]), {}),
    ("escala x1000", cycle(BASE), stream_rows(BASE, scale=1000.0), {}),
    ("frecuencia de 30 min", cycle(BASE), stream_rows(BASE, freq="30min"), {}),
    ("paquete del modelo ausente", cycle(BASE), stream_rows(BASE), {"bundle": False}),
    ("conteo declarado distinto", cycle(BASE, expected=99), stream_rows(BASE), {}),
    ("supabase caido para el recibo", cycle(BASE), stream_rows(BASE), {"supabase_receipt_fails": True}),
    ("stream con otro esquema", cycle(BASE), [{"station": "x", "ts": 1}], {}),
])
def test_siempre_se_envia_un_batch_valido(world, name, cyc, rows, kwargs):
    sent = world(cyc, rows, **kwargs)
    assert inf.main() == 0, name
    assert_valid_batch(sent, cyc)


def test_un_fallo_inesperado_en_la_inferencia_usa_la_red_de_emergencia(world, monkeypatch):
    cyc = cycle(BASE)
    sent = world(cyc, stream_rows(BASE))
    monkeypatch.setattr(inf, "infer_predictions_with_report", lambda *a, **k: (_ for _ in ()).throw(KeyError("columna nueva")))
    assert inf.main() == 0
    assert_valid_batch(sent, cyc)
    assert sent["payload"]["predictions"][0]["value"] > 1  # persistencia real, no la constante
