"""Demanda del stream en los esquemas 1 y 2 (measurement)."""

import httpx
import pandas as pd

import scripts.infer_and_submit as inf
from scripts.stream_schema import row_demand

V2 = {"station_id": "02300", "observed_at": "2026-09-20T12:15:00Z", "released_at": "2026-10-03T23:19:15Z",
      "schema_version": 2, "measurement": {"value": "546.00", "unit": "passengers", "quality": "observed"}}


def test_esquema_1_y_2():
    assert row_demand({"demand": 360}) == 360.0
    assert row_demand(V2) == 546.0
    assert row_demand({**V2, "demand": None}) == 546.0


def test_unidades_y_calidad():
    assert row_demand({"measurement": {"value": "1.5", "unit": "thousand_passengers"}}) == 1500.0
    assert row_demand({"measurement": {"value": "12", "unit": "missing"}}) == 12.0  # unidad rara: avisa y usa el valor
    assert row_demand({"measurement": {"value": "100", "unit": "passengers", "quality": "missing"}}) is None
    assert row_demand({"measurement": {"value": "", "unit": "passengers"}}) is None
    assert row_demand({"demand": None}) is None


def test_la_inferencia_lee_el_esquema_2_del_stream(tmp_path, monkeypatch):
    monkeypatch.setitem(inf.SAMPLE_DATA_PATHS, "observations", tmp_path / "obs.csv")
    rows = [V2, {**V2, "station_id": "03000", "measurement": {"value": "61.00", "unit": "passengers", "quality": "observed"}},
            {**V2, "station_id": "05000", "measurement": {"value": "9", "unit": "passengers", "quality": "missing"}}]

    def handler(request):
        return httpx.Response(200, json={"data": rows, "next_cursor": None})

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api") as client:
        inf.refresh_observations_from_stream(client, pd.Timestamp("2026-09-20T13:00Z"))
    saved = pd.read_csv(tmp_path / "obs.csv", dtype={"station_id": str})
    assert dict(zip(saved.station_id, saved.demand)) == {"02300": 546.0, "03000": 61.0}
